"""Audit and execute the fixed Stage 3B-2 matrix using the existing Stage 3B runner."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import uuid
from stage3b2_audit import (ROOT, MATRIX, BUDGETS, expected_config, discover, audit_candidate,
                            check_dataset, digest_file, value_label, read)


def command(cfg, dataset, destination, resume=None, collect=False):
    p=cfg['dynamics_shift']['parameter']
    args=[sys.executable,'-u',str(ROOT/'tdmpc2/adapt_shift.py'),'--config-name','stage3b',
          'task=pick-cube','model_size=5','episodic=false',
          'checkpoint='+json.dumps(cfg['checkpoint']),f"dynamics_shift={p['type']}",'schedule=fixed',
          f"dynamics_shift.parameter.shifted={p['shifted']}",
          'stage3a.mode='+('frozen' if collect else 'dynamics_only'),'stage3a.seed_set=0',
          'stage3a.fixed_data_path='+json.dumps(str(dataset)),
          'stage3a.output='+json.dumps(cfg['stage3a']['output']),
          'stage3b.output_dir='+json.dumps(str(destination)),
          'stage3b.run_id='+json.dumps(cfg['stage3b']['run_id'])]
    if collect:args.append('stage3a.fixed_data_collect=true')
    if resume:args.append('stage3b.resume='+json.dumps(str(resume)))
    return args


def plan(root, checkpoint, run_id):
    root=Path(root).resolve();checkpoint=Path(checkpoint).resolve()
    checksum=digest_file(checkpoint)
    unclassified=[]
    found=discover(root,unclassified);entries=[]
    for kind,values in MATRIX.items():
        for value in values:
            cfg=expected_config(kind,value,checkpoint)
            candidates=[audit_candidate(p,cfg,checksum) for p in found.get((kind,float(value)),[])]
            complete=[x for x in candidates if x['status']=='REUSE_COMPLETE']
            partial=sorted((x for x in candidates if x['status']=='RESUME_PARTIAL'),key=lambda x:x['budget'],reverse=True)
            selected=(complete or partial or [None])[0]
            target=root/'runs'/kind/'set-0'/'dynamics_only'/value_label(value)
            status=selected['status'] if selected else ('INCOMPATIBLE' if candidates or target.exists() else 'RUN_MISSING')
            entry=dict(shift_type=kind,shift_value=value,set_id=0,status=status,candidates=candidates,
                       result_path=selected['path'] if complete else None,commands=[],requested_config=cfg)
            if not complete:
                # Each fresh condition starts from the nominal pretrained checkpoint.
                # Every continuation gets a new destination and keeps its original paths.
                if selected:
                    source=read(Path(selected['path'])/'metadata.json')['config']
                    original=Path(source['checkpoint'])
                    cfg['checkpoint']=str(original)
                    dataset=source['stage3a']['fixed_data_path']
                    target=target/'versions'/run_id
                    entry['resume']=selected['resume']
                    entry['source_path']=selected['path']
                else:
                    dataset=None
                    # Reuse only independently verified fixed datasets of this severity.
                    choices=[root/'data'/kind/f'set-0.pt']
                    choices+=sorted((root/'data'/'stage3b2'/kind/value_label(value)).glob('*.pt'))
                    for candidate in choices:
                        if candidate.is_file():
                            try:check_dataset(candidate,cfg,checksum)
                            except Exception:continue
                            dataset=str(candidate);break
                    if target.exists():target=target/'versions'/run_id
                    if dataset is None:
                        dataset=str(root/'data'/'stage3b2'/kind/value_label(value)/(run_id+'.pt'))
                        entry['commands'].append(command(cfg,dataset,target,collect=True))
                if target.exists():raise FileExistsError(f'New destination already exists: {target}; use a new run-id.')
                entry.update(target=str(target),dataset=str(dataset),action='RESUME' if selected else 'START_NOMINAL')
                entry['commands'].append(command(cfg,dataset,target,entry.get('resume')))
            else:entry['action']='REUSE'
            entries.append(entry)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    from common.severity_protocol import source_digest
    return dict(protocol='stage3b-2-v1',run_id=run_id,root=str(root),checkpoint=str(checkpoint),
                checkpoint_sha256=checksum,code_revision=revision,source_sha256=source_digest(ROOT),
                budgets=BUDGETS,seed_sets=[0],unclassified_metadata=unclassified,entries=entries)


@contextmanager
def execution_lock(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.stage3b2.lock').open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another Stage 3B-2 launcher holds the root lock.')
        stream.seek(0);stream.truncate();stream.write(f'{socket.gethostname()} pid={os.getpid()}\n');stream.flush()
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def active_runs_guard():
    # Linux server: also catch legacy launches on another GPU in this repository.
    # Existing jobs are never killed or reassigned.
    processes=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
    for line in processes.splitlines():
        parts=line.strip().split(None,1)
        if len(parts)!=2 or int(parts[0])==os.getpid():continue
        if not any(name in parts[1] for name in ('tdmpc2/adapt_shift.py','scripts/run_stage3b.py')):continue
        cwd=Path('/proc')/parts[0]/'cwd'
        try:same_repo=cwd.resolve(strict=True)==ROOT
        except OSError:same_repo=str(ROOT) in parts[1]
        if same_repo:raise RuntimeError('Existing Stage 3 adaptation process in this repository: '+line.strip())


def gpu_guard():
    """No allocation changes; refuse ambiguous visibility or an occupied GPU."""
    import torch
    active_runs_guard()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one allocated CUDA GPU with CUDA_VISIBLE_DEVICES; no GPU was selected automatically.')
    uuid_value=getattr(torch.cuda.get_device_properties(0),'uuid',None)
    if uuid_value is None:raise RuntimeError('Cannot identify the visible GPU UUID with this PyTorch version.')
    selected=str(uuid_value)
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    matches=[x.strip() for x in devices if x.strip().removeprefix('GPU-')==selected.removeprefix('GPU-')]
    if len(matches)!=1:raise RuntimeError(f'Cannot map visible GPU UUID {selected}; refusing an ambiguous allocation.')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                    '--format=csv,noheader'],text=True)
    busy=[line for line in active.splitlines() if line.split(',')[0].strip()==matches[0]
          and line.split(',')[1].strip()!=str(os.getpid())]
    if busy:raise RuntimeError('Allocated GPU already has compute processes: '+'; '.join(busy))
    return dict(visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_uuid=matches[0],processes=active)


def save(path, payload):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2));temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=ROOT/'results/stage3b-generalization')
    parser.add_argument('--run-id',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    parser.add_argument('--execute',action='store_true',help='Default: audit only; no simulator launched.')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id):parser.error('Unsafe run-id')
    os.chdir(ROOT)
    # The lock covers audit too: another invocation cannot start between audit and execution.
    with execution_lock(args.root.resolve()):
        manifest=plan(args.root,args.checkpoint,args.run_id)
        report_root=args.root.resolve()/'ttt_severity'/args.run_id
        report_root.mkdir(parents=True,exist_ok=False)
        manifest_path=report_root/'manifest.json'
        save(manifest_path,manifest)
        for entry in manifest['entries']:
            print(f"{entry['status']:16} {entry['shift_type']}={entry['shift_value']} set-0  {entry['action']}",flush=True)
            for candidate in entry['candidates']:
                print('  '+candidate['path']+': '+candidate['status']+' — '+candidate['reason'],flush=True)
            for cmd in entry['commands']:print(shlex.join(cmd),flush=True)
        print(f'Manifest: {manifest_path}',flush=True)
        failure=None
        try:
            if args.execute:
                for entry in manifest['entries']:
                    if entry['action']=='REUSE':continue
                    entry['gpu']=gpu_guard();save(manifest_path,manifest)
                    for index,cmd in enumerate(entry['commands']):
                        log=report_root/f"{entry['shift_type']}-{value_label(entry['shift_value'])}-{index}.log"
                        if index:entry['gpu']=gpu_guard()
                        with log.open('x') as stream:
                            print(f'RUN {shlex.join(cmd)}\nLog: {log}',flush=True)
                            subprocess.run(cmd,cwd=ROOT,check=True,stdout=stream,stderr=subprocess.STDOUT)
                    result=audit_candidate(Path(entry['target'])/'metadata.json',entry['requested_config'],manifest['checkpoint_sha256'])
                    entry['execution_audit']=result
                    if result['status']!='REUSE_COMPLETE':raise RuntimeError('Post-run verification failed: '+result['reason'])
                    entry.update(result_path=entry['target'],execution_status='COMPLETE')
                    save(manifest_path,manifest)
        except (Exception,KeyboardInterrupt) as exc:
            failure=exc;manifest['execution_error']=f'{type(exc).__name__}: {exc}'
            if 'entry' in locals() and entry.get('target') and (Path(entry['target'])/'metadata.json').is_file():
                audit=audit_candidate(Path(entry['target'])/'metadata.json',entry['requested_config'],manifest['checkpoint_sha256'])
                entry['execution_audit']=audit
                if audit['status']=='RESUME_PARTIAL':
                    entry['source_path']=audit['path'];entry['candidates'].append(audit)
                elif audit['status']=='REUSE_COMPLETE':entry['result_path']=audit['path']
        finally:
            save(manifest_path,manifest)
            from analyze_stage3b2 import report
            try:report(manifest_path,report_root/'report')
            except Exception as exc:
                manifest['report_error']=f'{type(exc).__name__}: {exc}';save(manifest_path,manifest)
                if failure is None:raise
        if failure:raise failure


if __name__=='__main__':main()
