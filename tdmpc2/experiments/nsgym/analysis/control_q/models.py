"""Migrated historical computations; numerical formulas preserved."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
from experiments.nsgym.analysis.control_q.config import GRID, TASKS
from q_planning_core import compatible
from analyze_q_planning import make_model
from experiments.nsgym.analysis.control_q.metrics import read, require, sha256


class Sources:
    def __init__(self): self.files = {}; self.inventory = []
    def track(self, path, category, expected=None, required=True):
        path=Path(path).resolve()
        if not path.is_file():
            self.inventory.append(dict(path=str(path),category=category,status='MISSING',required=required))
            if required: raise FileNotFoundError(str(path))
            return path
        if str(path) not in self.files:
            self.files[str(path)]=dict(sha256=sha256(path),size=path.stat().st_size)
            self.inventory.append(dict(path=str(path),category=category,status='PRESENT',required=required,**self.files[str(path)]))
        if expected: require(self.files[str(path)]['sha256']==expected, f'Source SHA mismatch: {path}')
        return path
    def verify(self):
        return [p for p,r in self.files.items() if not Path(p).is_file() or sha256(Path(p))!=r['sha256']]


def snapshot_paths(root, task, family, k, seed):
    p=TASKS[task]['families'][family]; suffix=f'k{k:g}'
    if task=='halfcheetah':
        return {'D':root/f'dynamics-only-750k-v1/runs/fixed-decision/{p}-{suffix}-decision-seed{seed}/snapshot-0003000.pt',
         'DQ':root/f'dynamics-q-750k-v1/runs/fixed-decision/{p}-{suffix}-dynamics-q-seed{seed}/snapshot-0003000.pt',
         'J':root/f'full-joint-750k-v1/runs-fixed/{p}-{suffix}-full-joint-seed{seed}/snapshot-0003000.pt'}
    return {'D':root/f'reacher/preview-v1/adapt-home-01/{family}-{suffix}-decision-seed{seed}/snapshot-0010000.pt',
     'DQ':root/f'reacher-dynamics-q-v1/runs/{family}-{suffix}-dynamics-q-seed{seed}/snapshot-0010000.pt',
     'J':root/f'reacher/preview-v1/full-joint-home-01/runs-fixed/{p}-{suffix}-full-joint-seed{seed}/snapshot-0010000.pt'}


def inventory(root, src, smoke):
    """Inventory all requested streams/banks before any forward call."""
    jobs=[]; gaps=[]; configs={}; banks={}
    for task,t in TASKS.items():
        try:
            cp=root/t['checkpoint']
            src.track(cp/'model.pt','nominal checkpoint',t['sha'])
            cfg=read(src.track(cp/'metadata.json','model configuration'))['config']
            require(not cfg['episodic'] and not cfg['multitask'],'Only saved non-episodic single-task contracts supported')
            require(cfg['task']==f'nsgym-{task}', 'Checkpoint task mismatch')
            if task=='halfcheetah': require(cfg['horizon']==3,'HalfCheetah horizon must be 3')
            configs[task]=cfg
            if task=='halfcheetah':
                base=root/'physical-severity-750k-v1/frozen'
                integ=read(src.track(base/'model_integrity.json','Frozen bank provenance'))
                require(integ['file_sha256']['model.pt']==t['sha'],'Frozen bank checkpoint mismatch')
                require(read(src.track(base/'completion.json','Frozen completion'))['status']=='PASS','Incomplete Frozen bank')
                manifest=read(src.track(base/'manifest.json','Frozen configuration'))
                require(manifest['seeds']==t['evaluation'] and manifest['seed_split']=='evaluation','Frozen seed split')
                for r in read(src.track(base/'dataset_index.json','bank index')):
                    c=r['condition']; require(c['schedule']=='fixed','Bank schedule mismatch')
                    banks[(task,c['family'],float(c['severity']),int(r['seed']))]=(base/r['file'],r['sha256'])
            else:
                base=root/'reacher/preview-v1/frozen-home-01'
                frozen_manifest=read(src.track(base/'manifest.json','Frozen bank provenance'))
                require(frozen_manifest['checkpoint_sha256']==t['sha'],'Reacher Frozen bank origin mismatch')
                require(frozen_manifest['config']['evaluation_seeds']==t['evaluation'],'Reacher Frozen evaluation seeds mismatch')
                require(read(src.track(base/'completion.json','Frozen completion'))['status']=='PASS','Incomplete Reacher Frozen bank')
                selected=read(src.track(root/'reacher/preview-v1/nominal-home-01/selected_reacher_checkpoint.json','checkpoint selection'))
                require(selected['model_sha256']==t['sha'],'Reacher selection mismatch')
                for r in pd.read_csv(src.track(base/'reacher_frozen_severity.csv','bank index')).itertuples():
                    banks[(task,r.family,float(r.severity),int(r.seed))]=(base/r.data_file,r.data_sha256)
            for family in t['families']:
                for k in GRID:
                    for seed in t['adaptation']:
                        selected=not smoke or (k==2 and family in ('damping','mass','D0','M0') and seed==t['adaptation'][0])
                        paths=snapshot_paths(root,task,family,k,seed); valid=True
                        try:
                            for method,p in paths.items():
                                m=read(src.track(p.parent/'manifest.json','adaptation manifest'))
                                d=read(src.track(p.parent/'completion.json','adaptation completion'))
                                require(d['status']=='PASS',f'Incomplete {p.parent}')
                                i=m['identity']
                                require(i['parameter']==t['families'][family] and float(i['severity'])==k and int(i['adaptation_seed'])==seed,'Identity mismatch')
                                protocol=m.get('protocol',m.get('configuration',m.get('config')))
                                require(protocol['adaptation_seeds']==t['adaptation'] and protocol['evaluation_seeds']==t['evaluation'],'Adaptation/evaluation seed split mismatch')
                                origin=m.get('checkpoint_sha256',m.get('selection',{}).get('model_sha256'))
                                require(origin==t['sha'],'Adaptation nominal checkpoint mismatch')
                                src.track(p,'adaptation snapshot',d.get('snapshot_sha256'))
                            for e in t['evaluation']:
                                path,h=banks[(task,family,k,e)];src.track(path,'Frozen held-out bank',h)
                        except (OSError,KeyError,ValueError) as exc:
                            valid=False;gaps.append(dict(task=task,family=family,severity=k,adaptation_seed=seed,reason=str(exc)))
                        if selected: jobs.append(dict(task=task,family=family,severity=k,adaptation_seed=seed,paths=paths,valid=valid))
        except (OSError,KeyError,ValueError) as exc: gaps.append(dict(task=task,reason=str(exc)))
    # Historical archives and summaries are inventoried, not silently substituted.
    for pattern,category in [
        ('q-planning-mechanism-750k-v1/immediate/*.npz','existing immediate archive'),
        ('q-planning-mechanism-750k-v1/terminal/*.npz','existing terminal archive'),
        ('unified-analysis/control-prediction-q-corrected-v3/q_inference/*.npz','existing Reacher Q archive'),
        ('unified-analysis/*/canonical_records.csv','canonical summary'),
        ('dynamics-q-750k-v1/comparison/*.csv','HalfCheetah prediction/control summary'),
        ('reacher-dynamics-q-v1/comparison/*.csv','Reacher prediction/control summary')]:
        found=list(root.glob(pattern))
        if not found:src.inventory.append(dict(path=str(root/pattern),category=category,status='MISSING',required=False))
        for p in found:src.track(p,category,required=False)
    return jobs,configs,banks,gaps


def tree_hash(value):
    """training_support.tree_hash byte contract without importing its env stack."""
    import torch
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            x=v.detach().cpu().contiguous()
            h.update(str((x.dtype,x.shape)).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def load_models(root, job, cfg, src, device):
    import torch
    task=job['task'];t=TASKS[task]
    nominal=torch.load(root/t['checkpoint']/'model.pt',map_location='cpu',weights_only=False)['model']
    models={'F':make_model(cfg,nominal,device)}
    for m,p in job['paths'].items():
        payload=torch.load(p,map_location='cpu',weights_only=False)
        i=payload['identity']
        require(i['parameter']==t['families'][job['family']] and float(i['severity'])==job['severity'] and int(i['adaptation_seed'])==job['adaptation_seed'],'Payload identity mismatch')
        require(payload.get('checkpoint_sha256',payload.get('selection',{}).get('model_sha256'))==t['sha'],'Payload nominal origin mismatch')
        done=read(p.parent/'completion.json')
        if m=='D':
            require(payload['adapter']['counters'][1]==t['budget'],'Wrong final D snapshot budget')
            require(payload['adapter']['counters'][3]==(118 if task=='halfcheetah' else 398),'Wrong D updates')
            require(payload['model_hash']==done.get('final_model_hash',done.get('model_hash')),'D saved final model identity')
        elif m=='J': require(payload['transitions']==t['budget'],'Wrong Joint snapshot budget')
        else: require(done['transitions']==t['budget'],'Wrong DQ snapshot budget')
        models[m]=make_model(cfg,nominal if m=='D' else payload['model'],device)
        if m=='D': models[m]._dynamics.load_state_dict(payload['adapter']['predictor'],strict=True)
        expected=payload.get('model_hash') if m=='D' else done.get('final_model_hash')
        if expected:require(tree_hash(models[m].state_dict())==expected,'Reconstructed model differs from recorded final state')
    return models,compatible(models)


def track_implementation(src):
    """Hash the whole migrated implementation, not only its small CLI shim."""
    for path in Path(__file__).parent.rglob('*.py'):
        src.track(path, 'control_q implementation')
