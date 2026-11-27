#!/usr/bin/env python3
"""Final 3000-step Decision TTT versus existing Frozen; separate prediction/control/retention."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_dynamics_only_ttt as t

def read(p):return json.loads(Path(p).read_text())
def save(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def csv_rows(p,records):
    if not records:raise ValueError(f'Empty CSV: {p}')
    keys=list(dict.fromkeys(k for r in records for k in r))
    with Path(p).open('x',newline='') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(records)
def iter_csv(p):
    with Path(p).open() as f:yield from csv.DictReader(f)
def record_key(i):return (i['parameter'],float(i['severity']),int(i['adaptation_seed']))
def label(i):return f"{i['parameter']}-k{float(i['severity']):g}-decision-seed{i['adaptation_seed']}"
def conditions(c):return [(x['parameter'],float(x['severity'])) for x in c['conditions']]
class ReferenceCheckNeeded(AssertionError):pass
def run_manifest(root,method,expected,c):
    stage=Path(root)/f'fixed-{method}'
    assert read(stage/'completion.json')['status']=='PASS',f'{stage} not complete'
    manifest={};done={};paths={}
    for p in sorted(stage.iterdir()):
        if not p.is_dir():continue
        m=read(p/'manifest.json');d=read(p/'completion.json');i=m['identity'];key=record_key(i)
        assert i['method']==method and i['schedule']=='fixed' and i['synthetic'] is False
        assert (i['parameter'],float(i['severity'])) in conditions(c)
        assert i['adaptation_seed'] in c['adaptation_seeds'] and key not in manifest
        assert d['status']=='PASS' and d['identity']==i
        assert m['checkpoint_sha256']==expected and m['configuration']==c
        assert d['metrics']['adaptation_transitions']==3000 and d['metrics']['adaptation_episodes']==3
        assert d['initial_fixed_hash']==d['final_fixed_hash']
        assert d['auxiliary_hash']==m['initial_auxiliary_hash']
        assert d['finite_losses']
        assert d['initial_model_hash']==m['initial_model_hash']
        if method=='frozen':
            assert d['metrics']['gradient_update_steps']==0 and d['initial_model_hash']==d['final_model_hash']
        else:
            assert d['metrics']['gradient_update_steps']==118 and d['initial_model_hash']!=d['final_model_hash']
        final=p/'snapshot-0003000.pt';assert final.is_file()
        assert len(list(p.glob('snapshot-*.pt')))==5
        manifest[key]=m;done[key]=d;paths[key]=p
    expected_keys={(p,k,s) for p,k in conditions(c) for s in c['adaptation_seeds']}
    assert set(manifest)==expected_keys,f'{method} run coverage differs: {len(manifest)} vs {len(expected_keys)}'
    return manifest,done,paths

def check_steps(path,method):
    expected_updates={x:(x-50)//25 for x in range(75,3001,25)}
    count=0;updates=0;wall=0.;first=None;last=None
    for row in iter_csv(path/'online_steps.csv'):
        count+=1;step=int(row['adaptation_steps']);assert step==count
        expected=1 if method=='decision' and step in expected_updates else 0
        assert int(row['updates_this_transition'])==expected,(path,step)
        updates+=expected
        assert int(row['gradient_updates'])==updates
        assert int(row['episode'])==(step-1)//1000
        assert int(row['t'])==(step-1)%1000
        assert float(row['transition_seconds'])>=0
        wall+=float(row['transition_seconds'])
        if expected:
            import math
            assert row['training_loss'] and row['gradient_norm']
            assert math.isfinite(float(row['training_loss'])) and math.isfinite(float(row['gradient_norm']))
            if first is None:first=step
            last=step
    assert count==3000 and updates==(118 if method=='decision' else 0)
    assert first==(75 if method=='decision' else None)
    assert last==(3000 if method=='decision' else None)
    return dict(steps=count,updates=updates,first_update=first,last_update=last,online_transition_seconds=wall)

def static_sources(frozen_metadata,manifest):
    old=frozen_metadata.get('source_hashes',{});new=manifest.get('source_hashes',{})
    checked=[]
    for suffix in ('tdmpc2/tdmpc2.py','tdmpc2/common/world_model.py','tdmpc2/envs/nsgym/severity.py'):
        left=[v for k,v in old.items() if k.endswith(suffix)]
        right=[v for k,v in new.items() if k.endswith(suffix)]
        if not (len(left)==len(right)==1 and left[0]==right[0]):
            raise ReferenceCheckNeeded(f'Source hash mismatch {suffix}')
        checked.append(suffix)
    return checked

def baseline(reference,checkpoint_sha,c):
    base=Path(reference);frozen=base/'frozen';latent=base/'heldout-latent'
    fm=read(frozen/'manifest.json');fi=read(frozen/'model_integrity.json');fc=read(frozen/'completion.json');fmeta=read(frozen/'metadata.json')
    lm=read(latent/'manifest.json');lc=read(latent/'completion.json');lp=read(latent/'dataset_provenance.json')
    assert fc['status']==lc['status']=='PASS'
    assert fm['stage']=='frozen' and lm['stage']=='latent'
    assert fm['expected_checkpoint_sha256']==lm['expected_checkpoint_sha256']==fi['file_sha256']['model.pt']==checkpoint_sha
    assert fi['optimizer_updates']==fm['optimizer_updates']==0
    assert fm['seed_split']=='evaluation' and fm['seeds']==lm['seeds']==c['evaluation_seeds']
    assert fm['horizon']==lm['horizon']==c['horizon']==1000
    assert fm['clock']=='episode_local_pre_action_zero_based' and fm['persistent_params'] is False
    assert fm['policy']=='primary' and not fm['smoke']
    assert fm['schedule_timing']['onset']==c['onset']==250
    assert all(x['schedule']=='fixed' for x in fm['conditions'])
    assert sorted((x['parameter'],float(x['severity'])) for x in fm['conditions'])==sorted(conditions(c))
    assert digest(frozen/'dataset_index.json')==lp['index_sha256']
    assert fi['checkpoint_metadata']['model_sha256']==checkpoint_sha
    assert lp['observation_transform']=='unchanged state vector -> float32; no additional normalization'
    episodes={};source_files={}
    for r in iter_csv(frozen/'episodes.csv'):
        key=(r['parameter'],float(r['severity']),int(r['seed']))
        assert key not in episodes and r['schedule']=='fixed' and int(r['episode_length'])==1000
        assert r['terminated']=='False' and r['truncated']=='True'
        episodes[key]=r;source_files[key]=r['data_file']
    expected_keys={(p,k,s) for p,k in conditions(c) for s in c['evaluation_seeds']}
    assert set(episodes)==expected_keys and len(episodes)==240
    index=read(frozen/'dataset_index.json');filehash={x['file']:x['sha256'] for x in index}
    assert set(source_files.values())==set(filehash)
    assert all(lp['data_hashes'][file]==filehash[file] for file in filehash)
    values={};counts={}
    for r in iter_csv(latent/'latent_errors.csv'):
        if int(r['horizon'])!=1:continue
        key=(r['parameter'],float(r['severity']),int(r['seed']))
        assert r['schedule']=='fixed' and r['source_file']==source_files[key]
        values[key]=values.get(key,0.)+float(r['latent_mse'])
        counts[key]=counts.get(key,0)+1
    assert set(counts)==expected_keys and set(counts.values())=={1000}
    means={key:values[key]/1000 for key in values}
    return dict(frozen_manifest=fm,frozen_integrity=fi,frozen_metadata=fmeta,latent_manifest=lm,latent_provenance=lp,episodes=episodes,files=source_files,hashes=filehash,mse=means)

def audit(a,c,allow_source_mismatch=False):
    frozen,fd,fp=run_manifest(a.runs,'frozen',t.SHA,c)
    decision,dd,dp=run_manifest(a.runs,'decision',t.SHA,c)
    assert set(frozen)==set(decision)
    assert len({m['initial_model_hash'] for m in frozen.values()})==1
    assert len({m['initial_model_hash'] for m in decision.values()})==1
    assert {m['initial_model_hash'] for m in frozen.values()}=={m['initial_model_hash'] for m in decision.values()}
    baseline_data=baseline(a.reference,t.SHA,c)
    reference_config=baseline_data['frozen_integrity']['checkpoint_metadata']['config']
    expected_versions=baseline_data['frozen_metadata']['versions']
    checks=[]
    for key in sorted(decision):
        fm=frozen[key];dm=decision[key];f=fd[key];d=dd[key]
        assert fm['checkpoint_metadata']['config']==dm['checkpoint_metadata']['config']==reference_config
        assert fm['checkpoint_metadata']['model_sha256']==dm['checkpoint_metadata']['model_sha256']==t.SHA
        assert fm['checkpoint_metadata']['optimizer_sha256']==dm['checkpoint_metadata']['optimizer_sha256']==baseline_data['frozen_integrity']['file_sha256']['optimizers.pt']
        assert fm['versions']==dm['versions']
        for pkg in ('ns_gym','gymnasium','mujoco'):
            assert dm['versions'][pkg]==expected_versions[pkg]
        # A later runner revision added resume bookkeeping. Compare the
        # algorithm, environment and config sources, not that CLI file.
        for path,value in fm['source_hashes'].items():
            if path.endswith('/scripts/run_dynamics_only_ttt.py'):continue
            assert dm['source_hashes'].get(path)==value,f'Frozen/Decision source mismatch: {path}'
        try:static_sources(baseline_data['frozen_metadata'],dm)
        except ReferenceCheckNeeded:
            if not allow_source_mismatch:raise
        assert f['initial_fixed_hash']==d['initial_fixed_hash']
        assert fm['initial_auxiliary_hash']==dm['initial_auxiliary_hash']
        fs=check_steps(fp[key],'frozen');ds=check_steps(dp[key],'decision')
        assert ds['updates']==d['metrics']['gradient_update_steps']==118
        final=dp[key]/'snapshot-0003000.pt'
        checks.append(dict(parameter=key[0],severity=key[1],adaptation_seed=key[2],checkpoint_sha256=t.SHA,
            initial_model_hash=dm['initial_model_hash'],final_model_hash=d['final_model_hash'],fixed_hash=d['final_fixed_hash'],
            snapshot_sha256=digest(final),snapshot_path=str(final.resolve()),
            optimizer_steps=118,first_update=75,last_update=3000,interaction_count=3000,episodes=3,
            online_transition_wall_seconds=ds['online_transition_seconds'],gpu_update_seconds=d['metrics']['update_compute_time'],
            frozen_transition_wall_seconds=fs['online_transition_seconds']))
    return checks,baseline_data

def reference_check(a,c,base):
    """Only if old/new simulator or core source hashes cannot establish reuse."""
    import numpy as np
    t.dependencies()
    agent,_=load_original_agent(a)
    adapter=t.make_adapter(agent,c,'frozen',c['adaptation_seeds'][0])
    cond=dict(parameter='bthigh_damping',severity=2.)
    rows=[]
    with adapter.held_out():
        for seed in c['evaluation_seeds']:
            t.seeded(seed);env=t.environment(c,cond,'fixed')
            try:
                steps=[x[0] for x in t.rollout(agent,adapter,c,cond,'fixed',seed,env,0,False)]
                measured=sum(x['reward'] for x in steps)
                expected=float(base['episodes'][('bthigh_damping',2.,seed)]['episode_return'])
                tolerance=max(1e-4,abs(expected)*1e-5)
                assert len(steps)==1000 and abs(measured-expected)<=tolerance,(seed,measured,expected)
                rows.append(dict(seed=seed,reference_return=expected,current_frozen_return=measured,
                                 absolute_difference=abs(measured-expected),tolerance=tolerance))
            finally:env.close()
    assert adapter.gradient_steps==0
    return rows

def validate_snapshot(payload,manifest,completion,c):
    i=payload['identity'];assert i==manifest['identity']==completion['identity']
    assert i['method']=='decision' and i['schedule']=='fixed' and not i['synthetic']
    assert payload['version']==1 and payload['checkpoint_sha256']==t.SHA
    assert payload['configuration']==manifest['configuration']==c
    assert payload['model_hash']==completion['final_model_hash']
    assert payload['fixed_hash']==completion['final_fixed_hash']==manifest['initial_fixed_hash']
    assert payload['auxiliary_hash']==completion['auxiliary_hash']==manifest['initial_auxiliary_hash']
    state=payload['adapter'];assert state['counters'][1]==3000 and state['counters'][3]==118
    return i

def load_original_agent(a):
    # This evaluator only loads the real 750k checkpoint. The adaptation
    # runner's loader also supports synthetic fixtures, so set that explicitly.
    return t.load_agent(argparse.Namespace(checkpoint=a.checkpoint,synthetic=False))

def load_adapted(a,c,path,expected):
    import torch
    t.dependencies()
    payload=torch.load(path/'snapshot-0003000.pt',map_location='cpu',weights_only=False)
    manifest=read(path/'manifest.json');completion=read(path/'completion.json')
    i=validate_snapshot(payload,manifest,completion,c)
    assert digest(path/'snapshot-0003000.pt')==expected['snapshot_sha256']
    agent,meta=load_original_agent(a)
    adapter=t.make_adapter(agent,c,'decision',i['adaptation_seed'])
    assert t.tree_hash(adapter.fixed_state())==payload['fixed_hash']
    adapter.load_state_dict(payload['adapter'])
    assert t.tree_hash(agent.model.state_dict())==payload['model_hash']
    assert t.tree_hash(t.aux_state(agent))==payload['auxiliary_hash']
    return agent,adapter,payload

def evaluate_prediction(agent,adapter,condition,base,c):
    import numpy as np
    import torch
    rows=[]
    # Manifest may contain the original server path; resolve against the explicit
    # reference supplied to this command, not against a copied path string.
    frozen=Path(condition['reference'])/'frozen'
    param=condition['parameter'];severity=condition['severity']
    for seed in c['evaluation_seeds']:
        key=(param,severity,seed);filename=base['files'][key];path=frozen/filename
        assert digest(path)==base['hashes'][filename],f'Frozen bank changed: {path}'
        with np.load(path) as data:
            obs=np.asarray(data['observations'],dtype=np.float32);actions=np.asarray(data['actions'],dtype=np.float32)
        assert obs.shape==(1001,17) and actions.shape==(1000,6)
        total=0.;count=0
        with torch.no_grad():
            for start in range(0,1000,256):
                end=min(start+256,1000)
                o=torch.as_tensor(obs[start:end],device=agent.device)
                n=torch.as_tensor(obs[start+1:end+1],device=agent.device)
                act=torch.as_tensor(actions[start:end],device=agent.device)
                z=agent.model.encode(o,None);target=agent.model.encode(n,None)
                errors=(agent.model.next(z,act,None)-target).square().mean(-1)
                total+=float(errors.sum().item());count+=end-start
        mse=total/count;assert np.isfinite(mse)
        frozen_mse=base['mse'][key]
        rows.append(dict(parameter=param,severity=severity,adaptation_seed=condition['adaptation_seed'],evaluation_seed=seed,
            method='decision',budget=3000,optimizer_steps=118,bank_file=filename,bank_sha256=base['hashes'][filename],samples=count,
            frozen_latent_mse=frozen_mse,decision_latent_mse=mse,delta_latent_mse=mse-frozen_mse,
            latent_mse_ratio=mse/frozen_mse if frozen_mse>1e-10 else None))
    return rows

def evaluate_control(agent,adapter,condition,base,c,mode):
    import numpy as np
    param=condition['parameter'];severity=condition['severity'];cond=dict(parameter=param,severity=severity)
    rows=[]
    for seed in c['evaluation_seeds']:
        t.seeded(seed);env=t.environment(c,cond,'fixed' if mode=='shifted' else 'nominal')
        try:
            step_rows=[]
            for step,_,_ in t.rollout(agent,adapter,c,cond,'fixed' if mode=='shifted' else 'nominal',seed,env,0,False):
                step_rows.append(step)
            assert len(step_rows)==1000 and step_rows[-1]['truncated'] and not step_rows[-1]['terminated']
            key=(param,severity if mode=='shifted' else 1.,seed)
            ref=base['episodes'][key]
            shifted_return=sum(x['reward'] for x in step_rows)
            frozen_return=float(ref['episode_return'])
            row=dict(parameter=param,severity=severity,adaptation_seed=condition['adaptation_seed'],evaluation_seed=seed,
                kind=mode,method='decision',budget=3000,optimizer_steps=118,episode_length=len(step_rows),
                frozen_return=frozen_return,decision_return=shifted_return,paired_return_difference=shifted_return-frozen_return,
                frozen_velocity=float(ref['forward_velocity']),decision_velocity=float(np.mean([x['velocity'] for x in step_rows])),
                frozen_control_cost=float(ref['control_cost']),decision_control_cost=float(sum(x['control_cost'] for x in step_rows)),
                frozen_action_abs=float(ref['action_abs_mean']),decision_action_abs=float(np.mean([x['action_abs'] for x in step_rows])),
                frozen_action_saturation=float(ref['action_saturation']),decision_action_saturation=float(np.mean([x['action_component_saturation'] for x in step_rows])),
                decision_force_rms=float(np.mean([x['actuator_force_rms'] for x in step_rows])),
                decision_force_saturation=float(np.mean([x['actuator_force_saturation'] for x in step_rows])) if step_rows[0]['force_limited_count'] else None,
                force_limited_count=step_rows[0]['force_limited_count'],
                actual_parameter_min=min(x['actual_parameter'] for x in step_rows),actual_parameter_max=max(x['actual_parameter'] for x in step_rows),
                terminated=step_rows[-1]['terminated'],truncated=step_rows[-1]['truncated'],
                onpolicy_latent_mse=float(np.mean([x['latent_mse_preupdate'] for x in step_rows])))
            assert np.isfinite(row['decision_return']) and np.isfinite(row['onpolicy_latent_mse'])
            rows.append(row)
        finally:env.close()
    return rows

def batch(a,c,checks,baseline_data):
    assert a.mode in ('prediction','shifted','nominal')
    targets={(p,k,s) for p,k in conditions(c) for s in c['adaptation_seeds']}
    expected={record_key(x):x for x in checks}
    assert targets<=set(expected)
    out=Path(a.output)
    if a.resume:
        assert out.is_dir()
        m=read(out/'manifest.json')
        assert m['mode']==a.mode and m['evaluation_scope']=='full_grid_final_snapshot_v2' and m['audit_sha256']==digest(a.audit)
        previous=out/'completion.json'
        if previous.exists():
            assert read(previous)['status']=='PARTIAL','Completed batch cannot be resumed'
            previous.rename(out/f'completion-attempt-{uuid4().hex[:8]}.json')
    else:
        out.mkdir(parents=True,exist_ok=False)
        save(out/'manifest.json',dict(mode=a.mode,evaluation_scope='full_grid_final_snapshot_v2',audit=str(Path(a.audit).resolve()),audit_sha256=digest(a.audit),
            checkpoint_sha256=t.SHA,configuration=c,evaluation_seeds=c['evaluation_seeds'],identities=[dict(parameter=p,severity=k,adaptation_seed=s) for p,k,s in sorted(targets)],
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=digest(__file__)))
    passed=[];failed=[];skipped=[]
    for key in sorted(targets):
        expected_row=expected[key];directory=Path(a.runs)/'fixed-decision'/label(expected_row)
        target=out/label(expected_row)
        if target.exists():
            if not a.resume:raise FileExistsError(target)
            if (target/'completion.json').exists() and read(target/'completion.json').get('status')=='PASS':
                skipped.append(target.name);continue
            archived=out.parent/f'{out.name}-interrupted-{target.name}-{uuid4().hex[:8]}'
            target.rename(archived)
        target.mkdir()
        started=time.perf_counter()
        try:
            agent,adapter,payload=load_adapted(a,c,directory,expected_row)
            before=t.tree_hash(adapter.state_dict());aux=t.tree_hash(t.aux_state(agent))
            condition=dict(parameter=key[0],severity=key[1],adaptation_seed=key[2],reference=a.reference)
            with adapter.held_out(allow_in_episode=True):
                if a.mode=='prediction':rr=evaluate_prediction(agent,adapter,condition,baseline_data,c)
                else:rr=evaluate_control(agent,adapter,condition,baseline_data,c,a.mode)
            assert t.tree_hash(adapter.state_dict())==before and t.tree_hash(t.aux_state(agent))==aux
            assert len(rr)==20 and [x['evaluation_seed'] for x in rr]==c['evaluation_seeds']
            rr=[dict(x,checkpoint_sha256=t.SHA,snapshot_sha256=expected_row['snapshot_sha256']) for x in rr]
            csv_rows(target/f'{a.mode}.csv',rr)
            save(target/'manifest.json',dict(identity=payload['identity'],mode=a.mode,checkpoint_sha256=t.SHA,
                snapshot=str((directory/'snapshot-0003000.pt').resolve()),snapshot_sha256=expected_row['snapshot_sha256'],
                baseline_reference=str(Path(a.reference).resolve()),fixed_hash=payload['fixed_hash'],adapter_hash_before=before))
            save(target/'completion.json',dict(status='PASS',evaluation_wall_seconds=time.perf_counter()-started,
                model_and_adapter_unchanged=True,optimizer_steps=118,episodes_or_banks=20))
            passed.append(target.name);print('PASS',target.name,flush=True)
        except Exception as exc:
            save(target/'failure.json',dict(error=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-started))
            failed.append(dict(identity=target.name,error=str(exc)));print('FAIL',target.name,exc,flush=True)
            if isinstance(exc,AttributeError):raise
    status='PASS' if not failed and len(passed)+len(skipped)==len(targets) else 'PARTIAL'
    save(out/'completion.json',dict(status=status,mode=a.mode,completed=passed,skipped=skipped,failed=failed,expected=len(targets)))
    return status

def collect_stage(root,mode):
    if not root:return [],[]
    root=Path(root);manifest=read(root/'manifest.json');status=read(root/'completion.json')['status']
    assert manifest['mode']==mode and status in ('PASS','PARTIAL')
    rows=[];failures=[];seen=set()
    for item in sorted(root.iterdir()):
        if not item.is_dir():continue
        done=item/'completion.json'
        if not done.is_file() or read(done)['status']!='PASS':
            failure=item/'failure.json'
            failures.append(dict(path=str(item),**(read(failure) if failure.is_file() else {'error':'INCOMPLETE'})))
            continue
        m=read(item/'manifest.json');assert m['mode']==mode and m['checkpoint_sha256']==t.SHA
        key=record_key(m['identity']);assert key not in seen;seen.add(key)
        rr=list(iter_csv(item/f'{mode}.csv'));assert len(rr)==20
        rows.extend(dict(x,evaluation_wall_seconds=read(done)['evaluation_wall_seconds'],snapshot_sha256=m['snapshot_sha256']) for x in rr)
    return rows,failures

def unit_stats(values):
    import numpy as np
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(n_adaptation_seeds=0,mean=None,median=None,std=None,ci_low=None,ci_high=None)
    ci=[None,None]
    if len(a)>1:
        rng=np.random.default_rng(20261111)
        means=a[rng.integers(len(a),size=(10000,len(a)))].mean(axis=1)
        ci=[float(x) for x in np.quantile(means,[.025,.975])]
    return dict(n_adaptation_seeds=len(a),mean=float(a.mean()),median=float(np.median(a)),
        std=float(a.std(ddof=1)) if len(a)>1 else None,ci_low=ci[0],ci_high=ci[1])

def aggregate(a,c,checks,base):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pred,pfail=collect_stage(a.prediction,'prediction')
    control,cfail=collect_stage(a.shifted,'shifted')
    retention,nfail=collect_stage(a.nominal,'nominal') if a.nominal else ([],[])
    assert pred or control,'No completed prediction/control results'
    audit_ids={record_key(x):x for x in checks}
    def assert_unique(data):
        ids=[(x['parameter'],float(x['severity']),int(x['adaptation_seed']),int(x['evaluation_seed'])) for x in data]
        assert len(ids)==len(set(ids))
        assert set((x[0],x[1],x[2]) for x in ids)<=set(audit_ids)
    for data in (pred,control,retention):assert_unique(data)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    if pred:csv_rows(out/'ttt_final_prediction.csv',pred)
    if control:csv_rows(out/'ttt_final_control.csv',control)
    if retention:csv_rows(out/'ttt_nominal_retention.csv',retention)
    failures=pfail+cfail+nfail
    if failures:csv_rows(out/'ttt_final_failures.csv',failures)
    perrep=[];summary=[]
    for parameter,severity in conditions(c):
        for seed in c['adaptation_seeds']:
            key=(parameter,severity,seed)
            pr=[x for x in pred if (x['parameter'],float(x['severity']),int(x['adaptation_seed']))==key]
            cr=[x for x in control if (x['parameter'],float(x['severity']),int(x['adaptation_seed']))==key]
            nr=[x for x in retention if (x['parameter'],float(x['severity']),int(x['adaptation_seed']))==key]
            status='COMPLETE' if len(pr)==len(cr)==20 else 'PARTIAL' if pr or cr else 'MISSING'
            audited=audit_ids[key]
            perrep.append(dict(parameter=parameter,severity=severity,adaptation_seed=seed,status=status,
                prediction_seeds=len(pr),shifted_control_seeds=len(cr),nominal_retention_seeds=len(nr),
                frozen_latent_mse=float(np.mean([float(x['frozen_latent_mse']) for x in pr])) if pr else None,
                decision_latent_mse=float(np.mean([float(x['decision_latent_mse']) for x in pr])) if pr else None,
                delta_latent_mse=float(np.mean([float(x['delta_latent_mse']) for x in pr])) if pr else None,
                frozen_return=float(np.mean([float(x['frozen_return']) for x in cr])) if cr else None,
                decision_return=float(np.mean([float(x['decision_return']) for x in cr])) if cr else None,
                delta_return=float(np.mean([float(x['paired_return_difference']) for x in cr])) if cr else None,
                delta_velocity=float(np.mean([float(x['decision_velocity'])-float(x['frozen_velocity']) for x in cr])) if cr else None,
                delta_control_cost=float(np.mean([float(x['decision_control_cost'])-float(x['frozen_control_cost']) for x in cr])) if cr else None,
                nominal_retention_delta=float(np.mean([float(x['paired_return_difference']) for x in nr])) if nr else None,
                online_interactions=audited['interaction_count'],online_optimizer_steps=audited['optimizer_steps'],
                online_transition_wall_seconds=audited['online_transition_wall_seconds'],gpu_update_seconds=audited['gpu_update_seconds'],
                prediction_eval_wall_seconds=float(pr[0]['evaluation_wall_seconds']) if pr else None,
                shifted_eval_wall_seconds=float(cr[0]['evaluation_wall_seconds']) if cr else None,
                snapshot_sha256=audited['snapshot_sha256'],checkpoint_sha256=t.SHA))
        subset=[x for x in perrep if x['parameter']==parameter and x['severity']==severity and x['status']=='COMPLETE']
        values=[float(x['delta_return']) for x in subset]
        err=[float(x['delta_latent_mse']) for x in subset]
        summary.append(dict(parameter=parameter,severity=severity,status='COMPLETE' if len(subset)==3 else 'PARTIAL' if subset else 'MISSING',
            **{f'return_{k}':v for k,v in unit_stats(values).items()},
            **{f'prediction_{k}':v for k,v in unit_stats(err).items()},
            interpretation='CI across independent adaptation streams; fixed 20 evaluation seeds'))
    csv_rows(out/'ttt_final_replicate_summary.csv',perrep)
    csv_rows(out/'ttt_final_condition_summary.csv',summary)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for row,param in enumerate(('bthigh_damping','torso_mass')):
        group=sorted([x for x in perrep if x['parameter']==param and x['status']=='COMPLETE'],key=lambda x:x['severity'])
        for col,(fkey,dkey,ylabel) in enumerate([('frozen_return','decision_return','Held-out shifted return'),('frozen_latent_mse','decision_latent_mse','Fixed-bank one-step latent MSE')]):
            ax=axs[row,col]
            x=sorted(set(x['severity'] for x in group))
            if x:
                ax.plot(x,[np.mean([v[fkey] for v in group if v['severity']==k]) for k in x],'o--',label='Existing Frozen')
                ax.plot(x,[np.mean([v[dkey] for v in group if v['severity']==k]) for k in x],'o-',label='Final Decision, 3000')
            ax.set(title=param,xlabel='Severity multiplier',ylabel=ylabel);ax.legend()
    fig.tight_layout();fig.savefig(out/'ttt_final_vs_frozen.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for param in ('bthigh_damping','torso_mass'):
        rr=[x for x in perrep if x['parameter']==param and x['status']=='COMPLETE']
        if rr:ax.scatter([x['delta_latent_mse'] for x in rr],[x['delta_return'] for x in rr],label=param,s=25)
    ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8)
    ax.set(xlabel='Decision − Frozen fixed-bank latent MSE',ylabel='Decision − Frozen paired shifted return',
           title='One point per adapted stream / condition; association is descriptive')
    ax.legend();fig.tight_layout();fig.savefig(out/'ttt_prediction_vs_control.png',dpi=160);plt.close(fig)
    complete=[x for x in summary if x['status']=='COMPLETE']
    text=['# Final 3000-transition Dynamics-only TTT versus existing Frozen','',
          f"Measured complete conditions: {len(complete)}/12; incomplete/missing: {12-len(complete)}.",
          f"Independent adaptation streams per condition: 3; held-out evaluation seeds per stream: 20.",
          'Selection: final predeclared snapshot only. No intermediate or shifted-result checkpoint selection.','',
          'Existing Frozen returns and one-step latent MSE are reused after manifest/config/source/hash checks. No Frozen rollout was rerun.',
          'Prediction is measured on the identical hashed Frozen state/action/next-state bank. Control is a fresh shifted rollout using the same evaluation seeds.',
          'Nominal retention is a separate optional stage and is never inferred from shifted control.',
          'Bootstrap uncertainty uses independent adaptation streams after averaging 20 paired evaluation seeds within each stream. One nominal training seed limits generalization.',
          'Online transition seconds sum per-step measurements and omit loading/saving overhead. Evaluation wall seconds are measured separately. GPU update seconds are the adapter counter.',
          'The fixed-data prediction/control association is descriptive and does not establish causality. Negative and missing results remain in the tables.','']
    for row in summary:
        text.append(f"- {row['parameter']} ×{row['severity']:g}: {row['status']}; Δreturn={row['return_mean']}; ΔMSE={row['prediction_mean']}")
    (out/'TTT_FINAL_SNAPSHOT_COMPARISON.md').write_text('\n'.join(text)+'\n')
    save(out/'manifest.json',dict(audit=str(Path(a.audit).resolve()),audit_sha256=digest(a.audit),prediction=a.prediction,shifted=a.shifted,nominal=a.nominal,checkpoint_sha256=t.SHA,configuration=c))
    save(out/'completion.json',dict(status='PASS' if len(complete)==12 else 'PARTIAL',complete_conditions=len(complete),failed_evaluations=failures))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['audit','evaluate','aggregate'])
    parser.add_argument('--checkpoint',required=True,help='Fixed 750k checkpoint directory')
    parser.add_argument('--runs',required=True,help='Dynamics-only runs/ root')
    parser.add_argument('--reference',required=True,help='Existing physical-severity-750k-v1 root')
    parser.add_argument('--config',default=str(t.DEFAULT))
    parser.add_argument('--output',required=True,help='New directory; --resume only for incomplete evaluate batch')
    parser.add_argument('--audit',help='PASS audit.json produced by audit command')
    parser.add_argument('--mode',choices=['prediction','shifted','nominal'],help='Only for evaluate; nominal retention is separate')
    parser.add_argument('--prediction',help='Completed final prediction batch for aggregate')
    parser.add_argument('--shifted',help='Completed final shifted control batch for aggregate')
    parser.add_argument('--nominal',help='Optional completed separate nominal retention batch')
    parser.add_argument('--resume',action='store_true',help='Skip PASS identities and preserve interrupted directories')
    parser.add_argument('--dry-run',action='store_true',help='Print 750k/final-budget/seed/condition plan; no CUDA or physics')
    a=parser.parse_args()
    if a.command=='evaluate' and (not a.mode or not a.audit):parser.error('evaluate needs --mode and --audit')
    if a.command=='aggregate' and (not a.audit or not a.prediction or not a.shifted):parser.error('aggregate needs --audit, --prediction and --shifted')
    if a.resume and a.command!='evaluate':parser.error('--resume applies only to evaluate')
    a.checkpoint=str(Path(a.checkpoint).resolve());a.runs=str(Path(a.runs).resolve());a.reference=str(Path(a.reference).resolve())
    c=t.configuration(argparse.Namespace(config=a.config,command='run'))
    if a.dry_run:
        total=len(conditions(c))*len(c['adaptation_seeds'])*len(c['evaluation_seeds'])*1000
        print(json.dumps(dict(command=a.command,mode=a.mode,checkpoint=a.checkpoint,checkpoint_sha256=t.SHA,
            runs=a.runs,reference=a.reference,output=a.output,final_snapshot=3000,
            condition_count=len(conditions(c)),
            adaptation_streams=len(c['adaptation_seeds']),evaluation_seeds=c['evaluation_seeds'],
            shifted_rollout_transitions_new=total if a.command=='evaluate' and a.mode in ('shifted','nominal') else 0,
            fixed_bank_prediction_samples_new=total if a.command=='evaluate' and a.mode=='prediction' else 0),indent=2));return
    assert digest(Path(a.checkpoint)/'model.pt')==t.SHA,'Checkpoint hash mismatch'
    if a.command=='audit':
        fallback=False
        try:checks,base=audit(a,c)
        except ReferenceCheckNeeded:
            checks,base=audit(a,c,allow_source_mismatch=True)
            fallback=True
        out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
        csv_rows(out/'run_integrity.csv',checks)
        if fallback:csv_rows(out/'one_condition_frozen_reference_check.csv',reference_check(a,c,base))
        save(out/'audit.json',dict(status='PASS',checkpoint_sha256=t.SHA,checkpoint=a.checkpoint,runs=a.runs,reference=a.reference,
            configuration=c,condition_count=12,independent_streams=36,final_budget=3000,decision_updates=118,
            schedule='fixed',baseline_return_rows=len(base['episodes']),baseline_one_step_mse_rows=240000,
            frozen_reference_reused=True,reference_check_required=fallback,
            compatibility='PASS: one-condition frozen replay matched old reference' if fallback else 'PASS: same config, environment source, checkpoint, MPPI, horizon, shift, seed set and hashed bank',
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        save(out/'completion.json',dict(status='PASS',condition_streams=36,reference_check_required=fallback))
        print('PASS audit: 36 Frozen + 36 Decision streams; 118 updates; reference compatible')
        return
    auditpath=Path(a.audit);aud=read(auditpath)
    assert aud['status']=='PASS' and aud['checkpoint_sha256']==t.SHA
    assert aud['checkpoint']==a.checkpoint and aud['runs']==a.runs and aud['reference']==a.reference and aud['configuration']==c
    checks,base=audit(a,c,allow_source_mismatch=aud.get('reference_check_required',False))
    assert len(checks)==aud['independent_streams']==36
    if a.command=='evaluate':
        status=batch(a,c,checks,base)
        print('BATCH',status)
    else:aggregate(a,c,checks,base)

if __name__=='__main__':main()
