#!/usr/bin/env python3
"""Reacher task binding for the unchanged HalfCheetah DynamicsQAdapter.

No automatic batches. A real checkpoint smoke gates every full adaptation run.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile
import numpy as np
import reacher_pipeline as r
import run_dynamics_q as shared

DEFAULT=r.ROOT/'experiments/nsgym/configs/reacher_dynamics_q_v1.json'


def contract(path=DEFAULT):
    binding=r.read(path);base=r.config(r.DEFAULT);c=r.read(shared.DEFAULT)
    assert binding['version']==1 and binding['collection_eval_mode'] is True
    c.update(method='reacher_dynamics_q_v1',task='reacher',transitions=base['snapshot_budgets'][-1],horizon=base['horizon'],
        batch_size=base['batch_size'],warmup=base['warmup'],capacity=base['capacity'],update_every=base['update_every_decisions'],
        dynamics_lr=base['lr'],consistency_coef=base['consistency_coef'],dynamics_grad_clip=base['grad_clip'],
        adaptation_seeds=base['adaptation_seeds'],evaluation_seeds=base['evaluation_seeds'],conditions=r.conditions(base))
    assert c['horizon']==50 and len(c['conditions'])==24 and len(c['adaptation_seeds'])==3
    assert not set(c['adaptation_seeds']) & set(c['evaluation_seeds'])
    assert c['warmup']==c['batch_size']==64 and c['capacity']>=c['transitions']
    assert binding['baseline_config_fingerprint']==r.fingerprint(base)
    return c,base


def count(c,budget):
    from common.dynamics_q import expected_updates
    return expected_updates(budget,max(c['batch_size'],c['warmup']),c['update_every'])


def identities(c,a=None):
    items=[dict(**x,method=c['method'],adaptation_seed=s,schedule='fixed') for x in c['conditions'] for s in c['adaptation_seeds']]
    if a and any(v is not None for v in (a.family,a.severity,a.seed)):
        if any(v is None for v in (a.family,a.severity,a.seed)):raise ValueError('Specify --family, --severity and --seed together')
        items=[x for x in items if (x['family'],x['severity'],x['adaptation_seed'])==(a.family,a.severity,a.seed)]
        if len(items)!=1:raise ValueError('Condition outside predefined grid')
    return items


def label(i):return f"{r.tag(i)}-dynamics-q-seed{i['adaptation_seed']}"
def condition(i):return {k:i[k] for k in ('family','parameter','severity')}
def sources():
    paths=[Path(__file__),DEFAULT,shared.DEFAULT,r.DEFAULT,Path(shared.__file__),Path(r.__file__),
        r.ROOT/'tdmpc2/common/dynamics_q.py',r.ROOT/'tdmpc2/tdmpc2.py',r.ROOT/'tdmpc2/common/world_model.py',
        r.ROOT/'tdmpc2/common/math.py',r.ROOT/'tdmpc2/envs/nsgym/reacher_severity.py',r.ROOT/'experiments/nsgym/scripts/training_support.py']
    return {str(p.relative_to(r.ROOT)):r.sha(p) for p in paths}


def inventory(a,c,base,require_model=False):
    root=a.reference;s=r.read(a.selection or root/'nominal-home-01/selected_reacher_checkpoint.json')
    cp=a.checkpoint or Path(s['checkpoint']);s=dict(s,checkpoint=str(cp.resolve()))
    assert s['pipeline_config_fingerprint']==r.fingerprint(base)
    cfg=s['configuration'];assert cfg['task']=='nsgym-reacher' and cfg['action_dim']==2 and cfg['obs_shape']['state']==[10]
    assert cfg['episode_length']==50 and cfg['tau']==.01
    assert cfg['lr']==c['q_lr'] and cfg['value_coef']==c['value_coef'] and cfg['grad_clip_norm']==c['q_grad_clip']
    stream_rows=[]
    for stage in ('frozen-home-01','adapt-home-01','prediction-home-01','control-home-02-resumed','full-joint-home-01/runs-fixed','full-joint-home-01/evaluation-full-01'):
        assert r.read(root/stage/'completion.json')['status']=='PASS',stage
        manifest=r.read(root/stage/'manifest.json')
        assert manifest['checkpoint_sha256']==s['model_sha256'],f'Nominal origin mismatch: {stage}'
        if 'config_fingerprint' in manifest:assert manifest['config_fingerprint']==r.fingerprint(base)
    expected={(i['family'],i['severity'],i['adaptation_seed']) for i in identities(c)}
    actual=set()
    for p in (root/'adapt-home-01').glob('*/manifest.json'):
        m=r.read(p);d=r.read(p.with_name('completion.json'));assert d['status']=='PASS' and m['selection']['model_sha256']==s['model_sha256']
        assert m['config']==base and not m['synthetic']
        if m['identity']['method']!='decision':continue
        i=m['identity'];key=(i['family'],i['severity'],i['adaptation_seed']);assert key not in actual;actual.add(key)
        assert d['metrics']['adaptation_transitions']==c['transitions'] and d['metrics']['gradient_update_steps']==count(c,c['transitions'])
        stream_rows.append(dict(**i,transitions=d['metrics']['adaptation_transitions'],dynamics_steps=d['metrics']['gradient_update_steps']))
    assert actual==expected
    joint=set()
    for p in (root/'full-joint-home-01/runs-fixed').glob('*/manifest.json'):
        m=r.read(p);assert m['checkpoint_sha256']==s['model_sha256'];i=m['identity'];joint.add((i['family'],i['severity'],i['adaptation_seed']))
    assert joint==expected
    available=(cp/'model.pt').is_file() and (cp/'optimizers.pt').is_file()
    if require_model:assert available,'Selected model.pt/optimizers.pt required on CUDA server'
    if available:assert r.sha(cp/'model.pt')==s['model_sha256']
    report=dict(status='PASS_METADATA',selection=s,checkpoint_bytes_verified=available,baseline_streams=stream_rows,
        dynamics_steps=count(c,c['transitions']),conditions=24,adaptation_seeds=3,streams=72,
        configuration=c,baseline_configuration=base,source_hashes=sources(),
        real_checkpoint_smoke='NOT_RUN',heldout_evaluation='NOT_RUN')
    return s,report


def plan(a,c):
    items=identities(c,a)
    if a.command=='smoke':items=items[:1]
    n=len(items);b=100 if a.command=='smoke' else c['transitions']
    return dict(task='reacher',conditions=len({(i['family'],i['severity']) for i in items}),streams=n,
        adaptation_seed_count=len({i['adaptation_seed'] for i in items}),transitions_per_stream=b,
        adaptation_environment_transitions=n*b,dynamics_optimizer_steps=n*count(c,b),q_optimizer_steps=n*count(c,b),
        heldout_episodes_per_mode=n*len(c['evaluation_seeds']),heldout_environment_transitions_per_control_mode=n*len(c['evaluation_seeds'])*50,
        prediction_transitions_read_only=n*len(c['evaluation_seeds'])*50,configuration=c,
        note='Prediction reads existing data. Control and nominal retention are separate explicit commands; no automatic launch.')


def cpu_agent():
    from types import SimpleNamespace,MethodType
    import torch
    from common.world_model import WorldModel
    from tdmpc2 import TDMPC2
    cfg=deepcopy(shared.cpu_agent().cfg);cfg.obs_shape={'state':(10,)};cfg.action_dim=2
    agent=SimpleNamespace(cfg=cfg,model=WorldModel(cfg),discount=.95,device=torch.device('cpu'))
    agent._td_target=MethodType(TDMPC2._td_target,agent)
    return agent


def fixture(a,c):
    import torch
    from common.dynamics_q import DynamicsQAdapter
    from common.stage3a import equal_state
    torch.set_num_threads(1);r.seeded(17);agent=cpu_agent();i=identities(c)[0];ad=DynamicsQAdapter(agent,c,i)
    o=torch.randn(10);boundary=None
    for t in range(100):
        ep,et=divmod(t,50);ac=torch.randn(2).tanh();n=o+.01*torch.randn(10)
        ad.observe(o,ac,n,-float(n.square().sum()),False,et==49,ep,et,i['severity'],1.)
        if et==49:
            boundary=n.clone();o=torch.randn(10)+10
        else:o=n
    assert not torch.equal(ad.replay[49]['next_observation'],ad.replay[50]['observation'])
    assert ad.dynamics_steps==ad.q_steps==ad.target_steps==count(c,100)
    audit=ad.parameter_audit();assert all(x['changed']==(x['component'] in ('dynamics','q','target_q')) for x in audit)
    a.output.mkdir(parents=True,exist_ok=False)
    path=a.output/'fixture-roundtrip.pt';torch.save(dict(model=agent.model.state_dict(),adapter=ad.state_dict()),path)
    payload=torch.load(path,weights_only=False);restored=cpu_agent();other=DynamicsQAdapter(restored,c,i)
    # Frozen state must be the original nominal fixture, not a fresh random model.
    restored.model.load_state_dict(ad.initial_model);other=DynamicsQAdapter(restored,c,i)
    restored.model.load_state_dict(payload['model']);other.load_state_dict(payload['adapter'])
    assert equal_state(other.state_dict(),ad.state_dict())
    with other.held_out():other.model.Q(other.model.encode(torch.zeros(1,10),None),torch.zeros(1,2),None)
    r.table(a.output/'parameter_invariance.csv',audit)
    r.write(a.output/'completion.json',dict(status='PASS',cpu_fixture=True,transitions=100,dynamics_steps=ad.dynamics_steps,q_steps=ad.q_steps,
        metrics=ad.last_metrics,real_checkpoint_physics='NOT_RUN',configuration=c,source_hashes=sources()))


def train_one(a,c,base,s,i,out,budget):
    import torch
    from common.dynamics_q import DynamicsQAdapter
    from training_support import tree_hash
    started=time.perf_counter();agent,meta=r.load(s['checkpoint'],base);r.seeded(i['adaptation_seed'])
    assert agent.discount==.95
    aux=tree_hash(shared.ttt.aux_state(agent));ad=DynamicsQAdapter(agent,c,i);e=r.env(condition(i));episodes=[]
    r.write(out/'manifest.json',dict(identity=i,configuration=c,selection=s,source_hashes=sources(),optimizers=shared.optimizer_record(ad),
        checkpoint_metadata=meta,collection_eval_mode=True,episode_horizon=50,nominal_parameter=e.nominal_value))
    try:
        o,_=e.reset(seed=i['adaptation_seed']);episode_return=0.
        with (out/'training.jsonl').open('x') as log:
            for t in range(budget):
                ep,et=divmod(t,50);assert np.shape(o)==(10,)
                ac=r.action(agent,o,et);assert np.shape(ac)==(2,)
                n,reward,term,trunc,info=e.step(ac)
                metrics=ad.observe(o,ac,n,reward,term,trunc,ep,et,e.readback(),e.nominal_value)
                diagnostic=r.step_record(e,ac,n,reward,term,trunc,info)
                episode_return+=reward
                row=dict(global_step=t+1,episode=ep,episode_step=et,dynamics_steps=ad.dynamics_steps,q_steps=ad.q_steps,
                    target_steps=ad.target_steps,replay_size=len(ad.replay),metrics=metrics,**diagnostic)
                log.write(json.dumps(row,allow_nan=False)+'\n');o=n
                if trunc:
                    episodes.append(dict(**i,episode=ep,episode_return=episode_return));episode_return=0.
                    if t+1<budget:o,_=e.reset()
                if (t+1)%100==0:log.flush();print(f'{label(i)}: {t+1}/{budget}; D={ad.dynamics_steps}, Q={ad.q_steps}',flush=True)
        assert ad.dynamics_steps==ad.q_steps==ad.target_steps==count(c,budget)
        audit=ad.parameter_audit();assert all(x['changed']==(x['component'] in ('dynamics','q','target_q')) for x in audit)
        assert tree_hash(shared.ttt.aux_state(agent))==aux
        payload=dict(identity=i,configuration=c,selection=s,source_hashes=sources(),model=agent.model.state_dict(),adapter=ad.state_dict(),
            fixed_hash=tree_hash(ad.fixed_state()),original_optimizer_hash=aux,resume='evaluation_only; failed stream restarts, no exact midstream resume')
        snapshot=out/f'snapshot-{budget:07d}.pt';torch.save(payload,snapshot)
        assert tree_hash(torch.load(snapshot,weights_only=False,map_location=agent.device))==tree_hash(payload)
        assert r.sha(Path(s['checkpoint'])/'model.pt')==s['model_sha256']
        assert r.sha(Path(s['checkpoint'])/'optimizers.pt')==meta['optimizer_sha256']
        r.table(out/'online_episodes.csv',episodes);r.table(out/'parameter_invariance.csv',audit)
        r.done(out,identity=i,transitions=budget,dynamics_steps=ad.dynamics_steps,q_steps=ad.q_steps,target_steps=ad.target_steps,
            snapshot=str(snapshot.resolve()),snapshot_sha256=r.sha(snapshot),fixed_hash=payload['fixed_hash'],parameter_invariance=audit,
            wall_seconds=time.perf_counter()-started,update_seconds=ad.update_seconds,replay_size=len(ad.replay),heldout_evaluation='NOT_RUN')
    finally:e.close()


def train(a,c,base,s):
    items=identities(c,a)
    if a.command=='smoke':items=items[:1]
    else:
        assert a.smoke_result,'--smoke-result required'
        d=r.read(a.smoke_result/'completion.json');m=r.read(a.smoke_result/'manifest.json')
        assert d['status']=='PASS' and m['command']=='smoke' and not m['cpu_fixture']
        assert m['configuration']==c and m['selection']['model_sha256']==s['model_sha256'] and m['source_hashes']==sources()
    manifest=dict(command=a.command,cpu_fixture=False,configuration=c,selection=s,identities=items,source_hashes=sources())
    out=shared.open_stage(a.output,manifest,a.resume);passed=[];failed=[]
    for i in items:
        target,skip=shared.destination(out,label(i))
        try:
            if skip:
                m=r.read(target/'manifest.json');d=r.read(target/'completion.json')
                assert m['configuration']==c and m['source_hashes']==sources() and m['identity']==i
                assert r.sha(target/Path(d['snapshot']).name)==d['snapshot_sha256']
            else:train_one(a,c,base,s,i,target,100 if a.command=='smoke' else c['transitions'])
            passed.append(label(i));print('PASS',label(i),flush=True)
        except Exception:
            trace=traceback.format_exc();shared.write(target/'failure.json',dict(traceback=trace));failed.append(dict(identity=i,traceback=trace))
    shared.finish(out,dict(status='PASS' if not failed else 'PARTIAL',completed=passed,failed=failed,expected=len(items)))
    (out/'REACHER_DQ_COMPLETION.md').write_text(f'# Reacher D+Q {a.command}\n\nCompleted {len(passed)}/{len(items)} streams; failures {len(failed)}. Held-out evaluation NOT_RUN.\n')
    if failed:raise RuntimeError(f'{len(failed)} failed streams; see failure.json')


def restore(s,base,c,path):
    import torch
    from common.dynamics_q import DynamicsQAdapter
    from training_support import tree_hash
    d=r.read(path/'completion.json');m=r.read(path/'manifest.json');assert d['status']=='PASS'
    assert m['configuration']==c and m['source_hashes']==sources() and m['selection']['model_sha256']==s['model_sha256']
    snapshot=path/Path(d['snapshot']).name;assert r.sha(snapshot)==d['snapshot_sha256']
    agent,_=r.load(s['checkpoint'],base);ad=DynamicsQAdapter(agent,c,d['identity'])
    p=torch.load(snapshot,map_location=agent.device,weights_only=False)
    assert p['configuration']==c and p['source_hashes']==sources() and p['selection']['model_sha256']==s['model_sha256']
    assert p['identity']==d['identity'];agent.model.load_state_dict(p['model']);ad.load_state_dict(p['adapter'])
    assert tree_hash(ad.fixed_state())==d['fixed_hash']==p['fixed_hash']
    assert tree_hash(shared.ttt.aux_state(agent))==p['original_optimizer_hash']
    assert ad.dynamics_steps==ad.q_steps==ad.target_steps==count(c,d['transitions'])
    assert ad.transitions==d['transitions'] and len(ad.replay)==min(c['capacity'],d['transitions'])
    for j,rec in enumerate(ad.replay):
        assert rec['global_step']==j+1 and rec['episode']==j//50 and rec['episode_step']==j%50
        assert rec['truncated']==((j+1)%50==0) and not rec['terminated'] and rec['discount']==.95
        assert rec['adaptation_seed']==d['identity']['adaptation_seed']
        assert np.isclose(rec['actual_parameter'],rec['nominal_parameter']*d['identity']['severity'],rtol=0,atol=1e-10)
        if j and j%50:assert torch.equal(ad.replay[j-1]['next_observation'],rec['observation'])
    audit=ad.parameter_audit();assert all(x['changed']==(x['component'] in ('dynamics','q','target_q')) for x in audit)
    return agent,ad,d


def completed_items(a,c):
    top=r.read(a.runs/'manifest.json');d=r.read(a.runs/'completion.json')
    assert d['status']=='PASS' and top['command']=='run' and top['configuration']==c and top['source_hashes']==sources()
    wanted=identities(c,a)
    assert {label(i) for i in wanted}<={*d['completed']}
    return wanted


def audit(a,c,base,s):
    items=completed_items(a,c);a.output.mkdir(parents=True,exist_ok=False);records=[]
    for i in items:
        agent,ad,d=restore(s,base,c,a.runs/label(i));seen=0
        with (a.runs/label(i)/'training.jsonl').open() as log:
            for line in log:
                row=json.loads(line);seen+=1;assert row['global_step']==seen
                assert row['dynamics_steps']==row['q_steps']==row['target_steps']==count(c,seen)
                assert row['truncated']==(seen%50==0) and not row['terminated']
                assert all(math.isfinite(v) for v in row['metrics'].values())
        assert seen==c['transitions']
        records.append(dict(**i,transitions=seen,dynamics_steps=ad.dynamics_steps,q_steps=ad.q_steps,snapshot_sha256=d['snapshot_sha256']))
        del ad,agent
    r.table(a.output/'inventory.csv',records);r.done(a.output,streams=len(records),source_hashes=sources(),checkpoint_sha256=s['model_sha256'])


def evaluate(a,c,base,s):
    items=completed_items(a,c)
    manifest=dict(command='evaluate',mode=a.mode,configuration=c,selection=s,source_hashes=sources(),identities=items)
    out=shared.open_stage(a.output,manifest,a.resume);passed=[];failed=[]
    frozen=r.csvread(a.reference/'frozen-home-01/reacher_frozen_severity.csv')
    for i in items:
        dest,skip=shared.destination(out,label(i))
        try:
            agent,ad,d=restore(s,base,c,a.runs/label(i))
            if skip:
                dd=r.read(dest/'completion.json');assert dd['snapshot_sha256']==d['snapshot_sha256'] and dd['csv_sha256']==r.sha(dest/f'{a.mode}.csv')
            else:
                records=[]
                with ad.held_out():
                    if a.mode=='prediction':
                        bank=[x for x in frozen if x['family']==i['family'] and float(x['severity'])==i['severity']]
                        assert sorted(int(x['seed']) for x in bank)==sorted(c['evaluation_seeds'])
                        for x in bank:
                            path=a.reference/'frozen-home-01'/x['data_file'];assert r.sha(path)==x['data_sha256']
                            with np.load(path,allow_pickle=False) as z:
                                assert z['observations'].shape==(51,10) and z['actions'].shape==(50,2)
                                errors=[r.error(agent,o,ac,n) for o,ac,n in zip(z['observations'][:-1],z['actions'],z['observations'][1:])]
                            records.append(dict(**i,seed=int(x['seed']),latent_mse=float(np.mean(errors)),frozen_latent_mse=float(x['latent_mse']),
                                source_sha256=x['data_sha256'],samples=len(errors),snapshot_sha256=d['snapshot_sha256']))
                    else:
                        cond=condition(i)
                        if a.mode=='nominal':cond=dict(cond,severity=1.)
                        for seed in c['evaluation_seeds']:
                            rec,_=r.episode(agent,cond,seed)
                            rec.update(i);rec.update(kind=a.mode,budget=c['transitions'],snapshot_sha256=d['snapshot_sha256'])
                            records.append(rec)
                r.table(dest/f'{a.mode}.csv',records)
                r.done(dest,identity=i,rows=len(records),snapshot_sha256=d['snapshot_sha256'],csv_sha256=r.sha(dest/f'{a.mode}.csv'),optimizer_steps=0,replay_writes=0)
            passed.append(label(i));print('PASS',label(i),flush=True)
        except Exception:
            trace=traceback.format_exc();shared.write(dest/'failure.json',dict(traceback=trace));failed.append(dict(identity=i,traceback=trace))
    shared.finish(out,dict(status='PASS' if not failed else 'PARTIAL',completed=passed,failed=failed,expected=len(items)))
    (out/'REACHER_DQ_HELDOUT_RESULTS.md').write_text(f'# Reacher D+Q {a.mode}\n\nCompleted {len(passed)}/{len(items)} models on {len(c["evaluation_seeds"])} held-out seeds. Failed {len(failed)}. Per-seed metrics are in child CSVs. No optimizer updates or replay writes.\n')
    if failed:raise RuntimeError('Incomplete evaluation; see failure.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['inventory','smoke','run','audit','evaluate','aggregate','package'])
    p.add_argument('--config',type=Path,default=DEFAULT);p.add_argument('--reference',type=Path)
    p.add_argument('--selection',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runs',type=Path);p.add_argument('--smoke-result',type=Path);p.add_argument('--mode',choices=['prediction','shifted','nominal'])
    p.add_argument('--prediction',type=Path);p.add_argument('--control',type=Path);p.add_argument('--nominal',type=Path)
    p.add_argument('--family',choices=['D0','D1','M0','M1']);p.add_argument('--severity',type=float);p.add_argument('--seed',type=int)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--cpu-fixture',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--allow-shared-gpu',action='store_true')
    a=p.parse_args();c,base=contract(a.config)
    if a.dry_run:print(json.dumps(plan(a,c),indent=2));return
    if a.cpu_fixture:
        if a.command!='smoke':p.error('--cpu-fixture requires smoke')
        fixture(a,c);return
    if a.command=='package':package(a);return
    if a.reference is None:p.error('--reference required')
    if a.command=='aggregate':aggregate(a,c);return
    s,info=inventory(a,c,base,require_model=a.command!='inventory')
    if a.command=='inventory':
        a.output.mkdir(parents=True,exist_ok=False);r.write(a.output/'inventory.json',info);r.done(a.output,checkpoint_bytes_verified=info['checkpoint_bytes_verified']);print(json.dumps(plan(a,c),indent=2));return
    if a.command in ('audit','evaluate') and a.runs is None:p.error('--runs required')
    if a.command=='evaluate' and a.mode is None:p.error('--mode required')
    from run_full_joint_rl import gpu_status
    gpu=gpu_status(a.gpu,8192,a.allow_shared_gpu);os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    if a.command in ('smoke','run'):train(a,c,base,s)
    elif a.command=='audit':audit(a,c,base,s)
    else:evaluate(a,c,base,s)


# Aggregation and packaging are defined below; main is invoked at file end.
def aggregate(a,c):
    from collections import defaultdict
    from analyze_existing_mechanisms import unique,stats
    assert a.prediction and a.control and a.runs,'--runs, --prediction, --control required'
    selected,metadata=inventory(a,c,r.config(r.DEFAULT))
    assert r.read(a.runs/'completion.json')['status']=='PASS'
    run_manifest=r.read(a.runs/'manifest.json')
    assert run_manifest['configuration']==c and run_manifest['source_hashes']==sources()
    expected={(i['family'],float(i['severity']),i['adaptation_seed'],s) for i in identities(c) for s in c['evaluation_seeds']}
    def key(x):return x['family'],float(x['severity']),int(x['adaptation_seed']),int(x.get('seed',x.get('evaluation_seed')))
    def dq_stage(path,mode):
        top=r.read(path/'completion.json');m=r.read(path/'manifest.json')
        assert top['status']=='PASS' and m['mode']==mode and m['configuration']==c and m['source_hashes']==sources()
        assert m['selection']['model_sha256']==selected['model_sha256']
        data=[]
        for i in identities(c):
            child=path/label(i);d=r.read(child/'completion.json');assert d['status']=='PASS'
            assert r.sha(child/f'{mode}.csv')==d['csv_sha256']
            run=r.read(a.runs/label(i)/'completion.json');assert d['snapshot_sha256']==run['snapshot_sha256']
            rr=r.csvread(child/f'{mode}.csv');assert {key(x) for x in rr}=={(i['family'],i['severity'],i['adaptation_seed'],s) for s in c['evaluation_seeds']}
            data.extend(rr)
        result=unique(data,key);assert set(result)==expected;return result
    dqpred=dq_stage(a.prediction,'prediction');dqcontrol=dq_stage(a.control,'shifted')
    dqnom=dq_stage(a.nominal,'nominal') if a.nominal else None
    def baseline(stage,file,kind=None):
        assert r.read(a.reference/stage/'completion.json')['status']=='PASS'
        data=[]
        for p in (a.reference/stage).glob('*/'+file):
            marker=r.read(p.with_name('completion.json'));assert marker['status']=='PASS'
            for x in r.csvread(p):
                if int(x['budget'])!=c['transitions']:continue
                if stage.startswith(('prediction','control')) and x['method']!='decision':continue
                if kind and x['kind']!=kind:continue
                data.append(x)
        result=unique(data,key);assert set(result)==expected;return result
    dp=baseline('prediction-home-01','predict.csv')
    dc=baseline('control-home-02-resumed','control.csv','shifted')
    jc=baseline('full-joint-home-01/evaluation-full-01','control.csv','shifted')
    dn=baseline('control-home-02-resumed','control.csv','nominal') if dqnom else None
    jn=baseline('full-joint-home-01/evaluation-full-01','control.csv','nominal') if dqnom else None
    frozen=r.csvread(a.reference/'frozen-home-01/reacher_frozen_severity.csv')
    fb=unique(frozen,lambda x:(x['family'],float(x['severity']),int(x['seed'])))
    paired=[]
    for mode,dq,dd,jj in [('shifted',dqcontrol,dc,jc)]+([('nominal',dqnom,dn,jn)] if dqnom else []):
        for k in sorted(expected):
            f=fb[(k[0],k[1],k[3])] if mode=='shifted' else fb[('D0',1.,k[3])]
            row=dict(family=k[0],parameter=f['parameter'],severity=k[1],adaptation_seed=k[2],evaluation_seed=k[3],kind=mode,
                nominal_parameter=float(f['nominal_parameter']),actual_parameter=float(dq[k]['actual_parameter']))
            for method,x in [('Frozen',f),('Dynamics-only',dd[k]),('Dynamics+Q',dq[k]),('Joint RL',jj[k])]:
                for metric in ('episode_return','distance_mean','distance_final','reward_dist','reward_ctrl','action_abs','action_saturation'):
                    row[method+'_'+metric]=float(x[metric])
            row.update(dq_minus_frozen=row['Dynamics+Q_episode_return']-row['Frozen_episode_return'],
                dq_minus_dynamics=row['Dynamics+Q_episode_return']-row['Dynamics-only_episode_return'],
                joint_minus_dq=row['Joint RL_episode_return']-row['Dynamics+Q_episode_return'],
                dq_minus_dynamics_distance=row['Dynamics+Q_distance_final']-row['Dynamics-only_distance_final'])
            if mode=='shifted':
                assert dqpred[k]['source_sha256']==dp[k]['source_sha256']==f['data_sha256']
                row.update(frozen_mse=float(f['latent_mse']),dynamics_mse=float(dp[k]['latent_mse']),dq_mse=float(dqpred[k]['latent_mse']))
                row.update(dq_minus_dynamics_mse=row['dq_mse']-row['dynamics_mse'],dq_minus_frozen_mse=row['dq_mse']-row['frozen_mse'],
                    relative_mse_improvement_vs_frozen=1-row['dq_mse']/row['frozen_mse'],relative_mse_improvement_vs_dynamics=1-row['dq_mse']/row['dynamics_mse'])
            paired.append(row)
    a.output.mkdir(parents=True,exist_ok=False);figdir=a.output/'figures';figdir.mkdir()
    r.table(a.output/'four_way_per_seed.csv',paired)
    groups=defaultdict(list)
    for x in paired:groups[(x['family'],x['severity'],x['kind'],x['adaptation_seed'])].append(x)
    metric_keys=[k for k in paired[0] if k not in ('family','parameter','severity','adaptation_seed','evaluation_seed','kind','actual_parameter','nominal_parameter')]
    per=[]
    for (f,k,kind,s),rr in sorted(groups.items()):
        per.append(dict(family=f,severity=k,kind=kind,adaptation_seed=s,n_eval=len(rr),**{m:float(np.mean([x[m] for x in rr])) for m in metric_keys if all(m in x for x in rr)}))
    r.table(a.output/'four_way_per_adaptation_seed.csv',per);summaries=[]
    for f in r.config(r.DEFAULT)['families']:
        for k in r.config(r.DEFAULT)['severities']:
            for kind in ['shifted']+(['nominal'] if dqnom else []):
                rr=[x for x in per if x['family']==f and x['severity']==k and x['kind']==kind]
                for metric in metric_keys:
                    if all(metric in x for x in rr):
                        values=[x[metric] for x in rr];unit='adaptation_stream'
                        if metric.startswith('Frozen_') or metric=='frozen_mse':
                            values=list({x['evaluation_seed']:x[metric] for x in paired if x['family']==f and x['severity']==k and x['kind']==kind}.values());unit='unique_evaluation_seed'
                        summaries.append(dict(family=f,severity=k,kind=kind,metric=metric,unit=unit,**stats(values)))
    r.table(a.output/'four_way_summary.csv',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=[('01_four_way_return',[m+'_episode_return' for m in ['Frozen','Dynamics-only','Dynamics+Q','Joint RL']],'Episode return'),
        ('02_four_way_distance',[m+'_distance_final' for m in ['Frozen','Dynamics-only','Dynamics+Q','Joint RL']],'Final target distance [m]'),
        ('03_reward_decomposition',[m+'_'+v for m in ['Frozen','Dynamics-only','Dynamics+Q','Joint RL'] for v in ['reward_dist','reward_ctrl']],'Episode reward contribution'),
        ('04_paired_return',['dq_minus_frozen','dq_minus_dynamics','joint_minus_dq'],'Paired return difference'),
        ('05_paired_distance',['dq_minus_dynamics_distance'],'D+Q − D final distance [m]'),
        ('06_prediction',['frozen_mse','dynamics_mse','dq_mse'],'Fixed-bank latent MSE')]
    if dqnom:specs.append(('10_nominal_retention',[m+'_episode_return' for m in ['Frozen','Dynamics-only','Dynamics+Q','Joint RL']],'Nominal retention return'))
    for filename,metrics,ylabel in specs:
        fig,axes=plt.subplots(1,4,figsize=(18,5),layout='constrained');kind='nominal' if filename.startswith('10') else 'shifted'
        for ax,(family,param) in zip(axes,r.config(r.DEFAULT)['families'].items()):
            for metric in metrics:
                rr=[x for x in summaries if x['family']==family and x['kind']==kind and x['metric']==metric]
                xx=[x['severity'] for x in rr];line,=ax.plot(xx,[x['mean'] for x in rr],'o-',label=metric)
                ax.fill_between(xx,[x['ci_low'] for x in rr],[x['ci_high'] for x in rr],color=line.get_color(),alpha=.15)
            nominal=next(float(x['nominal_parameter']) for x in paired if x['family']==family)
            ax.set(title=f'{param}; nominal={nominal:.6g} '+('kg' if param.endswith('mass') else 'N m s/rad'),xlabel='k; actual = nominal × k' if kind=='shifted' else 'Adaptation k; evaluate nominal',ylabel=ylabel);ax.legend(fontsize=6)
        fig.suptitle('Adapted: 3 streams ×20 paired seeds; Frozen: 20 unique seeds; descriptive bootstrap')
        fig.savefig(figdir/(filename+'.png'),dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    for family in r.config(r.DEFAULT)['families']:
        rr=[x for x in per if x['family']==family and x['kind']=='shifted']
        ax.scatter([x['dq_minus_dynamics_mse'] for x in rr],[x['dq_minus_dynamics'] for x in rr],label=family)
    ax.axhline(0,color='gray');ax.axvline(0,color='gray');ax.set(xlabel='D+Q − D latent MSE',ylabel='D+Q − D return',title='Each point: adaptation stream mean; correlation is not causation');ax.legend()
    fig.savefig(figdir/'07_prediction_control.png',dpi=160);plt.close(fig)
    costs=[];diagnostics=[]
    for i in identities(c):
        p=a.runs/label(i);d=r.read(p/'completion.json')
        costs.append(dict(**i,transitions=d['transitions'],dynamics_steps=d['dynamics_steps'],q_steps=d['q_steps'],wall_seconds=d['wall_seconds'],update_seconds=d['update_seconds']))
        with (p/'training.jsonl').open() as h:
            for line in h:
                x=json.loads(line)
                if x['metrics']:diagnostics.append(dict(**i,transition=x['global_step'],**x['metrics']))
    comparison_costs=[]
    for i in identities(c):
        dq=next(x for x in costs if x['family']==i['family'] and x['severity']==i['severity'] and x['adaptation_seed']==i['adaptation_seed'])
        comparison_costs.append(dict(dq,method='Dynamics+Q',total_optimizer_steps=dq['dynamics_steps']+dq['q_steps'],collection_eval_mode=True))
        for code,method in [('frozen','Frozen'),('decision','Dynamics-only')]:
            name=f"{r.tag(i)}-{code}-seed{i['adaptation_seed']}"
            d=r.read(a.reference/'adapt-home-01'/name/'completion.json')
            comparison_costs.append(dict(**condition(i),adaptation_seed=i['adaptation_seed'],method=method,
                transitions=d['metrics']['adaptation_transitions'],dynamics_steps=d['metrics']['gradient_update_steps'],q_steps=0,
                total_optimizer_steps=d['metrics']['gradient_update_steps'],wall_seconds=d['wall_seconds'],update_seconds=d['metrics']['update_compute_time'],collection_eval_mode=True))
        name=f"{i['parameter']}-k{i['severity']:g}-full-joint-seed{i['adaptation_seed']}"
        d=r.read(a.reference/'full-joint-home-01/runs-fixed'/name/'completion.json')
        comparison_costs.append(dict(**condition(i),adaptation_seed=i['adaptation_seed'],method='Joint RL',transitions=d['transitions'],
            dynamics_steps=d['updates'],q_steps=d['updates'],total_optimizer_steps=d['optimizer_step_calls'],wall_seconds=d['wall_seconds'],update_seconds=d['update_seconds'],collection_eval_mode=False))
    r.table(a.output/'update_compute.csv',comparison_costs);r.table(a.output/'update_diagnostics.csv',diagnostics)
    fig,axes=plt.subplots(2,4,figsize=(18,8),layout='constrained')
    for col,(f,param) in enumerate(r.config(r.DEFAULT)['families'].items()):
        for row,m in enumerate(['dynamics_loss','q_loss']):
            for k in r.config(r.DEFAULT)['severities']:
                g=defaultdict(list)
                for x in diagnostics:
                    if x['family']==f and x['severity']==k:g[x['transition']].append(x[m])
                xx=sorted(g);axes[row,col].plot(xx,[np.mean(g[t]) for t in xx],label=f'k={k:g}')
            axes[row,col].set(title=param,xlabel='Transitions',ylabel=m);axes[row,col].legend(fontsize=7)
    fig.savefig(figdir/'08_training_diagnostics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    methods=['Frozen','Dynamics-only','Dynamics+Q','Joint RL']
    for ax,metric,title in zip(axes,['wall_seconds','total_optimizer_steps'],['Wall seconds (hardware/load confounded)','Optimizer step calls']):
        for j,method in enumerate(methods):
            values=[x[metric] for x in comparison_costs if x['method']==method]
            ax.scatter([j]*len(values),values,alpha=.3,s=12)
        ax.set(xticks=range(4),xticklabels=methods,ylabel=title)
        ax.tick_params(axis='x',rotation=20)
    fig.savefig(figdir/'09_compute.png',dpi=160);plt.close(fig)
    r.write(a.output/'manifest.json',dict(configuration=c,source_hashes=sources(),selection=selected,
        inputs={str(p.resolve()):r.sha(p/'completion.json') for p in [a.runs,a.prediction,a.control]+([a.nominal] if a.nominal else [])}))
    (a.output/'REACHER_DQ_FOUR_WAY_COMPARISON.md').write_text('# Four-way Reacher comparison\n\n'
        'Complete paired grid: 72 adaptation streams and 20 held-out seeds per stream. No Joint own-encoder MSE is pooled with fixed-encoder errors. '
        'Per-seed differences and per-stream summaries are separate. All nominal family labels are retained but not interpreted as independent tasks. '
        'Dynamics-only and D+Q schedules match, but changed Q changes collection trajectories; this is an algorithm-level ablation, not pure causal Q isolation. '
        'Joint uses exploratory collection and updates other heads. No test-return-based tuning. Q entropy is not calibrated uncertainty.\n')
    r.done(a.output,paired_shifted_episodes=1440,streams=72,nominal_retention='PASS' if dqnom else 'NOT_RUN')


def package(a):
    root=a.runs or a.reference
    assert root and root.is_dir(),'--runs must name the package root'
    dest=a.output
    assert dest.suffix=='.zip' and not dest.exists()
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix not in ('.md','.csv','.json','.png','.py'):continue
            if p.name in ('steps.csv','online_steps.csv'):continue
            if 'interrupted-' in str(p):continue
            z.write(p,Path(root.name)/p.relative_to(root))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None;number=len(z.namelist())
    sha=r.sha(dest);dest.with_suffix('.zip.sha256').write_text(sha+'  '+dest.name+'\n')
    print(json.dumps(dict(zip=str(dest.resolve()),sha256=sha,bytes=dest.stat().st_size,files=number),indent=2))


if __name__=='__main__':main()
