#!/usr/bin/env python3
"""Read-only HalfCheetah Q/planning analysis. Never trains or steps a simulator."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[3]))
from analyze_existing_mechanisms import digest, SHA
from q_planning_core import (Config, require, distributions, js, candidate_bank, compatible,
    component_hashes, state_hash, ReadOnlyHybrid, score, ranks, CORE)

PARAMETERS=('bthigh_damping','torso_mass')
SEVERITIES=(.5,.75,1.,1.25,1.5,2.)
SEEDS=(42000001,42000002,42000003)
EVAL_SEEDS=tuple(range(43000001,43000021))


def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def table(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        if data:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
def key(r):return r['parameter'],float(r['severity']),int(r['adaptation_seed'])
def identity(k):return dict(parameter=k[0],severity=k[1],adaptation_seed=k[2])
def label(k):return f'{k[0]}-k{k[1]:g}-seed{k[2]}'
def selected(a,k):
    return (not a.parameter or k[0]==a.parameter) and (a.severity is None or k[1]==a.severity) and (a.seed is None or k[2]==a.seed)
from experiments.nsgym.analysis.q_statistics import stats
def torchload(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)
def config(a):return read(a.checkpoint/'metadata.json')['config']
def make_model(cfg,state,device):
    from common.world_model import WorldModel
    from common.layers import api_model_conversion
    m=WorldModel(Config(**cfg)).to(device)
    m.load_state_dict(api_model_conversion(m.state_dict(),state),strict=True)
    m.requires_grad_(False);m.eval();return m


def discover(a):
    """Identify by manifests and payload counters, never by a guessed snapshot filename."""
    cp=a.checkpoint/'model.pt'
    if not cp.is_file():
        excluded=[dict(method='all',path=str(cp),reason='Nominal checkpoint missing; model inference unavailable')]
        table(a.output/'tables/model_inventory.csv',[]);table(a.output/'tables/excluded_models.csv',excluded)
        write(a.output/'provenance/inventory.json',dict(status='NOT_RUN',matched=[],excluded=excluded))
        print('Inventory NOT_RUN: nominal checkpoint missing');return
    require(digest(cp)==SHA,'Wrong primary checkpoint')
    meta=read(a.checkpoint/'metadata.json');require(meta['model_sha256']==SHA,'Checkpoint metadata mismatch')
    cfg=meta['config'];require(cfg['task']=='nsgym-halfcheetah' and cfg['episode_length']==1000,'Environment mismatch')
    require(cfg['num_bins']==101 and cfg['vmin']==-10 and cfg['vmax']==10,'Unexpected support')
    paths={'D':a.results_root/'dynamics-only-750k-v1/runs/fixed-decision',
           'DQ':a.results_root/'dynamics-q-750k-v1/runs/fixed-decision',
           'J':a.results_root/'full-joint-750k-v1/runs-fixed'}
    expected={(p,k,s) for p in PARAMETERS for k in SEVERITIES for s in SEEDS if selected(a,(p,k,s))}
    found={m:{} for m in paths};excluded=[]
    nominal=torchload(cp)['model']
    frozen=make_model(cfg,nominal,a.device)
    for method,root in paths.items():
        if not root.exists():
            excluded.append(dict(method=method,path=str(root),reason='Run directory missing'));continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():continue
            try:
                manifest=read(child/'manifest.json');done=read(child/'completion.json')
                i=manifest['identity'];k=key(i)
                if k not in expected:continue
                require(done['status']=='PASS' and i['schedule']=='fixed','Incomplete or non-fixed stream')
                require(i['method']=={'D':'decision','DQ':'dynamics_q_118','J':'full_joint_rl'}[method],'Method identity mismatch')
                require(manifest['checkpoint_sha256']==SHA,'Initial checkpoint mismatch')
                protocol=manifest.get('configuration',manifest.get('protocol'))
                require(tuple(protocol['adaptation_seeds'])==SEEDS and tuple(protocol['evaluation_seeds'])==EVAL_SEEDS,'Seed split mismatch')
                candidates=[]
                for path in child.glob('*.pt'):
                    # Check known completion hash before deserializing J/DQ files.
                    sha=digest(path)
                    if method!='D' and sha!=done['snapshot_sha256']:continue
                    payload=torchload(path)
                    if method=='D':
                        if payload.get('adapter',{}).get('counters',[0,0])[1]!=3000:continue
                        require(payload['adapter']['counters'][3]==118,'D update count')
                        require(payload['model_hash']==done['final_model_hash'],'D final model hash')
                        require(payload['fixed_hash']==done['final_fixed_hash']==manifest['initial_fixed_hash'],'D frozen hash')
                        updates=118
                    elif method=='DQ':
                        require(done['transitions']==3000 and done['dynamics_steps']==done['q_steps']==118,'DQ update count')
                        require(payload['checkpoint_config']==cfg,'DQ model config');updates=236
                    else:
                        require(payload['transitions']==3000 and payload['updates']==2001,'J update count')
                        require(payload['configuration']==cfg,'J model config');updates=4002
                    require(payload['identity']==i and payload['checkpoint_sha256']==SHA,'Payload identity mismatch')
                    require(payload.get('protocol',payload.get('configuration'))==protocol,'Snapshot training configuration mismatch')
                    if method=='J':require(manifest.get('evaluation_seed_replay_access') is False,'Joint held-out replay contract')
                    require(k not in found[method],'Duplicate stream identity')
                    candidates.append(dict(**identity(k),method=method,snapshot=str(path.resolve()),snapshot_sha256=sha,
                        manifest=str((child/'manifest.json').resolve()),manifest_sha256=digest(child/'manifest.json'),
                        completion=str((child/'completion.json').resolve()),completion_sha256=digest(child/'completion.json'),
                        interactions=3000,optimizer_steps=updates,initial_checkpoint_sha256=SHA,
                        source_revision=manifest.get('source_revision','NOT_RECORDED')))
                require(len(candidates)==1,'Expected one valid final snapshot')
                found[method][k]=candidates[0]
            except (KeyError,ValueError,FileNotFoundError,IndexError) as exc:
                excluded.append(dict(method=method,path=str(child),reason=str(exc)))
    inventory=[];matched=[]
    for k in sorted(expected):
        if not all(k in found[m] for m in paths):
            excluded.append(dict(**identity(k),method='matched',reason='Missing/incompatible method'));continue
        items={m:found[m][k] for m in paths}
        try:
            models=load_models(a,items,frozen=frozen,nominal=nominal)
            hashes=compatible(models)
            for m in ('F','D','DQ','J'):
                inventory.append(dict(**identity(k),method=m,environment=cfg['environment_id'],
                    initial_checkpoint_sha256=SHA,interactions=0 if m=='F' else 3000,
                    optimizer_steps=0 if m=='F' else items[m]['optimizer_steps'],
                    dynamics_updates={'F':0,'D':118,'DQ':118,'J':2001}[m],
                    critic_updates={'F':0,'D':0,'DQ':118,'J':2001}[m],
                    actor_updates=2001 if m=='J' else 0,
                    source_revision=meta.get('provenance',{}).get('git_revision','NOT_RECORDED') if m=='F' else items[m]['source_revision'],
                    snapshot_sha256=SHA if m=='F' else items[m]['snapshot_sha256'],**hashes[m]))
            matched.append(dict(**identity(k),models=items))
            del models
        except (ValueError,RuntimeError) as exc:
            excluded.append(dict(**identity(k),method='matched',reason=str(exc)))
    table(a.output/'tables/model_inventory.csv',inventory);table(a.output/'tables/excluded_models.csv',excluded)
    write(a.output/'provenance/inventory.json',dict(status='PASS' if len(matched)==len(expected) else 'PARTIAL',
        expected=len(expected),matched=matched,excluded=excluded,checkpoint=str(a.checkpoint.resolve()),
        checkpoint_sha256=SHA,configuration=cfg))
    print(f'Inventory: {len(matched)}/{len(expected)} compatible matched streams',flush=True)


def load_models(a,items,frozen=None,nominal=None):
    from training_support import tree_hash
    cfg=config(a)
    if nominal is None:nominal=torchload(a.checkpoint/'model.pt')['model']
    models={'F':frozen if frozen is not None else make_model(cfg,nominal,a.device)}
    for method,item in items.items():
        require(digest(item['snapshot'])==item['snapshot_sha256'],'Snapshot changed since inventory')
        for name in ('manifest','completion'):
            require(digest(item[name])==item[name+'_sha256'],'Metadata changed since inventory')
        payload=torchload(item['snapshot'])
        if method=='D':
            models[method]=make_model(cfg,nominal,a.device)
            models[method]._dynamics.load_state_dict(payload['adapter']['predictor'],strict=True)
        else:models[method]=make_model(cfg,payload['model'],a.device)
        expected=payload['model_hash'] if method=='D' else read(item['completion']).get('final_model_hash')
        if expected is not None:require(tree_hash(models[method].state_dict())==expected,'Reconstructed full model hash mismatch')
    compatible(models)
    return models


def bank(a):
    frozen=a.reference/'frozen';manifest=read(frozen/'manifest.json');integrity=read(frozen/'model_integrity.json')
    require(read(frozen/'completion.json')['status']=='PASS','Frozen bank not complete')
    require(tuple(manifest['seeds'])==EVAL_SEEDS and manifest['seed_split']=='evaluation','Held-out seed mismatch')
    require(integrity['file_sha256']['model.pt']==SHA,'Frozen checkpoint provenance')
    cfg=integrity['checkpoint_metadata']['config']
    if (a.checkpoint/'metadata.json').exists():require(cfg==config(a),'Bank/evaluation configuration mismatch')
    inventory=[];target=a.output/'banks';target.mkdir(parents=True,exist_ok=False)
    episode={r['data_file']:r for r in rows(frozen/'episodes.csv')}
    for item in read(frozen/'dataset_index.json'):
        c=item['condition'];k=(c['parameter'],float(c['severity']),a.seed or SEEDS[0])
        if not selected(a,k):continue
        require(item['seed'] in EVAL_SEEDS and item['source']=='closed_loop_frozen','Bank origin')
        require(c['schedule']=='fixed','Non-fixed bank')
        path=frozen/item['file'];require(digest(path)==item['sha256'],'Raw bank hash mismatch')
        ep=episode[item['file']]
        require(ep['terminated']=='False' and ep['truncated']=='True' and int(ep['episode_length'])==1000,'Cannot infer per-step terminal flags')
        with np.load(path) as raw:
            require(raw['actions'].shape==(1000,6) and raw['observations'].shape==(1001,17),'Bank dimensions')
            ix=np.unique(np.linspace(0,999,a.observations,dtype=int))
            act=raw['actions'][ix];require(np.isfinite(act).all() and np.max(np.abs(act))<=1,'Invalid stored action')
            data=dict(observations=raw['observations'][ix],actions=act,next_observations=raw['observations'][ix+1],
                rewards=raw['rewards'][ix],timestep=ix,terminated=np.zeros(len(ix),dtype=bool),truncated=ix==999,
                actual_parameter=raw['actual_parameter'][ix],root_velocity=raw['qvel'][ix,0],
                action_saturated=np.any(np.abs(act)>=.99,axis=-1),
                observation_regime=np.where(ix<100,'early','later'),
                velocity_regime=np.where(raw['qvel'][ix,0]<0,'backward',np.where(raw['qvel'][ix,0]<5,'0_to_5','above_5')))
            require(np.allclose(raw['actual_parameter'],float(ep['actual_parameter_min']),rtol=0,atol=1e-10)
                    and float(ep['actual_parameter_min'])==float(ep['actual_parameter_max']),'Fixed physical parameter mismatch')
            for name in ('qacc','actuator_force'):
                if name in raw:data[name]=raw[name][ix]
            if 'qacc' in raw:
                response=np.linalg.norm(raw['qacc'][:-1],axis=-1)
                edges=np.quantile(response,[1/3,2/3])
                data['physical_response_regime']=np.array(['low','middle','high'])[np.searchsorted(edges,response[ix])]
                data['physical_response_tertile_edges']=edges
            require(all(np.isfinite(v).all() for v in data.values() if v.dtype.kind in 'fi'),'Nonfinite bank')
            dest=target/item['file'];np.savez_compressed(dest,**data)
        inventory.append(dict(parameter=k[0],severity=k[1],evaluation_seed=item['seed'],environment=cfg['environment_id'],
            samples=len(ix),bank_file=str(dest.resolve()),bank_sha256=digest(dest),source_trajectory=str(path.resolve()),
            source_sha256=item['sha256'],origin=item['source'],checkpoint_sha256=SHA,
            actual_parameter=float(data['actual_parameter'][0]),nominal_parameter=float(data['actual_parameter'][0])/k[1],
            terminal_provenance='episode metadata: no termination, TimeLimit at 1000',
            simulator_versions=json.dumps(integrity['checkpoint_metadata'].get('provenance',{}).get('versions',{}))))
    require(inventory,'No bank inputs selected')
    table(a.output/'tables/common_bank_inventory.csv',inventory)
    write(a.output/'provenance/bank.json',dict(status='PASS',files=len(inventory),samples=sum(x['samples'] for x in inventory),
        index_sha256=digest(frozen/'dataset_index.json'),integrity_sha256=digest(frozen/'model_integrity.json'),
        checkpoint_sha256=SHA,environment=cfg['environment_id'],observation_regime='time since reset; no learned clustering',
        force_saturation='NOT_MEASURED: actuator limits not in bank; action saturation is separate'))
    print(f'Bank: {len(inventory)} held-out episodes',flush=True)


def inputs(a):
    inv=read(a.output/'provenance/inventory.json')
    require(inv['checkpoint_sha256']==digest(a.checkpoint/'model.pt')==SHA,'Checkpoint changed')
    banks=rows(a.output/'tables/common_bank_inventory.csv')
    for entry in inv['matched']:
        k=key(entry)
        if not selected(a,k):continue
        selected_banks=[r for r in banks if r['parameter']==k[0] and float(r['severity'])==k[1]]
        require({int(r['evaluation_seed']) for r in selected_banks}==set(EVAL_SEEDS),'Need all 20 held-out bank seeds')
        yield entry,selected_banks


def bank_data(row):
    require(digest(row['bank_file'])==row['bank_sha256'],'Canonical bank mutated')
    with np.load(row['bank_file']) as d:return dict(d)


def inference(a,terminal=False):
    import torch
    from common import math as tdmath
    cfg=Config(**config(a));discount=min(max(1-cfg.discount_denom/cfg.episode_length,cfg.discount_min),cfg.discount_max)
    out=a.output;metrics=[];divergences=[];planning=[];rankings=[];terminalrows=[];counter=[]
    stage='terminal' if terminal else 'immediate';archives=out/stage;archives.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();completed=[]
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device=a.device)
    for entry,banks in inputs(a):
        k=key(entry)
        if terminal and not a.full_grid and k[1]!=2:continue
        models=load_models(a,entry['models']);before={m:state_hash(v) for m,v in models.items()}
        with torch.inference_mode():
            if not terminal:
                for b in banks:
                    data=bank_data(b);o=tensor(data['observations']);act=tensor(data['actions']);n=tensor(data['next_observations'])
                    packs={};saved={}
                    base=dict(**identity(k),evaluation_seed=int(b['evaluation_seed']),bank_sha256=b['bank_sha256'],
                              actual_parameter=float(b['actual_parameter']),nominal_parameter=float(b['nominal_parameter']))
                    for method,model in models.items():
                        z=model.encode(o,None);logits=model.Q(z,act,None,return_type='all');pack=distributions(logits,cfg);packs[method]=pack
                        # Original Bellman rule, no MC calibration claim. Fork RNG preserves caller state.
                        devices=[torch.device(a.device).index or 0] if str(a.device).startswith('cuda') else []
                        with torch.random.fork_rng(devices=devices):
                            torch.manual_seed(310000+int(b['evaluation_seed']))
                            zn=model.encode(n,None);nextact,_=model.pi(zn,None)
                            target=tensor(data['rewards']).reshape(-1,1)+discount*(1-tensor(data['terminated']).reshape(-1,1))*model.Q(zn,nextact,None,return_type='min',target=True)
                        residual=pack['scalar']-target.flatten()[None]
                        projection=tdmath.two_hot(target,cfg)
                        ce=-(projection.unsqueeze(0)*logits.log_softmax(-1)).sum(-1)
                        for h in range(cfg.num_q):
                            for j,t in enumerate(data['timestep']):
                                metrics.append(dict(**base,method=method,head=h,timestep=int(t),
                                    q=float(pack['scalar'][h,j]),entropy=float(pack['entropy'][h,j]),
                                    max_probability=float(pack['maximum'][h,j]),boundary_mass=float(pack['boundary'][h,j]),
                                    ensemble_q_mean=float(pack['scalar'][:,j].mean()),ensemble_std=float(pack['ensemble_std'][j]),
                                    interhead_js=float(pack['interhead_js'][j]),bellman_residual=float(residual[h,j]),
                                    bellman_soft_ce=float(ce[h,j]),action_saturation=bool(data['action_saturated'][j]),
                                    observation_regime=str(data['observation_regime'][j]),velocity_regime=str(data['velocity_regime'][j]),
                                    physical_response_regime=str(data['physical_response_regime'][j]) if 'physical_response_regime' in data else 'NOT_MEASURED'))
                        saved[method+'_probability']=pack['probability'].cpu().numpy();saved[method+'_logits']=logits.cpu().numpy()
                    torch.testing.assert_close(packs['F']['probability'],packs['D']['probability'],rtol=1e-6,atol=1e-7)
                    for left,right in [('F','D'),('F','DQ'),('F','J'),('DQ','J')]:
                        distance=js(packs[left]['mixture'],packs[right]['mixture'])
                        for j,t in enumerate(data['timestep']):
                            divergences.append(dict(**base,left=left,right=right,timestep=int(t),
                                js=float(distance[j]),q_difference=float(packs[right]['scalar'][:,j].mean()-packs[left]['scalar'][:,j].mean())))
                    saved['symlog_support']=np.linspace(cfg.vmin,cfg.vmax,cfg.num_bins);saved['timestep']=data['timestep']
                    np.savez_compressed(archives/f'{label(k)}-eval{b["evaluation_seed"]}.npz',**saved)
            else:
                variants={'F':models['F'],'D':models['D'],'DQ-dyn-only':ReadOnlyHybrid(models['DQ'],models['F']),
                          'DQ-full':models['DQ'],'TTT+DQ-Q':ReadOnlyHybrid(models['D'],models['DQ']),
                          'F+DQ-Q':ReadOnlyHybrid(models['F'],models['DQ']),'J':models['J']}
                # One observation from each of the first N held-out seeds; no future input.
                for b in banks[:a.planning_observations]:
                    data=bank_data(b);o=tensor(data['observations'][0]);es=int(b['evaluation_seed'])
                    tape=candidate_bank(a.horizon or cfg.horizon,a.candidates,cfg.action_dim,es+8123,cfg.max_std)
                    require(a.candidates>=cfg.num_elites,'Candidate count must cover native num_elites')
                    file=archives/f'{label(k)}-eval{es}-candidates.npz';np.savez_compressed(file,actions=tape,observation=o.cpu().numpy())
                    base=dict(**identity(k),evaluation_seed=es,timestep=int(data['timestep'][0]),
                        nominal_parameter=float(b['nominal_parameter']),actual_parameter=float(b['actual_parameter']),
                        candidate_sha256=digest(file),bank_sha256=b['bank_sha256'],horizon=len(tape))
                    captures={};terminalpacks={};saved={}
                    for method,model in variants.items():
                        cap=score(model,o,tensor(tape),cfg,discount,es+9921);captures[method]=cap
                        # Controlled terminal-Q diagnostics fix a_H to the Frozen sampled action.
                        fixed_action=captures['F']['terminal_action']
                        critic=model.critic if isinstance(model,ReadOnlyHybrid) else model
                        q=critic.Q(cap['terminal_latent'],fixed_action,None,return_type='all')
                        pack=distributions(q,cfg);terminalpacks[method]=pack
                        saved[method+'_latent']=cap['terminal_latent'].cpu().numpy()
                        saved[method+'_native_action']=cap['terminal_action'].cpu().numpy()
                        saved[method+'_fixed_action_probability']=pack['probability'].cpu().numpy()
                        saved[method+'_native_probability']=distributions(cap['terminal_logits'],cfg)['probability'].cpu().numpy()
                        elite=cap['elite'].cpu().tolist();weights=dict(zip(elite,cap['weight'].cpu().tolist()))
                        for j in range(a.candidates):
                            planning.append(dict(**base,method=method,candidate=j,reward_sum=float(cap['reward'][j,0]),
                                terminal_contribution=float(cap['terminal'][j,0]),total_score=float(cap['score'][j]),
                                terminal_abs_fraction=float(cap['terminal'][j,0].abs()/(cap['reward'][j,0].abs()+cap['terminal'][j,0].abs()+1e-12)),
                                predicted_rank=float(__import__('analyze_existing_mechanisms').rankdata(-cap['score'].cpu().numpy())[j]),
                                elite_weight=weights.get(j,0),ess=cap['ess']))
                            terminalrows.append(dict(**base,method=method,candidate=j,q=float(pack['scalar'][:,j].mean()),
                                entropy=float(pack['entropy'][:,j].mean()),ensemble_std=float(pack['ensemble_std'][j]),
                                terminal_action_rule='common Frozen sampled a_H; distinct from native planning action'))
                        saved[method+'_score']=cap['score'].cpu().numpy()
                    for left,right in [('F','D'),('D','DQ-full'),('F','DQ-full'),('F','J'),('DQ-dyn-only','DQ-full'),('D','TTT+DQ-Q')]:
                        le=set(captures[left]['elite'].cpu().tolist());re=set(captures[right]['elite'].cpu().tolist())
                        rankings.append(dict(**base,left=left,right=right,elite_set_overlap=len(le&re)/len(le),
                            **ranks(captures[left]['score'].cpu().numpy(),captures[right]['score'].cpu().numpy(),min(a.top_k,a.candidates))))
                    qmean=lambda m:terminalpacks[m]['scalar'].mean(0)
                    for j in range(a.candidates):
                        counter.append(dict(**base,candidate=j,dynamics_effect=float((qmean('DQ-dyn-only')-qmean('F'))[j]),
                            q_head_effect=float((qmean('DQ-full')-qmean('DQ-dyn-only'))[j]),
                            interaction=float((qmean('DQ-full')-qmean('DQ-dyn-only')-qmean('F+DQ-Q')+qmean('F'))[j])))
                    saved['fixed_terminal_action']=captures['F']['terminal_action'].cpu().numpy()
                    np.savez_compressed(archives/f'{label(k)}-eval{es}-terminal.npz',**saved)
        require(all(state_hash(models[m])==before[m] for m in models),'Inference mutated model state')
        completed.append(identity(k));print(f'{stage} PASS {label(k)}',flush=True)
        del models
    if terminal:
        for name,data in [('terminal_q_metrics',terminalrows),('planning_score_components',planning),
                          ('candidate_ranking_comparison',rankings),('terminal_counterfactuals',counter)]:table(out/f'tables/{name}.csv',data)
    else:
        table(out/'tables/immediate_q_metrics.csv',metrics);table(out/'tables/q_distribution_divergence.csv',divergences)
    write(out/f'provenance/{stage}.json',dict(status='PASS' if completed else 'NOT_RUN',completed=completed,
        seconds=time.monotonic()-started,optimizer_steps=0,environment_transitions=0,model_state_unchanged=True,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        calibration='NOT_MEASURED',ranking_correctness='NOT_MEASURED'))


def aggregate(a):
    from collections import defaultdict
    out=a.output;immediate=rows(out/'tables/immediate_q_metrics.csv')
    divergence=rows(out/'tables/q_distribution_divergence.csv')
    # Within each adaptation stream average heads/observations, then evaluation seeds.
    from experiments.nsgym.analysis.q_statistics import summarize
    stream,summ=summarize(immediate,['parameter','severity','method'],['q','entropy','ensemble_std','boundary_mass','interhead_js','bellman_residual','bellman_soft_ce'])
    table(out/'tables/immediate_per_stream.csv',stream);table(out/'tables/immediate_summary.csv',summ)
    divstream,divsumm=summarize(divergence,['parameter','severity','left','right'],['js','q_difference'])
    table(out/'tables/divergence_per_stream.csv',divstream);table(out/'tables/divergence_summary.csv',divsumm)
    for regime in ('action_saturation','observation_regime','velocity_regime','physical_response_regime'):
        st,su=summarize(immediate,['parameter','severity','method',regime],['q','entropy','ensemble_std'])
        table(out/f'tables/{regime}_summary.csv',su)
    for name,group,fields in [('terminal_q_metrics',['parameter','severity','method'],['q','entropy','ensemble_std']),
        ('planning_score_components',['parameter','severity','method'],['reward_sum','terminal_contribution','total_score','ess']),
        ('candidate_ranking_comparison',['parameter','severity','left','right'],['top_k_overlap','argmax_disagreement']),
        ('terminal_counterfactuals',['parameter','severity'],['dynamics_effect','q_head_effect','interaction'])]:
        path=out/f'tables/{name}.csv'
        if path.exists() and path.stat().st_size:
            st,su=summarize(rows(path),group,fields);table(out/f'tables/{name}_per_stream.csv',st);table(out/f'tables/{name}_summary.csv',su)
    association=[]
    control=a.comparison/'per_adaptation_seed_comparison.csv'
    if control.exists():
        require(read(a.comparison/'completion.json')['status']=='PASS','Control comparison not PASS')
        controls={key(r):r for r in rows(control)}
        from analyze_existing_mechanisms import Sources,load_stage
        sources=Sources();raw_control={}
        inventory=read(out/'provenance/inventory.json')
        for method,relative in [('D','dynamics-only-750k-v1/final-snapshot-v2/shifted'),
                                ('DQ','dynamics-q-750k-v1/shifted'),('J','joint-final-evaluation-v1/shifted')]:
            raw,metadata,stage=load_stage(a.results_root/relative,'shifted',sources)
            raw_control[method]=raw
            for entry in inventory['matched']:
                require(metadata[key(entry)]['snapshot_sha256']==entry['models'][method]['snapshot_sha256'],
                        'Control outcomes refer to a different snapshot')
                rr=[r for r in raw if key(r)==key(entry)]
                require({int(r['evaluation_seed']) for r in rr}==set(EVAL_SEEDS),'Control seed pairing mismatch')
                field={'D':'decision_return','DQ':'episode_return','J':'joint_return'}[method]
                cached={'D':'ttt_return','DQ':'dq_return','J':'joint_return'}[method]
                require(np.isclose(np.mean([float(r[field]) for r in rr]),float(controls[key(entry)][cached]),rtol=1e-10,atol=1e-6),'Cached control mean mismatch')
        write(out/'provenance/control_inputs.json',sources.files)
        for r in divstream:
            if r['left']!='F':continue
            c=controls.get(key(r))
            if c is None:continue
            target={'D':'ttt','DQ':'dq','J':'joint'}[r['right']]
            association.append(dict(r,return_difference=float(c[target+'_return'])-float(c['frozen_return']),
                **{k:v for k,v in c.items() if k.endswith('_return') or 'velocity' in k or 'control_cost' in k}))
            diagnostics=next(x for x in stream if key(x)==key(r) and x['method']==r['right'])
            association[-1].update({k:diagnostics[k] for k in ('entropy','ensemble_std','boundary_mass','bellman_residual')})
            for method,velocity,cost in [('D','decision_velocity','decision_control_cost'),
                                         ('DQ','forward_velocity','control_cost'),('J','joint_forward_velocity','joint_control_cost')]:
                rr=[x for x in raw_control[method] if key(x)==key(r)]
                association[-1][method+'_forward_velocity']=float(np.mean([float(x[velocity]) for x in rr]))
                association[-1][method+'_control_cost']=float(np.mean([float(x[cost]) for x in rr]))
                if method=='D':
                    association[-1]['F_forward_velocity']=float(np.mean([float(x['frozen_velocity']) for x in rr]))
                    association[-1]['F_control_cost']=float(np.mean([float(x['frozen_control_cost']) for x in rr]))
            rankpath=out/'tables/candidate_ranking_comparison_per_stream.csv'
            if rankpath.exists():
                match=next((x for x in rows(rankpath) if key(x)==key(r) and x['left']=='F' and x['right']=={'D':'D','DQ':'DQ-full','J':'J'}[r['right']]),None)
                if match:association[-1].update({k:match[k] for k in ('top_k_overlap','argmax_disagreement')})
    table(out/'tables/q_control_association.csv',association)
    write(out/'provenance/aggregate.json',dict(status='PASS',association_rows=len(association),
        control_source=str(control),control_sha256=digest(control) if control.exists() else None,
        inference='descriptive; bootstrap adaptation seeds n=3; no time-step independence; no pooled nominal duplicates'))


def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=a.output;figs=out/'figures';figs.mkdir(exist_ok=True)
    require(not any(figs.glob('*.png')),'Figures already exist; use a fresh versioned output')
    summary=rows(out/'tables/immediate_summary.csv')
    def curves(data,metric,file,title,models=('F','D','DQ','J')):
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        banks=rows(out/'tables/common_bank_inventory.csv')
        for ax,p in zip(axes,PARAMETERS):
            for m in models:
                rr=sorted([r for r in data if r['parameter']==p and r.get('method')==m and r['metric']==metric],key=lambda r:float(r['severity']))
                if rr:
                    ax.plot([float(r['severity']) for r in rr],[float(r['mean']) for r in rr],marker='o',label=m)
                    ax.fill_between([float(r['severity']) for r in rr],[float(r['ci_low']) for r in rr],[float(r['ci_high']) for r in rr],alpha=.1)
            nominal=next((r['nominal_parameter'] for r in banks if r['parameter']==p),'unavailable')
            ax.set(title=f'{p}; nominal={nominal}',xlabel='Physical multiplier',ylabel=title)
            if ax.lines:ax.legend()
        fig.savefig(figs/file,dpi=160);plt.close(fig)
    curves(summary,'q','scalar_q_vs_severity.png','Decoded scalar Q')
    curves(summary,'entropy','q_entropy_by_severity.png','Categorical entropy (nats)')
    curves(summary,'ensemble_std','q_ensemble_disagreement.png','Between-head scalar SD')
    ds=rows(out/'tables/divergence_summary.csv')
    for r in ds:r['method']=r['left']+'→'+r['right']
    curves(ds,'js','q_js_divergence.png','Ensemble-mixture JS (nats)',('F→D','F→DQ','F→J','DQ→J'))
    # Categorical heatmaps use shared symlog coordinates. Scalar decode is NOT E[symexp(bin)].
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for row,p in enumerate(PARAMETERS):
        for col,m in enumerate(('F','DQ','J')):
            means=[];labels=[];axis=axes[row,col]
            for severity in SEVERITIES:
                files=list((out/'immediate').glob(f'{p}-k{severity:g}-seed*-eval*.npz'))
                if not files:continue
                values=[]
                for f in files:
                    with np.load(f) as d:values.append(d[m+'_probability'].mean(axis=(0,1)));bins=d['symlog_support']
                means.append(np.mean(values,axis=0));labels.append(severity)
            if means:
                im=axis.imshow(np.stack(means),aspect='auto',origin='lower',vmin=0,vmax=1,extent=[bins[0],bins[-1],-.5,len(labels)-.5])
                axis.set_yticks(range(len(labels)),labels);fig.colorbar(im,ax=axis,label='Mean bin probability')
            else:axis.text(.5,.5,'NOT_MEASURED',ha='center',transform=axis.transAxes)
            nominal=next((r['nominal_parameter'] for r in rows(out/'tables/common_bank_inventory.csv') if r['parameter']==p),'unavailable')
            axis.set(title=f'{p} / {m}; nominal={nominal}',xlabel='symlog value support',ylabel='Severity')
    fig.savefig(figs/'q_distribution_by_severity.png',dpi=160);plt.close(fig)
    variants=('F','D','DQ-dyn-only','DQ-full','TTT+DQ-Q','F+DQ-Q','J')
    for name,metric,file,title in [('terminal_q_metrics','q','terminal_q_decomposition.png','Terminal Q; common Frozen a_H'),
        ('planning_score_components','reward_sum','planning_reward_decomposition.png','Discounted predicted reward'),
        ('planning_score_components','terminal_contribution','planning_terminal_decomposition.png','Discounted native terminal Q'),
        ('planning_score_components','total_score','planning_score_decomposition.png','Native candidate score')]:
        path=out/f'tables/{name}_summary.csv'
        if path.exists():curves(rows(path),metric,file,title,variants)
    path=out/'tables/candidate_ranking_comparison_summary.csv'
    if path.exists():
        rr=rows(path)
        for r in rr:r['method']=r['left']+'→'+r['right']
        curves(rr,'top_k_overlap','ranking_disagreement.png','Top-k overlap',tuple(sorted({r['method'] for r in rr})))
    assoc=rows(out/'tables/q_control_association.csv') if (out/'tables/q_control_association.csv').stat().st_size else []
    if assoc:
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,p in zip(axes,PARAMETERS):
            for m in ('D','DQ','J'):
                rr=[r for r in assoc if r['parameter']==p and r['right']==m and float(r['severity'])!=1]
                ax.scatter([float(r['js']) for r in rr],[float(r['return_difference']) for r in rr],label=m)
            ax.axhline(0,color='gray');ax.set(title=p,xlabel='JS vs Frozen',ylabel='Paired return difference vs Frozen');ax.legend()
        fig.savefig(figs/'q_change_vs_control.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,metric in zip(axes,('entropy','ensemble_std')):
            for p in PARAMETERS:
                for m in ('D','DQ','J'):
                    rr=[r for r in assoc if r['parameter']==p and r['right']==m and float(r['severity'])!=1]
                    ax.scatter([float(r[metric]) for r in rr],[float(r['return_difference']) for r in rr],label=p+'/'+m)
            ax.axhline(0,color='gray');ax.set(xlabel=metric,ylabel='Return difference vs Frozen');ax.legend(fontsize=6)
        fig.savefig(figs/'q_uncertainty_vs_control.png',dpi=160);plt.close(fig)
    terminal=out/'tables/terminal_q_metrics_per_stream.csv'
    if terminal.exists():
        rr=rows(terminal);fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,p in zip(axes,PARAMETERS):
            kk=sorted({float(r['severity']) for r in rr if r['parameter']==p})
            for k in kk:
                ff={int(r['adaptation_seed']):float(r['q']) for r in rr if r['parameter']==p and float(r['severity'])==k and r['method']=='F'}
                dd=[float(r['q'])-ff[int(r['adaptation_seed'])] for r in rr if r['parameter']==p and float(r['severity'])==k and r['method']=='D']
                ax.scatter([k]*len(dd),dd,label=f'terminal k={k:g}')
            ax.axhline(0,color='gray',label='Immediate F=D (asserted)')
            ax.set(title=p,xlabel='Severity',ylabel='Terminal D−F, common action');ax.legend(fontsize=7)
        fig.savefig(figs/'immediate_vs_terminal_q.png',dpi=160);plt.close(fig)
    write(out/'provenance/plot.json',dict(status='PASS',figures=[p.name for p in figs.glob('*.png')]))


def report(a):
    out=a.output;reports=out/'reports';reports.mkdir(exist_ok=True)
    require(not any(reports.glob('*.md')) and not (out/'README.md').exists(),'Reports already exist')
    inventory=read(out/'provenance/inventory.json') if (out/'provenance/inventory.json').exists() else {'matched':[],'status':'NOT_RUN'}
    statuses={name:read(out/f'provenance/{name}.json') if (out/f'provenance/{name}.json').exists() else {'status':'NOT_RUN'} for name in ('bank','immediate','terminal','aggregate')}
    common=('No training, optimizer construction or simulator transitions. Existing results are read-only.\n\n'
        'Model-only hydration follows original state formats: F model.pt via api_model_conversion; '
        'D adapter.predictor into nominal dynamics; DQ/J full saved model including target Q. Original loaders construct '
        'optimizers/environments, so are deliberately not invoked; no original files were modified.\n\n'
        'Single nominal training seed; three adaptation replicas per condition. k=1 labels remain separate facets but '
        'must not be pooled as independent physical tasks. Fixed-bank rows/time steps are not independent model replicas.\n')
    texts={
        'Q_DISTRIBUTION_AUDIT.md':f'# Q distribution audit\n\nInventory: {inventory["status"]}, {len(inventory["matched"])} matched streams.\n\n'+common+
            'OBSERVED (source): 5 heads, 101 uniform symlog bins [-10,10]. Softmax logits, then symexp(sum p*symlog_bin). '
            'This differs from sum p*symexp(bin). Scalar ensemble mean is descriptive; native MPPI uses average of two randomly selected heads. '
            'Boundary mass uses first/last two bins; model JS uses ensemble probability mixtures, not head-index matching.\n',
        'MODEL_COMPATIBILITY.md':'# Compatibility\n\n'+common+'See model_inventory.csv and excluded_models.csv. '
            'F/D encoder,reward,Q,actor,target-Q equality is required; F/DQ encoder,reward,actor equality required. '
            'Changed trained groups are asserted. Joint hybrids across encoder spaces are prohibited.\n',
        'IMMEDIATE_Q_ANALYSIS.md':f'# Immediate Q\n\nStatus: {statuses["immediate"]["status"]}.\n\n'+
            'SUPPORTED INTERPRETATION: F and D immediate Q must coincide because the encoder and critic parameters are equal. '
            'This is asserted at probability level for every measured bank input. It does not imply equal terminal Q.\n',
        'TERMINAL_Q_ANALYSIS.md':f'# Terminal Q\n\nStatus: {statuses["terminal"]["status"]}.\n\n'+
            'Controlled terminal comparisons use a shared Frozen sampled terminal action on each native rollout latent. '
            'Dynamics effect, head effect and interaction are in terminal_counterfactuals.csv. Native planner scores instead '
            'use each rollout actor with common random numbers, including identical random-head-selection seeds. '
            'Joint uses its own encoder; no Joint module swaps. Raw terminal latents/actions/probabilities are archived.\n',
        'PLANNING_RANKING_ANALYSIS.md':f'# Planning ranking\n\nStatus: {statuses["terminal"]["status"]}.\n\n'+
            'Original TDMPC2._estimate_value is called directly. Reward is decoded before each transition; terminal '
            'actor is sampled after H transitions; terminal Q is average of two sampled heads, discounted by gamma^H. '
            'Candidate bank is an exogenous clipped Gaussian with nominal max_std, no future observations or policy prior. '
            'This is not endogenous multi-iteration MPPI. The native elite exponent weighting is reported on this bank; '
            'argmax disagreement is not the native stochastic selected action. No real candidate returns: ranking accuracy/regret NOT_MEASURED. '
            '2× cases are result-selected case studies, not representative conditions.\n',
        'Q_CONTROL_ASSOCIATION.md':f'# Q and control\n\nAggregation: {statuses["aggregate"]["status"]}.\n\n'+
            'Existing paired controls are joined by physical parameter, severity and adaptation seed. '
            'Associations are descriptive; within-condition n=3 is too small for reliable correlation inference. '
            'Joint degradation caused by Q: NOT ESTABLISHED. Larger entropy, JS or ensemble disagreement is not inherently better/worse.\n',
        'LIMITATIONS_AND_NEXT_EXPERIMENTS.md':'# Limitations\n\n'+common+
            'Empirical Q calibration: NOT_MEASURED. Frozen trajectory continuation does not equal each adapted actor policy. '
            'Off-policy Bellman residual and soft cross entropy use actual reward/next observation and saved target-Q; '
            'TimeLimit truncation bootstraps. Small TD residual is not empirical accuracy. '
            'No actual-candidate-return ranking, no causal closed-loop attribution, no optimal severity selection. '
            'Q-update necessity remains HYPOTHESIS. Fixed-bank latent errors for Joint cannot share the nominal encoder metric. '
            'Missing checkpoints or banks require server execution; local CPU fixtures are not scientific findings.\n'}
    artifact_lines=['# Available source artifacts','']
    for name in ['physical-severity-750k-v1/'+x for x in ('bank','frozen','physical-latent','heldout-latent','planning')]+[
        'dynamics-only-750k-v1/runs/fixed-decision','dynamics-only-750k-v1/final-snapshot-v2',
        'dynamics-only-750k-v1/analysis-distribution-v3','dynamics-q-750k-v1',
        'full-joint-750k-v1','joint-final-evaluation-v1']:
        path=a.results_root/name
        counts={ext:sum(1 for _ in path.rglob('*.'+ext)) for ext in ('pt','npz','csv','json')} if path.exists() else {}
        artifact_lines.append(f'- {path}: exists={path.exists()}, file counts={counts}')
    texts['AVAILABLE_ARTIFACTS.md']='\n'.join(artifact_lines)+'\n'
    # Append measured summaries rather than assert a generic result.
    for filename,source in [('IMMEDIATE_Q_ANALYSIS.md','divergence_summary.csv'),
                            ('TERMINAL_Q_ANALYSIS.md','terminal_counterfactuals_summary.csv'),
                            ('PLANNING_RANKING_ANALYSIS.md','candidate_ranking_comparison_summary.csv')]:
        p=out/'tables'/source
        if p.exists() and p.stat().st_size:
            texts[filename]+='\nOBSERVED summaries (seed-level bootstrap intervals; n=3 adaptation seeds):\n\n'
            for r in rows(p):texts[filename]+='- '+json.dumps(r)+'\n'
        else:texts[filename]+='\nOBSERVED scientific results: NOT_RUN / unavailable.\n'
    for name,text in texts.items():(reports/name).write_text(text) if not (reports/name).exists() else require(False,'Report exists; use new output')
    write(out/'provenance/completion.json',dict(status='PASS' if inventory['status']=='PASS' and all(v['status']=='PASS' for v in statuses.values()) else 'PARTIAL',
        inventory=inventory['status'],stages={k:v['status'] for k,v in statuses.items()},training='NOT_RUN',environment_rollouts=0))
    (out/'README.md').write_text('Read reports/Q_DISTRIBUTION_AUDIT.md and provenance/completion.json first.\n'
        'Commands: reproduction/RUN_Q_PLANNING_ANALYSIS.md. NPZ inference archives and canonical banks are excluded from review ZIP.\n')


def package(a):
    out=a.output;repro=out/'reproduction';repro.mkdir(exist_ok=False)
    for p in [Path(__file__),Path(__file__).with_name('q_planning_core.py'),Path(__file__).with_name('mechanism_forward.py'),
              Path(__file__).with_name('analyze_existing_mechanisms.py'),Path(__file__).parents[1]/'RUN_Q_PLANNING_ANALYSIS.md',
              Path(__file__).parents[1]/'tests/test_q_planning.py']:
        shutil.copy2(p,repro/p.name)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.md','.json','.csv','.png','.py') and p.parts[len(out.parts)] not in ('banks','immediate','terminal'):
                z.write(p,p.relative_to(out.parent))
    with zipfile.ZipFile(archive) as z:require(z.testzip() is None,'ZIP integrity');count=len(z.infolist())
    print(json.dumps(dict(zip=str(archive.resolve()),bytes=archive.stat().st_size,files=count,sha256=digest(archive)),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['inventory','bank','immediate','terminal','planning','aggregate','plot','report','package','smoke'])
    p.add_argument('--results-root',type=Path,default=Path('results/nsgym'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--comparison',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--parameter',choices=PARAMETERS)
    p.add_argument('--severity',type=float,choices=SEVERITIES)
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--observations',type=int,default=16,help='Evenly spaced transitions per held-out episode (Tier A)')
    p.add_argument('--planning-observations',type=int,default=2,help='First observation from each of N held-out episodes (Tier B)')
    p.add_argument('--candidates',type=int,default=64)
    p.add_argument('--horizon',type=int,help='Default saved planner horizon; override is analysis-only')
    p.add_argument('--top-k',type=int,default=8)
    p.add_argument('--tier',choices=['A','B'],help='Explicit validation of immediate=A / terminal,planning=B')
    p.add_argument('--full-grid',action='store_true',help='Expand Tier B beyond the two 2x case studies')
    p.add_argument('--models',nargs='+',choices=['F','D','DQ','J'],default=['F','D','DQ','J'],help='Matched four-method contract; all four required')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    a.checkpoint=a.checkpoint or a.results_root/'nominal-server4-20261108-nohup-01/checkpoint-000750000'
    a.reference=a.reference or a.results_root/'physical-severity-750k-v1'
    a.comparison=a.comparison or a.results_root/'dynamics-q-750k-v1/comparison'
    require(1<=a.observations<=1000 and 1<=a.planning_observations<=20,'Sample bounds')
    require(a.candidates>=2 and 1<=a.top_k<=a.candidates and (a.horizon is None or a.horizon>0),'Planning bounds')
    require(set(a.models)=={'F','D','DQ','J'},'This pipeline requires all four models for matched inference')
    if a.tier:require((a.command=='immediate' and a.tier=='A') or (a.command in ('terminal','planning') and a.tier=='B'),'Tier/command mismatch')
    if a.dry_run:
        print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2));return
    a.output.mkdir(parents=True,exist_ok=True)
    if a.command=='smoke':
        import subprocess
        test=Path(__file__).parents[1]/'tests/test_q_planning.py'
        result=subprocess.run([sys.executable,str(test)],check=False)
        require(result.returncode==0,'Bounded CPU tests failed')
        write(a.output/'smoke.json',dict(status='PASS',fixture='synthetic CPU WorldModel; no scientific checkpoint evidence'))
        return
    prov=a.output/'provenance';prov.mkdir(exist_ok=True)
    if not (prov/'analysis_config.json').exists():
        write(prov/'analysis_config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
        write(prov/'input_manifest.json',dict(source_hashes={str(x):digest(x) for x in
            [Path(__file__),Path(__file__).with_name('q_planning_core.py'),CORE/'tdmpc2.py',CORE/'common/world_model.py',CORE/'common/math.py']}))
    else:
        for path,sha in read(prov/'input_manifest.json')['source_hashes'].items():
            require(digest(path)==sha,'Source changed during analysis; start a new versioned output')
    if a.command=='inventory':discover(a)
    elif a.command=='bank':bank(a)
    elif a.command=='immediate':inference(a)
    elif a.command in ('terminal','planning'):inference(a,True)
    else:globals()[a.command](a)


if __name__=='__main__':main()
