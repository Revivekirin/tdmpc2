"""Paired closed-loop evaluation through the original task-specific episode loops.

CUDA and original simulator package versions are required. Nothing runs on import.
"""
import importlib.metadata
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from .config import TASKS, SCRIPTS
from .metrics import read, write, require, sha256
from .models import Sources, inventory, tree_hash
from .diagnostics.q_swap import swap_online_critic


def runtime_check():
    import torch
    require(torch.cuda.is_available(), 'Historical CUDA evaluator unavailable; run on the original server')
    return {name: importlib.metadata.version(name) for name in ('torch','numpy','mujoco','gymnasium','ns_gym')}


def label(job):
    return f"{job['task']}-{job['family']}-k{job['severity']:g}-seed{job['adaptation_seed']}"


def historical(root, job, src):
    task=job['task'];parameter=TASKS[task]['families'][job['family']]
    if task=='halfcheetah':
        parent=root/'dynamics-q-750k-v1/shifted'
        name=f"{parameter}-k{job['severity']:g}-dynamics-q-seed{job['adaptation_seed']}"
    else:
        parent=root/'reacher-dynamics-q-v1/control'
        name=f"{job['family']}-k{job['severity']:g}-dynamics-q-seed{job['adaptation_seed']}"
    require(read(src.track(parent/'completion.json','historical heldout completion'))['status']=='PASS','Historical heldout batch incomplete')
    child=parent/name
    done=read(src.track(child/'completion.json','historical heldout stream'))
    require(done['status']=='PASS','Historical stream incomplete')
    path=src.track(child/'shifted.csv','historical episode results',done.get('csv_sha256'))
    frame=pd.read_csv(path);seed='evaluation_seed' if 'evaluation_seed' in frame else 'seed'
    require(not frame[seed].duplicated().any(),'Duplicate historical evaluation seed')
    require(sorted(frame[seed])==TASKS[task]['evaluation'],'Historical seed mismatch')
    return frame.set_index(seed)


def load_controller(root, job, swapped):
    import torch
    import run_dynamics_only_ttt as ttt
    import reacher_pipeline as r
    task=job['task'];cp=root/TASKS[task]['checkpoint']
    if task=='halfcheetah':
        agent,meta=ttt.load_agent(SimpleNamespace(checkpoint=cp,synthetic=False))
    else:
        agent,meta=r.load(cp,r.config(r.DEFAULT))
    nominal=agent.model
    payload=torch.load(job['paths']['DQ'],map_location=agent.device,weights_only=False)
    from analyze_q_planning import make_model
    dq=make_model(meta['config'],payload['model'],str(agent.device))
    model,isolation=swap_online_critic(dq,nominal)
    agent.model=model if swapped else dq
    agent.model.requires_grad_(False);agent.model.eval()
    return agent,isolation


def episode(agent,root,job,seed,trajectory_dir):
    import run_dynamics_only_ttt as ttt
    import reacher_pipeline as r
    parameter=TASKS[job['task']]['families'][job['family']]
    cond=dict(parameter=parameter,severity=job['severity'])
    if job['task']=='reacher':
        rec,steps=r.episode(agent,dict(cond,family=job['family']),seed,out=trajectory_dir)
        return rec,steps
    # Exact historical seed -> environment -> rollout order; adapter is read-only bookkeeping.
    c=read(SCRIPTS.parent/'configs/dynamics_only_ttt.json')
    ttt.seeded(seed);env=ttt.environment(c,cond,'fixed')
    adapter=SimpleNamespace(gradient_steps=0,transitions=0,update_compute_time=0)
    try:
        steps=[]
        for row,observations,actions in ttt.rollout(agent,adapter,c,cond,'fixed',seed,env,0,False):steps.append(row)
        np.savez_compressed(trajectory_dir/f'seed{seed}.npz',observations=np.array(observations),actions=np.array(actions))
        require(len(steps)==1000 and steps[-1]['truncated'] and not steps[-1]['terminated'],'Episode termination mismatch')
        rec=dict(episode_return=sum(x['reward'] for x in steps),episode_length=len(steps),terminated=False,truncated=True,
            forward_velocity=float(np.mean([x['velocity'] for x in steps])),control_cost=sum(x['control_cost'] for x in steps),
            action_abs=float(np.mean([x['action_abs'] for x in steps])),action_saturation=float(np.mean([x['action_component_saturation'] for x in steps])))
        return rec,steps
    finally:env.close()


def evaluate(root,out,smoke=False,smoke_result=None):
    """Smoke uses four predeclared streams × first heldout seed × two controllers."""
    runtime=runtime_check();src=Sources();jobs,configs,banks,gaps=inventory(root,src,smoke)
    require(not gaps and all(j['valid'] for j in jobs),'Incomplete validated source inventory')
    if not smoke:
        require(smoke_result is not None,'Full evaluation requires --smoke-result')
        gate=read(smoke_result/'completion.json')
        require(gate['status']=='PASS' and gate['smoke'] and gate['streams']==4,'Smoke parity gate failed')
        previous=read(smoke_result/'provenance.json')
        require(previous['runtime']==runtime,'Smoke runtime changed')
        for path,record in previous['sources'].items():
            require(Path(path).is_file() and sha256(Path(path))==record['sha256'],'Smoke source changed: '+path)
    require(not out.exists(),'Use a new output directory; historical results are read-only')
    out.mkdir(parents=True);started=time.monotonic();completed=[]
    # Include evaluator, wrappers, model, and protocol code in gate identity.
    for p in list(Path(__file__).parent.rglob('*.py'))+[SCRIPTS/n for n in ('run_dynamics_only_ttt.py','reacher_pipeline.py','training_support.py')]+list((SCRIPTS.parents[2]/'tdmpc2/envs/nsgym').glob('*.py')):
        src.track(p,'evaluation implementation')
    for p in [SCRIPTS.parents[2]/'tdmpc2'/n for n in ('tdmpc2.py','common/world_model.py','common/math.py')]+list((SCRIPTS.parent/'configs').glob('*.json')):
        src.track(p,'native implementation/protocol')
    try:
        for job in jobs:
            dest=out/label(job);dest.mkdir();records=[];baseline=historical(root,job,src)
            for controller,swapped in [('DQ',False),('DQ_F',True)]:
                agent,isolation=load_controller(root,job,swapped)
                trajectory_dir=dest/controller;trajectory_dir.mkdir()
                write(dest/f'{controller}-isolation.json',isolation)
                before=tree_hash(agent.model.state_dict())
                import run_dynamics_only_ttt as ttt
                aux=tree_hash(ttt.aux_state(agent))
                for seed in (TASKS[job['task']]['evaluation'][:1] if smoke else TASKS[job['task']]['evaluation']):
                    rec,steps=episode(agent,root,job,seed,trajectory_dir)
                    require(np.isfinite(rec['episode_return']),'Nonfinite return')
                    for row in steps:
                        require(np.isclose(row['actual_parameter'],row['nominal_parameter']*job['severity'],rtol=0,atol=1e-10),'Physical readback mismatch')
                    delta=float(rec['episode_return']-baseline.loc[seed,'episode_return'])
                    # No post-hoc tolerance: deterministic historical check, 1e-6 absolute.
                    if not swapped:require(abs(delta)<=1e-6,f'Historical DQ parity failed {label(job)} seed {seed}: delta={delta}')
                    identity={k:job[k] for k in ('task','family','severity','adaptation_seed')}
                    records.append(dict(**identity,evaluation_seed=seed,method=controller,protocol='historical-fixed-eval',shift_schedule='fixed',
                        checkpoint_sha256=TASKS[job['task']]['sha'],snapshot_sha256=sha256(job['paths']['DQ']),historical_dq_delta=delta,**rec))
                    pd.DataFrame(steps).to_csv(dest/f'{controller}-seed{seed}-trajectory.csv',index=False)
                    require(tree_hash(agent.model.state_dict())==before and tree_hash(ttt.aux_state(agent))==aux,'Evaluation mutated state')
                del agent
            pd.DataFrame(records).to_csv(dest/'episodes.csv',index=False)
            write(dest/'completion.json',dict(status='PASS',episodes=len(records),optimizer_steps=0,replay_writes=0,
                files={str(p.relative_to(dest)):sha256(p) for p in dest.rglob('*') if p.is_file()}))
            completed.append(label(job));print('PASS',label(job),flush=True)
        require(not src.verify(),'Source changed during evaluation')
        write(out/'provenance.json',dict(runtime=runtime,sources=src.files,configs=configs,rng='Historical seed before each reset; native stochastic planner; paired seed, no imposed action tape'))
        write(out/'completion.json',dict(status='PASS',smoke=smoke,streams=len(completed),completed=completed,episodes=len(completed)*(2 if smoke else 40),optimizer_steps=0,seconds=time.monotonic()-started))
    except Exception as exc:
        write(out/'provenance.json',dict(runtime=runtime,sources=src.files))
        write(out/'completion.json',dict(status='PARTIAL',smoke=smoke,completed=completed,error=str(exc)))
        raise


def main(argv=None):
    """Simulator evaluation runner, separate from the read-only figure CLI."""
    import argparse
    parser=argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-result',type=Path)
    args=parser.parse_args(argv)
    evaluate(args.results_root.resolve(),args.output.resolve(),args.smoke,
             args.smoke_result.resolve() if args.smoke_result else None)


if __name__=='__main__':main()
