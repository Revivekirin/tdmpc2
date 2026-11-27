"""Paired full-state one-step and open-loop physical response experiments."""
import argparse
import json
from pathlib import Path
import sys

from research_utils import ROOT, output_dir, metadata, write_json, csv_rows
import gymnasium as gym
import mujoco
import numpy as np
from envs.nsgym import HalfCheetah

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
# mj_setConst derives these arrays from physical inputs. They must be allowed to
# change after mass edits; they are not independently perturbed parameters.
DERIVED = {'body_subtreemass','body_invweight0','dof_M0','dof_invweight0',
           'cam_poscom0','light_poscom0','actuator_acc0'}


def state(env):
    model,data=env.unwrapped.model,env.unwrapped.data
    value=np.empty(mujoco.mj_stateSize(model,STATE))
    mujoco.mj_getState(model,data,value,STATE)
    return value


def restore(env, value, atol):
    model,data=env.unwrapped.model,env.unwrapped.data
    mujoco.mj_setState(model,data,value,STATE)
    mujoco.mj_forward(model,data)
    # Forward may change acceleration caches but must preserve the complete
    # integration state, including warmstart, ctrl, applied forces and time.
    np.testing.assert_allclose(state(env),value,atol=atol,rtol=0)


def model_arrays(env):
    model=env.unwrapped.model
    return {key:getattr(model,key).copy() for key in dir(model)
            if isinstance(getattr(model,key),np.ndarray)}


def options(env):
    opt=env.unwrapped.model.opt
    return {key:np.asarray(getattr(opt,key)).copy() for key in dir(opt)
            if not key.startswith('_') and isinstance(getattr(opt,key),(int,float,np.ndarray))}


def physical(env):
    d=env.unwrapped.data
    return np.concatenate([d.qpos.copy(),d.qvel.copy(),d.qacc.copy()])


def run(config, destination):
    out=output_dir(destination)
    write_json(out/'metadata.json',metadata(config))
    scales=config['mass_scales']; horizon=config['horizon']; atol=config['atol']
    if 1.0 not in scales or len(scales)!=len(set(scales)) or horizon<1:
        raise ValueError('Require unique scales including 1.0 and positive horizon')
    actions=np.stack([np.random.default_rng(s).uniform(-config['action_bound'],config['action_bound'],(horizon,6)) for s in config['action_seeds']])
    trajectories=[]; rewards=[]; components=[]; states=[]; checks=[]; mass_reads=[]
    template=gym.make('HalfCheetah-v5').unwrapped
    try:
        nq,nv=template.model.nq,template.model.nv
        dt=template.dt
        nominal_mass=float(template.model.body('torso').mass[0])
        np.savez_compressed(out/'nominal_model_arrays.npz',**model_arrays(template))
        mujoco.mj_saveLastXML(str(out/'nominal_model.xml'),template.model)
        for seed in config['initial_state_seeds']:
            template.reset(seed=seed)
            states.append(state(template))
        for initial in states:
            for tape in actions:
                reference=None
                for scale in scales:
                    env=HalfCheetah(mode='fixed',mass_scale=scale,max_episode_steps=horizon)
                    try:
                        env.reset(seed=0)
                        before=model_arrays(env);opt_before=options(env)
                        restore(env,initial,atol)
                        trajectory=[physical(env)]; rs=[]; cs=[]; ms=[]
                        for action in tape:
                            _,reward,terminated,truncated,info=env.step(action)
                            trajectory.append(physical(env));rs.append(reward)
                            cs.append([info.get('reward_run',0),info.get('reward_ctrl',0)])
                            ms.append(float(env.unwrapped.model.body('torso').mass[0]))
                        np.testing.assert_allclose(ms,nominal_mass*scale,atol=atol,rtol=0)
                        after=model_arrays(env)
                        expected=before['body_mass'].copy();expected[env.unwrapped.model.body('torso').id]*=scale
                        np.testing.assert_allclose(after['body_mass'],expected,atol=atol,rtol=0)
                        changed=[k for k in before if not np.array_equal(before[k],after[k])]
                        unexpected=set(changed)-DERIVED-{'body_mass'}
                        assert not unexpected, f'Unintended model arrays: {unexpected}'
                        for key,value in opt_before.items():np.testing.assert_array_equal(options(env)[key],value)
                        np.testing.assert_array_equal(env.unwrapped.model.opt.gravity,template.model.opt.gravity)
                        env.reset(seed=0)
                        np.testing.assert_array_equal(env.unwrapped.model.body_mass,before['body_mass'])
                        checks.append(dict(scale=scale,changed_model_arrays=changed,reset_restored=True))
                        trajectories.append(trajectory);rewards.append(rs);components.append(cs);mass_reads.append(ms)
                        if scale==1:
                            reference=np.array(trajectory)
                            # Repeat from exactly the saved integration state.
                            restore(env,initial,atol); repeat=[physical(env)]
                            for action in tape: env.step(action);repeat.append(physical(env))
                            np.testing.assert_allclose(repeat,reference,atol=atol,rtol=0)
                            # Compare unwrapped Gymnasium to nominal NS-Gym,
                            # including reward; ignore reset qacc (derived cache).
                            template.reset(seed=0);restore(template,initial,atol)
                            raw=[physical(template)]; raw_rewards=[]
                            for action in tape:
                                _,r,_,_,_=template.step(action);raw_rewards.append(r);raw.append(physical(template))
                            np.testing.assert_allclose(raw,reference,atol=atol,rtol=0)
                            np.testing.assert_allclose(raw_rewards,rs,atol=atol,rtol=0)
                    finally: env.close()
    finally: template.close()
    shape=(len(states),len(actions),len(scales))
    trajectories=np.array(trajectories).reshape(*shape,horizon+1,nq+2*nv)
    rewards=np.array(rewards).reshape(*shape,horizon)
    components=np.array(components).reshape(*shape,horizon,2)
    mass_reads=np.array(mass_reads).reshape(*shape,horizon)
    np.savez_compressed(out/'raw.npz',initial_integration_states=states,actions=actions,
        physical=trajectories,rewards=rewards,reward_components=components,mass_readbacks=mass_reads,
        dt=dt,nq=nq,nv=nv,mass_scales=scales)
    write_json(out/'checks.json',dict(status='PASS',checks=checks,state_spec='mjSTATE_INTEGRATION',
        dt=dt,frame_skip=5,layout='physical[..., :nq]=qpos; next nv=qvel; last nv=qacc',
        axes=['initial_state','action_tape','mass_scale','time','physical_variable']))
    summarize(out)
    return out


def summarize(out):
    out=Path(out)
    for name in ['summary.csv','condition_summary.csv','physical_response.png','REPORT.md']:
        if (out/name).exists():raise FileExistsError(out/name)
    d=np.load(out/'raw.npz');x=d['physical'];scales=d['mass_scales'];nq=int(d['nq']);nv=int(d['nv']);dt=float(d['dt'])
    base=x[:,:,list(scales).index(1.0)]; rows=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,grid=plt.subplots(2,2,figsize=(11,7),sharex=True)
    axes=grid.ravel()
    for k,scale in enumerate(scales):
        delta=x[:,:,k]-base
        pos=np.linalg.norm(delta[...,:nq],axis=-1);vel=np.linalg.norm(delta[...,nq:nq+nv],axis=-1)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                row=dict(state_index=i,action_index=j,mass_scale=float(scale),torso_mass=float(d['mass_readbacks'][i,j,k,0]),
                    one_step_position_l2=float(pos[i,j,1]),one_step_velocity_l2=float(vel[i,j,1]),
                    final_position_l2=float(pos[i,j,-1]),final_velocity_l2=float(vel[i,j,-1]),
                    cumulative_position_l2=float(pos[i,j,1:].sum()*dt),cumulative_velocity_l2=float(vel[i,j,1:].sum()*dt),
                    rollout_reward=float(d['rewards'][i,j,k].sum()))
                for q in range(nq):row[f'one_step_qpos_delta_{q}']=float(delta[i,j,1,q])
                for v in range(nv):
                    row[f'one_step_qvel_delta_{v}']=float(delta[i,j,1,nq+v])
                    row[f'one_step_fd_acceleration_{v}']=float((x[i,j,k,1,nq+v]-x[i,j,k,0,nq+v])/dt)
                rows.append(row)
        curves=[np.linalg.norm(delta[...,:2],axis=-1),np.linalg.norm(delta[...,2:nq],axis=-1),
                np.linalg.norm(delta[...,nq:nq+2],axis=-1),np.linalg.norm(delta[...,nq+2:nq+nv],axis=-1)]
        for ax,value in zip(axes,curves):
            mean=value.mean((0,1));std=value.std((0,1),ddof=1)
            t=np.arange(value.shape[-1])*dt
            ax.plot(t,mean,label=str(scale));ax.fill_between(t,np.maximum(mean-std,0),mean+std,alpha=.15)
    for ax,label in zip(axes,['Root translation difference (m)','Angular position difference (rad)',
                              'Root velocity difference (m/s)','Angular velocity difference (rad/s)']):
        ax.set_ylabel(label);ax.set_xlabel('Time (seconds)')
    axes[0].legend(title='Mass scale')
    fig.tight_layout();fig.savefig(out/'physical_response.png');plt.close(fig)
    csv_rows(out/'summary.csv',rows)
    aggregate=[]
    for scale in scales:
        group=[r for r in rows if r['mass_scale']==scale]
        entry={'mass_scale':float(scale),'pairs':len(group)}
        for key in ['one_step_position_l2','one_step_velocity_l2','final_position_l2','final_velocity_l2','rollout_reward']:
            vals=[r[key] for r in group];entry[key+'_mean']=float(np.mean(vals));entry[key+'_std']=float(np.std(vals,ddof=1))
        aggregate.append(entry)
    csv_rows(out/'condition_summary.csv',aggregate)
    (out/'REPORT.md').write_text('# Physical validation: PASS\n\nFull integration-state restoration, nominal/repeated trajectory and reward equivalence, requested mass readback, reset restoration and model-array invariants passed. Raw qpos/qvel/qacc, actions, full initial states and reward components are in raw.npz. The first transition of each tape is the paired one-step experiment; remaining transitions are the paired open-loop experiment. Nine state/tape pairs per scale use three independent initial states crossed with three independent action tapes.\n\nPlots show descriptive mean ± standard deviation across crossed pairs, not independent-sample confidence intervals. Plots separate root translation (m), angular position (rad), linear velocity (m/s) and angular velocity (rad/s). The summary CSV additionally contains mixed-coordinate L2 summaries; these are descriptive, not physical severity scores. Joint-wise signed values and separate position/velocity columns are in summary.csv. No claim about controllability or trained-policy robustness follows. Values are exploratory, not calibrated severities.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/nsgym/configs/physical_validation.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--expanded',action='store_true')
    parser.add_argument('--summarize',type=Path,help='Rebuild missing summary files from saved raw.npz; existing tables refuse overwrite')
    args=parser.parse_args()
    if args.summarize: summarize(args.summarize);return
    if args.output is None:parser.error('--output is required')
    config=json.loads(args.config.read_text())
    if args.expanded:config['mass_scales']=config['expanded_mass_scales']
    existed=args.output.exists()
    try:run(config,args.output)
    except Exception as exc:
        if not existed and args.output.is_dir() and not (args.output/'failure.json').exists():
            write_json(args.output/'failure.json',dict(status='FAIL',error=repr(exc)))
        raise
    print(f'PASS: {args.output}')

if __name__=='__main__': main()
