"""Physical measurement and immutable frozen evaluation; imported after dry-run."""
import csv
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import gymnasium as gym
import mujoco

from severity_protocol import ROOT
sys.path.insert(0, str(ROOT/'tdmpc2'))
from envs.nsgym.severity import SeverityHalfCheetah
from validate_physics import state, restore, model_arrays, options, DERIVED


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def js(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def csv_write(path, rows):
    if not rows:
        raise ValueError(f'No measurements for {path}')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def provenance(c):
    import ns_gym.base
    import ns_gym.schedulers
    import ns_gym.update_functions.single_param
    import ns_gym.wrappers.mujoco_env
    modules = [ns_gym.base, ns_gym.schedulers, ns_gym.update_functions.single_param,
               ns_gym.wrappers.mujoco_env, __import__('envs.nsgym.severity',fromlist=['x'])]
    paths = [Path(inspect.getfile(m)) for m in modules]
    paths += list((ROOT/'experiments/nsgym/scripts').glob('*severity*.py'))
    paths += [ROOT/'tdmpc2/tdmpc2.py', ROOT/'tdmpc2/common/world_model.py']
    import torch
    model=Path(c['checkpoint'])/'model.pt'
    return dict(python=sys.executable, cuda_available=torch.cuda.is_available(),versions={k:importlib.metadata.version(k) for k in
                ('ns_gym','gymnasium','mujoco','numpy','torch')},
                source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
                source_hashes={str(p):digest(p) for p in paths}, configuration=c,
                environment_id='HalfCheetah-v5', checkpoint_sha256=digest(model) if model.exists() else None,
                optimizer_updates=0, status='STARTED')


def env_for(c, condition, horizon=None):
    return SeverityHalfCheetah(parameter=condition['parameter'], severity=condition['severity'],
        schedule=condition['schedule'], max_episode_steps=horizon or c['horizon'], **c['schedule_timing'])


def shift_onset(c,condition):
    if condition['schedule']=='nominal':return None
    return 0 if condition['schedule']=='fixed' else c['schedule_timing']['onset']


def alignment_reference(c,condition):
    onset=shift_onset(c,condition)
    return c['schedule_timing']['onset'] if onset is None else onset


def physical_snapshot(env):
    d = env.unwrapped.data; m = env.unwrapped.model
    contacts = []
    for i in range(d.ncon):
        force = np.zeros(6); mujoco.mj_contactForce(m, d, i, force)
        ct = d.contact[i]
        contacts.append(dict(geom1=int(ct.geom1), geom2=int(ct.geom2), distance=float(ct.dist),
                             position=ct.pos.tolist(), frame=ct.frame.tolist(), force=force.tolist()))
    snapshot=dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), qacc=d.qacc.copy(),
                root_orientation=d.xquat[m.body('torso').id].copy(),
                root_body_velocity=d.cvel[m.body('torso').id].copy(),
                ctrl=d.ctrl.copy(), actuator_force=d.actuator_force.copy(),
                contact_count=np.array(d.ncon),
                contact_force_norm=np.array(sum(np.linalg.norm(x['force'][:3]) for x in contacts)))
    if not all(np.isfinite(v).all() for v in snapshot.values()):
        raise FloatingPointError('Nonfinite physical state/actuator/contact measurement')
    return snapshot,contacts


def pack_snapshots(snapshots):
    return {k:np.stack([x[k] for x in snapshots]) for k in snapshots[0]}


def inventory(c, out):
    import ns_gym.base as base
    import ns_gym.wrappers.mujoco_env as wrapper
    from ns_gym.wrappers.mujoco_env import param_look_up
    e = gym.make('HalfCheetah-v5'); e.reset(seed=c['seeds'][0]); raw=e.unwrapped
    rows=[]
    try:
        for name in base.TUNABLE_PARAMS['HalfCheetahEnv']:
            value = np.asarray(param_look_up('HalfCheetahEnv',name)[0][0](raw))
            if name.endswith('_mass'):
                body='ffoot' if name=='ffeet_mass' else name.removesuffix('_mass')
                target=f'body_mass[{raw.model.body(body).id}]'; unit='kg'; scope=body
                bounds='>1e-6; no upper bound in wrapper'; meaning='Body translational mass; body_inertia is unchanged'
            elif name.endswith('_damping'):
                joint='ffoot' if name=='ffeet_damping' else name.removesuffix('_damping')
                idx=raw.model.joint(joint).id
                target=f'dof_damping[{idx}]'; unit='N m s / rad'; scope=joint
                bounds='>=0; no upper bound in wrapper'; meaning='Viscous hinge joint damping opposing joint velocity'
                if idx != raw.model.jnt_dofadr[idx]:
                    raise ValueError('Upstream DOF index assumption invalid')
            elif name=='floor_friction':
                target=f'geom_friction[{raw.model.geom("floor").id},0]';unit='dimensionless';scope='floor sliding coefficient'
                bounds='No friction-specific wrapper check';meaning='Sliding contact friction; torsional/rolling coefficients unchanged'
            else:
                target='opt.gravity (update changes z only)';unit='m/s^2';scope='world'
                bounds='No gravity-specific wrapper check';meaning='Gravitational acceleration'
            rows.append(dict(parameter=name,mujoco_target=target,nominal_value=json.dumps(value.tolist()),
                unit=unit,scope=scope,supported_range=bounds,update_method='MujocoWrapper setter + dependency resolver',
                schedule_support='Fixed and dynamic, official StepWiseUpdate/DiscreteScheduler',physical_meaning=meaning,
                research_relevance='Contact/locomotion acceleration and energy dissipation',
                setter_source=inspect.getsource(param_look_up('HalfCheetahEnv',name)[0][1]).strip()))
        csv_write(out/'physical_parameter_inventory.csv',rows)
        for label, obj in [('mujoco_wrapper',wrapper),('halfcheetah_reward',type(raw)),('ns_base',base)]:
            (out/f'{label}_source.py').write_text(inspect.getsource(obj))
        js(out/'environment_contract.json',dict(observation_shape=list(e.observation_space.shape),
            action_shape=list(e.action_space.shape),dt=raw.dt,frame_skip=raw.frame_skip,
            actuator_gear=raw.model.actuator_gear.tolist(),actuator_ctrlrange=raw.model.actuator_ctrlrange.tolist(),
            body_inertia=raw.model.body_inertia.tolist(),body_ipos=raw.model.body_ipos.tolist(),
            body_subtreemass=raw.model.body_subtreemass.tolist(),
            reward_formula='forward_reward_weight * x_velocity - ctrl_cost_weight * sum(action^2)',
            forward_reward_weight=raw._forward_reward_weight,ctrl_cost_weight=raw._ctrl_cost_weight,
            no_direct_parameter_reward_term=True))
    finally:e.close()
    return dict(parameters=len(rows),transitions=0)


def validate(c, out):
    if c['budget']['maximum_transitions']>2000:
        raise ValueError('Schedule validator is bounded; use --smoke (max 2000 transitions)')
    rows=[];checks=[];global_step=0
    for cond in c['conditions']:
        for seed in c['seeds']:
            env=env_for(c,cond)
            try:
                nominal=model_arrays(env); opt=options(env)
                tape=np.random.default_rng(seed).uniform(-.6,.6,(c['horizon'],6))
                repetitions=[]
                for episode in range(2):
                    obs,_=env.reset(seed=seed); trajectory=[]
                    np.testing.assert_array_equal(env.unwrapped.model.body_mass,nominal['body_mass'])
                    np.testing.assert_array_equal(env.unwrapped.model.dof_damping,nominal['dof_damping'])
                    for t,a in enumerate(tape):
                        before=obs.copy();obs,reward,term,trunc,info=env.step(a)
                        assert obs.shape==(17,) and not ({'relative_time','env_change','delta_change'} & info.keys())
                        raw=env.unwrapped
                        expected_mass=nominal['body_mass'].copy();expected_damping=nominal['dof_damping'].copy()
                        if cond['family']=='mass':
                            body=cond['parameter'].removesuffix('_mass')
                            expected_mass[raw.model.body(body).id]=env.readback()
                        else:
                            joint=cond['parameter'].removesuffix('_damping')
                            expected_damping[raw.model.joint(joint).id]=env.readback()
                        np.testing.assert_array_equal(raw.model.body_mass,expected_mass)
                        np.testing.assert_array_equal(raw.model.dof_damping,expected_damping)
                        np.testing.assert_array_equal(raw.model.body_inertia,nominal['body_inertia'])
                        np.testing.assert_array_equal(raw.model.body_ipos,nominal['body_ipos'])
                        expected=raw._forward_reward_weight*info['x_velocity']-raw._ctrl_cost_weight*float(a@a)
                        np.testing.assert_allclose(reward,expected,rtol=0,atol=1e-10)
                        assert not term and trunc==(t==c['horizon']-1)
                        trajectory.append(obs.copy())
                        rows.append(dict(**cond,seed=seed,episode=episode,global_step=global_step,episode_step=t,
                            shift_onset=shift_onset(c,cond),actual_parameter=env.readback(),expected_parameter=env.expected[t],
                            scheduler_state=json.dumps(env.scheduler_state()),reward=reward,
                            observation_before=json.dumps(before.tolist()),observation_after=json.dumps(obs.tolist()),
                            terminated=term,truncated=trunc))
                        global_step+=1
                    repetitions.append(np.asarray(trajectory))
                np.testing.assert_allclose(*repetitions,atol=1e-10,rtol=0)
                after=model_arrays(env)
                changed={k for k in nominal if not np.array_equal(nominal[k],after[k])}
                allowed=DERIVED|({'body_mass'} if cond['family']=='mass' else {'dof_damping'})
                assert not changed-allowed,changed-allowed
                for k in opt:np.testing.assert_array_equal(options(env)[k],opt[k])
                np.testing.assert_array_equal(after['body_inertia'],nominal['body_inertia'])
                np.testing.assert_array_equal(after['body_ipos'],nominal['body_ipos'])
                env.reset(seed=seed)
                np.testing.assert_array_equal(env.unwrapped.model.body_mass,nominal['body_mass'])
                np.testing.assert_array_equal(env.unwrapped.model.dof_damping,nominal['dof_damping'])
                checks.append(dict(**cond,reset_restored=True,repeat_equal=True,changed_arrays=sorted(changed)))
            finally:env.close()
    csv_write(out/'schedule_readbacks.csv',rows);js(out/'checks.json',dict(status='PASS',checks=checks))
    return dict(transitions=global_step,checks=len(checks))


class FrozenAgent:
    def __init__(self,c):
        from training_support import load_checkpoint, learned_state, tree_hash
        self.path=Path(c['checkpoint']);self.hashes={k:digest(self.path/k) for k in ('model.pt','optimizers.pt','metadata.json')}
        if self.hashes['model.pt']!=c['expected_checkpoint_sha256']:
            raise ValueError('Checkpoint does not match the fixed primary/sensitivity SHA256')
        self.agent,self.meta=load_checkpoint(self.path)
        cfg=self.meta['config']
        expected_versions=self.meta['provenance']['versions']
        for package in ('ns_gym','gymnasium','mujoco'):
            if importlib.metadata.version(package)!=expected_versions[package]:
                raise ValueError(f'{package} differs from nominal training environment: '
                                 f'{importlib.metadata.version(package)} != {expected_versions[package]}')
        for key,expected in dict(task='nsgym-halfcheetah',action_dim=6,episode_length=1000,horizon=3,
                                 num_samples=512,iterations=6,mpc=True,episodic=False,seed=17).items():
            if cfg[key]!=expected:raise ValueError(f'Checkpoint contract mismatch: {key}')
        if cfg['obs_shape']['state'] not in ([17],(17,)):
            raise ValueError('Observation contract mismatch')
        if set(c['seeds']) & set(cfg['eval_seeds']) or any(cfg['seed']<=s<=cfg['seed']+cfg['steps'] for s in c['seeds']):
            raise ValueError('Evaluation seed overlap')
        def forbidden(*a,**kw):raise RuntimeError('Optimizer updates forbidden')
        self.agent.update=forbidden;self.agent.optim.step=forbidden;self.agent.pi_optim.step=forbidden
        self.agent.model.eval()
        self.before=tree_hash(learned_state(self.agent))

    def verify(self,out):
        from training_support import learned_state,tree_hash
        after=tree_hash(learned_state(self.agent))
        assert after==self.before,'Learned state changed'
        assert all(digest(self.path/k)==v for k,v in self.hashes.items()),'Checkpoint files changed'
        js(out/'model_integrity.json',dict(before=self.before,after=after,file_sha256=self.hashes,optimizer_updates=0,
                                         checkpoint_metadata=self.meta))


def collect(c,out,bank=False):
    import torch
    from common.seed import set_seed
    if not bank and c['policy']!='primary':raise ValueError('Frozen evaluation requires pretrained policy')
    frozen=None if bank and c['policy']=='exploration' else FrozenAgent(c)
    conditions=[dict(family='mass',parameter='torso_mass',severity=1.,schedule='nominal')] if bank else c['conditions']
    rows=[];global_step=0;index=[]
    for ci,cond in enumerate(conditions):
        for seed in c['seeds']:
            set_seed(seed);rng=np.random.default_rng(seed);env=env_for(c,cond)
            try:
                obs,_=env.reset(seed=seed);observations=[obs];snapshots=[];states=[];actions=[];rewards=[];step_rows=[]
                p,contact=physical_snapshot(env);snapshots.append(p)
                contact_log=[]
                with torch.no_grad():
                    for t in range(c['horizon']):
                        states.append(state(env))
                        action=(rng.uniform(-.6,.6,6) if frozen is None else
                                frozen.agent.act(torch.as_tensor(obs,dtype=torch.float32),t0=t==0,eval_mode=True).numpy())
                        new,reward,term,trunc,info=env.step(action)
                        p,contact=physical_snapshot(env);snapshots.append(p);contact_log.append(contact)
                        actions.append(action);rewards.append(reward);observations.append(new)
                        step_rows.append(dict(**cond,seed=seed,episode_step=t,global_step=global_step,
                            aligned_step=t-alignment_reference(c,cond),shift_onset=shift_onset(c,cond),
                            alignment_reference_step=alignment_reference(c,cond),actual_parameter=env.readback(),
                            nominal_parameter=env.nominal_value,forward_velocity=info['x_velocity'],reward=reward,
                            control_cost=-info['reward_ctrl'],action_abs_mean=float(np.abs(action).mean()),
                            action_saturation=float((np.abs(action)>=.99).mean()),
                            action_change_rms=float(np.sqrt(np.square(action-actions[-2]).mean())) if len(actions)>1 else 0.,
                            scheduler_state=json.dumps(env.scheduler_state()),terminated=term,truncated=trunc))
                        obs=new;global_step+=1
                        if term or trunc:break
                a=np.asarray(actions);r=np.asarray(rewards)
                name=f'condition-{ci:02d}-seed-{seed}'
                np.savez_compressed(out/f'{name}.npz',observations=observations,actions=a,rewards=r,
                                    actual_parameter=[x['actual_parameter'] for x in step_rows],
                                    integration_states=states,dt=env.unwrapped.dt,**pack_snapshots(snapshots))
                js(out/f'{name}-contacts.json',contact_log)
                csv_write(out/f'{name}-steps.csv',step_rows)
                index.append(dict(file=f'{name}.npz',steps=f'{name}-steps.csv',contacts=f'{name}-contacts.json',
                                  sha256=digest(out/f'{name}.npz'),
                                  seed=seed,condition=cond,source='nominal_primary_policy' if bank and frozen else
                                  'bounded_exploration' if bank else 'closed_loop_frozen',dt=env.unwrapped.dt))
                vals=[x['actual_parameter'] for x in step_rows];exposed=[not np.isclose(x,env.nominal_value) for x in vals]
                row=dict(**cond,seed=seed,episode_return=float(r.sum()),episode_length=len(r),
                    return_per_step=float(r.mean()),forward_velocity=float(np.mean([x['forward_velocity'] for x in step_rows])),
                    control_cost=float(sum(x['control_cost'] for x in step_rows)),terminated=term,truncated=trunc,
                    action_abs_mean=float(np.abs(a).mean()),action_rms=float(np.sqrt(np.square(a).mean())),
                    action_saturation=float((np.abs(a)>=.99).mean()),
                    action_smoothness_rms=float(np.sqrt(np.square(np.diff(a,axis=0)).mean())) if len(a)>1 else 0.,
                    exposure_fraction=float(np.mean(exposed)),
                    ended_before_onset=len(r)<=shift_onset(c,cond) if shift_onset(c,cond) is not None else False,
                    reward_time_integral=float(r.sum()*env.unwrapped.dt),
                    actual_parameter_min=min(vals),actual_parameter_max=max(vals),
                    source=index[-1]['source'],data_file=index[-1]['file'])
                for j in range(6):
                    row[f'action_{j}_mean']=float(a[:,j].mean());row[f'action_{j}_std']=float(a[:,j].std())
                rows.append(row)
                print(f'{cond} seed={seed} return={row["episode_return"]:.2f}',flush=True)
            finally:env.close()
    js(out/'dataset_index.json',index)
    csv_write(out/'episodes.csv',rows)
    if frozen:frozen.verify(out)
    return dict(transitions=global_step,episodes=len(rows),source='bank' if bank else 'closed_loop')


def physical(c,out):
    source=Path(c['input']);source_config=json.loads((source/'manifest.json').read_text())
    if source_config['stage']!='bank':raise ValueError('Physical probes require a saved bank')
    if source_config['seed_split']!='calibration':raise ValueError('Physical bank must use calibration seeds')
    if source_config['expected_checkpoint_sha256']!=c['expected_checkpoint_sha256']:
        raise ValueError('Physical bank checkpoint identity differs')
    if source_config['protocol_sha256']!=c['protocol_sha256']:
        raise ValueError('Bank protocol changed before physical calibration')
    if source_config['policy']=='exploration' and not c['smoke']:
        raise ValueError('Full physical calibration requires nominal primary policy bank; exploration is smoke-only')
    index=json.loads((source/'dataset_index.json').read_text());records=[];result_index=[];steps=0
    if not any(x['severity']==1 for x in c['conditions']):raise ValueError('Physical comparisons require nominal severity 1')
    for entry in index:
        if entry['seed'] not in c['seeds']:continue
        if entry.get('sha256') and digest(source/entry['file'])!=entry['sha256']:raise ValueError('Bank hash mismatch')
        bank=np.load(source/entry['file']);h=c['probe_horizon']
        starts=np.unique(np.linspace(0,len(bank['actions'])-h,c['samples_per_seed'],dtype=int))
        if len(starts)!=c['samples_per_seed'] or starts[0]<0:raise ValueError('Bank too short for requested probe samples')
        for start in starts:
            references={}
            ordered=sorted(c['conditions'],key=lambda x:x['severity']!=1)
            for ci,cond in enumerate(ordered):
                if cond['schedule']!='fixed':raise ValueError('Controlled physical response uses fixed schedule only')
                env=env_for(c,cond,horizon=h)
                try:
                    obs,_=env.reset(seed=entry['seed']);restore(env,bank['integration_states'][start],1e-10)
                    obs=env.unwrapped._get_obs().copy()
                    np.testing.assert_allclose(obs,bank['observations'][start],atol=1e-10,rtol=0)
                    original=model_arrays(env)
                    snapshots=[];observations=[obs];rewards=[];contacts=[];parameters=[]
                    initial,_=physical_snapshot(env);snapshots.append(initial)
                    tape=bank['actions'][start:start+h]
                    for a in tape:
                        obs,r,term,trunc,info=env.step(a);steps+=1
                        snap,ct=physical_snapshot(env);snapshots.append(snap);contacts.append(ct)
                        observations.append(obs);rewards.append(r);parameters.append(env.readback())
                    data=pack_snapshots(snapshots)
                    np.testing.assert_array_equal(env.unwrapped.model.body_inertia,original['body_inertia'])
                    np.testing.assert_array_equal(env.unwrapped.model.body_ipos,original['body_ipos'])
                    name=f'{cond["parameter"]}-k{cond["severity"]}-seed{entry["seed"]}-t{start}'
                    np.savez_compressed(out/f'{name}.npz',observations=observations,actions=tape,rewards=rewards,
                        initial_integration_state=bank['integration_states'][start],actual_parameter=parameters,
                        dt=env.unwrapped.dt,**data)
                    js(out/f'{name}-contacts.json',contacts)
                    result_index.append(dict(file=f'{name}.npz',contacts=f'{name}-contacts.json',seed=entry['seed'],
                        sha256=digest(out/f'{name}.npz'),
                        source_step=int(start),source=entry['source'],condition=cond,
                        source_file=str(source/entry['file']),source_sha256=digest(source/entry['file'])))
                    if cond['severity']==1:
                        references[cond['parameter']]=(data,contacts)
                        env.reset(seed=entry['seed']);restore(env,bank['integration_states'][start],1e-10)
                        for i,a in enumerate(tape):
                            ro,rr,*_=env.step(a);steps+=1
                            np.testing.assert_allclose(ro,observations[i+1],atol=1e-10,rtol=0)
                            np.testing.assert_allclose(rr,rewards[i],atol=1e-10,rtol=0)
                        np.testing.assert_allclose(observations,bank['observations'][start:start+h+1],atol=1e-9,rtol=0)
                    ref,ref_contacts=references[cond['parameter']]
                    dv=data['qvel'][1]-ref['qvel'][1];dq=data['qpos'][1]-ref['qpos'][1]
                    row=dict(**cond,seed=entry['seed'],source_step=int(start),source=entry['source'],numerically_valid=True,
                        actual_parameter=parameters[0],nominal_parameter=env.nominal_value,
                        joint_position_l2=float(np.linalg.norm(dq[3:])),joint_velocity_l2=float(np.linalg.norm(dv[3:])),
                        joint_acceleration_difference=float(np.linalg.norm(dv[3:])/env.unwrapped.dt),
                        root_translation_l2=float(np.linalg.norm(dq[:2])),root_velocity_l2=float(np.linalg.norm(dv[:2])),
                        root_acceleration_difference=float(np.linalg.norm(dv[:2])/env.unwrapped.dt),
                        root_orientation_difference=float(abs(dq[2])),
                        mixed_state_l2=float(np.linalg.norm(np.r_[dq,dv])),
                        contact_count_difference=int(data['contact_count'][1]-ref['contact_count'][1]),
                        contact_force_difference=float(data['contact_force_norm'][1]-ref['contact_force_norm'][1]),
                        contact_pairs_changed=sorted((x['geom1'],x['geom2']) for x in contacts[0])!=sorted((x['geom1'],x['geom2']) for x in ref_contacts[0]),
                        final_joint_velocity_l2=float(np.linalg.norm(data['qvel'][-1,3:]-ref['qvel'][-1,3:])))
                    for j in range(len(dq)):
                        row[f'qpos_delta_{j}']=float(dq[j]);row[f'qvel_delta_{j}']=float(dv[j])
                        row[f'qacc_delta_{j}']=float(data['qacc'][1,j]-ref['qacc'][1,j])
                    records.append(row)
                except Exception as exc:
                    js(out/'invalid_condition.json',dict(condition=cond,seed=entry['seed'],source_step=int(start),
                        numerically_valid=False,error=repr(exc)))
                    if records:csv_write(out/'partial_physical_response_pairs.csv',records)
                    raise
                finally:env.close()
    if not records:raise ValueError('No matching bank seeds')
    csv_write(out/'physical_response_pairs.csv',records);js(out/'dataset_index.json',result_index)
    js(out/'checks.json',dict(status='PASS',full_integration_state=True,nominal_repeat_and_bank_replay_equal=True,
                            latent_predictions_used=False,state_spec='mjSTATE_INTEGRATION'))
    return dict(transitions=steps,paired_probes=len(records))


def latent(c,out):
    import torch
    from common import math as tdmath
    source=Path(c['input']);entries=json.loads((source/'dataset_index.json').read_text())
    source_config=json.loads((source/'manifest.json').read_text())
    if source_config['expected_checkpoint_sha256']!=c['expected_checkpoint_sha256']:
        raise ValueError('Data and diagnostic checkpoint identities differ')
    frozen=FrozenAgent(c);model=frozen.agent.model;rows=[]
    with torch.no_grad():
        for entry in entries:
            if entry['seed'] not in c['seeds']:continue
            if entry.get('sha256') and digest(source/entry['file'])!=entry['sha256']:raise ValueError('Diagnostic data hash mismatch')
            d=np.load(source/entry['file']);cond=entry['condition']
            obs=torch.as_tensor(d['observations'],device=frozen.agent.device,dtype=torch.float32)
            act=torch.as_tensor(d['actions'],device=frozen.agent.device,dtype=torch.float32)
            target=model.encode(obs,None)
            for horizon in (1,3,10):
                if len(act)<horizon:continue
                z=target[:-horizon].clone();n=len(z)
                for j in range(horizon):
                    if j==0:
                        pred_reward=tdmath.two_hot_inv(model.reward(z,act[:n],None),frozen.agent.cfg).squeeze(-1)
                    z=model.next(z,act[j:j+n],None)
                errors=(z-target[horizon:]).square().mean(-1).cpu().numpy()
                rerr=(pred_reward-torch.as_tensor(d['rewards'][:n],device=obs.device)).square().cpu().numpy()
                if not np.isfinite(errors).all() or not np.isfinite(rerr).all():
                    raise FloatingPointError(f'Nonfinite prediction diagnostics: {entry["file"]}')
                for t in range(n):
                    rows.append(dict(**cond,seed=entry['seed'],source=entry['source'],source_file=entry['file'],
                        source_step=entry.get('source_step',0)+t,episode_step=t,horizon=horizon,latent_mse=float(errors[t]),
                        aligned_step=t-alignment_reference(c,cond),
                        reward_squared_error=float(rerr[t]),action_regime='saturated' if np.any(np.abs(d['actions'][t])>=.99) else 'unsaturated',
                        velocity_regime='forward' if d['qvel'][t,0]>0 else 'nonforward',
                        actual_parameter=float(d['actual_parameter'][t]) if 'actual_parameter' in d else None))
    csv_write(out/'latent_errors.csv',rows);frozen.verify(out)
    js(out/'dataset_provenance.json',dict(input=str(source),index_sha256=digest(source/'dataset_index.json'),
        data_hashes={e['file']:digest(source/e['file']) for e in entries},
        observation_transform='unchanged state vector -> float32; no additional normalization',
        physical_outcome_prediction='NOT_APPLICABLE: model has no state decoder'))
    return dict(transitions=0,prediction_windows=len(rows))


def environment_check(c,out):
    from envs.nsgym.halfcheetah import HalfCheetah
    raw=gym.make('HalfCheetah-v5',max_episode_steps=12)
    adapted=HalfCheetah(mode='nominal',mass_scale=.7,max_episode_steps=12)
    scheduled=SeverityHalfCheetah(parameter='bthigh_damping',severity=.5,schedule='nominal',max_episode_steps=12)
    try:
        obs=[e.reset(seed=c['seeds'][0])[0] for e in (raw,adapted,scheduled)]
        for x in obs[1:]:np.testing.assert_allclose(x,obs[0],atol=1e-10,rtol=0)
        for a in np.random.default_rng(42).uniform(-.5,.5,(12,6)):
            outputs=[e.step(a) for e in (raw,adapted,scheduled)]
            for x in outputs[1:]:
                np.testing.assert_allclose(x[0],outputs[0][0],atol=1e-10,rtol=0)
                np.testing.assert_allclose(x[1],outputs[0][1],atol=1e-10,rtol=0)
                assert x[2:4]==outputs[0][2:4]
        js(out/'nominal_equivalence.json',dict(status='PASS',steps_per_env=12,
            original_adapter_inactive_mass_scale=.7,experimental_inactive_severity=.5,
            observation_reward_terminal_equivalent_to_gym=True))
    finally:
        raw.close();adapted.close();scheduled.close()
    return dict(transitions=36)


def planning(c,out):
    """Existing trace hooks capture actual MPPI candidates and stochastic decisions."""
    import torch
    from common.seed import set_seed
    from common import math as tdmath
    source=Path(c['input']);entries=json.loads((source/'dataset_index.json').read_text())
    source_config=json.loads((source/'manifest.json').read_text())
    if source_config['stage']!='physical' or source_config['expected_checkpoint_sha256']!=c['expected_checkpoint_sha256']:
        raise ValueError('Planning analysis requires controlled physical data from the same checkpoint')
    frozen=FrozenAgent(c);agent=frozen.agent
    selected=[];seen=set()
    requested={(x['parameter'],x['severity']) for x in c['conditions']}
    for e in entries:
        cond=e['condition'];key=(cond['parameter'],cond['severity'],e['seed'])
        if e['seed'] in c['seeds'] and (cond['parameter'],cond['severity']) in requested and key not in seen:
            selected.append(e);seen.add(key)
    if len(selected)>12:raise ValueError('Planning diagnostics are bounded to <=12 examples; choose --severity and --seeds')
    if not selected:raise ValueError('No matching planning examples')

    class Trace:
        def __init__(self):self.data={};self.i=0
        def encoded(self,z):self.z=z.detach().clone();self.data['encoded']=z.cpu().numpy().copy()
        def value(self,g,terminal):
            self.data[f'reward_sum_{self.i}']=g.cpu().numpy().copy()
            self.data[f'discounted_terminal_q_{self.i}']=terminal.cpu().numpy().copy()
        def iteration(self,actions,value,elite_idxs,num_pi_trajs):
            self.data[f'actions_{self.i}']=actions.cpu().numpy().copy()
            self.data[f'score_{self.i}']=value.cpu().numpy().copy()
            self.data[f'elite_indices_{self.i}']=elite_idxs.cpu().numpy().copy()
            self.data[f'ranking_{self.i}']=torch.argsort(value.squeeze(-1),descending=True).cpu().numpy()
            self.i+=1
        def selected(self,idx,action):
            self.data['selected_index']=idx.cpu().numpy().copy();self.data['selected_action']=action.cpu().numpy().copy()

    def trace(obs,seed):
        set_seed(seed);agent._prev_mean.zero_();tr=Trace();agent._mechanism_trace=tr
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        with torch.no_grad():
            agent.act(torch.tensor(obs,dtype=torch.float32),t0=True,eval_mode=True)
            for i in range(tr.i):
                actions=torch.tensor(tr.data[f'actions_{i}'],device=agent.device)
                z=tr.z.repeat(agent.cfg.num_samples,1);zs=[z.cpu().numpy()];rewards=[]
                for a in actions:
                    rewards.append(tdmath.two_hot_inv(agent.model.reward(z,a,None),agent.cfg).cpu().numpy())
                    z=agent.model.next(z,a,None);zs.append(z.cpu().numpy())
                tr.data[f'latent_rollout_{i}']=np.stack(zs);tr.data[f'predicted_reward_{i}']=np.stack(rewards)
        agent._mechanism_trace=None
        if not all(np.isfinite(v).all() for v in tr.data.values()):raise FloatingPointError('Nonfinite planning trace')
        return tr.data,rng

    comparisons=[]
    for i,entry in enumerate(selected):
        d=np.load(source/entry['file']);cond=entry['condition'];seed=entry['seed']
        nominal=next(e for e in entries if e['condition']['parameter']==cond['parameter'] and
                     e['condition']['severity']==1 and e['seed']==seed and e.get('source_step')==entry.get('source_step'))
        baseline=np.load(source/nominal['file'])
        a,rng=trace(d['observations'][0],seed);b,_=trace(baseline['observations'][0],seed)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        changed,_=trace(d['observations'][1],seed);nominal_next,_=trace(baseline['observations'][1],seed)
        for label,data in [('same_observation',a),('shifted_next_observation',changed),('nominal_next_observation',nominal_next)]:
            np.savez_compressed(out/f'example-{i:02d}-{label}.npz',**data)
        torch.save(rng,out/f'example-{i:02d}-rng.pt')
        comparisons.append(dict(**cond,seed=seed,source_file=entry['file'],same_observation_all_candidates_scores_actions_equal=True,
            next_observation_action_l2=float(np.linalg.norm(changed['selected_action']-nominal_next['selected_action'])),
            candidate_actual_returns_measured=False,ranking_correctness='NOT_MEASURED'))
    csv_write(out/'planning_comparisons.csv',comparisons);frozen.verify(out)
    return dict(transitions=0,examples=len(comparisons))
