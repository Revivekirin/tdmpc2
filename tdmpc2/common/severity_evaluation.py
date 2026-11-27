"""Read-only native-step measurements for Frozen severity evaluation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def model_digest(model):
    import torch
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        if isinstance(value, torch.Tensor):
            data = value.detach().cpu().contiguous()
            h.update(str((data.dtype, tuple(data.shape))).encode())
            h.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, torch.Size):
            h.update(b'torch.Size:')
            h.update(repr(tuple(value)).encode())
        elif isinstance(value, torch.device):
            h.update(b'torch.device:')
            h.update(str(value).encode())
        elif value is None:
            h.update(b'NoneType:None')
        elif isinstance(value, (bool, int, float, str, torch.dtype)):
            h.update(f'{type(value).__name__}:{value!r}'.encode())
        else:
            raise TypeError(f'Unsupported model state entry {key}: {type(value).__name__}')
    return h.hexdigest()


def measured_pose(native):
    tcp = getattr(native, 'tcp', None)
    if tcp is None:
        tcp = getattr(getattr(native, 'agent', None), 'tcp', None)
    if tcp is None:
        raise RuntimeError('Measured EEF logging requires simulator tcp.pose; no command integration fallback.')
    pose = tcp.pose if hasattr(tcp, 'pose') else tcp.get_pose()
    xyz = np.asarray(pose.p,dtype=float).copy()
    quat = np.asarray(pose.q,dtype=float).copy() if hasattr(pose,'q') else np.full(4,np.nan)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('Invalid measured EEF position.')
    return xyz, quat


def initial_kinematics(native, observation):
    """Observable reset state, excluding simulator bookkeeping and physical constants."""
    if hasattr(observation, 'detach'):
        observation = observation.detach().cpu().numpy()
    state = dict(observation=np.asarray(observation,dtype=float).copy())
    actor = getattr(native, 'obj', None)
    if actor is not None:
        pose = actor.get_pose()
        state['object_pose_xyz'] = np.asarray(pose.p,dtype=float).copy()
        state['object_pose_quaternion_wxyz'] = np.asarray(pose.q,dtype=float).copy()
        state['object_velocity_xyz'] = np.asarray(actor.get_velocity(),dtype=float).copy()
        state['object_angular_velocity_xyz'] = np.asarray(actor.get_angular_velocity(),dtype=float).copy()
    robot = getattr(getattr(native, 'agent', None), 'robot', None)
    if robot is not None:
        state['robot_qpos'] = np.asarray(robot.get_qpos(),dtype=float).copy()
        state['robot_qvel'] = np.asarray(robot.get_qvel(),dtype=float).copy()
    return state


def same_initial_kinematics(left, right, atol=1e-6):
    if left.keys() != right.keys():
        return False
    return all(np.shape(left[key]) == np.shape(right[key]) and
               np.allclose(left[key],right[key],rtol=0,atol=atol)
               for key in left)


class NativeTrajectory:
    def __init__(self, shift, expected, log_eef):
        self.shift,self.expected,self.log_eef=shift,expected,log_eef
        self.rows=[]; self.commands=[]; self.initial=None; self.effective=None

    def reset(self, env, seed, observation=None):
        native=self.shift.unwrapped
        state=np.asarray(native.get_state()).copy()
        if observation is None:
            observation = native.get_obs() if hasattr(native, 'get_obs') else state
        kinematics=initial_kinematics(native,observation)
        self.initial=dict(evaluation_seed=seed,initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                          initial_state=state.tolist(),
                          initial_kinematic_state={key:value.tolist() for key,value in kinematics.items()})
        if self.log_eef:
            xyz,q=measured_pose(native)
            self.initial.update(initial_eef_xyz=xyz.tolist(),initial_eef_quaternion_wxyz=q.tolist())
            if hasattr(native, 'obj') and hasattr(native, 'goal_pos'):
                self.initial['initial_object_goal_distance_m'] = float(np.linalg.norm(
                    np.asarray(native.obj.pose.p)-np.asarray(native.goal_pos)))
        if self.shift.parameter['type']=='action_delay' and len(self.shift.queue):
            raise AssertionError('Delay queue not reset at episode boundary.')

    def observe(self, observation, record):
        shift=self.shift;kind=shift.parameter['type']
        if record['applied_parameter'] != self.expected:
            raise AssertionError('Applied schedule differs from evaluation condition.')
        cmd=np.asarray(record['commanded_action']).copy();exe=np.asarray(record['executed_action']).copy()
        self.commands.append(cmd)
        effective=dict(applied_parameter=record['applied_parameter'])
        if kind in ('action_gain','action_delay'):
            expected=cmd.copy()
            if kind=='action_gain':
                expected[shift.indices]*=self.expected
                effective.update(unclipped_scaled_action=expected.tolist(),clipping='native action-space bounds')
            else:
                old=self.commands[-int(self.expected)-1] if len(self.commands)>self.expected else shift.initial_action
                expected[shift.indices]=old[shift.indices]
                effective.update(requested_delay=int(self.expected),queue_length=len(shift.queue),
                                 queue_capacity=shift.max_delay+1)
                if len(shift.queue)!=min(len(self.commands),shift.max_delay+1):
                    raise AssertionError('Delay queue length mismatch.')
            np.testing.assert_allclose(exe,np.clip(expected,shift.low,shift.high),rtol=1e-6,atol=1e-7)
        else:
            # Actual simulator getter readback, not the cached shift label.
            rb=shift.backend.readback()
            if kind=='object_mass':
                np.testing.assert_allclose(rb['mass'],rb['nominal_mass']*self.expected,rtol=1e-5)
                np.testing.assert_allclose(rb['inertia'],shift.backend.baseline['inertia']*self.expected,rtol=1e-5)
            else:
                baseline=np.asarray(rb['nominal_materials']);actual=np.asarray(rb['materials'])
                np.testing.assert_allclose(actual,baseline*np.array([self.expected,self.expected,1]),rtol=1e-5)
            effective.update(rb)
        self.effective=deepcopy(effective)
        row=dict(timestep=record['episode_native_step'], commanded_action=cmd,executed_action=exe,
                 reward=record['reward'],success=record['success'],done=record['done'],
                 observation=np.asarray(observation).copy(),queue_length=effective.get('queue_length',-1))
        if self.log_eef:
            row['eef_xyz'],row['eef_quaternion_wxyz']=measured_pose(shift.unwrapped)
            native = shift.unwrapped
            if hasattr(native, 'obj') and hasattr(native, 'goal_pos'):
                row['object_goal_distance_m'] = float(np.linalg.norm(
                    np.asarray(native.obj.pose.p)-np.asarray(native.goal_pos)))
        self.rows.append(row)

    def save(self, path, metadata, result):
        if len(self.rows)!=result['native_steps'] or not self.rows:
            raise AssertionError('Trajectory/native-step count mismatch.')
        arrays={key:np.asarray([r[key] for r in self.rows]) for key in self.rows[0]}
        if any(a.dtype==object for a in arrays.values()):
            raise ValueError('Trajectory arrays must not require pickle.')
        arrays['episode_success']=np.full(len(self.rows),bool(result['success']))
        with path.with_suffix('.npz').open('xb') as f:
            np.savez_compressed(f,**arrays)
        with path.with_suffix('.json').open('x') as f:
            json.dump(dict(metadata,**self.initial,effective_parameter=self.effective,
                           result=result,pose_sample='post native env.step',
                           quaternion_order='wxyz',position_units='meters; simulator world frame'),f,default=serial,indent=2)


def evaluate_frozen(agent, cfg, make_env, episode_fn, output, metadata, find_shift):
    """No adapter/replay/update calls. Same episode loop/seeding as Stage 3B."""
    import random
    import torch
    previous=agent._prev_mean.clone()
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
         torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    before=model_digest(agent.model)
    results={}; effective={}; initial_ids={}; initial_kinematic_states={}
    try:
        for condition in ('shifted','nominal'):
            config=deepcopy(cfg)
            if condition=='nominal': config.schedule['type']='nominal'
            env=make_env(config)
            shift=find_shift(env)
            expected=shift.parameter['shifted' if condition=='shifted' else 'nominal']
            results[condition]=[];effective[condition]=[];initial_ids[condition]=[]
            initial_kinematic_states[condition]=[]
            try:
                for seed in cfg.stage3a['evaluation_seeds']:
                    trajectory=NativeTrajectory(shift,expected,cfg.severity['log_eef'])
                    shift.native_step_observer=trajectory.observe
                    with torch.no_grad():
                        result=episode_fn(agent,env,seed,on_reset=trajectory.reset)
                    if model_digest(agent.model)!=before:
                        raise AssertionError('Frozen model state changed during evaluation.')
                    trajectory.save(output/f'{condition}-seed-{seed}',dict(metadata,condition=condition),result)
                    results[condition].append(result)
                    effective[condition].append(trajectory.effective)
                    initial_ids[condition].append(trajectory.initial['initial_state_sha256'])
                    initial_kinematic_states[condition].append(trajectory.initial['initial_kinematic_state'])
                    with (output/'severity-episodes.jsonl').open('a') as f:
                        f.write(json.dumps(dict(condition=condition,**result))+'\n')
            finally:
                shift.__dict__.pop('native_step_observer',None)
                try:
                    if 'backend' in shift.__dict__: shift.backend.restore()
                    if 'queue' in shift.__dict__: shift.queue.clear()
                finally:
                    env.close()
        for seed, shifted, nominal in zip(cfg.stage3a['evaluation_seeds'],
                initial_kinematic_states['shifted'],initial_kinematic_states['nominal']):
            if not same_initial_kinematics(shifted,nominal):
                raise AssertionError(f'Shifted/nominal initial kinematics differ for seed {seed}.')
        return dict(evaluation=results,effective_physical_values=effective,initial_state_ids=initial_ids,
                    initial_kinematic_states=initial_kinematic_states,
                    model_state_sha256=before,parameter_drift=0,gradient_updates=0,replay_updates=0,
                    nominal_restored=True)
    finally:
        agent._prev_mean.copy_(previous)
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        if rng[3] is not None: torch.cuda.set_rng_state_all(rng[3])
