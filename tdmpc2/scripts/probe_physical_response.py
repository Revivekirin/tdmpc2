"""Bounded SAPIEN 2 probe, no model/checkpoint/training imports.

Run on the existing Linux simulator environment with a prescribed native-action
tape (.npy, shape [T,4], or severity .npz commanded_action). Maximum 80 native
steps per condition. Writes only to a new output directory. A contact-free tape
cannot validate grasp/friction sensitivity; contact presence is recorded.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import subprocess
import sys
import dataclasses
from pathlib import Path

import numpy as np


def serialize(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    if isinstance(x, type):
        return x.__module__ + '.' + x.__qualname__
    raise TypeError(f'Unsupported JSON value: {type(x).__name__}')


def vector(value, size=3):
    """Fail explicitly rather than silently stringify SAPIEN vector bindings."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        arr = np.asarray([getattr(value, k) for k in ('x','y','z')], dtype=float)
    if arr.shape != (size,) or not np.isfinite(arr).all():
        raise ValueError(f'Expected finite numeric vector of size {size}')
    return arr.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=4000)
    parser.add_argument('--max-native-steps', type=int, default=40, choices=range(1,81))
    parser.add_argument('--shift', choices=['action_gain','action_delay','object_mass','contact_friction'], required=True)
    parser.add_argument('--values', type=float, nargs='+', required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--npy-native-tape', action='store_true', help='Explicitly assert a prescribed NPY has one command per native step.')
    args = parser.parse_args()
    if len(args.values)>5 or len(set(args.values)) != len(args.values):
        parser.error('Use at most five distinct conditions.')
    if any(not np.isfinite(v) or (v<0 if args.shift=='action_delay' else v<=0) for v in args.values):
        parser.error('Invalid severity.')
    if args.shift=='action_delay' and any(v!=int(v) or v>3 for v in args.values):
        parser.error('This bounded probe supports integer delays 0..3.')
    loaded = np.load(args.actions, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        actions = loaded['commanded_action'].copy()
        if 'timestep' not in loaded or not np.array_equal(loaded['timestep'],np.arange(len(actions))):
            parser.error('NPZ must contain contiguous native timestep starting at zero.')
        loaded.close()
        sidecar = args.actions.with_suffix('.json')
        if not sidecar.exists(): parser.error('Native trajectory JSON sidecar required for seed/count provenance.')
        tape_meta = json.loads(sidecar.read_text())
        if tape_meta.get('evaluation_seed') != args.seed or tape_meta.get('result',{}).get('native_steps') != len(actions):
            parser.error('Tape seed or native-step count disagrees with sidecar.')
    else:
        if not args.npy_native_tape: parser.error('NPY requires --npy-native-tape; never supply policy-decision actions.')
        actions = loaded
        tape_meta = dict(origin='user-prescribed native-step tape',seed=args.seed)
    actions = np.asarray(actions[:args.max_native_steps], dtype=np.float32)
    if actions.ndim!=2 or actions.shape[1]!=4 or len(actions)==0 or not np.isfinite(actions).all() or np.abs(actions).max()>1:
        parser.error('Action tape must be nonempty [T,4], finite and within [-1,1].')
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.repo.resolve()
    sys.path.insert(0,str(root/'tdmpc2'))
    import gym
    import mani_skill2.envs
    from envs.wrappers.dynamics_shift.factory import make_shift
    from common.severity_protocol import source_digest
    def git(*cmd):
        return subprocess.check_output(['git','-C',str(root),*cmd],text=True).strip()
    meta = dict(git_head=git('rev-parse','HEAD'),git_status=git('status','--short'),
                actions_sha256=hashlib.sha256(args.actions.read_bytes()).hexdigest(),
                seed=args.seed, shift=args.shift, values=args.values,
                tape_metadata=tape_meta, tape_clock='native_control_step',
                max_native_steps=args.max_native_steps, no_learning=True,
                versions={k:importlib.metadata.version(k) for k in ['mani-skill2','sapien','gym','numpy']})
    meta['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (root/'tdmpc2/envs/wrappers/dynamics_shift').glob('*.py')}
    meta['repository_source_sha256'] = source_digest(root)
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    baseline_initial = None
    for value in args.values:
        value = int(value) if args.shift=='action_delay' else value
        config = dict(enabled=True, application='reset' if args.shift in ['object_mass','contact_friction'] else 'step',
                      parameter=dict(type=args.shift, nominal=0 if args.shift=='action_delay' else 1,
                                     shifted=value,target='arm',actor='obj',surfaces=['@plane']),
                      schedule=dict(type='fixed',clock='global_native_step'))
        raw = gym.make('PickCube-v0',obs_mode='state',control_mode='pd_ee_delta_pos')
        env = make_shift(raw,config,task='pick-cube')
        try:
            obs = env.reset(seed=args.seed); base=env.unwrapped
            ctrl = base.agent.controller; arm=ctrl.controllers['arm']
            fingers = [link for link in base.agent.robot.get_links() if link.get_name() in ('panda_leftfinger','panda_rightfinger')]
            finger_ids = {link.get_id() for link in fingers}
            plane_ids = {a.get_id() for a in base.obj.get_scene().get_all_actors()
                         if any(type(s.geometry).__name__=='PlaneGeometry' for s in a.get_collision_shapes())}
            def state():
                return dict(qpos=base.agent.robot.get_qpos().copy(),qvel=base.agent.robot.get_qvel().copy(),
                            eef_pose=np.r_[base.tcp.pose.p,base.tcp.pose.q],
                            object_pose=np.r_[base.obj.pose.p,base.obj.pose.q],
                            object_velocity=base.obj.get_velocity().copy(),
                            object_angular_velocity=base.obj.get_angular_velocity().copy(),
                            object_in_tcp=np.r_[(base.tcp.pose.inv()*base.obj.pose).p,(base.tcp.pose.inv()*base.obj.pose).q],
                            object_in_fingers={link.get_name():np.r_[(link.pose.inv()*base.obj.pose).p,(link.pose.inv()*base.obj.pose).q] for link in fingers})
            initial=state()
            # Kinematic equality check only; no repeated pairing audit.
            if baseline_initial is None: baseline_initial=initial
            else:
                for key in ['qpos','qvel','eef_pose','object_pose','object_velocity','object_angular_velocity']:
                    np.testing.assert_allclose(initial[key],baseline_initial[key],atol=1e-6,rtol=0)
            inventory=dict(action_low=env.action_space.low,action_high=env.action_space.high,
                           obs_shape=np.shape(obs),sim_freq=base.sim_freq,control_freq=base.control_freq,
                           controllers={k:vars(c.config) for k,c in ctrl.controllers.items()},
                           mass=base.obj.get_mass(),inertia=base.obj.get_inertia(),
                           physics_readback=env.backend.readback() if hasattr(env,'backend') else None,
                           robot_base_pose=np.r_[base.agent.robot.pose.p,base.agent.robot.pose.q],
                           center_of_mass=np.r_[base.obj.get_cmass_local_pose().p,base.obj.get_cmass_local_pose().q],
                           surfaces=[])
            for actor in list(base.obj.get_scene().get_all_actors())+list(base.agent.robot.get_links()):
                for shape in actor.get_collision_shapes():
                    m=shape.get_physical_material()
                    inventory['surfaces'].append(dict(actor=actor.get_name(),geometry=type(shape.geometry).__name__,
                        friction=[m.get_static_friction(),m.get_dynamic_friction()],restitution=m.get_restitution()))
            sources={}
            for concrete in [type(base),type(base.agent),type(ctrl),type(arm),type(ctrl.controllers['gripper'])]:
                for thing in concrete.__mro__:
                    if thing.__module__.startswith('mani_skill2'):
                        path=Path(inspect.getfile(thing)); text=path.read_text()
                        sources[str(path)]=dict(sha256=hashlib.sha256(text.encode()).hexdigest(),text=text)
            rows=[]; substeps=[]
            original_after = base._after_simulation_step
            def capture_substep():
                original_after()
                contacts=[]
                for contact in base.obj.get_scene().get_contacts():
                    if base.obj.get_id() in [contact.actor0.get_id(), contact.actor1.get_id()]:
                        other=contact.actor1 if contact.actor0.get_id()==base.obj.get_id() else contact.actor0
                        kind='object-finger' if other.get_id() in finger_ids else 'object-plane' if other.get_id() in plane_ids else 'object-other'
                        contacts.append(dict(kind=kind,actors=[contact.actor0.get_name(),contact.actor1.get_name()],
                            points=[dict(position=vector(p.position),normal=vector(p.normal),impulse=vector(p.impulse),separation=float(p.separation)) for p in contact.points]))
                world_target=base.agent.robot.pose*arm._target_pose
                substeps.append(dict(physics_substep=len(substeps),contacts=contacts,state=state(),
                                     target_pose_world=np.r_[world_target.p,world_target.q]))
            base._after_simulation_step=capture_substep
            for t,action in enumerate(actions):
                substeps=[]
                before=state(); obs,reward,done,info=env.step(action); after=state()
                if len(substeps)!=base.sim_freq//base.control_freq:
                    raise RuntimeError('Physics substep observer did not capture the complete native step.')
                world_target=base.agent.robot.pose*arm._target_pose
                rows.append(dict(native_step=t,commanded=action.copy(),executed=info['shift_record']['executed_action'],
                    arm_target_qpos=arm._target_qpos.copy(),target_pose_base=np.r_[arm._target_pose.p,arm._target_pose.q],
                    target_pose_world=np.r_[world_target.p,world_target.q],
                    before=before,after=after,reward=reward,success=info.get('success'),done=done,substeps=substeps,
                    contacts_scope='all physics substeps; position m; impulse N*s; not force',
                    average_object_acceleration=(after['object_velocity']-before['object_velocity'])*base.control_freq))
                if done: break
            result=dict(inventory=inventory,initial=initial,rows=rows,installed_sources=sources,
                        interpretation='Prescribed native open-loop tape; all physics substeps. Relative object pose is measured motion, not automatically contact slip.')
            (args.output/f'{args.shift}-{value:g}.json').write_text(json.dumps(result,default=serialize,indent=2,allow_nan=False)+'\n')
        finally:
            env.close()
    print(args.output)


if __name__=='__main__':
    main()
