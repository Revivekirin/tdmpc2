"""Audit the installed ManiSkill2/SAPIEN stack, without a model/checkpoint.

All artifacts go into a new directory. This does not edit benchmark code or logs.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import numpy as np
from audit_shift_logs import output_directory, check, close

ROOT=Path(__file__).resolve().parents[1]


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    if value is None or isinstance(value,(bool,str,int,float)): return value
    return repr(value)


def tree_equal(a,b,atol):
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(tree_equal(a[k],b[k],atol) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(tree_equal(x,y,atol) for x,y in zip(a,b))
    if a is None or b is None: return a is b
    if isinstance(a,str) or isinstance(b,str): return a==b
    return close(a,b,atol)


def snapshot(native):
    controller=native.agent.controller
    if not callable(getattr(controller,'get_state',None)) or not callable(getattr(controller,'set_state',None)):
        raise NotImplementedError('Installed controller lacks get_state/set_state; identical controller initialization cannot be certified.')
    return dict(simulator=np.asarray(native.get_state()).copy(),controller=deepcopy(controller.get_state()))


def restore(native,state):
    native.set_state(state['simulator'].copy())
    native.agent.controller.set_state(deepcopy(state['controller']))


def first_divergence(reference,candidate,atol):
    for index,(a,b) in enumerate(zip(reference,candidate)):
        for key in ('observation','reward','simulator','success','done'):
            exact=key in {'success','done'}
            if not close(a[key],b[key],0 if exact else atol):
                x,y=np.asarray(a[key]),np.asarray(b[key])
                return dict(native_step=index,decision_step=index//2,field=key,
                            max_absolute_error=float(np.max(np.abs(x-y))) if x.shape==y.shape and not exact else None,
                            reference=plain(a[key]) if exact else None,candidate=plain(b[key]) if exact else None)
    if len(reference)!=len(candidate): return dict(native_step=min(len(reference),len(candidate)),field='trajectory_length')
    return None


class Trace:
    def __init__(self,env): self.env=env; self.records=[]
    def __getattr__(self,key): return getattr(self.env,key)
    @property
    def unwrapped(self): return self.env.unwrapped
    def reset(self,**kwargs): self.records=[]; return self.env.reset(**kwargs)
    def step(self,action):
        obs,reward,done,info=self.env.step(action)
        self.records.append(dict(observation=np.asarray(obs).copy(),reward=float(reward),done=bool(done),
                                 success=bool(info['success']),simulator=np.asarray(self.unwrapped.get_state()).copy()))
        return obs,reward,done,info
    def close(self): return self.env.close()


def config(kind='action_gain',schedule='fixed',value=1.,enabled=True):
    return dict(enabled=enabled,application='reset' if kind in {'object_mass','contact_friction'} else 'step',
                parameter=dict(type=kind,target='arm',nominal=0 if kind=='action_delay' else 1.,shifted=value,
                               actor='obj',surfaces=['@plane']),
                schedule=dict(type=schedule,clock='global_native_step',reset_on_episode=False))


def actor_properties(actor):
    shapes=[]
    for shape in actor.get_collision_shapes():
        geometry=shape.geometry
        values={}
        for key in ('half_lengths','radius','half_length','scale','vertices','indices'):
            if hasattr(geometry,key): values[key]=plain(getattr(geometry,key))
        material=shape.get_physical_material()
        pose=shape.get_local_pose()
        shapes.append(dict(geometry=type(geometry).__name__,geometry_values=values,local_pose=[plain(pose.p),plain(pose.q)],
                           groups=plain(shape.get_collision_groups()),friction=[material.get_static_friction(),material.get_dynamic_friction()],
                           restitution=material.get_restitution()))
    pose,com=actor.get_pose(),actor.get_cmass_local_pose()
    return dict(mass=actor.get_mass(),inertia=plain(actor.get_inertia()),com=[plain(com.p),plain(com.q)],
                pose=[plain(pose.p),plain(pose.q)],velocity=plain(actor.get_velocity()),angular_velocity=plain(actor.get_angular_velocity()),shapes=shapes)


def references(native,actor):
    names=[]
    for key,value in vars(native).items():
        if value is actor: names.append(key)
        if isinstance(value,(list,tuple)):
            names.extend(f'{key}[{i}]' for i,item in enumerate(value) if item is actor)
        if isinstance(value,dict):
            names.extend(f'{key}[{k!r}]' for k,item in value.items() if item is actor)
    return names


def installed_evidence(native,output):
    objects=[native,native.agent,native.agent.controller]
    components=getattr(native.agent.controller,'controllers',{})
    if isinstance(components,dict): objects.extend(components.values())
    files={}; excerpts=[]
    for obj in objects:
        for cls in type(obj).__mro__:
            try: filename=inspect.getsourcefile(cls)
            except TypeError: continue
            if filename and Path(filename).is_file():
                source=Path(filename); digest=hashlib.sha256(source.read_bytes()).hexdigest()
                dest=output/f'source-{digest[:12]}-{source.name}'
                if not dest.exists(): dest.write_bytes(source.read_bytes())
                files[str(source)]=dict(sha256=digest,copied_to=dest.name)
        for name in ('set_action','_preprocess_action','before_simulation_step','step_action','get_done','compute_target_pose'):
            method=getattr(obj,name,None)
            if callable(method):
                try: excerpts.append(dict(owner=type(obj).__qualname__,method=name,source=inspect.getsource(method)))
                except (TypeError,OSError): pass
    versions={}
    for name in ('mani-skill2','sapien','gym','gymnasium','torch'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    actor=native.obj
    actor_api={name:dict(available=callable(getattr(actor,name,None)),doc=getattr(getattr(actor,name,None),'__doc__',None))
               for name in ('get_builder','get_mass','get_inertia','get_cmass_local_pose','set_mass','set_inertia','set_mass_and_inertia')}
    ctrl=native.agent.controller
    return dict(versions=versions,installed_source_files=files,method_sources=excerpts,actor_api=actor_api,
                sim_freq=getattr(native,'sim_freq',None),control_freq=getattr(native,'control_freq',None),
                sim_steps_per_control=getattr(native,'_sim_steps_per_control',None),
                action_bounds=dict(low=plain(native.action_space.low),high=plain(native.action_space.high)),
                controller_type=type(ctrl).__qualname__,controller_config=repr(getattr(ctrl,'config',None)),
                controller_components={k:dict(type=type(v).__qualname__,config=repr(getattr(v,'config',None))) for k,v in components.items()} if isinstance(components,dict) else {})


def contact_probe(make_native,make_shift,kind,seed,atol):
    """Controlled force-response diagnostic; it is not a policy success experiment."""
    traces=[]; initial=None
    for factor in (1.,1.,.5,1.5):
        raw=make_native()
        try:
            env=make_shift(raw,config(kind,value=factor),task='pick-cube')
            env.reset(seed=seed)
            if initial is None: initial=snapshot(raw.unwrapped)
            restore(raw.unwrapped,initial)
            actor=raw.unwrapped.obj
            force=getattr(actor,'add_force_torque',None)
            scene=actor.get_scene()
            if not callable(force) or not callable(getattr(scene,'get_contacts',None)):
                raise NotImplementedError('Installed force/contact inspection APIs are unavailable.')
            trajectory=[]; contacts=0; impulses=[]; contact_api_known=True
            for _ in range(20):
                # Equal force call before each native control step. No claim that
                # the force persists across all internal physics substeps.
                force(np.array([20.,0.,0.]),np.zeros(3))
                _,_,done,_=env.step(np.zeros(raw.action_space.shape,dtype=np.float32))
                trajectory.append(np.r_[actor.get_pose().p,actor.get_velocity()])
                for contact in scene.get_contacts():
                    pair=[getattr(contact,'actor0',None),getattr(contact,'actor1',None)]
                    if any(a is None for a in pair): contact_api_known=False; continue
                    if not any(a.get_id()==actor.get_id() for a in pair): continue
                    contacts+=1
                    for point in getattr(contact,'points',[]):
                        impulse=getattr(point,'impulse',None)
                        if impulse is not None: impulses.append(float(np.linalg.norm(impulse)))
                if done: break
            traces.append(dict(factor=factor,trajectory=np.asarray(trajectory),contacts=contacts,
                               contact_api_known=contact_api_known,impulse_norms=impulses,readback=plain(env.backend.readback())))
        finally: raw.close()
    def difference(a,b):
        if a.shape!=b.shape: return None
        return float(np.max(np.abs(a-b)))
    control_error=difference(traces[0]['trajectory'],traces[1]['trajectory'])
    differences=[difference(traces[0]['trajectory'],row['trajectory']) for row in traces[2:]]
    observed=all(r['contacts'] and r['contact_api_known'] and any(v>0 for v in r['impulse_norms']) for r in traces)
    resolved=control_error is not None and control_error<=atol and observed and all(d is not None and d>atol for d in differences)
    return dict(status='PASS' if resolved else 'UNVERIFIED',seed=seed,parameter=kind,
                nominal_repeat_max_error=control_error,shifted_trajectory_max_errors=differences,
                nonzero_object_contact_observed=observed,traces=plain(traces),
                scope='Controlled response only; interpret mass only after actor replacement/restore tests pass.')


def run(args,output):
    sys.path.insert(0,str(ROOT/'tdmpc2'))
    import gym
    import mani_skill2.envs
    from envs.maniskill import ManiSkillWrapper
    from envs.wrappers.timeout import Timeout
    from envs.wrappers.dynamics_shift import make_shift
    def make_native():
        return gym.make('PickCube-v0',obs_mode='state',control_mode='pd_ee_delta_pos',render_camera_cfgs=dict(width=384,height=384))
    report=dict(status='UNVERIFIED',scope='identity_and_property_checks_only',atol=args.atol,rtol=0.,identities=[],physics=[],limitations=[
        'No model inference; frozen parameter hashes must be checked separately for a model-run claim.',
        'Copied Python installation sources and exposed API docstrings do not include SAPIEN C++ implementation.',
        'Public state lacks PhysX internal solver caches; repeated-disabled control calibrates reproducibility.'])
    for seed in args.seeds:
        actions=np.random.default_rng(seed+10000).uniform(-.3,.3,(args.decisions,4)).astype(np.float32)
        raw=make_native()
        try:
            raw.reset(seed=seed)
            initial=snapshot(raw.unwrapped)
            initial_obs=np.asarray(raw.unwrapped.get_obs()).copy()
            initial_properties=actor_properties(raw.unwrapped.obj)
            if 'installed' not in report: report['installed']=installed_evidence(raw.unwrapped,output)
            trace=Trace(raw); outer=Timeout(ManiSkillWrapper(trace,None),max_episode_steps=100)
            # Native reset already happened; reset outer counters then restore exactly.
            outer.reset(seed=seed); restore(raw.unwrapped,initial)
            for action in actions:
                _,_,done,_=outer.step(action)
                if done: break
            reference=deepcopy(trace.records)
        finally: raw.close()
        cases=[('disabled_repeat',None),('factory_disabled',config(enabled=False)),
               ('enabled_nominal',config(schedule='nominal',value=.7)),('gain_1',config()),
               ('delay_0',config('action_delay',value=0)),('mass_1',config('object_mass')),
               ('friction_1',config('contact_friction'))]
        calibration=True
        for label,cfg in cases:
            raw=make_native()
            try:
                trace=Trace(raw); shifted=make_shift(trace,cfg,task='pick-cube') if cfg else trace
                outer=Timeout(ManiSkillWrapper(shifted,None),max_episode_steps=100)
                obs=outer.reset(seed=seed)
                reset_equal=(close(initial['simulator'],raw.unwrapped.get_state(),args.atol) and close(initial_obs,obs,args.atol)
                             and tree_equal(plain(initial['controller']),plain(raw.unwrapped.agent.controller.get_state()),args.atol))
                restore(raw.unwrapped,initial)
                for action in actions:
                    _,_,done,_=outer.step(action)
                    if done: break
                divergence=first_divergence(reference,trace.records,args.atol)
                status='PASS' if reset_equal and divergence is None else 'FAIL'
                if label=='disabled_repeat': calibration=status=='PASS'
                elif not calibration: status='UNVERIFIED'
                report['identities'].append(dict(seed=seed,case=label,status=status,reset_equal=reset_equal,
                                                  first_divergence=divergence,native_steps=len(trace.records)))
            except Exception as exc:
                report['identities'].append(dict(seed=seed,case=label,status='FAIL' if isinstance(exc,AssertionError) else 'UNVERIFIED',error=f'{type(exc).__name__}: {exc}'))
                if label=='disabled_repeat': calibration=False
            finally: raw.close()
        for kind in ('object_mass','contact_friction'):
            raw=make_native()
            try:
                trace=Trace(raw); env=make_shift(trace,config(kind,value=.5),task='pick-cube')
                raw.reset(seed=seed); old=raw.unwrapped.obj; before=actor_properties(old); before_refs=references(raw.unwrapped,old)
                env.reset(seed=seed); after=actor_properties(raw.unwrapped.obj)
                stale=references(raw.unwrapped,old) if old is not raw.unwrapped.obj else []
                rb=deepcopy(env.readback)
                invariants={key:tree_equal(before[key],after[key],args.atol) for key in ('com','pose','velocity','angular_velocity')}
                if kind=='object_mass': invariants['shapes']=tree_equal(before['shapes'],after['shapes'],args.atol)
                invariants['reference_slots']=before_refs==references(raw.unwrapped,raw.unwrapped.obj)
                # Restore using the EXISTING backend only; do not invent unsupported physics APIs.
                env.backend.restore()
                restored=actor_properties(raw.unwrapped.obj)
                properties_restored=tree_equal(restored,initial_properties,args.atol)
                outer=Timeout(ManiSkillWrapper(trace,None),max_episode_steps=100)
                obs=outer.reset(seed=seed)
                reset_equal=(close(initial['simulator'],raw.unwrapped.get_state(),args.atol) and close(initial_obs,obs,args.atol)
                             and tree_equal(plain(initial['controller']),plain(raw.unwrapped.agent.controller.get_state()),args.atol))
                restore(raw.unwrapped,initial)
                for action in actions:
                    _,_,done,_=outer.step(action)
                    if done: break
                divergence=first_divergence(reference,trace.records,args.atol)
                valid=all(invariants.values()) and not stale and properties_restored and reset_equal and divergence is None
                report['physics'].append(dict(seed=seed,parameter=kind,status=('PASS' if valid else 'FAIL') if calibration else 'UNVERIFIED',
                    readback=plain(rb),invariants=invariants,stale_shallow_references=stale,properties_restored=properties_restored,
                    reset_equal=reset_equal,first_divergence=divergence,
                    contact_response=check('UNVERIFIED','This minimal identity test does not establish effective friction or contact forces.'),
                    all_references_preserved=check('UNVERIFIED','Only direct/list/dict attributes of the native environment were inspected; nested/controller caches need source review.')))
            except Exception as exc:
                report['physics'].append(dict(seed=seed,parameter=kind,status='FAIL' if isinstance(exc,AssertionError) else 'UNVERIFIED',error=f'{type(exc).__name__}: {exc}'))
            finally: raw.close()
    if args.contact_probe:
        report['contact_probes']=[]
        for kind in ('object_mass','contact_friction'):
            try: report['contact_probes'].append(contact_probe(make_native,make_shift,kind,args.seeds[0],args.atol))
            except Exception as exc: report['contact_probes'].append(dict(parameter=kind,status='UNVERIFIED',error=f'{type(exc).__name__}: {exc}'))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root',type=Path,default=Path('audits'))
    parser.add_argument('--seeds',nargs='+',type=int,default=[17])
    parser.add_argument('--decisions',type=int,default=12)
    parser.add_argument('--atol',type=float,default=1e-6)
    parser.add_argument('--contact-probe',action='store_true',help='Also measure a controlled force/contact response; no policy/checkpoint.')
    args=parser.parse_args()
    if args.decisions<1 or args.atol<0: parser.error('decisions must be positive and atol nonnegative')
    output=output_directory(args.out_root)
    try: report=run(args,output)
    except Exception as exc: report=dict(status='UNVERIFIED',error=f'{type(exc).__name__}: {exc}')
    (output/'simulator_audit.json').write_text(json.dumps(plain(report),indent=2)+'\n')
    print(output.resolve())


if __name__=='__main__': main()
