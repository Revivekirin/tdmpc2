import json
import types
import unittest
from copy import deepcopy
import numpy as np
from benchmark_imports import package
from benchmark_shift.logging import encode_state, decode_state
from benchmark_shift.schedule import Schedule
from common.shift_protocol import evaluation_seeds, make_protocol, validate_probe_holdout
from test_dynamics_shift import Native


def config(kind='action_gain', schedule='fixed', nominal=1., shifted=.7, **schedule_options):
    return dict(enabled=True, application='step', parameter=dict(type=kind, target='arm', nominal=nominal, shifted=shifted),
                schedule=dict(type=schedule, clock='global_native_step', reset_on_episode=False, **schedule_options))


class BenchmarkTests(unittest.TestCase):
    def test_nominal_all_action_types(self):
        for kind, nominal, shifted in [('action_gain',1,.7), ('action_bias',0,.3), ('action_delay',0,2),
                                       ('action_coupling',np.eye(3).tolist(), [[1,.2,0],[0,1,0],[0,0,1]])]:
            raw, env = Native(), package.ActionShiftWrapper(Native(), config(kind, 'nominal', nominal, shifted))
            raw.reset(); env.reset()
            for action in np.random.default_rng(12).uniform(-1,1,(20,4)).astype(np.float32):
                a, b = raw.step(action), env.step(action)
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1:3], b[1:3])
                np.testing.assert_array_equal(action, b[3]['shift_record']['commanded_action'])

    def test_gain_bias_coupling_clipping_gripper(self):
        cases = [('action_gain',1,2,[1,-1,.4,.3]), ('action_bias',0,.3,[1,-.5,.5,.3]),
                 ('action_coupling',np.eye(3).tolist(),[[1,1,0],[0,1,0],[0,0,1]],[0,-.8,.2,.3])]
        for kind, nominal, shifted, expected in cases:
            env = package.ActionShiftWrapper(Native(), config(kind,'fixed',nominal,shifted)); env.reset()
            action = np.array([.8,-.8,.2,.3],dtype=np.float32)
            obs, _, _, info = env.step(action)
            np.testing.assert_allclose(obs,expected,atol=1e-7)
            np.testing.assert_array_equal(info['shift_record']['applied_parameter'], shifted)

    def test_delay_queue_reset_and_preserve(self):
        for reset in [True, False]:
            cfg = config('action_delay', 'fixed', 0, 2)
            cfg['parameter']['reset_queue_on_episode'] = reset
            env = package.ActionShiftWrapper(Native(), cfg); env.reset()
            for i, expected in [(1,0),(2,0),(3,1)]:
                info = env.step(np.array([i/10]*3+[.8]))[3]['shift_record']
                np.testing.assert_allclose(info['executed_action'], [expected/10]*3+[.8])
            env.reset(seed=7)
            np.testing.assert_allclose(env.step(np.array([.4]*4))[3]['shift_record']['executed_action'],
                                       [0 if reset else .2]*3+[.4])

    def test_discrete_and_continuous_schedules(self):
        cases=[('abrupt',dict(switch_step=2),[1,1,.7,.7,.7]),
               ('repeated',dict(interval=2),[1,1,.7,.7,1]),
               ('sinusoidal',dict(interval=4),[1,.7,1,1.3,1]),
               ('linear_drift',dict(switch_step=1,duration=2),[1,1,.85,.7,.7])]
        for kind, options, expected in cases:
            s=Schedule(dict(type=kind, **options),1,.7)
            np.testing.assert_allclose([s.value(t)[0] for t in range(5)],expected)

    def test_sequence_and_probe_holdout(self):
        s=Schedule(dict(type='sequence',segments=[dict(duration=2,value=v) for v in [1,.7,1,.5]]),1,.7)
        self.assertEqual([s.value(t)[0] for t in range(9)],[1,1,.7,.7,1,1,.5,.5,.5])
        with self.assertRaisesRegex(ValueError,'reserved'):
            validate_probe_holdout(s,'train',[.5])
        validate_probe_holdout(s,'probe',[.5])
        validate_probe_holdout(s,'eval',[.3])

    def test_random_resume_including_delay_queue(self):
        cfg=config('action_delay','random_piecewise',0,2,interval=2,seed=19,values=[0,1,2])
        env=package.ActionShiftWrapper(Native(),cfg); env.reset()
        clone=package.ActionShiftWrapper(Native(),cfg); clone.reset()
        for i in range(7):
            env.step(np.full(4,i/100))
        snapshot=decode_state(json.loads(json.dumps(encode_state(env.state_dict()))))
        clone.load_state_dict(snapshot)
        for i in range(7,25):
            a=env.step(np.full(4,i/100))[3]['shift_record']
            b=clone.step(np.full(4,i/100))[3]['shift_record']
            for field in ['applied_parameter','native_step','shift_segment_id','shift_change_count']:
                self.assertEqual(a[field],b[field])
            np.testing.assert_array_equal(a['executed_action'],b['executed_action'])
        one=Schedule(cfg['schedule'],0,2); two=Schedule(cfg['schedule'],0,2)
        self.assertEqual([one.value(t) for t in range(20)],[two.value(t) for t in range(20)])

    def test_decision_repeat_and_clock(self):
        import ast
        from benchmark_imports import ROOT
        tree=ast.parse((ROOT/'tdmpc2/envs/maniskill.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ManiSkillWrapper')
        scope={'gym':types.SimpleNamespace(Env=object,spaces=types.SimpleNamespace(Box=lambda **kw:kw)),
               'np':np,'ShiftWrapper':package.ShiftWrapper}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'maniskill.py','exec'),scope)
        for clock, expected in [('global_native_step',[1,.7,.7,.7]),('global_decision_step',[1,1,.7,.7])]:
            cfg=config(schedule='abrupt',switch_step=1); cfg['schedule']['clock']=clock
            native=package.ActionShiftWrapper(Native(),cfg)
            outer=scope['ManiSkillWrapper'](native,None); outer.reset(seed=9)
            rows=[]
            for _ in range(2):
                rows+=outer.step(np.ones(4)*.1)[3]['shift_native_steps']
            self.assertEqual([r['applied_parameter'] for r in rows],expected)
            self.assertEqual([r['decision_step'] for r in rows],[0,0,1,1])
            self.assertEqual([r['native_step'] for r in rows],[0,1,2,3])
            outer.reset(seed=10)
            self.assertEqual((native.global_step,native.decision_step,native.episode_step),(4,2,0))

    def test_episode_clock_and_events(self):
        cfg=config(schedule='abrupt',switch_step=1)
        cfg['schedule'].update(clock='episode_native_step',reset_on_episode=True)
        env=package.ActionShiftWrapper(Native(),cfg); callback=[]; env.boundary_callbacks.append(callback.append)
        env.reset(seed=1)
        self.assertEqual([env.step(np.zeros(4))[3]['shift_record']['applied_parameter'] for _ in range(2)],[1,.7])
        env.reset(seed=2)
        self.assertEqual(env.step(np.zeros(4))[3]['shift_record']['applied_parameter'],1)
        self.assertEqual(env.change_count,2)
        self.assertEqual(len(callback),3)

    def test_supported_parameter_schedule_matrix(self):
        from test_shift_physics import PhysicsNative
        kinds=[('action_gain',1.,.7),('action_bias',0.,.2),('action_delay',0,2),
               ('action_coupling',np.eye(3).tolist(),[[1,.2,0],[0,1,0],[0,0,1]]),
               ('object_mass',1.,1.5),('contact_friction',1.,.5)]
        schedules=['nominal','fixed','abrupt','repeated','sinusoidal','linear_drift','random_piecewise','sequence']
        for kind, nominal, shifted in kinds:
            for schedule in schedules:
                with self.subTest(parameter=kind,schedule=schedule):
                    cfg=config(kind,schedule,nominal,shifted,interval=2,switch_step=1,duration=2,
                               values=[nominal,shifted],segments=[dict(duration=2,value=v) for v in [nominal,shifted,nominal]])
                    physics=kind in {'object_mass','contact_friction'}
                    if physics:
                        cfg['application']='reset'
                        cfg['parameter'].update(actor='obj',surfaces=['@plane'])
                    if kind=='action_delay' and schedule in {'sinusoidal','linear_drift'}:
                        with self.assertRaises(ValueError): package.make_shift(Native(),cfg)
                        continue
                    env=package.make_shift(PhysicsNative() if physics else Native(),cfg,task='pick-cube')
                    for episode in range(2):
                        env.reset(seed=episode)
                        for _ in range(4):
                            record=env.step(np.full(4,.1))[3]['shift_record']
                            self.assertEqual(record['shift_type'],kind)
                            self.assertEqual(record['schedule_type'],schedule)
                            self.assertTrue(np.all(np.isfinite(record['executed_action'])))

    def test_validation_and_seed_protocol(self):
        for schedule in ['sinusoidal','linear_drift']:
            with self.assertRaisesRegex(ValueError,'Integer action delay'):
                package.ActionShiftWrapper(Native(),config('action_delay',schedule,0,2))
        with self.assertRaises(ValueError):
            Schedule(dict(type='nominal',clock='episode_native_step'),1,.7)
        with self.assertRaises(ValueError):
            package.ActionShiftWrapper(Native(),config('action_delay','fixed',0,1.5))
        with self.assertRaises(ValueError):
            package.make_shift(Native(),config('unknown'))
        with self.assertRaises(NotImplementedError):
            make_protocol('dynamics_only')
        self.assertEqual(evaluation_seeds(7,3),[7,8,9])
        self.assertEqual(evaluation_seeds(7,3,[10,20,30]),[10,20,30])
        with self.assertRaises(ValueError):
            evaluation_seeds(7,3,[1,2])


if __name__=='__main__':
    unittest.main()
