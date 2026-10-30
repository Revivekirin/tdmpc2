"""V2 composition, legacy compatibility, exposure and artifact contracts."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from benchmark_imports import ROOT, package
from test_dynamics_shift import Native
from test_shift_benchmark import config
from common.benchmark_config import normalize, cli_aliases, config_hash, runtime_shift
from common.benchmark_validation import validate
from common.benchmark_metrics import exposure
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf


def composed(*args):
    overrides = cli_aliases(['task=pick-cube', *args])
    with initialize_config_dir(config_dir=str(ROOT/'tdmpc2'), version_base=None):
        return compose(config_name='config', overrides=overrides)


def canonical(*args):
    return normalize(composed(*args), cli_aliases(args))


class V2Tests(unittest.TestCase):
    def test_matrix_and_overrides(self):
        for parameter in ['nominal','action_gain','action_delay','object_mass','contact_friction']:
            for schedule in ['fixed','abrupt','repeated','sequence','sinusoidal','drift']:
                for evaluation in ['standard','paired']:
                    args=[f'dynamics_shift={parameter}',f'schedule={schedule}',f'evaluation={evaluation}']
                    invalid=(parameter in {'object_mass','contact_friction'} and schedule!='fixed') or (parameter=='action_delay' and schedule in {'sinusoidal','drift','sequence'})
                    with self.subTest(args=args):
                        if invalid:
                            with self.assertRaises(ValueError):
                                validate(OmegaConf.to_container(canonical(*args),resolve=True))
                        else:
                            cfg=canonical(*args)
                            validate(OmegaConf.to_container(cfg,resolve=True))
                            self.assertNotIn('schedule',cfg.dynamics_shift)
        cfg=canonical('dynamics_shift=action_gain','schedule=abrupt','schedule.switch_step=20')
        self.assertEqual(cfg.schedule.switch_step,20)
        self.assertEqual(cfg.schedule.clock,'episode_native_step')

    def test_invalid(self):
        for arg in ['adaptation=dynamics_only','adaptation=joint_rl','evaluation=continual','adaptation_mode=dynamics_only']:
            with self.assertRaises(NotImplementedError): canonical(arg)
        for args in [('schedule=abrupt','schedule.switch_step=-1'),('schedule=repeated','schedule.interval=0'),
                     ('schedule=drift','schedule.duration=0'),('schedule=sequence','schedule.segments=[{duration:0,value:1}]'),
                     ('schedule=abrupt','schedule.reset_on_episode=false'),
                     ('dynamics_shift=action_gain','dynamics_shift.application=reset'),
                     ('dynamics_shift.observability.expose_shift_parameter_to_agent=true',),
                     ('schedule=fixed','dynamics_shift.schedule.type=abrupt'),
                     ('evaluation.episodes=0',),('evaluation.pairing_atol=.nan',)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate(OmegaConf.to_container(canonical(*args),resolve=True))

    def test_legacy_archive_and_conversion(self):
        source=(ROOT/'benchmarks/dynamics_shift_suite.json').read_bytes()
        self.assertEqual(source,(ROOT/'benchmarks/archive/dynamics_shift_suite_v1.json').read_bytes())
        for condition in json.loads(source)['conditions']:
            old=composed(f"configs/dynamics_shift@dynamics_shift={condition['preset']}",*condition['overrides'])
            expected=OmegaConf.to_container(old.dynamics_shift,resolve=True)
            new=normalize(old,condition['overrides'])
            actual=runtime_shift(OmegaConf.to_container(new,resolve=True))
            expected['enabled']=True
            self.assertEqual(expected,actual)
            self.assertEqual(list(new.eval_seeds),list(range(1,11)))

    def test_hash(self):
        cfg=OmegaConf.to_container(canonical(),resolve=True)
        self.assertEqual(config_hash(cfg),config_hash(dict(reversed(list(cfg.items())))))
        other=deepcopy(cfg);other['schedule']['switch_step']+=1
        self.assertNotEqual(config_hash(cfg),config_hash(other))

    def test_exposure_boundaries(self):
        env=package.ActionShiftWrapper(Native(),config(schedule='abrupt',switch_step=2));env.reset()
        rows=[env.step(np.zeros(4))[3]['shift_record'] for _ in range(2)]
        self.assertEqual(exposure(rows)['shifted_native_steps'],0)
        rows.append(env.step(np.zeros(4))[3]['shift_record'])
        self.assertEqual(exposure(rows)['shifted_native_steps'],1)
        self.assertEqual(exposure(rows)['shift_transition_count'],1)
        env.reset()
        rows=[env.step(np.zeros(4))[3]['shift_record'] for _ in range(3)]
        self.assertEqual(exposure(rows)['exposure_fraction'],1)
        self.assertEqual(exposure(rows)['shift_transition_count'],0)
        for schedule,options,n,transitions in [('repeated',dict(interval=1),5,4),('sinusoidal',dict(interval=4),5,1)]:
            env=package.ActionShiftWrapper(Native(),config(schedule=schedule,**options));env.reset()
            metrics=exposure([env.step(np.zeros(4))[3]['shift_record'] for _ in range(n)])
            self.assertEqual(metrics['shift_transition_count'],transitions)
            if schedule=='sinusoidal':
                self.assertEqual(metrics['shifted_native_steps'],2)
                self.assertEqual(metrics['nominal_crossing_count'],1)
                self.assertEqual(metrics['nominal_native_steps'],3)

    def test_v2_artifacts(self):
        from common.parser import cfg_to_dataclass
        from common.benchmark_logging import BenchmarkLog
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'fixture.pt';checkpoint.write_bytes(b'fixture')
            cfg=canonical('dynamics_shift=action_gain','schedule=fixed','eval_episodes=1')
            cfg.work_dir=tmp;cfg.checkpoint=str(checkpoint)
            cfg=cfg_to_dataclass(cfg)
            log=BenchmarkLog(cfg,[1])
            env=package.ActionShiftWrapper(Native(),runtime_shift(dict(dynamics_shift=cfg.dynamics_shift,schedule=cfg.schedule)))
            env.reset()
            log.native_steps([env.step(np.zeros(4))[3]['shift_record']], True)
            log.episode(dict(episode_id=0,seed=1,episode_length=1,episode_return=0,success=0,shift_events=0))
            result=log.finish()
            for name in ['resolved_config.yaml','metadata.json','summary.json','episodes.csv','timesteps.jsonl']:
                self.assertTrue((log.path/name).is_file())
            saved=OmegaConf.to_container(OmegaConf.load(log.path/'resolved_config.yaml'),resolve=True)
            self.assertEqual(config_hash(saved),result['resolved_config_hash'])
            self.assertEqual(result['exposure_fraction'],1)
            self.assertTrue(result['actual_shift_observed'])
            self.assertEqual(result['shift_transition_count'],0)

class RunnerTests(unittest.TestCase):
    def test_frozen_runner_and_pairing_with_fixture(self):
        import importlib.util
        import sys
        import types
        from unittest.mock import patch
        from common.parser import cfg_to_dataclass
        from common.benchmark_metrics import compare_initial
        from test_shift_state_and_logging import ResumableNative, Tensor
        from benchmark_shift import logging
        spec=importlib.util.spec_from_file_location('benchmark_entry', ROOT/'tdmpc2/evaluate_shift.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        class Environment:
            def __init__(self,cfg):
                self.env=package.ActionShiftWrapper(ResumableNative(),runtime_shift(dict(dynamics_shift=cfg.dynamics_shift,schedule=cfg.schedule)))
            def reset(self,seed):
                self.count=0
                return Tensor(self.env.reset(seed=seed))
            def step(self,action):
                self.env.begin_decision();records=[];reward=0
                for _ in range(2):
                    obs,r,done,info=self.env.step(action.numpy());records.append(info['shift_record']);reward+=r
                self.env.end_decision();self.count+=1
                return Tensor(obs), reward, self.count==3, dict(success=False,shift_native_steps=records)
            def close(self): pass
        class Agent:
            def __init__(self,cfg): pass
            def load(self,path): pass
            def act(self,obs,t0,task): return Tensor(np.full(4,.2))
        modules={'envs':types.SimpleNamespace(make_env=Environment),
                 'envs.wrappers.dynamics_shift.logging':logging,
                 'tdmpc2':types.SimpleNamespace(TDMPC2=Agent),
                 'common.seed':types.SimpleNamespace(set_seed=lambda seed:None)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,modules):
            checkpoint=Path(tmp)/'fixture.pt';checkpoint.write_bytes(b'fixture')
            cfg=canonical('dynamics_shift=action_gain','schedule=abrupt','schedule.switch_step=3',
                          'evaluation=paired','eval_episodes=2','save_video=false')
            cfg.work_dir=tmp;cfg.checkpoint=str(checkpoint);cfg=cfg_to_dataclass(cfg)
            shifted=entry.run_condition(cfg,[1,2])
            cfg.schedule['type']='nominal'
            nominal=entry.run_condition(cfg,[1,2])
            self.assertEqual(shifted[3]['shifted_native_steps'],6)
            self.assertFalse(nominal[3]['actual_shift_observed'])
            for a,b in zip(nominal[2],shifted[2]):
                result=compare_initial(a,b,1e-6)
                self.assertEqual(result['observation'],'PASS')
                self.assertEqual(result['full_state'],'UNVERIFIED')
            result=compare_initial({'observation':[1.]},{'observation':[2.]},1e-6)
            self.assertEqual(result['observation'],'FAIL')
            self.assertEqual(result['controller'],'UNVERIFIED')
            cfg.evaluation['require_shift_exposure']=True
            with self.assertRaisesRegex(RuntimeError,'Exposure constraints'):
                entry.run_condition(cfg,[1,2])


if __name__ == '__main__': unittest.main()
