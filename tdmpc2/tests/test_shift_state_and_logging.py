import dataclasses
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
from benchmark_imports import package
from benchmark_shift.logging import capture_environment, restore_environment, encode_state, decode_state
from test_shift_benchmark import config
from test_dynamics_shift import Native
from common.benchmark_logging import BenchmarkLog
from common.shift_protocol import Transition


class Controller:
    def __init__(self): self.state={'target':np.array([.1,.2,.3])}
    def get_state(self): return self.state
    def set_state(self,state): self.state=state


class ResumableNative(Native):
    def __init__(self):
        super().__init__()
        self.agent=types.SimpleNamespace(controller=Controller())
        self._main_rng=np.random.RandomState(1)
        self._episode_rng=np.random.RandomState(2)
        self._elapsed_steps=0
    def get_state(self): return self.state.copy()
    def set_state(self,state): self.state=state.copy()
    def step(self,action):
        self._elapsed_steps+=1
        return super().step(action)


class Tensor:
    def __init__(self,value): self.value=np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


@dataclasses.dataclass
class Config:
    work_dir: str
    checkpoint: str
    dynamics_shift: dict
    benchmark_id: str='fixture'
    benchmark_split: str='probe'
    adaptation_mode: str='frozen'
    benchmark_save_trajectory: bool=True


class StateLogTests(unittest.TestCase):
    def test_full_environment_resume(self):
        cfg=config('action_delay','random_piecewise',0,2,interval=3,seed=12,values=[0,1,2])
        a=package.ActionShiftWrapper(ResumableNative(),cfg); a.reset()
        for _ in range(4): a.step(np.ones(4)*.2)
        saved=decode_state(json.loads(json.dumps(encode_state(capture_environment(a)))))
        b=package.ActionShiftWrapper(ResumableNative(),cfg); b.reset()
        restore_environment(b,saved)
        self.assertEqual(a.env._elapsed_steps,b.env._elapsed_steps)
        np.testing.assert_array_equal(a.env._episode_rng.rand(10),b.env._episode_rng.rand(10))
        for i in range(10):
            action=np.full(4,i/20)
            x,y=a.step(action),b.step(action)
            np.testing.assert_array_equal(x[0],y[0])
            self.assertEqual(x[1:3],y[1:3])
            np.testing.assert_array_equal(x[3]['shift_record']['executed_action'],y[3]['shift_record']['executed_action'])
        bad=package.ActionShiftWrapper(Native(),cfg); bad.reset()
        with self.assertRaises(NotImplementedError): capture_environment(bad)

    def test_benchmark_artifacts_flags_and_failure(self):
        for enabled in [True,False]:
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint=Path(tmp)/'fixture.pt'; checkpoint.write_bytes(b'synthetic checkpoint fixture')
                shift=config(); shift['logging']={key:enabled for key in ['save_step_trace','save_shift_events','save_episode_summary']}
                cfg=Config(tmp,str(checkpoint),shift)
                log=BenchmarkLog(cfg,[12])
                record=dict(native_step=0,commanded_action=np.ones(4),executed_action=np.ones(4)*.7,applied_parameter=.7)
                log.native_steps([record],True)
                log.events([dict(native_step=0,shift_segment_id=0)])
                log.transition(Transition(Tensor([0]),Tensor(np.ones(4)),1.,Tensor([1]),True,{}))
                log.episode(dict(episode_id=0,seed=12,episode_return=1.,success=0.,episode_length=1,shift_events=0))
                summary=log.finish(error='intentional fixture failure')
                self.assertEqual(summary['status'],'failed')
                self.assertTrue(summary['checkpoint_unchanged'])
                self.assertEqual(summary['episode_seeds'],[12])
                self.assertEqual(bool((log.path/'timesteps.jsonl').read_text()),enabled)
                self.assertEqual((log.path/'episodes.csv').exists(),enabled)
                with np.load(log.path/'episode-0000.npz',allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['commanded_actions'][0],np.ones(4))
                    np.testing.assert_allclose(data['diagnostic_native_executed_actions'][0],np.ones(4)*.7)
                self.assertEqual(json.loads((log.path/'summary.json').read_text())['summary_metrics']['episode_return_mean'],1.)
