import dataclasses
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from test_dynamics_shift import ROOT

spec = importlib.util.spec_from_file_location('shift_logging', ROOT / 'tdmpc2/common/shift_evaluation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@dataclasses.dataclass
class Config:
    work_dir: str
    checkpoint: str
    shift_mode: str = 'nominal'
    shift_clock: str = 'global'
    shift_gain: float = .7


class LogTests(unittest.TestCase):
    def test_artifacts_and_unique_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'fixture.pt'
            checkpoint.write_bytes(b'test fixture, not a real model')
            cfg = Config(tmp, str(checkpoint))
            log = module.EvaluationLog(cfg)
            log.step('fixture', 0, 0, None, 3., True, {'shift_native_steps': [
                dict(shift_gain=1., shift_global_step=0, reward=1., done=False),
                dict(shift_gain=.7, shift_global_step=1, reward=2., done=False)]})
            log.episode(dict(task='fixture', episode_return=3., success=1., episode_length=1))
            summary = log.finish()
            self.assertEqual(summary['fixture']['episode_return_mean'], 3.)
            self.assertEqual(len((log.path / 'timesteps.jsonl').read_text().splitlines()), 2)
            self.assertTrue(json.loads((log.path / 'summary.json').read_text())['checkpoint_unchanged'])
            self.assertTrue((log.path / 'episodes.csv').exists())
            other = module.EvaluationLog(cfg)
            other.finish()
            self.assertNotEqual(log.path, other.path)
