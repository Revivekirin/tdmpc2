"""Bounded CPU protocol/config smoke; does not claim CUDA simulator validation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_stage3a import Model, settings, data
from common.decision_ttt import DecisionAdapter
from common.stage3a import EpisodeAdapter, equal_state, validate
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_stage3b import commands
from analyze_stage3b import analyze


class DecisionTests(unittest.TestCase):
    def test_warmup_freeze_evaluation_terminal_resume(self):
        torch.set_num_threads(1)
        s = settings('decision_ttt')
        s.update(update_granularity='decision', updates_per_decision=1, update_every_decisions=1)
        model = Model()
        initial = deepcopy(model.state_dict())
        adapter = DecisionAdapter(model, s, 20, 20)
        adapter.begin_episode()
        for transition in data(15):
            adapter.observe(*transition, 2)
        self.assertEqual(adapter.gradient_steps, 0)
        adapter.end_episode()
        adapter.begin_episode()
        adapter.observe(*data(1)[0], 2)
        self.assertEqual(adapter.gradient_steps, 1)
        self.assertEqual(adapter.scheduler['first_actual_update_decision'], 16)
        self.assertFalse(equal_state(initial['_dynamics.0.weight'], model.state_dict()['_dynamics.0.weight']))
        adapter.assert_fixed()
        with adapter.held_out(allow_in_episode=True):
            adapter.loss(data())
            with self.assertRaises(RuntimeError): adapter.observe(*data(1)[0], 2)
            with self.assertRaises(RuntimeError): adapter.update()
        adapter.update()  # Same decision cannot update twice.
        self.assertEqual(adapter.gradient_steps, 1)
        adapter.end_episode()
        self.assertEqual(adapter.gradient_steps, 1)
        other = Model(); other.load_state_dict(initial)
        resumed = DecisionAdapter(other, s, 20, 20)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resume.pt'
            torch.save(adapter.state_dict(), path)
            resumed.load_state_dict(torch.load(path, weights_only=False))
        self.assertTrue(equal_state(adapter.state_dict(), resumed.state_dict()))
        with self.assertRaises(ValueError):
            resumed.load_state_dict(EpisodeAdapter(Model(),settings(),20,20).state_dict())
        self.assertEqual(adapter.native_steps, 2*adapter.transitions)

    def test_config_matrix(self):
        root=Path(__file__).resolve().parents[1]
        suite=json.loads((root/'benchmarks/stage3c_suite.json').read_text())
        planned=list(commands(suite,'nominal.pt',Path('/tmp/stage3c'),'adapt'))
        self.assertEqual(len(planned),8)
        from common.benchmark_config import cli_aliases, normalize
        with initialize_config_dir(config_dir=str(root/'tdmpc2'),version_base=None):
            for command in planned:
                overrides=cli_aliases(command[4:])
                cfg=normalize(compose(config_name='stage3c',overrides=overrides), overrides)
                s=OmegaConf.to_container(cfg.stage3a,resolve=True)
                s['adaptation_seeds']=list(range(1000,1000+s['episodes']))
                validate(s)
                self.assertEqual((s['warmup'],s['batch_size']),(16,16))

    def test_three_method_severity_report(self):
        from test_stage3b_report import fixture
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            original=tmp/'original'
            suite=fixture(original)
            suite.update(methods=['frozen','dynamics_only','decision_ttt'],seed_sets=[0],
                         shifts=[dict(type='action_delay',magnitude=1),dict(type='action_delay',magnitude=2)])
            for value in (1,2):
                for method in suite['methods']:
                    source=original/'action_delay'/'set-0'/('dynamics_only' if method=='decision_ttt' else method)/'1'
                    dest=tmp/'runs'/str(value)/method
                    shutil.copytree(source,dest)
                    meta=json.loads((dest/'metadata.json').read_text())
                    meta['active_method']=method
                    meta['config']['stage3a']['mode']=method
                    meta['config']['dynamics_shift']['parameter']['shifted']=value
                    if method=='decision_ttt':
                        meta['config']['stage3a'].update(update_granularity='decision',updates_per_decision=1,update_every_decisions=1)
                    (dest/'metadata.json').write_text(json.dumps(meta))
            result=analyze(tmp/'runs',tmp/'report',suite)
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(result['n_runs'],6)
            self.assertEqual(len(list((tmp/'report'/'figures').glob('*'))),18)
            # Duplicate versions are rejected; no silent selection of an outcome.
            shutil.copytree(tmp/'runs'/'1'/'decision_ttt',tmp/'runs'/'duplicate')
            with self.assertRaisesRegex(ValueError,'Duplicate run'):
                analyze(tmp/'runs',tmp/'bad',suite)


if __name__ == '__main__': unittest.main()
