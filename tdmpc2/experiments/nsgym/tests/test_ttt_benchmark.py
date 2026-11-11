import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ttt_benchmark as b


class BenchmarkPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = b'fixed-checkpoint'
        self.profile = dict(benchmark_id='fixture', evaluator='experiments/nsgym/scripts/evaluator.py',
            checkpoint='results/checkpoint', checkpoint_sha256=hashlib.sha256(self.model).hexdigest(),
            reference='results/reference', runs='results/runs', output='results/final',
            reference_files=['frozen/episodes.csv', 'heldout-latent/latent_errors.csv'],
            run_completion_stages=['fixed-frozen', 'fixed-decision'],
            tests=['experiments/nsgym/tests/test_ttt_final.py'])
        self.path = self.root / 'profile.json'
        self.path.write_text(json.dumps(self.profile))
        self.profile, self.paths = b.load_profile(self.path, self.root)

    def touch(self, path, content=''):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_profile_must_be_relocatable(self):
        for key in ('checkpoint', 'evaluator', 'output'):
            altered = dict(self.profile, **{key: '/nfs4/old/location'})
            self.path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                b.load_profile(self.path, self.root)

    def test_preflight_checks_inputs_hash_and_completed_runs(self):
        self.touch(self.paths['evaluator'])
        self.touch(self.paths['checkpoint'] / 'model.pt', self.model.decode())
        self.touch(self.paths['reference'] / 'frozen/episodes.csv')
        self.touch(self.paths['reference'] / 'heldout-latent/latent_errors.csv')
        for method in ('frozen', 'decision'):
            self.touch(self.paths['runs'] / f'fixed-{method}/completion.json', '{"status":"PASS"}')
        b.preflight(self.profile, self.paths)
        self.touch(self.paths['checkpoint'] / 'model.pt', 'wrong')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            b.preflight(self.profile, self.paths)

    def test_primary_order_and_separated_control(self):
        self.assertEqual(b.STAGES[0], 'test')
        self.assertEqual(b.STAGES[-1], 'aggregate')
        self.assertLess(b.STAGES.index('smoke-shifted'), b.STAGES.index('full-shifted'))
        cmd = b.command('full-shifted', self.profile, self.paths)
        self.assertEqual(cmd[cmd.index('--mode') + 1], 'shifted')
        self.assertEqual(cmd[cmd.index('--subset') + 1], 'full')
        self.assertIn('--smoke-prediction', cmd)
        self.assertIn('--smoke-shifted', cmd)
        self.assertNotIn('nominal', cmd)
        nominal = b.command('nominal', self.profile, self.paths)
        self.assertEqual(nominal[nominal.index('--mode') + 1], 'nominal')
        self.assertNotIn('nominal', b.STAGES)
        retention = b.command('aggregate-retention', self.profile, self.paths)
        self.assertIn('--nominal', retention)

    def test_gate_and_partial_resume(self):
        with self.assertRaisesRegex(RuntimeError, 'requires PASS audit'):
            b.run_stage('smoke-prediction', self.profile, self.paths, True)
        self.touch(self.paths['output'] / 'audit/completion.json', '{"status":"PASS"}')
        self.touch(self.paths['output'] / 'audit/audit.json', json.dumps(dict(status='PASS',
            checkpoint=str(self.paths['checkpoint']), runs=str(self.paths['runs']),
            reference=str(self.paths['reference']), checkpoint_sha256=self.profile['checkpoint_sha256'])))
        self.touch(self.paths['output'] / 'smoke-prediction/manifest.json', json.dumps(dict(
            mode='prediction', subset='smoke', checkpoint_sha256=self.profile['checkpoint_sha256'],
            audit_sha256=b.file_hash(self.paths['output'] / 'audit/audit.json'))))
        with patch.object(b.subprocess, 'run') as run:
            def finish(cmd, **kwargs):
                self.assertIn('--resume', cmd)
                self.touch(self.paths['output'] / 'smoke-prediction/completion.json', '{"status":"PASS"}')
            run.side_effect = finish
            b.run_stage('smoke-prediction', self.profile, self.paths, True)
        with patch.object(b.subprocess, 'run') as run:
            b.run_stage('smoke-prediction', self.profile, self.paths, True)
            run.assert_not_called()

    def test_existing_pass_with_wrong_provenance_is_rejected(self):
        self.touch(self.paths['output'] / 'audit/completion.json', '{"status":"PASS"}')
        self.touch(self.paths['output'] / 'audit/audit.json', json.dumps(dict(status='PASS',
            checkpoint='/nfs4/obsolete', runs=str(self.paths['runs']),
            reference=str(self.paths['reference']), checkpoint_sha256=self.profile['checkpoint_sha256'])))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            b.verify_pass('audit', self.profile, self.paths)


if __name__ == '__main__':
    unittest.main()
