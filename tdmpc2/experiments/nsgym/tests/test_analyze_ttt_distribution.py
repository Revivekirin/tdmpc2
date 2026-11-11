"""CSV-only analysis checks; no model, GPU, or simulator imports."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_ttt_distribution as analysis


class DistributionAnalysis(unittest.TestCase):
    def test_paired_bank_and_snapshot_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_path = root / 'audit.json'
            audit_path.write_text(json.dumps({'status': 'PASS'}))
            audit = {'checkpoint_sha256': 'checkpoint', '_path': str(audit_path),
                     'configuration': {'evaluation_seeds': [1, 2], 'horizon': 1000}}
            identity = ('torso_mass', 2.0, 9)
            name = 'torso_mass-k2-decision-seed9'
            child = root / 'prediction' / name
            child.mkdir(parents=True)
            (root / 'prediction' / 'manifest.json').write_text(json.dumps({
                'mode': 'prediction', 'checkpoint_sha256': 'checkpoint',
                'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest()}))
            (child / 'completion.json').write_text('{"status":"PASS"}')
            (child / 'manifest.json').write_text(json.dumps({
                'identity': {'parameter': 'torso_mass', 'severity': 2, 'adaptation_seed': 9},
                'mode': 'prediction', 'checkpoint_sha256': 'checkpoint',
                'snapshot_sha256': 'snapshot'}))
            data = []
            source = {}
            frozen = {}
            for seed in (1, 2):
                file = f'bank-{seed}.npz'
                source[file] = f'hash-{seed}'
                frozen['torso_mass', 2.0, seed] = {'data_file': file}
                data.append(dict(parameter='torso_mass', severity=2,
                    adaptation_seed=9, evaluation_seed=seed, bank_file=file,
                    bank_sha256=source[file], samples=1000, frozen_latent_mse=1,
                    decision_latent_mse=0.5 if seed == 1 else 1.5,
                    delta_latent_mse=-0.5 if seed == 1 else 0.5))
            with (child / 'prediction.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, data[0]); writer.writeheader(); writer.writerows(data)
            result, status, _ = analysis.stage(root / 'prediction', 'prediction',
                {identity: 'snapshot'}, audit, source, frozen)
            self.assertEqual(status[identity], 'COMPLETE')
            streams, _, per_seed = analysis.prediction_tables(result, audit['configuration'])
            self.assertEqual(streams[0]['fraction_eval_seed_means_improved'], 0.5)
            self.assertEqual(streams[0]['mse_improvement'], 0)
            self.assertEqual(len(per_seed), 2)
            data[0]['bank_sha256'] = 'wrong'
            with (child / 'prediction.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, data[0]); writer.writeheader(); writer.writerows(data)
            _, status, _ = analysis.stage(root / 'prediction', 'prediction',
                {identity: 'snapshot'}, audit, source, frozen)
            self.assertEqual(status[identity], 'INCOMPATIBLE')

    def test_quadrants_keep_neutral_and_negative_cases(self):
        prediction = [dict(parameter='torso_mass', severity=2,
            adaptation_seed=9, mse_improvement=0.4, frozen_mean_mse=1,
            ttt_mean_mse=0.6)]
        control = [dict(parameter='torso_mass', severity=2,
            adaptation_seed=9, delta_return=-20, frozen_mean_return=100,
            ttt_mean_return=80, velocity_difference=-0.2,
            control_cost_difference=2)]
        optimization = [dict(parameter='torso_mass', severity=2,
            adaptation_seed=9, gradient_update_steps=118,
            adaptation_transitions=3000)]
        self.assertEqual(analysis.make_pairs(prediction, control, optimization)[0]['quadrant'], 'P+/C-')
        control[0]['delta_return'] = 0
        self.assertEqual(analysis.make_pairs(prediction, control, optimization)[0]['quadrant'], 'P+/C0')


if __name__ == '__main__':
    unittest.main()
