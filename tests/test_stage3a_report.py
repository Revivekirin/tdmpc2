"""Synthetic artifacts exercise strict pairing validation and report rendering."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_stage3a_matched import analyze


class ReportTests(unittest.TestCase):
    def test_paired_report_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runs'
            for group in (0, 1):
                for mode in ('frozen', 'dynamics_only'):
                    path = root / f'set-{group}' / mode / '1'
                    path.mkdir(parents=True)
                    offset = group * 100
                    settings = dict(seed_set=group, mode=mode, native_budgets=[5],
                        adaptation_seeds=[offset+1], evaluation_seeds=[offset+2], fixed_data_seeds=[offset+3])
                    metadata = dict(config={'stage3a':settings}, active_method=mode,
                        checkpoint_sha256='shared-checkpoint', fixed_data_sha256=f'dataset-{group}',
                        fixed_data_collection_native_steps=4, fixed_data_collection_policy_decisions=2)
                    (path / 'metadata.json').write_text(json.dumps(metadata))
                    (path / 'complete.json').write_text(json.dumps({'adaptation_native_steps':5}))
                    rows = []
                    for budget in (0,5):
                        metrics = {'episode_means':{'mean':.1}, 'transition':{'mean':.1}}
                        rows.append(dict(native_budget=budget, adaptation_native_steps=budget,
                            adaptation_episodes=0, adaptation_policy_decisions=3 if budget else 0,
                            completed_policy_decisions=2 if budget else 0,
                            replay_native_steps=4 if budget else 0, in_flight_decisions=1 if budget else 0,
                            gradient_update_steps=0, reward_q_parameter_invariance=True,
                            evaluation_state_invariance=True, fixed_data_sha256=f'dataset-{group}',
                            fixed_data_prediction=dict(initial=metrics,current=metrics,paired_episode_delta={'mean':0}),
                            evaluation=dict(shifted={'success':.8, 'episode_length':10, 'successful_completion_length':5},
                                            nominal={'success':1}),
                            nominal_retention_success_delta=0, evaluation_native_steps=8,
                            evaluation_policy_decisions=4))
                    (path / 'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            output = Path(tmp) / 'report'
            self.assertEqual(len(analyze(root, output)), 8)
            self.assertTrue((output / 'adaptation_curve.png').is_file())
            self.assertTrue((output / 'interaction_budget_report.csv').is_file())
            with self.assertRaises(FileExistsError):
                analyze(root, output)
            metadata['fixed_data_sha256'] = 'different-data'
            (path / 'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                analyze(root, Path(tmp) / 'invalid-report')


if __name__ == '__main__':
    unittest.main()
