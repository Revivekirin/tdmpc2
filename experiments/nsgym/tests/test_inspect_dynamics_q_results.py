"""CPU-only checks of follow-up evidence, pairing and diagnostic completeness."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

from experiments.nsgym.analysis import diagnostics as m


class FollowupTests(unittest.TestCase):
    def test_trace_and_bad_update_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'training.jsonl'
            records=[]
            for step in range(1,3001):
                records.append(dict(global_step=step, adaptation_seed=42000001,
                    dynamics_steps=max(0,step//25-2),q_steps=max(0,step//25-2),
                    metrics={k:1.0 for k in m.METRICS} if step>=75 and step%25==0 else {}))
            p.write_text(''.join(json.dumps(r)+'\n' for r in records))
            trace=m.load_trace(p,('bthigh_damping',2,42000001))
            self.assertEqual((len(trace),trace[0]['global_step'],trace[-1]['global_step']),(118,75,3000))
            self.assertEqual(m.trace_summary(trace)['q_loss_last20_mean'],1)
            records[-1]['q_steps']=117
            p.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(AssertionError):m.load_trace(p,('bthigh_damping',2,42000001))

    def test_pairing_directions_ci_and_missing_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); comp=root/'comparison';comp.mkdir()
            (comp/'completion.json').write_text('{"status":"PASS"}')
            streams=[];paired=[];summary=[]
            for param in ('bthigh_damping','torso_mass'):
                for severity in (.5,.75,1,1.25,1.5,2):
                    for seed in range(42000001,42000004):
                        r=dict(parameter=param,severity=severity,adaptation_seed=seed,
                            frozen_return=100,ttt_return=110,dq_return=90,joint_return=80,
                            frozen_latent_mse=.3,ttt_latent_mse=.2,dq_latent_mse=.1,nominal_parameter=6)
                        streams.append(r)
                        paired.extend(dict(r,evaluation_seed=s) for s in range(43000001,43000021))
                    for metric,value in [('dq_minus_frozen',-10),('dq_minus_ttt',-20),('joint_minus_dq',-10)]:
                        summary.append(dict(parameter=param,severity=severity,metric=metric,
                            mean=value,median=value,ci_low=value-1,ci_high=value+2))
            for name,data in [('per_adaptation_seed_comparison.csv',streams),
                              ('four_method_paired_comparison.csv',paired),('condition_summary.csv',summary)]:
                m.save(comp/name,data)
            with redirect_stdout(io.StringIO()):m.inspect(comp,root/'runs',root/'output')
            done=m.read(root/'output/completion.json')
            self.assertEqual(done['status'],'PARTIAL')
            self.assertEqual(len(done['missing_logs']),36)
            condition=m.rows(root/'output/condition_comparison.csv')[0]
            self.assertEqual((float(condition['dq_minus_joint_ci_low']),float(condition['dq_minus_joint_ci_high'])),(8,11))
            row=m.rows(root/'output/stream_evidence.csv')[0]
            self.assertEqual(row['prediction_direction_vs_ttt'],'improved')
            self.assertEqual(row['control_direction_vs_ttt'],'worsened')
            self.assertEqual(row['ttt_gain_lost_below_frozen'],'True')
            self.assertTrue((root/'output/figures/prediction_control_alignment.png').is_file())
            self.assertFalse((root/'output/all_update_diagnostics.csv').exists())
            log=root/'runs/bthigh_damping-k0.5-dynamics-q-seed42000001/training.jsonl'
            log.parent.mkdir(parents=True)
            with log.open('w') as f:
                for step in range(1,3001):
                    f.write(json.dumps(dict(global_step=step,adaptation_seed=42000001,
                        dynamics_steps=max(0,step//25-2),q_steps=max(0,step//25-2),
                        metrics={k:1.0 for k in m.METRICS} if step>=75 and step%25==0 else {}))+'\n')
            with redirect_stdout(io.StringIO()):m.inspect(comp,root/'runs',root/'with-trace')
            self.assertEqual(len(m.rows(root/'with-trace/all_update_diagnostics.csv')),118)
            self.assertTrue((root/'with-trace/figures/bthigh_damping-k0.5-training.png').is_file())
            paired[0]['evaluation_seed']=43000002
            (comp/'four_method_paired_comparison.csv').unlink()
            m.save(comp/'four_method_paired_comparison.csv',paired)
            with self.assertRaises(AssertionError):m.inspect(comp,root/'runs',root/'invalid')
            self.assertFalse((root/'invalid').exists())


if __name__=='__main__':unittest.main()
