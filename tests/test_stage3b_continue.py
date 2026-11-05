"""Continuation selects completed recovery once and launches only missing jobs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_stage3b import commands, remaining_commands, archive_incomplete
from stage3b_results import selected_results


class ContinueTests(unittest.TestCase):
    def test_four_completed_twenty_remaining_and_provenance_guard(self):
        suite = json.loads((Path(__file__).resolve().parents[1] / 'benchmarks/stage3b_suite.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output, recovery = tmp/'stage3b-generalization', tmp/'recovered'/'runs'
            checkpoint = tmp/'pretrained.pt'
            checkpoint.write_bytes(b'fixture')
            for shift in suite['shifts']:
                for group in suite['seed_sets']:
                    data = output/'data'/shift['type']/f'set-{group}.pt'
                    data.parent.mkdir(parents=True, exist_ok=True)
                    data.write_bytes(b'fixed fixture')
            def make(group, method, complete, root):
                path = root/'action_delay'/f'set-{group}'/method/'1'
                path.mkdir(parents=True)
                cfg = dict(task=suite['task'], model_size=5, episodic=False, seed=1,
                    schedule={'type':'fixed'}, dynamics_shift={'parameter':{'type':'action_delay','shifted':1}},
                    stage3a=dict(seed_set=group, native_budgets=suite['budgets'][1:], output=str(root)),
                    stage3b=dict(run_id=output.name, resume='original.pt' if root==recovery else None))
                meta = dict(config=cfg, active_method=method, checkpoint_sha256=hashlib.sha256(b'fixture').hexdigest(),fixed_data_sha256=f'fixed-{group}')
                (path/'metadata.json').write_text(json.dumps(meta))
                budgets=suite['budgets'] if complete else suite['budgets'][:-1]
                (path/'evaluations.jsonl').write_text(''.join(json.dumps({'native_budget':b})+'\n' for b in budgets))
                if complete:
                    (path/'complete.json').write_text(json.dumps({'adaptation_native_steps':12800}))
                return path
            for group, method in [(0,'frozen'),(0,'dynamics_only'),(1,'frozen')]:
                make(group,method,True,output/'runs')
            original=make(1,'dynamics_only',False,output/'runs')
            restored=make(1,'dynamics_only',True,recovery)
            before={p:p.read_bytes() for root in (output,recovery) for p in root.rglob('*') if p.is_file()}
            planned=list(commands(suite,str(checkpoint),output,'adapt'))
            remaining, skipped, archives=remaining_commands(planned,suite,output,recovery,checkpoint)
            self.assertEqual(len(remaining),20)
            self.assertEqual(len(skipped),4)
            self.assertIn(str(restored),skipped)
            self.assertEqual(archives, [])  # Completed recovery protects its incomplete original.
            self.assertNotIn(str(original),skipped)
            self.assertIn('stage3a.seed_set=2',remaining[0])
            self.assertIn('stage3a.mode=frozen',remaining[0])
            self.assertEqual(len(selected_results(output/'runs',recovery)),4)
            self.assertTrue(all(p.read_bytes()==value for p,value in before.items()))
            restart, done, archives=remaining_commands(planned,suite,output,None,checkpoint)
            self.assertEqual((len(restart),len(done)),(21,3))
            self.assertEqual(archives,[original])
            self.assertTrue(all('stage3b.resume=' not in arg for cmd in restart for arg in cmd))
            self.assertTrue(all(p.read_bytes()==value for p,value in before.items()))
            meta=json.loads((restored/'metadata.json').read_text())
            meta['fixed_data_sha256']='different'
            (restored/'metadata.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                remaining_commands(planned,suite,output,recovery,checkpoint)

    def test_partial_metadata_and_recovery_archive_then_restart(self):
        suite = dict(task='pick-cube',model_size=5,episodic=False,methods=['frozen','dynamics_only'],
                     seed_sets=[0],budgets=[0,800,12800],shifts=[dict(type='action_delay',magnitude=1)])
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            output,recovery=tmp/'experiment',tmp/'recovered'/'runs'
            checkpoint=tmp/'pretrained.pt'; checkpoint.write_bytes(b'pretrained fixture')
            data=output/'data'/'action_delay'/'set-0.pt'
            data.parent.mkdir(parents=True); data.write_bytes(b'fixed fixture')
            original=output/'runs'/'action_delay'/'set-0'/'frozen'/'1'
            restored=recovery/'action_delay'/'set-0'/'frozen'/'1'
            for path in (original,restored):
                path.mkdir(parents=True)
                (path/'metadata.json').write_text('{')  # Interrupted write, no completion marker.
                (path/'resume-budget-800.pt').write_bytes(b'incomplete checkpoint fixture')
            planned=list(commands(suite,str(checkpoint),output,'adapt'))
            remaining, skipped, archives=remaining_commands(planned,suite,output,recovery,checkpoint)
            self.assertEqual(remaining,planned)
            self.assertEqual(skipped,[])
            self.assertEqual(set(archives),{original,restored})
            self.assertTrue(original.exists() and restored.exists())  # Dry-run preserved both.
            archive_incomplete(archives,'attempt-1')
            for path in (original,restored):
                archived=path.parents[3].parent/'archived-incomplete'/'attempt-1'/path.relative_to(path.parents[3])
                self.assertEqual((archived/'resume-budget-800.pt').read_bytes(),b'incomplete checkpoint fixture')
                self.assertFalse(path.exists())
            self.assertEqual(checkpoint.read_bytes(),b'pretrained fixture')
            self.assertEqual(data.read_bytes(),b'fixed fixture')
            self.assertEqual(selected_results(output/'runs',recovery),{})
            remaining, skipped, archives=remaining_commands(planned,suite,output,recovery,checkpoint)
            self.assertEqual((len(remaining),skipped,archives),(2,[],[]))

    def test_refuses_archiving_a_run_that_completed_after_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs'/'action_delay'/'set-0'/'frozen'/'1'
            path.mkdir(parents=True)
            (path/'complete.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'completed after planning'):
                archive_incomplete([path],'attempt')
            self.assertTrue(path.exists())


if __name__=='__main__': unittest.main()
