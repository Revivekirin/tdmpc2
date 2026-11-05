"""Missing-data preparation preserves completed-run provenance and existing files."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_stage3b_data import plan_data, execute_plan, install_candidate, sha256


class DataTests(unittest.TestCase):
    def test_collect_only_missing_and_bind_completed_hash(self):
        suite=dict(task='pick-cube',model_size=5,episodic=False,methods=['frozen','dynamics_only'],
                   seed_sets=[0,1],budgets=[0,800],shifts=[dict(type='action_delay',magnitude=1)])
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'experiment'; output.mkdir()
            cp=Path(tmp)/'model.pt'; cp.write_bytes(b'checkpoint')
            existing=output/'data/action_delay/set-0.pt'
            existing.parent.mkdir(parents=True); existing.write_bytes(b'original data')
            expected=sha256(existing)
            run=output/'runs/action_delay/set-1/dynamics_only/1'; run.mkdir(parents=True)
            cfg=dict(task='pick-cube',model_size=5,episodic=False,schedule={'type':'fixed'},
                     dynamics_shift={'parameter':{'type':'action_delay','shifted':1}},
                     stage3a={'seed_set':1,'native_budgets':[800]})
            (run/'metadata.json').write_text(json.dumps(dict(config=cfg,active_method='dynamics_only',
                checkpoint_sha256=sha256(cp),fixed_data_sha256=expected)))
            (run/'complete.json').write_text(json.dumps({'adaptation_native_steps':800}))
            (run/'evaluations.jsonl').write_text('{"native_budget":0}\n{"native_budget":800}\n')
            plan=plan_data(suite,cp,output)
            self.assertEqual([e['status'] for e in plan],['EXISTS','MISSING'])
            self.assertEqual(plan[1]['expected_sha256'],expected)
            self.assertFalse((output/'configs').exists())
            def collect(command,check):
                target=Path(next(a.split('=',1)[1] for a in command if a.startswith('stage3a.fixed_data_path=')))
                target.parent.mkdir(parents=True); target.write_bytes(b'original data')
            with patch('prepare_stage3b_data.subprocess.run',side_effect=collect) as launch:
                execute_plan(plan,output)
            self.assertEqual(launch.call_count,1)
            self.assertEqual(existing.read_bytes(),b'original data')
            self.assertEqual(sha256(output/'data/action_delay/set-1.pt'),expected)
            self.assertEqual([e['status'] for e in plan_data(suite,cp,output)],['EXISTS','EXISTS'])
            (output/'data/action_delay/set-1.pt').write_bytes(b'wrong data')
            with self.assertRaisesRegex(ValueError,'differs from completed'):
                plan_data(suite,cp,output)

    def test_mismatch_and_target_collision_preserve_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); candidate=tmp/'candidate.pt'; candidate.write_bytes(b'new')
            target=tmp/'data/set-1.pt'
            result=install_candidate(candidate,target,'wrong-hash')
            self.assertEqual(result['status'],'HASH_MISMATCH')
            self.assertFalse(target.exists()); self.assertTrue(candidate.exists())
            target.parent.mkdir(); target.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                install_candidate(candidate,target,None)
            self.assertEqual(target.read_bytes(),b'original')
            self.assertEqual(candidate.read_bytes(),b'new')


if __name__=='__main__': unittest.main()
