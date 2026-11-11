"""Bounded real HalfCheetah physics + synthetic model integration checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/nsgym/scripts'))
import run_dynamics_only_ttt as t


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):t.dependencies()

    def test_online_snapshot_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=SimpleNamespace(config=str(t.DEFAULT),command='smoke',synthetic=True,checkpoint='',fixed_data=None)
            c=t.configuration(a);seed=c['adaptation_seeds'][0];cond=c['conditions'][0]
            hashes=[]
            for method in ('frozen','decision','episode'):
                out=root/method;t.run_one(a,c,out,cond,method,'abrupt',seed)
                done=t.read(out/'completion.json');hashes.append(done['initial_model_hash'])
                self.assertEqual(done['initial_fixed_hash'],done['final_fixed_hash'])
                self.assertEqual(done['metrics']['gradient_update_steps'],0 if method=='frozen' else 12)
                with (out/'online_steps.csv').open() as f:rr=list(t.csv.DictReader(f))
                self.assertEqual(len(rr),48)
                for r in rr:
                    expected=6*(1.5 if int(r['t'])>=8 else 1)
                    self.assertAlmostEqual(float(r['actual_parameter']),expected)
                    if method=='episode' and int(r['t'])!=23:self.assertEqual(int(r['updates_this_transition']),0)
                for snap in sorted(out.glob('snapshot-*.pt')):
                    a.snapshot=str(snap);a.output=str(root/f'eval-{method}-{snap.stem}')
                    t.evaluate_snapshot(a)
                    self.assertFalse(t.read(Path(a.output)/'completion.json')['heldout_leakage'])
            self.assertEqual(len(set(hashes)),1)

    def test_frozen_and_leakage_guards(self):
        a=SimpleNamespace(config=str(t.DEFAULT),command='smoke')
        c=t.configuration(a);t.seeded(17);agent=t.tiny_agent()
        adapter=t.make_adapter(agent,c,'decision',c['adaptation_seeds'][0])
        x=t.torch.zeros(17);u=t.torch.zeros(6)
        with self.assertRaises(RuntimeError):adapter.observe(x,u,x,1)
        with adapter.held_out():
            with self.assertRaises(RuntimeError):adapter.observe(x,u,x,1)
            with self.assertRaises(RuntimeError):adapter.update()
        with t.torch.no_grad():agent.model._reward.weight.add_(1)
        with self.assertRaises(AssertionError):adapter.assert_fixed()

if __name__=='__main__':unittest.main()
