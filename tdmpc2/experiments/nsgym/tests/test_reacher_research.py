"""Bounded analysis-only tests; no model/environment imports."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_reacher_research import collapse, identity, num, unique

class ResearchTests(unittest.TestCase):
    def test_reused_reference_is_one_observation(self):
        rows=[dict(seed=1, value=2, attempt='original'),dict(seed=1,value=2,attempt='resumed')]
        self.assertEqual(len(collapse(rows,lambda x:x['seed'],['value'])),1)
    def test_conflicting_reused_reference_fails(self):
        with self.assertRaises(ValueError):
            collapse([dict(seed=1,value=2),dict(seed=1,value=3)],lambda x:x['seed'],['value'])
    def test_duplicate_episode_fails(self):
        with self.assertRaises(ValueError):unique([dict(seed=1),dict(seed=1)],lambda x:x['seed'])
    def test_seed_and_physics_are_pairing_keys(self):
        row=dict(family='M0',severity=1.5,adaptation_seed=1,seed=2,kind='shifted')
        self.assertNotEqual(identity(row),identity(dict(row,seed=3)))
        self.assertNotEqual(identity(row),identity(dict(row,kind='nominal')))
    def test_missing_not_zero(self):
        self.assertIsNone(num(''))
        self.assertEqual(num('0'),0)
        with self.assertRaises(ValueError):num('nan')

if __name__=='__main__':unittest.main()
