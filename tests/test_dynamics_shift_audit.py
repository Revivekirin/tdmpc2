"""Tests of audit decisions using synthetic logs, never reported as benchmark data."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_shift_logs import audit_run, intended, output_directory
from audit_shift_simulator import first_divergence, tree_equal


class AuditTests(unittest.TestCase):
    def fixture(self,path,mode='fixed',value=.7,corrupt=False):
        cfg=dict(parameter=dict(type='action_gain',target='arm',nominal=1.,shifted=value),
                 schedule=dict(type=mode,clock='episode_native_step',reset_on_episode=True,switch_step=100))
        metadata=dict(config=dict(benchmark_id='fixture'),shift_config=cfg)
        (path/'metadata.json').write_text(json.dumps(metadata))
        (path/'summary.json').write_text(json.dumps(dict(status='complete',checkpoint_unchanged=True,
            summary_metrics=dict(episode_return_mean=2.,episode_return_std=0.,success_mean=1.,success_std=0.,episode_length_mean=1.,episode_length_std=0.))))
        row=dict(episode_id=0,seed=1,success=1,episode_return=2.,episode_length=1,native_episode_length=2,native_start=0,native_end=2,shift_events=0)
        with (path/'episodes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row); w.writeheader();w.writerow(row)
        rows=[]
        for i in range(2):
            gain=value if mode=='fixed' else 1.
            rows.append(dict(episode_id=0,episode_step=i,native_step=i,global_step=i,decision_step=0,episode_decision_step=0,
                applied_parameter=.2 if corrupt else gain,commanded_action=[.1,.2,.3,.4],executed_action=[.1*gain,.2*gain,.3*gain,.4],reward=1.,done=i==1,decision_done=True))
        (path/'timesteps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_fixed_exposure_despite_zero_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path)
            report,_=audit_run(path)
            self.assertEqual(report['checks']['trace_contract']['status'],'PASS')
            self.assertEqual(report['episodes'][0]['exposure_fraction'],1.)
            self.assertEqual(report['episodes'][0]['shift_events'],0)
            self.assertEqual(report['episodes'][0]['applied_parameter_min'],.7)
            self.assertEqual(report['episodes'][0]['intended_parameter_first'],'0.7')
            self.assertEqual(report['checks']['paired_full_initial_state']['status'],'UNVERIFIED')

    def test_abrupt_terminated_before_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path,'abrupt')
            report,_=audit_run(path)
            self.assertEqual(report['episodes'][0]['shifted_native_steps'],0)
            self.assertIsNone(report['episodes'][0]['first_shift_native_step'])

    def test_wrong_applied_parameter_is_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path,corrupt=True)
            report,_=audit_run(path)
            self.assertEqual(report['checks']['trace_contract']['status'],'FAIL')

    def test_missing_trace_is_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path);(path/'timesteps.jsonl').write_text('')
            report,_=audit_run(path)
            self.assertEqual(report['checks']['trace_audit']['status'],'UNVERIFIED')

    def test_first_divergence_and_tolerances(self):
        reference=[dict(observation=np.zeros(2),simulator=np.zeros(3),reward=0.,done=False,success=False) for _ in range(3)]
        candidate=[dict(row) for row in reference]; candidate[1]['simulator']=np.array([0,0,2e-6])
        result=first_divergence(reference,candidate,1e-6)
        self.assertEqual((result['native_step'],result['decision_step'],result['field']),(1,0,'simulator'))
        self.assertIsNone(first_divergence(reference,candidate,3e-6))
        self.assertTrue(tree_equal({'shape':[1.,2.]},{'shape':[1.,2.+1e-7]},1e-6))

    def test_never_reuses_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotEqual(output_directory(tmp),output_directory(tmp))
