"""Bounded real-physics checks and explicit CPU fixture runner tests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import numpy as np
import torch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import reacher_pipeline as r

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.c=r.config(r.DEFAULT)
    def tearDown(self):self.tmp.cleanup()
    def test_real_fixed_reset_and_nominal_equivalence(self):
        from envs.nsgym.reacher import Reacher
        for cond in r.conditions(self.c):
            e=r.env(cond);base=Reacher();rng=np.random.default_rng(7)
            try:
                for seed in [7,8]:
                    o,_=e.reset(seed=seed);b,_=base.reset(seed=seed)
                    np.testing.assert_array_equal(o,b)
                    for t in range(4):
                        ac=rng.uniform(-1,1,2);n=e.step(ac)[0];bn=base.step(ac)[0]
                        self.assertAlmostEqual(e.readback(),e.nominal_value*cond['severity'])
                        if cond['severity']==1:np.testing.assert_allclose(n,bn,rtol=0,atol=1e-12)
            finally:e.close();base.close()
    def test_physical_counterfactuals_all_grid(self):
        c=dict(self.c,calibration_seeds=self.c['calibration_seeds'][:1]);rows=r.probe(c,self.root)
        self.assertEqual(len(rows),24*15)
        self.assertTrue(all(x['moving'] for x in rows))
        self.assertTrue(all(x['delta_tip_m']==0 for x in rows if x['severity']==1))
    def test_adapter_snapshot_evaluation_and_isolation(self):
        c=dict(self.c,evaluation_seeds=self.c['evaluation_seeds'][:2],adaptation_seeds=self.c['adaptation_seeds'][:1]);cond=r.conditions(c)[0];selection={'model_sha256':'fixture-only','checkpoint':'fixture'}
        evaluation_dirs=[]
        with patch.object(r,'load',side_effect=lambda *args:(r.tiny(),{})):
            for method in c['methods']:
                out=self.root/method
                result=r.adapt_one(c,selection,out,cond,method,c['adaptation_seeds'][0],[0,100],synthetic=False)
                self.assertEqual(result['metrics']['adaptation_transitions'],100)
                source=self.root/f'{method}-frozen';source.mkdir();rr=[]
                agent=r.tiny()
                for seed in c['evaluation_seeds']:rr.append(r.episode(agent,cond,seed,source)[0])
                r.table(source/'reacher_frozen_severity.csv',rr)
                for kind in ('predict','control'):
                    for budget in (0,100):
                        target=self.root/f'{method}-{kind}-{budget}';target.mkdir();evaluation_dirs.append(target)
                        r.evaluate_snapshot(c,selection,out/f'snapshot-{budget:07d}.pt',target,source,kind)
                        rows=r.csvread(target/f'{kind}.csv')
                        self.assertEqual(len(rows),2 if kind=='predict' else 4)
                        self.assertEqual(r.read(target/'completion.json')['status'],'PASS')
        from analyze_reacher_pipeline import aggregate
        report=self.root/'report';report.mkdir();aggregate(evaluation_dirs,report,c)
        outcomes=r.csvread(report/'adaptation_full_grid_outcomes.csv')
        self.assertEqual(len(outcomes),24)
        self.assertEqual(outcomes[0]['status'],'COMPLETE')
        self.assertTrue((report/'reacher_prediction_vs_control.png').is_file())
    def test_seed_split_and_gate(self):
        c=dict(self.c,adaptation_seeds=self.c['evaluation_seeds'][:1]);r.write(self.root/'bad.json',c)
        with self.assertRaises(AssertionError):r.config(self.root/'bad.json')
        r.write(self.root/'completion.json',{'status':'PASS'});r.write(self.root/'manifest.json',dict(stage='inventory',config_fingerprint=r.fingerprint(self.c)))
        r.gate(self.root,'inventory',self.c)
        with self.assertRaises(AssertionError):r.gate(self.root,'probe',self.c)
    def test_blocked_discovery_does_not_consume_nominal_output(self):
        discovery=self.root/'discovery';discovery.mkdir()
        result=r.discover(self.root,self.c,discovery)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertIn('No valid nominal Reacher checkpoint',result['reason'])
        nominal=self.root/'nominal'
        process=subprocess.run([sys.executable,str(SCRIPTS/'run_reacher_pipeline.py'),'nominal',
            '--discovery',str(discovery/'discovery.json'),'--output',str(nominal)],capture_output=True,text=True)
        self.assertNotEqual(process.returncode,0)
        self.assertIn('Discovery is BLOCKED',process.stderr)
        self.assertFalse(nominal.exists())
    def test_frozen_alias_metadata_and_plots(self):
        c=dict(self.c,evaluation_seeds=self.c['evaluation_seeds'][:2],calibration_seeds=self.c['calibration_seeds'][:1])
        frozen=self.root/'frozen';frozen.mkdir();probe=self.root/'probe';probe.mkdir()
        with patch.object(r,'load',side_effect=lambda *args:(r.tiny(),{})):
            rows=r.frozen(c,{'checkpoint':'fixture'},frozen)
            r.probe(c,probe,r.tiny())
        self.assertEqual(len(rows),48)
        mass=next(x for x in rows if x['family']=='M0' and x['severity']==1)
        self.assertAlmostEqual(mass['actual_parameter'],.03560471674068433)
        self.assertEqual(mass['data_file'],next(x['data_file'] for x in rows if x['family']=='D0' and x['severity']==1 and x['seed']==mass['seed']))
        from analyze_reacher_pipeline import frozen_plots,physical_plots
        physical_plots(probe,probe,c);frozen_plots(frozen,frozen,c,probe)
        self.assertTrue((frozen/'reacher_physical_latent_control.png').is_file())
    def test_streaming_training_update_audit(self):
        run=self.root/'run';run.mkdir();cp=run/'checkpoint-000100002';cp.mkdir()
        r.write(cp/'metadata.json',dict(config={'seed_steps':2}))
        with (run/'training.jsonl').open('w') as f:
            for step in [1,2,3,50001,100002]:f.write(json.dumps(dict(environment_transitions=step,gradient_updates=0 if step<=2 else step-1,consistency_loss=1/step))+'\n')
        out=self.root/'audit';out.mkdir();r.training_curves(run,out)
        audit=r.read(out/'training_audit.json');self.assertEqual(audit['rows'],5);self.assertEqual(audit['update_count_mismatches'],0)
        self.assertEqual(len(r.csvread(out/'training_bins.csv')),3)
    def test_seed_unit_statistics(self):
        from analyze_reacher_pipeline import summary
        rows=[dict(family='D0',severity=1,seed=1,error=0)]*100+[dict(family='D0',severity=1,seed=2,error=10)]
        stat=summary(rows,'error')[0];self.assertEqual(stat['n'],2);self.assertEqual(stat['mean'],5)

if __name__=='__main__':unittest.main()
