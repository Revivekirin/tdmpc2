"""Synthetic CSV/figure smoke. Fixture numbers are not experimental results."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_stage3b import analyze, normalized_auc


def fixture(root):
    suite=json.loads((Path(__file__).resolve().parents[1]/'benchmarks/stage3b_suite.json').read_text())
    suite['run_id']='SYNTHETIC-plot-smoke-NOT-experiment'
    for i,shift in enumerate(suite['shifts']):
        for method in suite['methods']:
            for group in suite['seed_sets']:
                path=root/shift['type']/f'set-{group}'/method/'1'
                path.mkdir(parents=True)
                s=dict(mode=method,seed_set=group,native_budgets=suite['budgets'][1:],
                    adaptation_seeds=[1000+group*10000],evaluation_seeds=[4000+group*10000,4001+group*10000],
                    fixed_data_seeds=[5000+group*10000,5001+group*10000])
                meta=dict(config=dict(stage3a=s,stage3b={'run_id':suite['run_id']},task='SYNTHETIC',
                    dynamics_shift={'parameter':{'type':shift['type'],'shifted':shift['magnitude']}},schedule={'type':'fixed'}),
                    checkpoint_sha256='SYNTHETIC-NO-CHECKPOINT',active_method=method,
                    fixed_data_sha256=f"SYNTHETIC-{shift['type']}-{group}",
                    fixed_data_collection_native_steps=20,fixed_data_collection_policy_decisions=10)
                (path/'metadata.json').write_text(json.dumps(meta))
                (path/'complete.json').write_text(json.dumps({'adaptation_native_steps':12800}))
                records=[]
                for j,budget in enumerate(suite['budgets']):
                    gain=(j*.025 if method=='dynamics_only' else 0)
                    success=.5+i*.1+group*.02+gain
                    evaluation={}
                    for condition in ('shifted','nominal'):
                        episodes=[dict(seed=seed,success=success if condition=='shifted' else 1.,
                            episode_return=20.,episode_length=10.) for seed in s['evaluation_seeds']]
                        evaluation[condition]=dict(episodes=episodes)
                    initial=[.001*(i+1)*(1+group*.1)]*2
                    current=[v*(1-gain) for v in initial]
                    records.append(dict(native_budget=budget,adaptation_native_steps=budget,
                        adaptation_policy_decisions=budget//2,adaptation_episodes=budget//40,
                        gradient_update_steps=0 if method=='frozen' else budget//40,
                        replay_size=budget//2,reward_q_parameter_invariance=True,evaluation_state_invariance=True,
                        fixed_data_sha256=meta['fixed_data_sha256'],evaluation=evaluation,
                        fixed_data_prediction=dict(initial={'per_episode_means':initial},current={'per_episode_means':current}),
                        parameter_drift=0. if method=='frozen' else gain))
                (path/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return suite


class ReportTest(unittest.TestCase):
    def test_csv_figures_auc_missing_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            suite=fixture(tmp/'runs')
            result=analyze(tmp/'runs',tmp/'report',suite)
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(len(list((tmp/'report'/'figures').glob('*'))),10)
            with (tmp/'report'/'tables'/'stage3b_aggregated_results.csv').open() as f:
                rows=list(csv.DictReader(f))
            r=next(r for r in rows if r['adaptation']=='frozen' and r['shift_type']=='action_delay'
                and r['metric']=='shifted_success' and r['budget_native_steps']=='0')
            self.assertAlmostEqual(float(r['mean']),.52)
            self.assertAlmostEqual(float(r['std']),.02)
            self.assertEqual(int(r['n_seed_sets']),3)
            with (tmp/'report'/'tables'/'stage3b_auc_seed_results.csv').open() as f:
                rows=list(csv.DictReader(f))
            r=next(r for r in rows if r['adaptation']=='dynamics_only' and r['shift_type']=='action_delay' and r['seed_set']=='0')
            expected=normalized_auc(suite['budgets'],[.5+j*.025 for j in range(6)])
            self.assertAlmostEqual(float(r['auc']),expected)
            self.assertAlmostEqual(float(r['auc_difference_vs_frozen']),expected-.5)
            path=tmp/'runs'/'action_delay'/'set-0'/'dynamics_only'/'1'/'evaluations.jsonl'
            records=path.read_text().splitlines(); records.pop(2)
            path.write_text('\n'.join(records)+'\n')
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                analyze(tmp/'runs',tmp/'rejected',suite)
            self.assertFalse((tmp/'rejected').exists())
            result=analyze(tmp/'runs',tmp/'partial',suite,allow_partial=True)
            self.assertEqual(result['status'],'PARTIAL')
            with (tmp/'partial'/'tables'/'stage3b_auc_seed_results.csv').open() as f:
                rows=list(csv.DictReader(f))
            r=next(r for r in rows if r['adaptation']=='dynamics_only' and r['shift_type']=='action_delay' and r['seed_set']=='0')
            self.assertEqual(r['auc'],'')
            meta=path.parent/'metadata.json'
            content=json.loads(meta.read_text()); content['fixed_data_sha256']='bad'
            meta.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError,'[Ff]ixed'):
                analyze(tmp/'runs',tmp/'bad',suite,allow_partial=True)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--render-smoke':
        output=Path(sys.argv[2]); output.mkdir(parents=True,exist_ok=False)
        suite=fixture(output/'runs')
        (output/'suite.json').write_text(json.dumps(suite,indent=2))
        analyze(output/'runs',output/'report',suite)
    else:
        unittest.main()
