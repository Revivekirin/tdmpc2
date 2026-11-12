"""Analysis contract tests. No simulator/model/optimizer construction."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import os
import zipfile
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_existing_mechanisms as m


class Mechanisms(unittest.TestCase):
    def test_average_tie_ranks_and_uncertainty(self):
        self.assertEqual(m.rankdata([3,1,1,2]).tolist(),[4,1.5,1.5,3])
        self.assertEqual(m.classify(m.stats([-1,1]),.01),'inconclusive')
        self.assertEqual(m.classify(m.stats([-2,-1]),.01),'negative')
        self.assertIsNone(m.correlation([1]*6,list(range(6)))[0])

    def test_missing_target_source_is_not_a_completed_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'completion.json').write_text('{"status":"PARTIAL"}')
            with self.assertRaisesRegex(ValueError,'Incomplete stage'):
                m.load_stage(root,'shifted',m.Sources())

    def test_joint_child_without_manifest(self):
        # Joint writes snapshot identity in completion.json, unlike TTT's child manifest.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'completion.json').write_text('{"status":"PASS"}')
            (root/'manifest.json').write_text(json.dumps(dict(mode='shifted',checkpoint_sha256=m.SHA)))
            for p in m.PARAMETERS:
                for k in m.SEVERITIES:
                    for seed in (1,2,3):
                        child=root/f'{p}-{k}-{seed}';child.mkdir()
                        (child/'completion.json').write_text(json.dumps(dict(status='PASS',snapshot_sha256='saved')))
                        m.table(child/'shifted.csv',[dict(parameter=p,severity=k,adaptation_seed=seed,evaluation_seed=e) for e in range(20)])
            rows,meta,_=m.load_stage(root,'shifted',m.Sources())
            self.assertEqual(len(rows),720);self.assertEqual(len(meta),36)
            self.assertEqual(meta[('torso_mass',2.,3)]['snapshot_sha256'],'saved')

    def test_optimizer_and_environment_not_constructed_by_forward_module(self):
        import ast
        source=Path(m.__file__).with_name('mechanism_forward.py').read_text()
        tree=ast.parse(source)
        calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
        for forbidden in ('TDMPC2','Adam','env.step','model.update','torch.optim.Adam'):
            self.assertFalse(any(call.endswith(forbidden) for call in calls))

    def test_full_csv_pipeline_and_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'results';out=Path(temp)/'analysis'
            def put(path,data):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
            c=dict(adaptation_seeds=[1,2,3],evaluation_seeds=list(range(10,30)),conditions=[dict(parameter=p,severity=k) for p in m.PARAMETERS for k in m.SEVERITIES])
            fa=root/'dynamics-only-750k-v1/final-snapshot-v2/audit/audit.json'
            ja=root/'joint-final-evaluation-v1/audit/audit.json'
            checks=[dict(parameter=p,severity=k,adaptation_seed=s,snapshot_sha256='snapshot') for p in m.PARAMETERS for k in m.SEVERITIES for s in c['adaptation_seeds']]
            put(fa,dict(status='PASS',checkpoint_sha256=m.SHA,configuration=c))
            put(ja,dict(status='PASS',checkpoint_sha256=m.SHA,configuration=c,checks=checks))
            m.table(fa.parent/'run_integrity.csv',checks)
            frozen=root/'physical-severity-750k-v1/frozen'
            put(frozen/'completion.json',dict(status='PASS'))
            put(frozen/'manifest.json',dict(seeds=c['evaluation_seeds']))
            put(frozen/'model_integrity.json',dict(file_sha256={'model.pt':m.SHA},optimizer_updates=0))
            F=[];index=[]
            for p in m.PARAMETERS:
                for k in m.SEVERITIES:
                    for e in c['evaluation_seeds']:
                        filename=f'{p}-{k}-{e}.npz';index.append(dict(file=filename,sha256='bank'))
                        F.append(dict(parameter=p,severity=k,seed=e,episode_return=100+e,forward_velocity=1,
                            control_cost=1,action_abs_mean=.1,action_saturation=0,actual_parameter_min=k,
                            actual_parameter_max=k,data_file=filename,episode_length=1000,terminated=False,truncated=True))
            m.table(frozen/'episodes.csv',F);put(frozen/'dataset_index.json',index)
            for owner,audit in [('dynamics-only-750k-v1/final-snapshot-v2',fa),('joint-final-evaluation-v1',ja)]:
                for mode in ('prediction','shifted'):
                    stage=root/owner/mode
                    put(stage/'completion.json',dict(status='PASS'))
                    put(stage/'manifest.json',dict(mode=mode,checkpoint_sha256=m.SHA,audit_sha256=m.digest(audit)))
                    for check in checks:
                        p,k,s=m.ident(check);child=stage/f'{p}-{k}-{s}'
                        put(child/'completion.json',dict(status='PASS',snapshot_sha256='snapshot'))
                        if owner.startswith('dynamics'):
                            put(child/'manifest.json',dict(mode=mode,checkpoint_sha256=m.SHA,snapshot_sha256='snapshot'))
                        rr=[]
                        for e in c['evaluation_seeds']:
                            row=dict(parameter=p,severity=k,adaptation_seed=s,evaluation_seed=e)
                            if mode=='prediction':
                                row.update(bank_file=f'{p}-{k}-{e}.npz',bank_sha256='bank',samples=1000,
                                           frozen_latent_mse=.01,decision_latent_mse=.005,joint_self_consistency_mse=.02)
                            else:
                                row.update(frozen_return=100+e,decision_return=105+e,joint_return=98+e,
                                    decision_velocity=1.,decision_control_cost=1.,decision_action_abs=.1,decision_action_saturation=0,
                                    joint_forward_velocity=1.,joint_control_cost=1.,joint_action_abs=.1,joint_action_component_saturation=0,
                                    episode_length=1000,terminated=False,truncated=True,actual_parameter_min=k,actual_parameter_max=k)
                            rr.append(row)
                        m.table(child/f'{mode}.csv',rr)
            put(root/'full-joint-750k-v1/runs-fixed/completion.json',dict(status='PASS'))
            for check in checks:
                p,k,s=m.ident(check);child=root/f'full-joint-750k-v1/runs-fixed/{p}-{k}-{s}'
                put(child/'manifest.json',dict(identity=check,checkpoint_sha256=m.SHA))
                put(child/'completion.json',dict(status='PASS',transitions=3000,updates=2001,optimizer_step_calls=4002,update_seconds=1,wall_seconds=2))
                with (child/'training.jsonl').open('w') as f:
                    for i in range(1,3001):f.write(json.dumps(dict(transition=i,updates=max(0,i-999),replay_episodes=i//1000,reward=1,metrics={'value_loss':.01} if i>=1000 else {}))+'\n')
            plan=root/'physical-severity-750k-v1/planning'
            put(plan/'completion.json',dict(status='PASS'))
            m.table(plan/'planning_comparisons.csv',[dict(parameter='bthigh_damping',severity=1.5,seed=1)])
            for label in ('same_observation','nominal_next_observation','shifted_next_observation'):
                np.savez(plan/f'example-00-{label}.npz',encoded=np.zeros((1,4)),actions_0=np.zeros((3,8,6)),reward_sum_0=np.arange(8.),discounted_terminal_q_0=np.ones(8),score_0=np.arange(8.)+1,elite_indices_0=np.array([6,7]))
            result=subprocess.run([sys.executable,m.__file__,'--results-root',str(root),'--output',str(out)],
                env=dict(os.environ,MPLCONFIGDIR=str(Path(temp)/'mpl')),text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads((out/'provenance/completion.json').read_text())['paired_episodes'],720)
            with zipfile.ZipFile(out.with_suffix('.zip')) as z:
                self.assertIsNone(z.testzip());self.assertFalse(any(n.endswith('.pt') for n in z.namelist()))


if __name__=='__main__':unittest.main()
