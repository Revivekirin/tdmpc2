"""Final snapshot audit and fixed-bank scoring; no full server experiment."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_ttt_final as f
import run_dynamics_only_ttt as t

_PROJECT=Path(__file__).resolve().parents[3]
REFERENCE=next((p/'results/nsgym/physical-severity-750k-v1' for p in (_PROJECT,_PROJECT.parent)
                if (p/'results/nsgym/physical-severity-750k-v1').exists()),
               _PROJECT/'results/nsgym/physical-severity-750k-v1')

class FinalSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (REFERENCE/'heldout-latent/latent_errors.csv').exists():raise unittest.SkipTest('Local Frozen reference unavailable')
        cls.c=t.configuration(SimpleNamespace(config=str(t.DEFAULT),command='run'))
        cls.base=f.baseline(REFERENCE,t.SHA,cls.c)

    def test_existing_frozen_bank_compatible(self):
        self.assertEqual(len(self.base['episodes']),240)
        self.assertEqual(len(self.base['mse']),240)
        self.assertEqual(self.base['frozen_manifest']['seeds'],self.c['evaluation_seeds'])
        self.assertGreater(self.base['mse'][('bthigh_damping',2.,43000001)],0)
        old=self.base['frozen_metadata']['source_hashes']
        sources={}
        for k,v in old.items():
            if k.endswith(('tdmpc2/tdmpc2.py','tdmpc2/common/world_model.py','tdmpc2/envs/nsgym/severity.py')):
                sources[str(Path(k).relative_to('/home/jhkim/repos/tdmpc2'))]=v
        self.assertEqual(len(f.static_sources(self.base['frozen_metadata'],{'source_hashes':sources})),3)

    def test_update_schedule_118_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rows=[];count=0
            for step in range(1,3001):
                update=int(step>=75 and step%25==0);count+=update
                rows.append(dict(adaptation_steps=step,gradient_updates=count,updates_this_transition=update,
                    episode=(step-1)//1000,t=(step-1)%1000,transition_seconds=.01,
                    training_loss=.1 if update else '',gradient_norm=.2 if update else ''))
            f.csv_rows(p/'online_steps.csv',rows)
            result=f.check_steps(p,'decision')
            self.assertEqual((result['updates'],result['first_update'],result['last_update']),(118,75,3000))
            rows[74]['updates_this_transition']=0
            with (p/'online_steps.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,rows[0]);writer.writeheader();writer.writerows(rows)
            with self.assertRaises(AssertionError):f.check_steps(p,'decision')

    def test_same_frozen_bank_prediction_scoring(self):
        t.dependencies();agent=t.tiny_agent()
        c=dict(self.c,evaluation_seeds=[43000001])
        cond=dict(parameter='bthigh_damping',severity=2.,adaptation_seed=42000001,reference=str(REFERENCE))
        rows=f.evaluate_prediction(agent,None,cond,self.base,c)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['samples'],1000)
        self.assertEqual(rows[0]['bank_sha256'],self.base['hashes'][rows[0]['bank_file']])
        self.assertAlmostEqual(rows[0]['delta_latent_mse'],rows[0]['decision_latent_mse']-rows[0]['frozen_latent_mse'])

    def test_full_36_stream_manifest_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta=self.base['frozen_integrity']['checkpoint_metadata']
            old=self.base['frozen_metadata']['source_hashes']
            source={str(Path(path).relative_to('/home/jhkim/repos/tdmpc2')):value
                    for path,value in old.items() if path.startswith('/home/jhkim/repos/tdmpc2/')
                    and path.endswith(('tdmpc2/tdmpc2.py','tdmpc2/common/world_model.py','tdmpc2/envs/nsgym/severity.py'))}
            self.assertEqual(len(source),3)
            versions={key:self.base['frozen_metadata']['versions'][key]
                      for key in ('torch','numpy','gymnasium','mujoco','ns_gym')}
            templates={}
            for method in ('frozen','decision'):
                rows=[];updates=0
                for step in range(1,3001):
                    hit=int(method=='decision' and step>=75 and step%25==0);updates+=hit
                    rows.append(dict(adaptation_steps=step,gradient_updates=updates,updates_this_transition=hit,
                                     episode=(step-1)//1000,t=(step-1)%1000,transition_seconds=.01,
                                     training_loss=.1 if hit else '',gradient_norm=.2 if hit else ''))
                templates[method]=root/f'{method}-steps.csv';f.csv_rows(templates[method],rows)
                stage=root/f'fixed-{method}';stage.mkdir();f.save(stage/'completion.json',{'status':'PASS'})
                for parameter,severity in f.conditions(self.c):
                    for seed in self.c['adaptation_seeds']:
                        identity=dict(parameter=parameter,severity=severity,adaptation_seed=seed,
                                      method=method,schedule='fixed',synthetic=False)
                        child=stage/f.label(identity).replace('-decision-',f'-{method}-');child.mkdir()
                        initial='initial-model';final='initial-model' if method=='frozen' else 'adapted-model'
                        f.save(child/'manifest.json',dict(identity=identity,checkpoint_sha256=t.SHA,
                            configuration=self.c,checkpoint_metadata=meta,versions=versions,source_hashes=source,
                            initial_model_hash=initial,initial_fixed_hash='fixed',initial_auxiliary_hash='aux'))
                        f.save(child/'completion.json',dict(status='PASS',identity=identity,finite_losses=True,
                            initial_model_hash=initial,final_model_hash=final,initial_fixed_hash='fixed',
                            final_fixed_hash='fixed',auxiliary_hash='aux',metrics=dict(adaptation_transitions=3000,
                            adaptation_episodes=3,gradient_update_steps=118 if method=='decision' else 0,
                            update_compute_time=1.)))
                        os.link(templates[method],child/'online_steps.csv')
                        (child/'snapshot-0003000.pt').write_bytes(b'fixture snapshot')
                        for n in (0,250,1000,2000):(child/f'snapshot-{n:07d}.pt').write_bytes(b'fixture')
            checks,_=f.audit(SimpleNamespace(runs=str(root),reference=str(REFERENCE)),self.c)
            self.assertEqual(len(checks),36)
            self.assertEqual({x['optimizer_steps'] for x in checks},{118})

    def test_shifted_only_control_scoring(self):
        t.dependencies();agent=t.tiny_agent()
        c=dict(self.c,evaluation_seeds=[43000001])
        adapter=t.make_adapter(agent,c,'decision',42000001)
        cond=dict(parameter='bthigh_damping',severity=2.,adaptation_seed=42000001)
        with adapter.held_out():
            rows=f.evaluate_control(agent,adapter,cond,self.base,c,'shifted')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['episode_length'],1000)
        self.assertEqual(rows[0]['kind'],'shifted')
        self.assertAlmostEqual(rows[0]['paired_return_difference'],rows[0]['decision_return']-rows[0]['frozen_return'])
        self.assertEqual(adapter.gradient_steps,0)

    def test_subset_and_statistics(self):
        self.assertEqual(len(f.selected(self.c,'smoke')),2)
        self.assertEqual(set(f.selected(self.c,'smoke')),{('bthigh_damping',2.),('torso_mass',2.)})
        self.assertEqual(len(f.selected(self.c,'full')),12)
        self.assertEqual(f.unit_stats([1.,2.,3.])['n_adaptation_seeds'],3)
        self.assertEqual(f.unit_stats([1.])['ci_low'],None)

    def test_partial_aggregation_keeps_full_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audit=root/'audit.json';f.save(audit,{'status':'PASS'})
            checks=[]
            for parameter,severity in f.conditions(self.c):
                for seed in self.c['adaptation_seeds']:
                    checks.append(dict(parameter=parameter,severity=severity,adaptation_seed=seed,
                        interaction_count=3000,optimizer_steps=118,online_transition_wall_seconds=100.,
                        gpu_update_seconds=10.,snapshot_sha256=f'{parameter}-{severity}-{seed}'))
            roots={mode:root/mode for mode in ('prediction','shifted')}
            for mode,parent in roots.items():
                parent.mkdir();f.save(parent/'manifest.json',{'mode':mode})
                f.save(parent/'completion.json',{'status':'PARTIAL'})
                for parameter,severity in f.selected(self.c,'smoke'):
                    for seed in self.c['adaptation_seeds']:
                        identity=dict(parameter=parameter,severity=severity,adaptation_seed=seed,method='decision',schedule='fixed',synthetic=False)
                        child=parent/f.label(identity);child.mkdir()
                        f.save(child/'manifest.json',dict(mode=mode,checkpoint_sha256=t.SHA,identity=identity,snapshot_sha256='fixture'))
                        f.save(child/'completion.json',dict(status='PASS',evaluation_wall_seconds=1.))
                        rows=[]
                        for evaluation_seed in self.c['evaluation_seeds']:
                            base=self.base['episodes'][(parameter,severity,evaluation_seed)]
                            mse=self.base['mse'][(parameter,severity,evaluation_seed)]
                            common=dict(parameter=parameter,severity=severity,adaptation_seed=seed,evaluation_seed=evaluation_seed)
                            if mode=='prediction':rows.append(dict(common,frozen_latent_mse=mse,decision_latent_mse=.9*mse,delta_latent_mse=-.1*mse))
                            else:rows.append(dict(common,frozen_return=float(base['episode_return']),decision_return=float(base['episode_return'])+1,
                                                   paired_return_difference=1.,frozen_velocity=0.,decision_velocity=1.,frozen_control_cost=1.,decision_control_cost=1.))
                        f.csv_rows(child/f'{mode}.csv',rows)
            args=SimpleNamespace(output=str(root/'summary'),audit=str(audit),prediction=str(roots['prediction']),shifted=str(roots['shifted']),nominal=None)
            f.aggregate(args,self.c,checks,self.base)
            summary=list(f.iter_csv(root/'summary/ttt_final_condition_summary.csv'))
            self.assertEqual(len(summary),12)
            self.assertEqual(sum(x['status']=='COMPLETE' for x in summary),2)
            self.assertEqual(f.read(root/'summary/completion.json')['status'],'PARTIAL')
            self.assertEqual(len(list(f.iter_csv(root/'summary/ttt_final_control.csv'))),120)
            self.assertTrue((root/'summary/ttt_prediction_vs_control.png').is_file())

if __name__=='__main__':unittest.main()
