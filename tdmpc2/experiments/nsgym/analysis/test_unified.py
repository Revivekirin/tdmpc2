"""End-to-end raw discovery contracts using temporary completed-run fixtures."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from .discovery import Catalog
from .sources import collect
from .metrics import paired_improvements
from .__main__ import main

CONFIG=json.loads(Path(__file__).with_name('config.json').read_text())

def write_json(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def fixture(root,seed=1,name='arbitrary-run',method='decision',task='halfcheetah'):
    parameter='bthigh_damping' if task=='halfcheetah' else 'joint0_damping'
    identity=dict(parameter=parameter,severity=2,adaptation_seed=seed,method=method,schedule='fixed')
    config=dict(task=task,evaluation_seeds=[100,101],adaptation_seeds=[seed])
    checkpoint='c'*64
    train=root/name/'training';train.mkdir(parents=True)
    snapshot=train/'snapshot-0003000.pt';snapshot.write_bytes(f'fixture {seed}'.encode());h=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    write_json(train/'manifest.json',dict(identity=identity,configuration=config,checkpoint_sha256=checkpoint))
    write_json(train/'completion.json',dict(status='PASS',transitions=3000,updates=118,snapshot_sha256=h))
    if method in ('dynamics_q','reacher_dynamics_q_v1'):
        write_json(train.parent/'completion.json',dict(status='PASS',completed=['training'],expected=1))
    base=dict(task=task,checkpoint_sha256=checkpoint,configuration=config)
    frozen=root/'shared-baseline'
    write_json(frozen/'manifest.json',base);write_json(frozen/'completion.json',dict(status='PASS'))
    write_csv(frozen/'episodes.csv',[dict(parameter=parameter,severity=2,seed=e,episode_return=10+e,episode_length=1000) for e in (100,101)])
    for mode in ('shifted','prediction'):
        batch=root/name/mode
        write_json(batch/'manifest.json',{**base,'mode':mode})
        write_json(batch/'completion.json',dict(status='PASS',completed=['condition'],expected=1))
        child=batch/'condition'
        write_json(child/'manifest.json',{**base,'mode':mode,'identity':identity,'snapshot':str(snapshot),'snapshot_sha256':h})
        write_json(child/'completion.json',dict(status='PASS',rows=2,snapshot=str(snapshot),snapshot_sha256=h))
        records=[dict(**identity,evaluation_seed=e,budget=3000,episode_return=12+e,episode_length=1000) if mode=='shifted' else dict(**identity,evaluation_seed=e,budget=3000,latent_mse=.1,frozen_latent_mse=.2,samples=10) for e in (100,101)]
        write_csv(child/(mode+'.csv'),records)
    return train


class RawPipelineTests(unittest.TestCase):
    def catalog(self,root):return Catalog(root,CONFIG['tasks'])
    def test_new_run_discovered_and_aggregated_without_configuration_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'results';fixture(root)
            first=collect(self.catalog(root));_,c=paired_improvements(first.rows)
            self.assertEqual(c[0]['n_adaptation_streams'],1)
            fixture(root,seed=2,name='new-completed-run-id')
            second=collect(self.catalog(root));_,c=paired_improvements(second.rows)
            self.assertEqual(c[0]['n_adaptation_streams'],2)
            self.assertEqual(c[0]['mean'],2)
            main(['full','--source-root',str(root),'--output',str(Path(tmp)/'report'),'--no-plots'])
            self.assertTrue((Path(tmp)/'report/UNIFIED_RESEARCH_REPORT.md').is_file())
    def test_incomplete_and_smoke_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);train=fixture(root)
            write_json(train/'completion.json',dict(status='PARTIAL'))
            rec=collect(self.catalog(root))
            self.assertFalse(any(r['method']=='dynamics_only' for r in rec.rows))
            self.assertTrue(rec.c.issues)
    def test_relocated_absolute_snapshot_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'nsgym';train=fixture(root)
            c=self.catalog(root)
            self.assertEqual(c.resolve('/old/server/results/nsgym/arbitrary-run/training/snapshot-0003000.pt'),(train/'snapshot-0003000.pt').resolve())
    def test_duplicate_copies_not_counted(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            shutil.copytree(root/'arbitrary-run/shifted',root/'resumed-copy')
            rec=collect(self.catalog(root));_,c=paired_improvements(rec.rows)
            self.assertEqual(c[0]['n_adaptation_streams'],1)
            self.assertEqual(c[0]['evaluation_seeds_per_stream'],2)
            self.assertTrue(rec.duplicates)
    def test_own_output_not_rediscovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            out=root/'anything-you-name-it'
            write_json(out/'manifest.json',dict(no_training_or_simulator=True))
            write_json(out/'completion.json',dict(status='PASS'))
            write_csv(out/'shifted.csv',[dict(parameter='bthigh_damping',severity=2,evaluation_seed=100,method='decision',adaptation_seed=99,episode_return=999)])
            c=self.catalog(root)
            self.assertFalse(any(p.is_relative_to(out) for p in c.files))
    def test_dq_requires_both_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root,method='reacher_dynamics_q_v1',task='reacher')
            write_json(root/'arbitrary-run/prediction/completion.json',dict(status='PARTIAL'))
            rec=collect(self.catalog(root))
            self.assertFalse(any(r['method']=='dynamics_q' for r in rec.rows))
            self.assertFalse(rec.c.issues)
            write_json(root/'arbitrary-run/prediction/completion.json',dict(status='PASS'))
            rec=collect(self.catalog(root))
            self.assertTrue(any(r['method']=='dynamics_q' for r in rec.rows))
    def test_running_dq_batch_and_unconsumed_log_are_not_source_errors(self):
        from unittest.mock import patch
        from . import engine
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'results';fixture(root,method='reacher_dynamics_q_v1',task='reacher')
            # Some real held-out exports identify the method through the
            # referenced training snapshot, not a child manifest identity.
            for mode in ('shifted','prediction'):
                manifest=root/'arbitrary-run'/mode/'condition/manifest.json'
                data=json.loads(manifest.read_text());data.pop('identity')
                write_json(manifest,data)
            write_json(root/'arbitrary-run/shifted/completion.json',dict(status='PARTIAL'))
            log=root/'arbitrary-run/postrun.log';log.write_text('running\n')
            original=engine.collect
            def with_live_log(catalog,filters):
                records=original(catalog,filters)
                log.write_text('running\nstill running\n')
                return records
            output=Path(tmp)/'report'
            with patch.object(engine,'collect',side_effect=with_live_log):
                main(['full','--source-root',str(root),'--output',str(output),'--no-plots'])
            done=json.loads((output/'completion.json').read_text())
            self.assertNotEqual(done['status'],'PARTIAL')
            self.assertEqual(done['invalid_sources'],0)
            excluded=list(csv.DictReader((output/'EXCLUDED_SOURCES.csv').open()))
            self.assertTrue(any('held-out batch not complete' in r['reason'] for r in excluded))
    def test_dq_leaf_pass_does_not_override_incomplete_training_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root,method='reacher_dynamics_q_v1',task='reacher')
            write_json(root/'arbitrary-run/completion.json',dict(status='PARTIAL',completed=['training'],expected=1))
            rec=collect(self.catalog(root))
            self.assertFalse(any(r['method']=='dynamics_q' for r in rec.rows))
            self.assertFalse(rec.c.issues)
            self.assertTrue(any('training batch not complete' in r['reason'] for r in rec.c.excluded))
    def test_running_nominal_retention_does_not_invalidate_complete_shifted_dq(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root,method='reacher_dynamics_q_v1',task='reacher')
            source=root/'arbitrary-run/shifted/condition'
            target=root/'arbitrary-run/nominal-retention/condition'
            manifest=json.loads((source/'manifest.json').read_text())
            manifest['mode']='nominal'
            write_json(target/'manifest.json',manifest)
            write_json(target/'completion.json',json.loads((source/'completion.json').read_text()))
            with (source/'shifted.csv').open(newline='') as stream:
                rows=list(csv.DictReader(stream))
            for row in rows:row['kind']='nominal'
            write_csv(target/'nominal.csv',rows)
            write_json(target.parent/'completion.json',dict(status='PARTIAL',completed=['condition'],expected=1))
            rec=collect(self.catalog(root))
            self.assertFalse(rec.c.issues)
            self.assertTrue(any(r['method']=='dynamics_q' and r['kind']=='shifted' for r in rec.rows))
            self.assertFalse(any(r['method']=='dynamics_q' and r['kind']=='nominal' for r in rec.rows))
            self.assertTrue(any('held-out batch not complete' in r['reason'] for r in rec.c.excluded))
    def test_snapshot_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);train=fixture(root)
            (train/'snapshot-0003000.pt').write_bytes(b'changed')
            rec=collect(self.catalog(root))
            self.assertFalse(any(r['method']=='dynamics_only' for r in rec.rows))
    def test_wrong_heldout_seed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            p=root/'arbitrary-run/shifted/condition/shifted.csv'
            rows=list(csv.DictReader(p.open()));rows[0]['evaluation_seed']='999'
            write_csv(p,rows);rec=collect(self.catalog(root))
            self.assertFalse(any(r['method']=='dynamics_only' and r['kind']=='shifted' for r in rec.rows))
    def test_conflicting_resume_excluded(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            shutil.copytree(root/'arbitrary-run/shifted',root/'resume')
            p=root/'resume/condition/shifted.csv';rows=list(csv.DictReader(p.open()));rows[0]['episode_return']='999'
            write_csv(p,rows);rec=collect(self.catalog(root))
            self.assertTrue(any('Conflicting' in r['reason'] for r in rec.c.issues))
            self.assertFalse(any(r['method']=='dynamics_only' and r['kind']=='shifted' and r['metric']=='episode_return' and r['evaluation_seed']=='100' for r in rec.rows))
    def test_independent_selections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';fixture(root)
            output=Path(tmp)/'report'
            main(['full','--source-root',str(root),'--output',str(output),'--task','halfcheetah','--method','dynamics_only','--shift-family','damping','--severity','2','--schedule','fixed','--no-plots'])
            rows=list(csv.DictReader((output/'comparisons.csv').open()))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['method'],'dynamics_only')

    def test_schedule_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);rec=collect(self.catalog(root))
            for r in rec.rows:
                if r['method']=='dynamics_only':r['schedule']='abrupt'
            self.assertEqual(paired_improvements(rec.rows)[1],[])

if __name__=='__main__':unittest.main()
