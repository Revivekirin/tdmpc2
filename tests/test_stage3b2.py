"""Synthetic artifact-management tests; no experimental performance claims."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage3b2_audit import *
from run_stage3b2 import plan,execution_lock,command
from analyze_stage3b2 import report,frozen_references


def fixture(root,checkpoint,kind='action_gain',value=.7,complete=True):
    cfg=expected_config(kind,value,checkpoint)
    cfg['stage3b']['run_id']='SYNTHETIC-stage3b2'
    # Legacy last component was the model seed, not severity.
    path=root/'runs'/kind/'set-0'/'dynamics_only'/'1';path.mkdir(parents=True)
    dataset=root/'data'/kind/'set-0.pt';dataset.parent.mkdir(parents=True)
    cfg['stage3a']['fixed_data_path']=str(dataset)
    checksum=digest_file(checkpoint)
    torch.save(dict(version=1,identity=fixed_identity(cfg,checksum),action=torch.ones(10,1),
                    observation=torch.ones(10,2),next_observation=torch.ones(10,2),episode_ids=torch.arange(10),
                    episodes=[{'seed':s} for s in cfg['stage3a']['fixed_data_seeds']]),dataset)
    dh=digest_file(dataset)
    meta=dict(config=cfg,checkpoint_sha256=checksum,active_method='dynamics_only',fixed_data_path=str(dataset),
              fixed_data_sha256=dh,fixed_data_collection_native_steps=20,fixed_data_collection_policy_decisions=10)
    (path/'metadata.json').write_text(json.dumps(meta))
    optimizer=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=cfg['stage3a']['lr']).state_dict()
    rows=[]
    for budget in BUDGETS if complete else BUDGETS[:3]:
        ev={c:dict(episodes=[dict(seed=s,success=1.,episode_return=10.,episode_length=5.) for s in cfg['stage3a']['evaluation_seeds']],
                   success=1.,episode_return=10.,episode_length=5.) for c in ('shifted','nominal')}
        row=dict(native_budget=budget,adaptation_native_steps=budget,adaptation_policy_decisions=budget//2,
                 adaptation_episodes=budget//10,gradient_update_steps=budget//10*50,replay_size=budget//2,
                 evaluation_state_invariance=True,reward_q_parameter_invariance=True,fixed_data_sha256=dh,evaluation=ev,
                 fixed_data_prediction={c:dict(per_episode_means=[.01]*10) for c in ('initial','current')})
        rows.append(row)
        updates=row['gradient_update_steps']
        optimizer['state']={0:dict(step=torch.tensor(float(updates)),exp_avg=torch.zeros(1),exp_avg_sq=torch.zeros(1))} if updates else {}
        state=dict(version=1,identity=dict(checkpoint_sha256=checksum,fixed_data_sha256=dh,
                   controls={k:v for k,v in cfg.items() if k not in ('stage3a','stage3b','work_dir')}),
                   rows=deepcopy(rows),budgets=[r['native_budget'] for r in rows],rollout=dict(native_steps=budget),
                   adapter=dict(version=1,settings=cfg['stage3a'],optimizer=optimizer,predictor={'weight':torch.ones(1)},
                                replay=[(torch.ones(2),torch.ones(1),torch.ones(2))]*(budget//2),in_episode=False,rng=None,counters=(budget//10,budget//2,budget,updates)),
                   environment=None,rng=None,planner=torch.zeros(1),episodes=[{}]*(budget//10))
        torch.save(state,path/f'resume-budget-{budget}.pt')
    (path/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for name in ('diagnostics.jsonl','adaptation-episodes.jsonl'):(path/name).write_text('')
    if complete:
        marker=dict(adaptation_native_steps=12800)
        (path/'complete.json').write_text(json.dumps(marker))
        torch.save(dict(checkpoint_sha256=checksum,fixed_data_sha256=dh,metrics=marker,
                        predictor={'weight':torch.ones(1)},optimizer=optimizer),path/'adapted.pt')
    return path,cfg


class Stage3B2Test(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cp=self.root/'nominal.pt';self.cp.write_bytes(b'synthetic nominal identity')
    def tearDown(self):self.tmp.cleanup()
    def test_reuse_uses_actual_severity_and_preserves_files(self):
        path,cfg=fixture(self.root,self.cp)
        before={str(p):digest_file(p) for p in path.iterdir()}
        result=plan(self.root,self.cp,'audit')
        entry=next(e for e in result['entries'] if e['shift_type']=='action_gain' and e['shift_value']==.7)
        self.assertEqual(entry['status'],'REUSE_COMPLETE');self.assertEqual(entry['commands'],[])
        self.assertEqual(len(result['entries']),8)
        self.assertEqual(before,{str(p):digest_file(p) for p in path.iterdir()})
        self.assertEqual(sum(e['action']=='START_NOMINAL' for e in result['entries']),7)
        for e in result['entries']:
            for cmd in e['commands']:self.assertIn('checkpoint='+json.dumps(str(self.cp.resolve())),cmd)
    def test_generated_overrides_resolve_to_requested_protocol(self):
        from hydra import compose,initialize_config_dir
        from common.benchmark_config import normalize,cli_aliases
        from omegaconf import OmegaConf
        cfg=expected_config('object_mass',3.0,self.cp.resolve())
        cmd=command(cfg,self.root/'data.pt',self.root/'new-destination')
        overrides=cli_aliases(cmd[5:])
        with initialize_config_dir(config_dir=str(ROOT/'tdmpc2'),version_base=None):
            actual=normalize(compose(config_name='stage3b',overrides=overrides),overrides)
        actual=OmegaConf.to_container(actual,resolve=True)
        actual['stage3a']['adaptation_seeds']=cfg['stage3a']['adaptation_seeds']
        self.assertEqual(controls(actual),controls(cfg))
        self.assertEqual(actual['stage3b']['output_dir'],str(self.root/'new-destination'))

    def test_partial_resumes_latest_into_new_directory(self):
        path,cfg=fixture(self.root,self.cp,complete=False)
        result=plan(self.root,self.cp,'resume-v1')
        e=next(e for e in result['entries'] if e['status']=='RESUME_PARTIAL')
        self.assertTrue(e['resume'].endswith('1600.pt'))
        self.assertNotEqual(e['target'],str(path));self.assertIn('/0.7/versions/resume-v1',e['target'])
        self.assertEqual(len(e['commands']),1)
        (path/'resume-budget-1600.pt').write_bytes(b'corrupt')
        audit=audit_candidate(path/'metadata.json',cfg,digest_file(self.cp))
        self.assertEqual(audit['budget'],800)
    def test_incompatible_settings_and_missing_prefix(self):
        path,cfg=fixture(self.root,self.cp)
        meta=read(path/'metadata.json');meta['config']['stage3a']['lr']=.1
        (path/'metadata.json').write_text(json.dumps(meta))
        self.assertEqual(audit_candidate(path/'metadata.json',cfg,digest_file(self.cp))['status'],'INCOMPATIBLE')
        meta['config']=cfg;(path/'metadata.json').write_text(json.dumps(meta))
        (path/'resume-budget-0.pt').unlink()
        self.assertEqual(audit_candidate(path/'metadata.json',cfg,digest_file(self.cp))['status'],'INCOMPATIBLE')
    def test_continuation_inherits_milestones_without_copying(self):
        import shutil
        path,cfg=fixture(self.root,self.cp)
        destination=self.root/'runs'/'action_gain'/'set-0'/'dynamics_only'/'0.7'/'versions'/'resumed'
        destination.mkdir(parents=True)
        for name in ('metadata.json','complete.json','adapted.pt','evaluations.jsonl','diagnostics.jsonl','adaptation-episodes.jsonl'):
            shutil.copyfile(path/name,destination/name)
        meta=read(destination/'metadata.json');meta['resumed_from']=str(path/'resume-budget-12800.pt')
        (destination/'metadata.json').write_text(json.dumps(meta))
        audit=audit_candidate(destination/'metadata.json',cfg,digest_file(self.cp))
        self.assertEqual(audit['status'],'REUSE_COMPLETE',audit['reason'])
        self.assertEqual(len(list(destination.glob('resume*.pt'))),0)

    def test_dataset_hash_mutation_and_lock(self):
        path,cfg=fixture(self.root,self.cp)
        Path(cfg['stage3a']['fixed_data_path']).write_bytes(b'wrong')
        audit=audit_candidate(path/'metadata.json',cfg,digest_file(self.cp))
        self.assertEqual(audit['status'],'INCOMPATIBLE');self.assertIn('SHA256',audit['reason'])
        with execution_lock(self.root):
            with self.assertRaisesRegex(RuntimeError,'lock'):
                with execution_lock(self.root):pass
    def test_report_all_conditions_and_no_false_baseline(self):
        path,cfg=fixture(self.root,self.cp)
        manifest=plan(self.root,self.cp,'report')
        mp=self.root/'manifest.json';mp.write_text(json.dumps(manifest))
        report(mp,self.root/'report')
        with (self.root/'report'/'baseline_comparison.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),8)
        done=next(r for r in rows if r['status']=='COMPLETE')
        self.assertEqual(done['adaptation_auc'],'1.0');self.assertEqual(done['frozen_compatible'],'False')
        self.assertEqual(done['success_delta_vs_frozen'],'')
        self.assertTrue((self.root/'report'/'action_gain-0.7'/'figures'/'shifted_success_by_adaptation.png').is_file())
    def test_frozen_different_seed_is_separate(self):
        cfg=expected_config('action_gain',.7,self.cp)
        folder=self.root/'runs'/'frozen';folder.mkdir(parents=True)
        meta=dict(config=cfg,shift_type='action_gain',shift_value=.7,set_id=0,
                  evaluation_seeds=[4000],checkpoint_sha256=digest_file(self.cp))
        (folder/'severity-metadata.json').write_text(json.dumps(meta))
        with patch('analyze_stage3b_severity.inspect',return_value=(dict(status='COMPLETE',reason='',success=1.,episode_length=5.,episode_return=10.,nominal_success=1.),None)):
            refs=frozen_references(self.root,dict(shift_type='action_gain',shift_value=.7,requested_config=cfg),digest_file(self.cp))
        self.assertFalse(refs[0]['compatible']);self.assertIn('seeds differ',refs[0]['reason'])


if __name__=='__main__':unittest.main()
