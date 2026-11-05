"""CPU protocol/physics API fixtures; not evidence of real SAPIEN rollout performance."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from benchmark_imports import package, ROOT
from test_shift_benchmark import config
from test_shift_physics import PhysicsNative
from test_dynamics_shift import Native
sys.path.insert(0,str(ROOT/'scripts'))
from stage3b_severity import add_arguments,plan,launch
from common.severity_protocol import severity,run_path,PROTOCOL
from common.severity_evaluation import NativeTrajectory,evaluate_frozen,model_digest
from analyze_stage3b_severity import inspect,summarize


class MeasuredNative(Native):
    def get_state(self): return self.state.copy()
    @property
    def tcp(self):
        return types.SimpleNamespace(pose=types.SimpleNamespace(p=self.state[:3].copy(),q=np.array([1,0,0,0])))
    def close(self): self.closed=True


class Repeat:
    def __init__(self,shift): self.shift=shift
    def reset(self,**kw): return torch.tensor(self.shift.reset(**kw))
    def step(self,action):
        records=[];reward=0
        self.shift.begin_decision()
        for _ in range(2):
            obs,r,done,info=self.shift.step(action.numpy());reward+=r;records.append(info['shift_record'])
        self.shift.end_decision()
        return torch.tensor(obs),reward,self.shift.episode_step>=4,dict(info,shift_native_steps=records)
    def close(self): self.shift.close()


class Agent:
    def __init__(self):
        self.model=torch.nn.Linear(4,4);self._prev_mean=torch.zeros(2,4)
    def act(self,obs,**kwargs):
        self._prev_mean+=1
        return torch.tensor([.2,.3,.4,.5])


class SeverityTests(unittest.TestCase):
    def args(self,extra=()):
        p=argparse.ArgumentParser();add_arguments(p)
        a=p.parse_args(extra)
        a.output=Path('/tmp/severity-tests');a.checkpoint='/tmp/model.pt'
        a.continue_suite=False;a.recovered_root=None;a.smoke=False;a.execute=False
        return a
    def suite(self):return json.loads((ROOT/'benchmarks/stage3b_suite.json').read_text())

    def test_default_and_pilot_matrix(self):
        m=plan(self.args(),self.suite())
        self.assertEqual([e['shift_value'] for e in m['entries']],[1,.7,1.5,.5])
        self.assertEqual(len(plan(self.args(['--pilot']),self.suite())['entries']),16)
        self.assertEqual(m['entries'][0]['seeds'],list(range(4000,4010)))

    def test_explicit_override_and_paths(self):
        a=self.args(['--shift-type','action_gain','--shift-values','.5','.3','--set-id','2','--run-id','pilot'])
        m=plan(a,self.suite())
        self.assertEqual([e['shift_value'] for e in m['entries']],[.5,.3])
        self.assertEqual(m['entries'][0]['seeds'][0],24000)
        self.assertTrue(m['entries'][0]['path'].endswith('action_gain/set-2/frozen/0.5/severity-eval/pilot'))

    def test_invalid_values(self):
        for kind in ('action_delay','action_gain','object_mass','contact_friction'):
            for value in ('nan','inf','-inf',-1,True):
                with self.subTest(kind=kind,value=value),self.assertRaises(ValueError):severity(kind,value)
        for kind in ('action_gain','object_mass','contact_friction'):
            with self.assertRaises(ValueError):severity(kind,0)
        for value in (.5,10001):
            with self.assertRaises(ValueError):severity('action_delay',value)
        with self.assertRaises(ValueError):severity('wrong',1)

    def test_invalid_combinations(self):
        for flags in (['--shift-value','.5'],['--method','dynamics_only'],['--set-id','-1'],
                      ['--shift-type','action_delay','--shift-values','1','1.0'],['--run-id','../bad']):
            with self.subTest(flags=flags),self.assertRaises(ValueError):plan(self.args(flags),self.suite())
        a=self.args(['--dry-run']);a.execute=True
        with self.assertRaises(ValueError):plan(a,self.suite())

    def test_nonoverwrite_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.args(['--shift-type','action_delay','--run-id','fixed']);a.output=Path(tmp)
            a.checkpoint=str(Path(tmp)/'model.pt');Path(a.checkpoint).write_bytes(b'fixture')
            target=Path(plan(a,self.suite())['entries'][0]['path']);target.mkdir(parents=True)
            (target/'keep').write_text('preserve');a.execute=True
            with patch('stage3b_severity.subprocess.run') as run,self.assertRaises(FileExistsError):launch(a,self.suite())
            run.assert_not_called();self.assertEqual((target/'keep').read_text(),'preserve')

    def test_native_eef_delay_correspondence_and_reset(self):
        env=package.ActionShiftWrapper(MeasuredNative(),config('action_delay','fixed',0,1))
        logger=NativeTrajectory(env,1,True);env.reset();logger.reset(env,4000);env.native_step_observer=logger.observe
        commands=[np.array([.2,.3,.4,.5]),np.array([.7,.8,.9,.1])]
        for action in commands:env.step(action)
        np.testing.assert_allclose(logger.rows[0]['executed_action'],[0,0,0,.5])
        np.testing.assert_allclose(logger.rows[1]['executed_action'],[.2,.3,.4,.1])
        np.testing.assert_allclose(logger.rows[1]['eef_xyz'],env.unwrapped.state[:3])
        self.assertEqual([r['timestep'] for r in logger.rows],[0,1])
        env.reset();self.assertEqual(len(env.queue),0)

    def test_gain_readback_and_logging_does_not_change_dynamics(self):
        for value in (.3,1,2):
            a=package.ActionShiftWrapper(MeasuredNative(),config('action_gain','fixed',1,value))
            b=package.ActionShiftWrapper(MeasuredNative(),config('action_gain','fixed',1,value))
            a.reset();b.reset();logger=NativeTrajectory(a,value,True);logger.reset(a,4000);a.native_step_observer=logger.observe
            for _ in range(3):
                action=np.array([.8,-.8,.2,.4]);np.testing.assert_array_equal(a.step(action)[0],b.step(action)[0])
            self.assertEqual(len(logger.rows),3)

    def test_physics_severity_exact_reset_and_restore(self):
        for kind in ('object_mass','contact_friction'):
            for value in (1,.3,3):
                c=config(kind,'fixed',1,value);c['application']='reset';c['parameter'].update(actor='obj',surfaces=['ground'])
                env=package.make_shift(PhysicsNative(),c,task='pick-cube')
                for _ in range(2):
                    env.reset();logger=NativeTrajectory(env,value,False);env.native_step_observer=logger.observe
                    env.step(np.zeros(4))
                    self.assertIsNotNone(logger.effective)
                env.backend.restore()
                if kind=='object_mass':self.assertEqual(env.backend.readback()['mass'],2)
                else:np.testing.assert_allclose(env.backend.readback()['materials'],[[.8,.6,.1]]*2)

    def test_frozen_episode_isolation_and_npz(self):
        from adapt_shift import episode
        agent=Agent();before=model_digest(agent.model);prev=agent._prev_mean.clone()
        cfg=types.SimpleNamespace(stage3a={'evaluation_seeds':[4000,4001]},severity={'log_eef':True},schedule={'type':'fixed'})
        environments=[]
        def make(c):
            env=Repeat(package.ActionShiftWrapper(MeasuredNative(),config('action_delay',c.schedule['type'],0,1)))
            environments.append(env);return env
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            result=evaluate_frozen(agent,cfg,make,episode,output,{},lambda e:e.shift)
            self.assertEqual(model_digest(agent.model),before);torch.testing.assert_close(agent._prev_mean,prev)
            self.assertEqual(result['gradient_updates'],0)
            self.assertTrue(all(e.shift.unwrapped.closed for e in environments))
            shifted=np.load(output/'shifted-seed-4000.npz');nominal=np.load(output/'nominal-seed-4000.npz')
            self.assertEqual(shifted['eef_xyz'].shape,(4,3))
            np.testing.assert_allclose(shifted['executed_action'][0,:3],0)
            np.testing.assert_allclose(nominal['executed_action'][0,:3],[.2,.3,.4])
            np.testing.assert_array_equal(shifted['timestep'],np.arange(4))

    def test_model_change_rejected(self):
        from adapt_shift import episode
        agent=Agent()
        def bad(*args,**kw):
            with torch.no_grad():agent.model.weight.add_(1)
            return episode(*args,**kw)
        cfg=types.SimpleNamespace(stage3a={'evaluation_seeds':[4000]},severity={'log_eef':False},schedule={'type':'fixed'})
        def make(c):return Repeat(package.ActionShiftWrapper(MeasuredNative(),config('action_gain',c.schedule['type'],1,.5)))
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(AssertionError,'Frozen'):
            evaluate_frozen(agent,cfg,make,bad,Path(tmp),{},lambda e:e.shift)

    def test_model_digest_handles_tdmpc2_state_metadata(self):
        class MetadataModel(torch.nn.Linear):
            def __init__(self):
                super().__init__(2,2)
                self.batch_size=torch.Size([3,2])
            def state_dict(self,*args,**kwargs):
                state=super().state_dict(*args,**kwargs)
                state['q_ensemble_batch_size']=self.batch_size
                state['device']=torch.device('cpu')
                state['optional']=None
                state['dtype']=torch.float32
                return state
        model=MetadataModel()
        first=model_digest(model)
        self.assertEqual(first,model_digest(model))
        model.batch_size=torch.Size([4,2])
        self.assertNotEqual(first,model_digest(model))
        with torch.no_grad():model.weight.add_(1)
        self.assertNotEqual(first,model_digest(model))

    def test_hydra_backward_compatibility_and_identity_severity(self):
        from hydra import initialize_config_dir,compose
        from common.benchmark_config import cli_aliases,normalize
        with initialize_config_dir(config_dir=str(ROOT/'tdmpc2'),version_base=None):
            old=compose(config_name='stage3b')
            self.assertEqual(old.stage3a.native_budgets,[800,1600,3200,6400,12800])
            for kind,value in [('action_delay',0),('object_mass',1),('action_gain',.3),('contact_friction',.7)]:
                overrides=cli_aliases([f'dynamics_shift={kind}','schedule=fixed',f'dynamics_shift.parameter.shifted={value}'])
                c=normalize(compose(config_name='severity',overrides=overrides),overrides)
                self.assertEqual(c.dynamics_shift.parameter.shifted,value)
                self.assertEqual(c.stage3a.mode,'frozen');self.assertEqual(list(c.stage3a.native_budgets),[])

    def test_summary_missing_partial_incompatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);entry=dict(path=str(p/'run'),shift_type='action_gain',shift_value=.5,set_id=0,seeds=[4000])
            manifest=dict(run_id='test',checkpoint_sha256='hash',task='pick-cube',model_size=5,episodic=False,log_eef=True)
            self.assertEqual(inspect(entry,manifest)[0]['status'],'MISSING')
            Path(entry['path']).mkdir();self.assertEqual(inspect(entry,manifest)[0]['status'],'PARTIAL')
            (Path(entry['path'])/'severity-metadata.json').write_text('{}')
            self.assertEqual(inspect(entry,manifest)[0]['status'],'PROTOCOL_INCOMPATIBLE')

    def test_complete_summary_and_duplicate_set_rejection(self):
        from adapt_shift import episode
        from common.severity_evaluation import serial
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'run';target.mkdir()
            manifest=dict(run_id='fixture',checkpoint_sha256='test-only',task='pick-cube',
                          model_size=5,episodic=False,log_eef=True)
            entry=dict(path=str(target),shift_type='action_gain',shift_value=.5,set_id=0,seeds=[4000])
            manifest['entries']=[entry]
            meta=dict(manifest,protocol=PROTOCOL,shift_type='action_gain',shift_value=.5,set_id=0,
                      evaluation_seeds=[4000],protocol_controls_sha256='fixture',code_revision='fixture',
                      source_sha256='fixture',environment_versions={'simulator':'CPU fixture'})
            meta['shift_definition_sha256']='fixture'
            cfg=types.SimpleNamespace(stage3a={'evaluation_seeds':[4000]},severity={'log_eef':True},schedule={'type':'fixed'})
            def make(c):return Repeat(package.ActionShiftWrapper(MeasuredNative(),config('action_gain',c.schedule['type'],1,.5)))
            result=evaluate_frozen(Agent(),cfg,make,episode,target,meta,lambda e:e.shift)
            (target/'severity-metadata.json').write_text(json.dumps(meta))
            (target/'severity-complete.json').write_text(json.dumps(result,default=serial))
            self.assertEqual(inspect(entry,manifest)[0]['status'],'COMPLETE')
            mp=root/'manifest.json';mp.write_text(json.dumps(manifest))
            rows=summarize([mp],root/'summary')
            self.assertEqual(rows[0]['episode_count'],1)
            self.assertTrue((root/'summary/action_gain.png').is_file())
            self.assertTrue((root/'summary/action_gain.pdf').is_file())
            with patch('analyze_stage3b_severity.plot'):
                rows=summarize([mp,mp],root/'duplicate')
                self.assertTrue(all(r['status']=='PROTOCOL_INCOMPATIBLE' and r['success']=='' for r in rows))
            # A completion marker cannot hide a truncated trajectory.
            with (target/'shifted-seed-4000.npz').open('wb') as f:np.savez(f,timestep=[0])
            self.assertEqual(inspect(entry,manifest)[0]['status'],'PARTIAL')

    def test_real_cli_dry_run_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'untouched'
            p=subprocess.run([sys.executable,str(ROOT/'scripts/run_stage3b.py'),'--phase','severity',
                              '--checkpoint','/not-needed/dry-run.pt','--output',str(output),'--pilot',
                              '--dry-run'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn('16 Frozen conditions',p.stdout)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
