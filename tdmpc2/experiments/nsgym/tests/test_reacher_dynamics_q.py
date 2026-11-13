"""Reacher binding tests using the unchanged real WorldModel and D+Q adapter."""
from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/nsgym/scripts'))
sys.path.insert(0,str(ROOT/'tdmpc2'))
import torch
import run_reacher_dynamics_q as q
from common.dynamics_q import DynamicsQAdapter
from common.stage3a import equal_state

class ReacherDQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(123);self.c,self.base=q.contract();self.agent=q.cpu_agent()
        self.i=q.identities(self.c)[0];self.ad=DynamicsQAdapter(self.agent,self.c,self.i)
    def observe(self,n):
        for t in range(self.ad.transitions,n):
            ep,et=divmod(t,50)
            self.ad.observe(torch.ones(10)*ep,torch.zeros(2),torch.ones(10)*(ep+.5),-1.,False,et==49,ep,et,self.i['severity'],1.)
    def test_full_update_schedule_and_reset_boundaries(self):
        self.observe(10000)
        self.assertEqual(self.ad.dynamics_steps,q.count(self.c,10000))
        self.assertEqual(self.ad.q_steps,398);self.assertEqual(self.ad.target_steps,398)
        self.assertEqual(sum(x['truncated'] for x in self.ad.replay),200)
        self.assertFalse(torch.equal(self.ad.replay[49]['next_observation'],self.ad.replay[50]['observation']))
        self.assertEqual(len(self.ad.replay),10000)
    def test_time_limit_bootstrap_uses_actual_next_encoding(self):
        self.observe(64);batch=list(self.ad.replay)
        original=self.agent.model.Q
        def value(z,a,task,return_type='min',target=False,**kwargs):
            return torch.full((len(z),1),4.) if target else original(z,a,task,return_type=return_type,**kwargs)
        with patch.object(self.agent.model,'Q',side_effect=value):
            dyn,loss,logits,target=self.ad.losses(batch)
        self.assertEqual(logits.shape,(5,64,101));self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(target[49]),-1+.95*4,places=5)
    def test_gradients_ema_and_frozen_state(self):
        self.observe(74);before=deepcopy(self.agent.model._target_Qs_params.state_dict());self.observe(75)
        online=self.agent.model._detach_Qs_params.state_dict()
        for k,v in self.agent.model._target_Qs_params.state_dict().items():
            if isinstance(v,torch.Tensor):torch.testing.assert_close(v,before[k].lerp(online[k],.01))
        changed={x['component']:x['changed'] for x in self.ad.parameter_audit()}
        self.assertEqual(changed,dict(encoder=False,dynamics=True,reward=False,q=True,target_q=True,actor=False))
    def test_heldout_and_roundtrip(self):
        self.observe(100);snapshot=deepcopy(self.ad.state_dict())
        with self.ad.held_out():
            with self.assertRaises(RuntimeError):self.observe(101)
            with self.assertRaises(RuntimeError):self.ad.update()
        self.ad.load_state_dict(snapshot);self.assertTrue(equal_state(snapshot,self.ad.state_dict()))
    def test_grid_and_disjoint_seeds(self):
        self.assertEqual(len(q.identities(self.c)),72)
        self.assertEqual(self.c['horizon'],50);self.assertEqual(self.agent.cfg.action_dim,2)
        self.assertEqual(self.agent.cfg.obs_shape,{'state':(10,)})
        self.assertFalse(set(self.c['adaptation_seeds']) & set(self.c['evaluation_seeds']))
        a=SimpleNamespace(family='M1',severity=2.,seed=53000003)
        self.assertEqual(len(q.identities(self.c,a)),1)
        a.seed=54000001
        with self.assertRaises(ValueError):q.identities(self.c,a)
    def test_leakage_and_bad_boundary_rejected(self):
        self.ad.identity['adaptation_seed']=self.c['evaluation_seeds'][0]
        with self.assertRaises(AssertionError):self.observe(1)
        self.ad.identity=self.i
        with self.assertRaises(AssertionError):
            self.ad.observe(torch.zeros(10),torch.zeros(2),torch.zeros(10),0.,False,True,0,0,.5,1.)
    def test_task_binding_train_restore_and_evaluation(self):
        import hashlib
        import numpy as np
        def tree_hash(value):
            h=hashlib.sha256()
            def visit(x):
                if isinstance(x,torch.Tensor):h.update(x.detach().cpu().contiguous().numpy().tobytes())
                elif isinstance(x,dict):
                    for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
                elif isinstance(x,(list,tuple)):
                    for v in x:visit(v)
                else:h.update(repr(x).encode())
            visit(value);return h.hexdigest()
        c=dict(self.c,transitions=100)
        class Env:
            nominal_value=1.
            def reset(self,seed=None):self.t=0;return np.zeros(10,dtype=np.float32),{}
            def readback(self):return self_i['severity']
            def step(self,ac):
                self.t+=1
                return np.full(10,self.t*.01,dtype=np.float32),-1.,False,self.t==50,dict(reward_dist=-.9,reward_ctrl=-.1)
            def close(self):pass
        self_i=self.i
        with TemporaryDirectory() as tmp:
            root=Path(tmp);cp=root/'checkpoint';cp.mkdir()
            (cp/'model.pt').write_bytes(b'fixture');(cp/'optimizers.pt').write_bytes(b'fixture optimizers')
            selection=dict(checkpoint=str(cp),model_sha256=q.r.sha(cp/'model.pt'))
            def loader(*args):
                torch.manual_seed(999);agent=q.cpu_agent()
                agent.optim=torch.optim.Adam(agent.model._dynamics.parameters())
                agent.pi_optim=torch.optim.Adam(agent.model._pi.parameters())
                agent.scale=SimpleNamespace(state_dict=lambda:{'x':torch.tensor(1.)})
                def act(o,t0=False,eval_mode=True):
                    self.assertTrue(eval_mode)
                    with torch.no_grad():return agent.model.pi(agent.model.encode(o[None],None),None)[1]['mean'][0]
                agent.act=act
                return agent,dict(optimizer_sha256=q.r.sha(cp/'optimizers.pt'))
            runs=root/'runs';runs.mkdir();child=runs/q.label(self.i);child.mkdir()
            args=SimpleNamespace(runs=runs,reference=root/'reference',resume=False,mode='prediction',output=root/'prediction')
            def physical(*x):return dict(reward=-1.,reward_dist=-.9,reward_ctrl=-.1,distance=.1,action_saturation=0.,terminated=False,truncated=x[5],actual_parameter=self.i['severity'],nominal_parameter=1.)
            with patch.dict(sys.modules,{'training_support':SimpleNamespace(tree_hash=tree_hash)}),patch.object(q.r,'load',side_effect=loader),patch.object(q.r,'env',return_value=Env()),patch.object(q.r,'step_record',side_effect=physical):
                q.train_one(args,c,self.base,selection,self.i,child,100)
                agent,adapter,d=q.restore(selection,self.base,c,child)
                self.assertEqual(d['dynamics_steps'],2);self.assertEqual(d['q_steps'],2)
                bank=args.reference/'frozen-home-01';bank.mkdir(parents=True)
                path=bank/'fixture.npz';np.savez(path,observations=np.zeros((51,10),np.float32),actions=np.zeros((50,2),np.float32))
                q.r.table(bank/'reacher_frozen_severity.csv',[dict(**q.condition(self.i),seed=seed,data_file=path.name,data_sha256=q.r.sha(path),latent_mse=.1) for seed in c['evaluation_seeds']])
                with patch.object(q,'completed_items',return_value=[self.i]):
                    q.evaluate(args,c,self.base,selection)
                    self.assertEqual(q.r.read(args.output/q.label(self.i)/'completion.json')['rows'],20)
                    args.mode='shifted';args.output=root/'control'
                    with patch.object(q.r,'episode',side_effect=lambda a,cond,seed:(dict(**cond,seed=seed,episode_return=-50.,distance_final=.1),[])):
                        q.evaluate(args,c,self.base,selection)
                    self.assertEqual(q.r.read(args.output/'completion.json')['status'],'PASS')

    def test_original_inventory_rejects_origin_or_budget_mismatch(self):
        import json
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            selection=dict(checkpoint=str(root/'missing-checkpoint'),model_sha256='nominal',pipeline_config_fingerprint=q.r.fingerprint(self.base),
                configuration=dict(task='nsgym-reacher',action_dim=2,obs_shape={'state':[10]},episode_length=50,tau=.01,lr=.0003,value_coef=.1,grad_clip_norm=20))
            (root/'nominal-home-01').mkdir();q.r.write(root/'nominal-home-01/selected_reacher_checkpoint.json',selection)
            stages=['frozen-home-01','adapt-home-01','prediction-home-01','control-home-02-resumed','full-joint-home-01/runs-fixed','full-joint-home-01/evaluation-full-01']
            for stage in stages:
                p=root/stage;p.mkdir(parents=True);q.r.write(p/'completion.json',dict(status='PASS'));q.r.write(p/'manifest.json',dict(checkpoint_sha256='nominal'))
            for i in q.identities(self.c):
                p=root/'adapt-home-01'/q.label(i);p.mkdir()
                q.r.write(p/'manifest.json',dict(identity=dict(i,method='decision'),selection=selection,config=self.base,synthetic=False))
                q.r.write(p/'completion.json',dict(status='PASS',metrics=dict(adaptation_transitions=10000,gradient_update_steps=398)))
                p=root/'full-joint-home-01/runs-fixed'/q.label(i);p.mkdir();q.r.write(p/'manifest.json',dict(identity=i,checkpoint_sha256='nominal'))
            args=SimpleNamespace(reference=root,selection=None,checkpoint=None)
            _,report=q.inventory(args,self.c,self.base)
            self.assertFalse(report['checkpoint_bytes_verified']);self.assertEqual(len(report['baseline_streams']),72)
            p=root/'frozen-home-01/manifest.json';p.write_text(json.dumps(dict(checkpoint_sha256='WRONG')))
            with self.assertRaises(AssertionError):q.inventory(args,self.c,self.base)

    def test_four_way_aggregation_preserves_paired_differences(self):
        import csv,json
        with TemporaryDirectory() as tmp:
            root=Path(tmp);ref=root/'reference';runs=root/'runs';runs.mkdir();ref.mkdir()
            args=SimpleNamespace(reference=ref,runs=runs,prediction=root/'prediction',control=root/'control',nominal=None,output=root/'comparison')
            q.r.write(runs/'completion.json',dict(status='PASS'))
            q.r.write(runs/'manifest.json',dict(configuration=self.c,source_hashes=q.sources()))
            for stage in ['prediction-home-01','control-home-02-resumed','full-joint-home-01/evaluation-full-01']:
                (ref/stage).mkdir(parents=True);q.r.write(ref/stage/'completion.json',dict(status='PASS'))
            for path,mode in [(args.prediction,'prediction'),(args.control,'shifted')]:
                path.mkdir();q.r.write(path/'completion.json',dict(status='PASS'))
                q.r.write(path/'manifest.json',dict(mode=mode,configuration=self.c,source_hashes=q.sources(),selection=dict(model_sha256='nominal')))
            frozen=[]
            for i in q.identities(self.c):
                child=runs/q.label(i);child.mkdir()
                q.r.write(child/'completion.json',dict(status='PASS',snapshot_sha256='snapshot',transitions=10000,dynamics_steps=398,q_steps=398,wall_seconds=1.,update_seconds=.1))
                (child/'training.jsonl').write_text(json.dumps(dict(global_step=75,metrics=dict(dynamics_loss=.1,q_loss=.2)))+'\n')
                for code in ['frozen','decision']:
                    child=ref/'adapt-home-01'/f"{q.r.tag(i)}-{code}-seed{i['adaptation_seed']}";child.mkdir(parents=True)
                    q.r.write(child/'completion.json',dict(metrics=dict(adaptation_transitions=10000,gradient_update_steps=398 if code=='decision' else 0,update_compute_time=.1),wall_seconds=1.))
                child=ref/'full-joint-home-01/runs-fixed'/f"{i['parameter']}-k{i['severity']:g}-full-joint-seed{i['adaptation_seed']}";child.mkdir(parents=True)
                q.r.write(child/'completion.json',dict(transitions=10000,updates=398,optimizer_step_calls=796,wall_seconds=1.,update_seconds=.1))
                control=[];pred=[]
                for seed in self.c['evaluation_seeds']:
                    row=dict(**i,seed=seed,budget=10000,kind='shifted',episode_return=-3.,distance_mean=.05,distance_final=.001,reward_dist=-2.5,reward_ctrl=-.5,action_abs=.1,action_saturation=0.,nominal_parameter=1.,actual_parameter=i['severity'],data_sha256='bank',latent_mse=.2)
                    if i['adaptation_seed']==self.c['adaptation_seeds'][0]:frozen.append(row)
                    control.append(dict(row,episode_return=-4.,reward_dist=-3.5))
                    pred.append(dict(row,method='decision',source_sha256='bank',latent_mse=.1))
                for stage,file,rows in [('prediction-home-01','predict.csv',pred),('control-home-02-resumed','control.csv',[dict(x,method='decision') for x in control]),('full-joint-home-01/evaluation-full-01','control.csv',[dict(x,method='full_joint_rl') for x in control])]:
                    ch=ref/stage/q.label(i);ch.mkdir();q.r.write(ch/'completion.json',dict(status='PASS'));q.r.table(ch/file,rows)
                for path,mode,rows in [(args.prediction,'prediction',pred),(args.control,'shifted',[dict(x,episode_return=-5.,reward_dist=-4.5) for x in control])]:
                    ch=path/q.label(i);ch.mkdir();q.r.table(ch/f'{mode}.csv',rows)
                    q.r.write(ch/'completion.json',dict(status='PASS',snapshot_sha256='snapshot',csv_sha256=q.r.sha(ch/f'{mode}.csv')))
            (ref/'frozen-home-01').mkdir();q.r.table(ref/'frozen-home-01/reacher_frozen_severity.csv',frozen)
            with patch.object(q,'inventory',return_value=(dict(model_sha256='nominal'),{})):
                q.aggregate(args,self.c)
                rr=q.r.csvread(args.output/'four_way_per_seed.csv')
                self.assertEqual(len(rr),1440)
                self.assertTrue(all(float(x['dq_minus_dynamics'])==-1 for x in rr))
                self.assertEqual(len(list((args.output/'figures').glob('*.png'))),9)
                (args.control/'completion.json').write_text(json.dumps(dict(status='PARTIAL')))
                args.output=root/'must-not-pass'
                with self.assertRaises(AssertionError):q.aggregate(args,self.c)

    def test_fresh_optimizers_match_validated_settings(self):
        for opt,lr in [(self.ad.dynamics_optimizer,.0001),(self.ad.q_optimizer,.0003)]:
            g=opt.param_groups[0];self.assertEqual(g['lr'],lr);self.assertEqual(g['betas'],(.9,.999))
            self.assertEqual(g['weight_decay'],0);self.assertEqual(g['eps'],1e-8)
            self.assertFalse(opt.state)

if __name__=='__main__':unittest.main()
