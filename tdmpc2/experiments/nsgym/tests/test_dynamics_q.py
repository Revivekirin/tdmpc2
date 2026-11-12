"""Actual CPU WorldModel/math tests; synthetic data never count as CUDA evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/nsgym/scripts'))
sys.path.insert(0,str(ROOT/'tdmpc2'))
import torch
from common import math
from common.dynamics_q import DynamicsQAdapter, expected_updates
from common.stage3a import equal_state
from tdmpc2 import TDMPC2
import run_dynamics_q as runner
import analyze_dynamics_q as analysis


class LearningContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)
        self.config, _ = runner.configuration()
        self.identity = dict(parameter='bthigh_damping',severity=2.,adaptation_seed=42000001,
                             method='dynamics_q_118',schedule='fixed')
        self.agent = runner.cpu_agent()
        self.adapter = DynamicsQAdapter(self.agent,self.config,self.identity)

    def observe(self, n):
        for step in range(self.adapter.transitions,n):
            ep,t = divmod(step,1000)
            self.adapter.observe(torch.randn(17),torch.randn(6).tanh(),torch.randn(17),
                                 float(step%7),False,t==999,ep,t,12.,6.)

    def batch(self):
        self.observe(64)
        return list(self.adapter.replay)

    def test_real_distributional_loss_finite_and_support(self):
        dyn,q,logits,target = self.adapter.losses(self.batch())
        self.assertEqual(logits.shape,(5,64,101))
        self.assertTrue(torch.isfinite(target).all())
        self.assertTrue(torch.isfinite(dyn) and torch.isfinite(q))
        projected=math.two_hot(target,self.agent.cfg)
        self.assertEqual(projected.shape,(64,101))
        torch.testing.assert_close(projected.sum(-1),torch.ones(64))
        self.assertTrue(((projected>0).sum(-1)<=2).all())
        explicit=torch.stack([-(projected*head.log_softmax(-1)).sum(-1).mean() for head in logits]).mean()
        torch.testing.assert_close(q,explicit)

    def test_bellman_reward_discount_time_limit_and_true_terminal(self):
        b = self.batch()
        b[0]['terminated']=True
        b[1]['truncated']=True
        rewards=torch.tensor([[r['reward']] for r in b])
        terminals=torch.tensor([[r['terminated']] for r in b],dtype=torch.float32)
        # Known target scalar isolates reward/discount/terminal semantics while
        # still calling the repository's original _td_target implementation.
        original=self.agent.model.Q
        def q_value(z,a,task,return_type='min',target=False,**kw):
            if target:
                self.assertEqual(return_type,'min')
                return torch.full((len(z),1),4.)
            return original(z,a,task,return_type=return_type,target=target,**kw)
        with patch.object(self.agent.model,'Q',side_effect=q_value):
            _,_,_,target=self.adapter.losses(b)
        torch.testing.assert_close(target,rewards+.995*(1-terminals)*4)
        self.assertEqual(float(target[0]),float(rewards[0]))
        self.assertAlmostEqual(float(target[1]-rewards[1]),.995*4,places=5)
        b[2]['discount']=.5
        with self.assertRaises(AssertionError):self.adapter.losses(b)

    def test_original_target_uses_online_actor_and_two_random_target_heads(self):
        model=self.agent.model
        z=torch.zeros(8,16); action=torch.zeros(8,6)
        logits=torch.full((5,8,101),-100.)
        for head in range(5):logits[head,:,50+head]=100.
        torch.manual_seed(9);selected=torch.randperm(5)[:2]
        expected=math.two_hot_inv(logits[selected],self.agent.cfg).min(0).values
        with patch.object(model._target_Qs,'forward',return_value=logits):
            torch.manual_seed(9)
            actual=model.Q(z,action,None,target=True,return_type='min')
        torch.testing.assert_close(actual,expected)
        with patch.object(model,'pi',wraps=model.pi) as pi, patch.object(model,'Q',wraps=model.Q) as q:
            self.agent._td_target(z,torch.ones(8,1),torch.zeros(8,1),None)
            self.assertEqual(pi.call_count,1)
            self.assertTrue(q.call_args.kwargs['target'])
            self.assertEqual(q.call_args.kwargs['return_type'],'min')

    def test_gradient_groups_are_isolated(self):
        dyn,q,_,_=self.adapter.losses(self.batch())
        q.backward()
        self.assertTrue(any(p.grad is not None for p in self.adapter.q_params))
        self.assertTrue(all(p.grad is None for p in self.adapter.dynamics_params))
        self.adapter.q_optimizer.zero_grad(set_to_none=True)
        dyn.backward()
        self.assertTrue(any(p.grad is not None for p in self.adapter.dynamics_params))
        self.assertTrue(all(p.grad is None for p in self.adapter.q_params))
        self.adapter.assert_fixed()

    def test_target_exact_ema_and_frozen_optimizer_ownership(self):
        self.observe(74)
        before=deepcopy(self.agent.model._target_Qs_params.state_dict())
        self.observe(75)
        online=self.agent.model._detach_Qs_params.state_dict()
        for k,v in self.agent.model._target_Qs_params.state_dict().items():
            if isinstance(v,torch.Tensor):torch.testing.assert_close(v,before[k].lerp(online[k],.01))
        changed={r['component']:r['changed'] for r in self.adapter.parameter_audit()}
        self.assertEqual(changed,dict(encoder=False,dynamics=True,reward=False,q=True,target_q=True,actor=False))
        for opt,params in [(self.adapter.q_optimizer,self.adapter.q_params),
                           (self.adapter.dynamics_optimizer,self.adapter.dynamics_params)]:
            self.assertEqual({id(p) for g in opt.param_groups for p in g['params']},{id(p) for p in params})
            self.assertTrue(all(int(s['step'])==1 for s in opt.state.values()))

    def test_3000_transitions_have_118_plus_118_updates(self):
        events=[]
        for step in range(1,3001):
            self.observe(step)
            if self.adapter.last_metrics:events.append(step)
        self.assertEqual(events,list(range(75,3001,25)))
        self.assertEqual(self.adapter.dynamics_steps,118)
        self.assertEqual(self.adapter.q_steps,118)
        self.assertEqual(self.adapter.target_steps,118)
        self.assertEqual(len(self.adapter.replay),3000)
        self.assertEqual(expected_updates(64),0)
        self.assertEqual(expected_updates(75),1)
        with self.assertRaises(AssertionError):self.adapter.update()

    def test_no_future_or_reset_transition_leakage_and_replay_copy(self):
        x=torch.ones(17); n=torch.ones(17)*2
        self.adapter.observe(x,torch.zeros(6),n,5.,False,False,0,0,12.,6.)
        x.fill_(100);n.fill_(100)
        self.assertTrue(torch.equal(self.adapter.replay[0]['observation'],torch.ones(17)))
        self.assertTrue(torch.equal(self.adapter.replay[0]['next_observation'],torch.ones(17)*2))
        self.observe(75)
        self.assertLessEqual(max(self.adapter.last_sample_steps),75)
        self.assertEqual(len(set(self.adapter.last_sample_steps)),64)
        with self.assertRaises(AssertionError):
            self.adapter.observe(x,torch.zeros(6),n,5.,False,False,1,0,12.,6.)
        self.adapter.identity['adaptation_seed']=43000001
        with self.assertRaises(AssertionError):self.observe(76)

    def test_mpc_terminal_value_uses_original_scalar_decode(self):
        agent=self.agent
        agent.cfg.horizon=3;agent.cfg.num_samples=4
        agent._estimate_value=MethodType(TDMPC2._estimate_value,agent)
        logits=torch.full((5,4,101),-100.);logits[:,:,55]=100.
        with patch.object(agent.model._Qs,'forward',return_value=logits):
            result=agent._estimate_value(torch.zeros(4,16),torch.zeros(3,4,6),None)
        expected=torch.full((4,1),.995**3*(torch.exp(torch.tensor(1.)).item()-1))
        torch.testing.assert_close(result,expected,atol=1e-5,rtol=1e-5)

    def test_heldout_blocks_mutation_and_snapshot_roundtrip(self):
        self.observe(75)
        before=deepcopy(self.agent.model.state_dict())
        with self.adapter.held_out():
            with self.assertRaises(RuntimeError):self.adapter.update()
            with self.assertRaises(RuntimeError):self.observe(76)
            self.agent.model.Q(torch.zeros(1,16),torch.zeros(1,6),None)
        self.assertTrue(equal_state(before,self.agent.model.state_dict()))
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'snapshot.pt'
            torch.save(dict(model=before,adapter=self.adapter.state_dict()),p)
            saved=torch.load(p,weights_only=False)
            self.assertTrue(equal_state(saved['adapter'],self.adapter.state_dict()))
            fresh=runner.cpu_agent();fresh.model.load_state_dict(saved['model'])
            for a,b in zip(self.agent.model.parameters(),fresh.model.parameters()):torch.testing.assert_close(a,b)


class EvaluationContract(unittest.TestCase):
    def test_runner_final_snapshot_audit_and_both_evaluation_modes(self):
        """Exercise real orchestration with actual small model and a labeled fake env."""
        import hashlib
        import numpy as np
        import random
        torch.set_num_threads(1)
        c,reference=runner.configuration()
        identity=runner.identities(c)[-1]
        nominal_cfg=dict(lr=.0003,value_coef=.1,grad_clip_norm=20,tau=.01)

        def tree_hash(value):
            h=hashlib.sha256()
            def visit(v):
                if isinstance(v,torch.Tensor):
                    h.update(str((v.dtype,v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
                elif isinstance(v,dict):
                    for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
                elif isinstance(v,(list,tuple)):
                    for x in v:visit(x)
                else:h.update(repr(v).encode())
            visit(value);return h.hexdigest()

        class FakeEnv:
            nominal_value=6.
            def __init__(self,config,condition,schedule):
                assert config['onset']==250
                self.scale=1. if schedule=='nominal' else condition['severity']
                self.unwrapped=SimpleNamespace(data=SimpleNamespace(actuator_force=np.zeros(6)),
                    model=SimpleNamespace(actuator_forcelimited=np.zeros(6,dtype=bool),actuator_forcerange=np.zeros((6,2))))
            def reset(self,seed=None):self.t=0;return np.zeros(17,dtype='float32'),{}
            def step(self,action):
                self.t+=1
                return np.full(17,self.t*.001,dtype='float32'),1.,False,self.t==1000,dict(x_velocity=2.,reward_ctrl=-.1)
            def readback(self):return self.nominal_value*self.scale
            def close(self):pass

        def seeded(seed):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

        with TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'checkpoint';checkpoint.mkdir()
            (checkpoint/'model.pt').write_bytes(b'fixture-not-a-scientific-checkpoint')
            (checkpoint/'optimizers.pt').write_bytes(b'fixture-optimizer')
            original_digest=runner.digest
            optim_hash=original_digest(checkpoint/'optimizers.pt')
            def digest(p):
                return runner.ttt.SHA if Path(p)==checkpoint/'model.pt' else original_digest(p)
            def loader(args):
                self.assertFalse(args.synthetic)
                seeded(17)
                agent=runner.cpu_agent()
                agent.optim=torch.optim.Adam(agent.model._dynamics.parameters(),lr=.0003)
                agent.pi_optim=torch.optim.Adam(agent.model._pi.parameters(),lr=.0003)
                agent.scale=SimpleNamespace(state_dict=lambda:{'value':torch.tensor(1.)})
                @torch.no_grad()
                def act(o,t0=False,eval_mode=True):
                    self.assertTrue(eval_mode)
                    return agent.model.pi(agent.model.encode(o[None],None),None)[1]['mean'][0]
                agent.act=act
                return agent,dict(config=nominal_cfg,optimizer_sha256=optim_hash)
            runs=root/'runs';runs.mkdir()
            child=runs/runner.name(identity);child.mkdir()
            args=SimpleNamespace(checkpoint=checkpoint,runs=runs,reference=root/'reference',mode='prediction')
            with patch.dict(sys.modules,{'training_support':SimpleNamespace(tree_hash=tree_hash)}), \
                 patch.object(runner.ttt,'load_agent',side_effect=loader), \
                 patch.object(runner.ttt,'seeded',side_effect=seeded), \
                 patch.object(runner.ttt,'environment',side_effect=FakeEnv), \
                 patch.object(runner.ttt,'np',np,create=True),patch.object(runner.ttt,'torch',torch,create=True), \
                 patch.object(runner,'digest',side_effect=digest):
                runner.train_one(args,c,identity,child,3000)
                runner.write(runs/'manifest.json',dict(command='run',configuration=c,source_hashes=runner.sources(),identities=[identity]))
                runner.write(runs/'completion.json',dict(status='PASS',completed=[child.name]))
                with patch.object(runner,'identities',return_value=[identity]):
                    inventory,parameters=runner.audit_runs(args,c)
                self.assertEqual(inventory[0]['total_optimizer_steps'],236)
                self.assertEqual(len(parameters),6)
                bankdir=args.reference/'frozen';bankdir.mkdir(parents=True)
                bank=bankdir/'fixture.npz'
                obs=np.repeat((np.arange(1001,dtype='float32')*.001)[:,None],17,axis=1)
                np.savez_compressed(bank,observations=obs,actions=np.zeros((1000,6),dtype='float32'))
                keys=[(identity['parameter'],identity['severity'],s) for s in c['evaluation_seeds']]
                baseline=dict(files={k:bank.name for k in keys},hashes={bank.name:original_digest(bank)},
                    mse={k:.1 for k in keys},episodes={k:dict(episode_return=1000.) for k in keys})
                predicted=runner.evaluate_one(args,c,reference,baseline,inventory[0])
                self.assertEqual(len(predicted),20)
                self.assertTrue(all(np.isfinite(r['latent_mse']) for r in predicted))
                args.mode='shifted'
                control=runner.evaluate_one(args,c,reference,baseline,inventory[0])
                self.assertEqual(len(control),20)
                self.assertTrue(all(r['episode_return']==1000. and r['episode_length']==1000 for r in control))
                self.assertTrue(all(r['actuator_force_saturation'] is None for r in control))

    def test_resume_only_preserves_pass_and_archives_partial(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp)
            p,skip=runner.destination(out,'stream')
            self.assertFalse(skip)
            (p/'partial').write_text('evidence')
            p,skip=runner.destination(out,'stream')
            self.assertFalse(skip)
            self.assertEqual(len(list(out.glob('interrupted-stream-*'))),1)
            runner.write(p/'completion.json',dict(status='PASS'))
            self.assertTrue(runner.destination(out,'stream')[1])

    def test_existing_frozen_manifest_grid_and_seeds(self):
        c,_=runner.configuration()
        # Available local baseline; production preflight additionally audits all
        # original completed adaptation logs and hash-linked raw evaluation data.
        p=ROOT.parent/'results/nsgym/physical-severity-750k-v1/frozen/manifest.json'
        if not p.exists():self.skipTest('Local baseline copy unavailable')
        m=runner.read(p)
        self.assertEqual(m['seeds'],c['evaluation_seeds'])
        self.assertEqual({(r['parameter'],r['severity']) for r in m['conditions']},
                         {(r['parameter'],r['severity']) for r in c['conditions']})
        self.assertEqual(m['expected_checkpoint_sha256'],runner.ttt.SHA)

    def test_aggregation_keeps_negative_effects_and_all_streams(self):
        c,_=runner.configuration()
        with TemporaryDirectory() as tmp:
            root=Path(tmp); audit=root/'audit.json'
            inventory=[dict(**i,snapshot=str(root/runner.name(i)/'snapshot-0003000.pt')) for i in runner.identities(c)]
            runner.write(audit,dict(status='PASS',configuration=c,checkpoint_sha256=runner.ttt.SHA,inventory=inventory))
            args=SimpleNamespace(audit=audit,prediction=root/'prediction',shifted=root/'shifted',
                ttt_final=root/'ttt',joint_final=root/'joint',nominal=None,output=root/'comparison')
            for method,mode,path in [('dq','prediction',args.prediction),('dq','shifted',args.shifted),
                ('ttt','prediction',args.ttt_final/'prediction'),('ttt','shifted',args.ttt_final/'shifted'),
                ('joint','shifted',args.joint_final/'shifted')]:
                path.mkdir(parents=True)
                runner.write(path/'manifest.json',dict(mode=mode,checkpoint_sha256=runner.ttt.SHA,
                    configuration=runner.configuration()[1] if method=='ttt' else c,
                    evaluation_seeds=c['evaluation_seeds'],audit_sha256=runner.digest(audit)))
                runner.write(path/'completion.json',dict(status='PASS'))
                for i in runner.identities(c):
                    label=runner.name(i)
                    if method=='ttt':label=runner.evaluation.label(i)
                    if method=='joint':label=label.replace('dynamics-q','full-joint')
                    child=path/label;child.mkdir()
                    data=[]
                    for seed in c['evaluation_seeds']:
                        data.append(dict(**i,evaluation_seed=seed,episode_return=8.,frozen_return=10.,
                            decision_return=11.,joint_return=7.,bank_sha256='fixture',bank_file='fixture.npz',
                            frozen_latent_mse=.1,decision_latent_mse=.08,latent_mse=.07,
                            forward_velocity=1.,control_cost=2.,action_saturation=.1,nominal_parameter=6.))
                    runner.table(child/f'{mode}.csv',data)
                    runner.write(child/'completion.json',dict(status='PASS',csv_sha256=runner.digest(child/f'{mode}.csv')))
                    runner.write(child/'manifest.json',dict(mode=mode,checkpoint_sha256=runner.ttt.SHA))
            analysis.aggregate(args,c)
            rr=runner.rows(args.output/'four_method_paired_comparison.csv')
            self.assertEqual(len(rr),720)
            self.assertTrue(all(float(r['dq_minus_ttt'])==-3 for r in rr))
            self.assertEqual(len(list((args.output/'figures').glob('*.png'))),6)
            self.assertEqual(runner.read(args.output/'completion.json')['nominal_retention'],'NOT_EVALUATED')
            # Missing/partial stages cannot be silently treated as complete.
            (args.shifted/'completion.json').write_text(json.dumps(dict(status='PARTIAL',failed=['fixture'])))
            with self.assertRaises(AssertionError):analysis.stage(args.shifted,'shifted',c,'dq',runner.digest(audit))


if __name__=='__main__':
    unittest.main()
