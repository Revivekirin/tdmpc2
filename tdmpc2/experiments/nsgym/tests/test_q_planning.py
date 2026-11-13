"""Bounded real WorldModel CPU tests; no scientific checkpoint or simulator claims."""
import sys
from pathlib import Path
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np

SCRIPTS=Path(__file__).parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
import q_planning_core as c
import analyze_q_planning as cli
import torch
from common.world_model import WorldModel
from common import math


def model():
    cfg=c.Config(multitask=False,episodic=False,obs='state',obs_shape={'state':(17,)},
        num_enc_layers=2,enc_dim=16,latent_dim=16,mlp_dim=16,action_dim=6,task_dim=0,
        simnorm_dim=8,num_q=5,num_bins=101,vmin=-10.,vmax=10.,bin_size=.2,dropout=.01,
        log_std_min=-10.,log_std_max=2.,tau=.01,horizon=3,num_samples=8,num_elites=4,temperature=.5)
    m=WorldModel(cfg);m.eval();m.requires_grad_(False)
    # Nonzero critics/rewards so all tests are sensitive to state/action and decoding.
    with torch.no_grad():
        for p in m._Qs.parameters():p.add_(torch.randn_like(p)*.05)
        for p in m._reward.parameters():p.add_(torch.randn_like(p)*.05)
    return m


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(4);self.f=model();self.o=torch.randn(5,17);self.a=torch.randn(5,6).tanh()
    def test_probability_decode_support_and_finite(self):
        q=self.f.Q(self.f.encode(self.o,None),self.a,None,return_type='all');p=c.distributions(q,self.f.cfg)
        torch.testing.assert_close(p['probability'].sum(-1),torch.ones(5,5))
        bins=torch.linspace(-10,10,101)
        torch.testing.assert_close(p['scalar'],math.symexp((p['probability']*bins).sum(-1)))
        self.assertTrue((p['interhead_js']>=-1e-6).all())
        with self.assertRaises(ValueError):c.distributions(q[...,:4],self.f.cfg)
        q[0,0,0]=float('nan')
        with self.assertRaises(ValueError):c.distributions(q,self.f.cfg)
    def test_immediate_invariance_and_q_change(self):
        d=deepcopy(self.f)
        with torch.no_grad():next(d._dynamics.parameters()).add_(.1)
        f=self.f.Q(self.f.encode(self.o,None),self.a,None,return_type='all')
        torch.testing.assert_close(f,d.Q(d.encode(self.o,None),self.a,None,return_type='all'),rtol=0,atol=0)
        with torch.no_grad():
            for p in d._Qs.parameters():p.add_(torch.randn_like(p)*.1)
        altered=d.Q(d.encode(self.o,None),self.a,None,return_type='all')
        self.assertGreater(float(c.js(f.softmax(-1).mean(0),altered.softmax(-1).mean(0)).mean()),0)
    def test_candidate_reproducibility_bounds_and_no_leak(self):
        a=c.candidate_bank(3,16,6,14,2)
        np.testing.assert_array_equal(a,c.candidate_bank(3,16,6,14,2))
        self.assertLessEqual(np.max(np.abs(a)),1)
        self.assertFalse(np.array_equal(a,c.candidate_bank(3,16,6,15,2)))
    def test_native_score_decomposition_no_optimizers_state_rng(self):
        before=c.state_hash(self.f);rng=torch.get_rng_state().clone()
        tape=torch.tensor(c.candidate_bank(3,8,6,11,1))
        with patch('torch.optim.Adam',side_effect=AssertionError('optimizer constructed')):
            first=c.score(self.f,self.o[0],tape,self.f.cfg,.995,19)
            second=c.score(self.f,self.o[0],tape,self.f.cfg,.995,19)
        torch.testing.assert_close(first['score'],second['score'],rtol=0,atol=0)
        torch.testing.assert_close(first['score'],(first['reward']+first['terminal']).flatten())
        z=self.f.encode(self.o[:1],None).repeat(8,1);reward=0
        for t in range(3):
            reward=reward+.995**t*math.two_hot_inv(self.f.reward(z,tape[t],None),self.f.cfg)
            z=self.f.next(z,tape[t],None)
        torch.testing.assert_close(reward,first['reward'])
        torch.testing.assert_close(first['terminal'],.995**3*first['native_terminal_q'])
        self.assertEqual(before,c.state_hash(self.f));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        bad=tape.clone();bad[0,0,0]=2
        with self.assertRaises(ValueError):c.score(self.f,self.o[0],bad,self.f.cfg,.995,19)
    def test_hybrid_and_ranking(self):
        d=deepcopy(self.f);c.ReadOnlyHybrid(d,self.f)
        with torch.no_grad():next(d._encoder.parameters()).add_(.1)
        with self.assertRaisesRegex(ValueError,'encoder'):c.ReadOnlyHybrid(d,self.f)
        r=c.ranks([1,2,3],[3,2,1],1)
        self.assertEqual(r['top_k_overlap'],0);self.assertEqual(r['argmax_disagreement'],1)
        self.assertAlmostEqual(r['spearman'],-1)
    def test_seed_filter_and_nominal_not_pooled(self):
        from types import SimpleNamespace
        a=SimpleNamespace(parameter='torso_mass',severity=1.,seed=42000002)
        self.assertTrue(cli.selected(a,('torso_mass',1.,42000002)))
        self.assertFalse(cli.selected(a,('bthigh_damping',1.,42000002)))
        self.assertFalse(set(cli.SEEDS)&set(cli.EVAL_SEEDS))
    def test_four_model_compatibility(self):
        models={'F':self.f,'D':deepcopy(self.f),'DQ':deepcopy(self.f),'J':deepcopy(self.f)}
        with torch.no_grad():
            for m,parts in [('D',['_dynamics']),('DQ',['_dynamics','_Qs','_target_Qs_params']),
                            ('J',['_encoder','_dynamics','_reward','_Qs','_pi'])]:
                for part in parts:
                    for p in getattr(models[m],part).state_dict().values():p.add_(torch.randn_like(p)*.01)
        c.compatible(models)
        with torch.no_grad():next(models['D']._pi.parameters()).add_(.1)
        with self.assertRaisesRegex(ValueError,'fixed actor'):c.compatible(models)

    def test_pipeline_tables_figures_and_missing_evidence(self):
        from types import SimpleNamespace
        from contextlib import redirect_stdout
        import io
        with TemporaryDirectory() as tmp:
            out=Path(tmp);banks=[]
            for e in cli.EVAL_SEEDS:
                path=out/f'bank{e}.npz'
                np.savez_compressed(path,observations=self.o[:2].numpy(),actions=self.a[:2].numpy(),
                    next_observations=self.o[1:3].numpy(),rewards=np.ones(2),terminated=np.zeros(2),
                    timestep=np.array([0,999]),action_saturated=np.array([True,False]),
                    observation_regime=np.array(['early','later']),velocity_regime=np.array(['0_to_5','above_5']))
                banks.append(dict(parameter='bthigh_damping',severity=2,evaluation_seed=e,bank_file=str(path),
                    bank_sha256=cli.digest(path),actual_parameter=12,nominal_parameter=6))
            cli.table(out/'tables/common_bank_inventory.csv',banks)
            cfg=vars(self.f.cfg)|dict(episode_length=1000,discount_denom=5,discount_min=.95,discount_max=.995,max_std=2.)
            a=SimpleNamespace(output=out,results_root=out/'results',device='cpu',checkpoint=out/'none',full_grid=False,horizon=None,
                candidates=8,top_k=4,planning_observations=1,comparison=out/'missing')
            entry=dict(parameter='bthigh_damping',severity=2,adaptation_seed=42000001,models={})
            models={name:deepcopy(self.f) for name in ('F','D','DQ','J')}
            with patch.object(cli,'config',return_value=cfg),patch.object(cli,'inputs',return_value=[(entry,banks)]),\
                 patch.object(cli,'load_models',return_value=models),patch('torch.optim.Adam',side_effect=AssertionError('optimizer')),redirect_stdout(io.StringIO()):
                cli.inference(a);cli.inference(a,True);cli.aggregate(a);cli.plot(a);cli.report(a)
            self.assertEqual(len(cli.rows(out/'tables/immediate_q_metrics.csv')),800)
            self.assertEqual(len(cli.rows(out/'tables/planning_score_components.csv')),56)
            self.assertEqual(cli.read(out/'provenance/completion.json')['status'],'PARTIAL')
            self.assertTrue((out/'figures/q_distribution_by_severity.png').is_file())
            import os,shutil
            if os.environ.get('Q_PLANNING_TEST_PREVIEW'):
                shutil.copytree(out/'figures',Path(os.environ['Q_PLANNING_TEST_PREVIEW']),dirs_exist_ok=True)
            with self.assertRaises(FileExistsError):cli.table(out/'tables/immediate_q_metrics.csv',[])

    def test_original_snapshot_formats_no_optimizer_and_hash_guard(self):
        from types import SimpleNamespace
        import json
        with TemporaryDirectory() as tmp:
            root=Path(tmp);items={}
            models={m:deepcopy(self.f) for m in ('D','DQ','J')}
            for m in models.values():m.init()
            with torch.no_grad():
                for m,parts in [('D',['_dynamics']),('DQ',['_dynamics','_Qs','_target_Qs_params']),
                                ('J',['_encoder','_dynamics','_reward','_Qs','_pi'])]:
                    for part in parts:
                        for tensor in getattr(models[m],part).state_dict().values():tensor.add_(torch.randn_like(tensor)*.02)
            for m in models:
                folder=root/m;folder.mkdir()
                path=folder/'arbitrary-final-filename.pt'
                payload=dict(adapter={'predictor':models[m]._dynamics.state_dict()},model_hash='verified') if m=='D' else dict(model=models[m].state_dict())
                torch.save(payload,path)
                item=dict(snapshot=str(path),snapshot_sha256=cli.digest(path))
                for name in ('manifest','completion'):
                    f=folder/(name+'.json');f.write_text('{}');item[name]=str(f);item[name+'_sha256']=cli.digest(f)
                items[m]=item
            a=SimpleNamespace(device='cpu',checkpoint=root)
            fake_support=SimpleNamespace(tree_hash=lambda state:'verified')
            with patch.dict(sys.modules,{'training_support':fake_support}),patch.object(cli,'config',return_value=vars(self.f.cfg)),\
                 patch('torch.optim.Adam',side_effect=AssertionError('optimizer constructed')):
                restored=cli.load_models(a,items,frozen=self.f,nominal=self.f.state_dict())
                for m in models:self.assertEqual(c.state_hash(restored[m]),c.state_hash(models[m]))
                Path(items['D']['snapshot']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Snapshot changed'):cli.load_models(a,items,frozen=self.f,nominal=self.f.state_dict())


if __name__=='__main__':unittest.main(verbosity=2)
