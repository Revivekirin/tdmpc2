import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
import torch
from test_planner_instrumentation import agent, load_planner
from common import math
from common.mechanism_analysis import MechanismAnalysis, prediction_error, rollout, execution_semantics, model_checksum
from common.mechanism_metrics import ranks, ranking_comparison, paired_summary
from common.mechanism_config import validate_analysis


def options():
    return dict(enabled=True, mode='frozen_mechanism', one_step=True, multi_step=True,
        planning=True, counterfactual=True, rollout_horizons=[1,3,5], decision_stride=1,
        max_logged_candidates=4, deterministic_probes=True, q_head_mode='all', policy_probe_mode='mean',
        save_latents=False, save_candidate_actions=True, save_candidate_scores=True, save_counterfactual_details=True)


class MechanismTests(unittest.TestCase):
    def test_checksum_includes_tensordict_metadata(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.]))
                self.batch_size = torch.Size([5])
                self.state_device = None

            def state_dict(self):
                return dict(weight=self.weight, batch_size=self.batch_size,
                            device=self.state_device)

        model = Model()
        original = model_checksum(model)
        model.batch_size = torch.Size([6])
        self.assertNotEqual(original, model_checksum(model))
        model.batch_size = torch.Size([5])
        model.state_device = torch.device('cpu')
        self.assertNotEqual(original, model_checksum(model))
        model.state_device = None
        model.weight.data.add_(1)
        self.assertNotEqual(original, model_checksum(model))

    def test_prediction_and_recorded_action_rollout(self):
        a = agent(load_planner())
        z = torch.zeros(1,2)
        actions = [torch.tensor([[1.,-.5]])]*3
        p = rollout(a.model, z, actions)
        torch.testing.assert_close(p, torch.tensor([[.6,-.3]]))
        error = prediction_error(p, torch.zeros_like(p))
        self.assertAlmostEqual(error['latent_prediction_mse'], .225, places=6)
        self.assertAlmostEqual(error['latent_prediction_squared_l2'], .45, places=6)
        with self.assertRaises(ValueError): prediction_error(p, torch.zeros(2))

    def test_distribution_decoding(self):
        cfg = types.SimpleNamespace(num_bins=3, vmin=-1., vmax=1.)
        logits = torch.tensor([[-100.,-100.,100.],[100.,-100.,-100.]])
        torch.testing.assert_close(math.two_hot_inv(logits,cfg).reshape(-1), torch.tensor([np.e-1, 1-np.e],dtype=torch.float32))
        cfg.num_bins=1
        torch.testing.assert_close(math.two_hot_inv(torch.ones(2,1),cfg), torch.full((2,1),np.e-1))

    def test_rank_ties(self):
        np.testing.assert_array_equal(ranks([4,2,2,1]), [1,2.5,2.5,4])
        result = ranking_comparison([4,2,2,1], [1,2,2,4])
        self.assertAlmostEqual(result['kendall_tau_b'], -1)
        self.assertAlmostEqual(result['spearman'], -1)
        self.assertIsNone(ranking_comparison([1,1],[1,1])['spearman'])
        self.assertEqual(paired_summary([1,2])['status'], 'INSUFFICIENT_SAMPLES')

    def test_execution_identifiability(self):
        action = np.ones(2)
        same = [dict(executed_action=np.zeros(2))]*2
        semantics, _, valid = execution_semantics(action, same, True)
        self.assertTrue(valid)
        self.assertAlmostEqual(semantics['commanded_executed_action_difference'], np.sqrt(2))
        for records, compatible in [(same[:1],True),(same,False),([same[0],dict(executed_action=action)],True)]:
            self.assertEqual(execution_semantics(action,records,compatible)[0]['executed_prediction_validity'],'NOT_IDENTIFIABLE')

    def test_config_errors(self):
        validate_analysis(options(), {'compile':False, 'multitask':'???'})
        for key, value in [('q_head_mode','avg'),('decision_stride',0),('save_latents','yes'),('rollout_horizons',[1,1])]:
            bad = options(); bad[key] = value
            with self.assertRaises(ValueError): validate_analysis(bad,{})
        with self.assertRaises(ValueError): validate_analysis(options(),{'compile':True})
        validate_analysis({'enabled':False},{'compile':True})

    def test_episode_artifacts_boundary_rng_and_frozen(self):
        a = agent(load_planner())
        cfg = a.cfg
        cfg.analysis=options(); cfg.checkpoint='fake.pt'; cfg.task='fake'
        cfg.dynamics_shift={'parameter':{'type':'action_delay'}}; cfg.schedule={'type':'fixed'}
        space=types.SimpleNamespace(shape=(2,),low=-np.ones(2),high=np.ones(2))
        digest=model_checksum(a.model)
        with tempfile.TemporaryDirectory() as path:
            analysis=MechanismAnalysis(cfg,a,path,space)
            for episode in range(2):
                analysis.begin_episode(episode,episode)
                obs=torch.zeros(2)
                for step in range(3):
                    analysis.before_action(step)
                    action=a.act(obs,t0=step==0)
                    next_obs=obs+.2*action
                    record=dict(global_native_step=step*2,applied_parameter=2,nominal_parameter=0,regime_id=0,
                                native_step=step*2,commanded_action=action.numpy(),executed_action=np.zeros(2))
                    info=dict(shift_native_steps=[record,dict(record,native_step=step*2+1)],terminated=False)
                    state=torch.random.get_rng_state().clone()
                    analysis.observe(obs,action,next_obs,1.,step==2,info,[])
                    self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
                    obs=next_obs
                analysis.end_episode(episode==0)
            analysis.finish()
            root=Path(path)/'mechanism'
            import csv
            with (root/'multistep.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),8)  # per episode: three h=1, one h=3, zero h=5
            self.assertEqual(model_checksum(a.model),digest)
            validity=json.loads((root/'validity_report.json').read_text())
            self.assertEqual(validity['exact_state_counterfactual'],'UNVERIFIED')
            self.assertTrue(validity['model_parameters_unchanged'])
            with np.load(root/'planner_candidates.npz', allow_pickle=False) as archive:
                self.assertLessEqual(len(archive['e0_d0_i0_candidate_id']),4)
                self.assertEqual(archive['e0_d0_i0_candidate_action_sequence'].shape,(3,4,2))

    def test_real_delay_queue_capture_and_disabled_record_compatibility(self):
        from benchmark_imports import package
        from test_shift_benchmark import config
        from test_dynamics_shift import Native
        env=package.ActionShiftWrapper(Native(),config('action_delay','fixed',0,2))
        env.reset()
        first=env.step(np.ones(4)*.3)[3]['shift_record']
        self.assertNotIn('analysis_delay_queue_before',first)
        env.analysis_capture_history=True
        second=env.step(np.ones(4)*.7)[3]['shift_record']
        np.testing.assert_array_equal(second['analysis_delay_queue_before'][0],np.ones(4)*.3)
        self.assertFalse(np.array_equal(second['commanded_action'],second['executed_action']))
