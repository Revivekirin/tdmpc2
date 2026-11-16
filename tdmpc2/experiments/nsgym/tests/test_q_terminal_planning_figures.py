"""Numerical contracts for the controlled forward-only diagnostic."""
import unittest
import numpy as np
from experiments.nsgym.analysis.q_terminal_planning_figures import (
    decomposition, pack_metrics, ranking_metrics, select_bank, centered_components, probability_window)


class TerminalPlanningTests(unittest.TestCase):
    def test_native_random_tape_replays_draws_independent_of_global_seed(self):
        import torch
        from common import math
        from experiments.nsgym.analysis.q_decision_relevance import RandomTape
        def sample():
            return (torch.randn(2,3),torch.randn_like(torch.zeros(4)),torch.randperm(5),math.gumbel_softmax_sample(torch.tensor([.2,.3,.5])))
        torch.manual_seed(7)
        with RandomTape() as tape:first=sample()
        torch.manual_seed(999)
        with RandomTape(tape.events):second=sample()
        for a,b in zip(first,second):torch.testing.assert_close(a,b,rtol=0,atol=0)
        with RandomTape() as selection:
            left=math.gumbel_softmax_sample(torch.tensor([1.,0.,0.]))
        with RandomTape(selection.events):
            right=math.gumbel_softmax_sample(torch.tensor([0.,1.,0.]))
        self.assertEqual(int(left),0)
        self.assertEqual(int(right),1)

    def test_constant_q_offset_is_not_action_dependent(self):
        from experiments.nsgym.analysis.q_decision_relevance import action_metrics
        x=np.array([1.,3.,2.,5.]);r=action_metrics(x,x+20,np.arange(4)[:,None])
        self.assertEqual(r['mean_shift'],20)
        self.assertEqual(r['action_dependent_sd'],0)
        self.assertEqual(r['argmax_disagreement'],0)
        self.assertAlmostEqual(r['spearman'],1)

    def test_centering_preserves_rank_and_does_not_assume_additive_medians(self):
        reward=np.array([[0.,10.,11.]])
        terminal=np.array([[10.,0.,11.]])
        total=reward+terminal
        r,q,g=centered_components(reward,terminal,total)
        np.testing.assert_array_equal(np.argsort(g),np.argsort(total))
        self.assertFalse(np.allclose(r+q,g))
        with self.assertRaises(ValueError):centered_components(reward,terminal,total+1)

    def test_zoom_retains_each_head_mass_not_only_ensemble_peak(self):
        support=np.arange(11,dtype=float)
        p=np.zeros((2,11));p[0,2]=1;p[1,8]=1
        self.assertEqual(probability_window(support,p),(1.,9.))

    def test_signed_mean_does_not_replace_statewise_absolute_change(self):
        frozen=dict(scalar=np.array([[1.,1.],[1.,1.]]),probability=np.full((2,2,3),1/3),entropy=np.ones((2,2)))
        other=dict(frozen,scalar=np.array([[2.,0.],[2.,0.]]))
        d=pack_metrics(other,frozen)
        self.assertEqual(d['signed_delta'].mean(),0.)
        self.assertEqual(d['absolute_delta'].mean(),1.)

    def test_additive_decomposition_including_distinct_dq_predictor(self):
        v={'F':np.array([10.,20.]),'D':np.array([12.,19.]),'DQ_F':np.array([13.,16.]),'DQ':np.array([15.,18.])}
        d=decomposition(v,.95**3)
        np.testing.assert_allclose(d['predictor_D'],.95**3*np.array([2.,-1.]))
        np.testing.assert_allclose(d['predictor_D_to_DQ'],.95**3*np.array([1.,-3.]))
        np.testing.assert_allclose(d['critic_DQ'],.95**3*np.array([2.,2.]))
        np.testing.assert_allclose(d['reconstruction_error'],0,atol=1e-12)

    def test_rank_reversal_and_action_change(self):
        r=ranking_metrics(np.array([1.,2.,3.]),np.array([3.,2.,1.]),np.array([[0.,0.],[1.,0.],[0.,2.]]),1)
        self.assertAlmostEqual(r['spearman'],-1.)
        self.assertEqual(r['top_k_jaccard'],0.)
        self.assertEqual(r['top_action_l2'],2.)

    def test_terminal_action_index_is_inside_bank(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bank.npz'
            a=np.arange(50,dtype=np.float32)[:,None]/50
            np.savez(p,observations=np.arange(51,dtype=np.float32)[:,None],actions=a)
            result=select_bank({('reacher','D0',2.,1):(p,'unused')},'reacher','D0',2.,[1],3,4)
            np.testing.assert_array_equal(result['starts'],[0,15,30,46])
            np.testing.assert_allclose(result['terminal_action'],a[[3,18,33,49]])
            np.testing.assert_allclose(result['tapes'][-1,:,0],a[46:49,0])

    def test_headwise_nonlinear_decode_differs_from_mixture_decode(self):
        import torch
        from common import math
        from q_planning_core import Config
        cfg=Config(num_bins=3,vmin=-2.,vmax=2.)
        logits=torch.tensor([[[-30.,30.,-30.]],[[-30.,-30.,30.]]])
        per_head=math.two_hot_inv(logits,cfg).squeeze(-1)
        mixture=math.two_hot_inv(logits.softmax(-1).mean(0).log(),cfg).squeeze(-1)
        self.assertGreater(abs(float(per_head.mean())-float(mixture)),1.)


if __name__=='__main__':unittest.main()
