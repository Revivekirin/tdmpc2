"""Numerical contracts for the read-only final figure pipeline."""
import numpy as np
import pytest

from experiments.nsgym.analysis.control_q.data import paired_terminal_arrays
from experiments.nsgym.analysis.control_q.metrics import ci, probability_js


def _array(path, corrupt=False):
    q=np.arange(80,dtype=np.float32)
    probs=np.tile(np.array([.5,.5],dtype=np.float32),(2,80,1))
    delta=np.linspace(-1,1,32,dtype=np.float32)[None,:]
    before=np.tile(np.arange(32,dtype=np.float32),(80,1))
    after=before+delta
    if corrupt:after[0,0]+=1
    fields=dict(evaluation_seeds=np.arange(20),candidate_ids=np.arange(32),
                F_immediate_scalar=np.zeros((2,80)),D_immediate_scalar=np.zeros((2,80)),
                DQ_F_scores=before,DQ_scores=after,DQ_F_terminal_scores=np.zeros_like(before),
                DQ_terminal_scores=np.broadcast_to(delta,before.shape),
                DQ_F_rewards=before,DQ_rewards=before, effect_critic_DQ=np.zeros(80))
    for name in ('predictor_D','predictor_D_to_DQ','total_DQ_minus_F'):
        fields['effect_'+name]=np.zeros(80)
    for method in ('F','D','DQ_F','DQ','J'):
        fields[method+'_terminal_scalar']=np.stack((q,q+1))
    np.savez(path,**fields)


def test_fixed_candidate_score_identity_and_immediate_invariant(tmp_path):
    p=tmp_path/'saved.npz';_array(p)
    a=paired_terminal_arrays(p)
    np.testing.assert_allclose(a['score']['DQ']-a['score']['DQ_F'],
                               a['terminal']['DQ']-a['terminal']['DQ_F'],atol=1e-6)
    _array(p,corrupt=True)
    with pytest.raises(ValueError,match='score identity'):paired_terminal_arrays(p)


def test_hierarchical_interval_and_js_are_reproducible():
    # Pinned output of the deleted historical helper at revision 99429838.
    matrix=np.arange(60,dtype=float).reshape(3,20)/7
    np.testing.assert_allclose(ci(matrix,'audit'),
                               (4.2142857142857135,1.4355357142857141,6.936071428571429),rtol=0,atol=1e-12)
    p=np.array([[.2,.8],[.7,.3]])
    np.testing.assert_array_equal(probability_js(p,p),np.zeros(2))
    np.testing.assert_allclose(probability_js(p,1-p),(.19274476,.08228288),atol=1e-8)
