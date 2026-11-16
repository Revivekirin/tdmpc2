"""Critic intervention and paired aggregation contracts; no simulator."""
from types import SimpleNamespace
import pandas as pd
import pytest
from experiments.nsgym.analysis.control_q.reporting.aggregate import return_bridge,hierarchical_interval
from experiments.nsgym.analysis.control_q.provenance import relocate


def test_paired_bridge_uses_streams_and_matched_seed():
    rows=[];historical=[]
    for stream in (1,2,3):
        for seed in (10,11):
            key=dict(task='halfcheetah',family='damping',severity=2.,adaptation_seed=stream,evaluation_seed=seed)
            rows.extend([dict(**key,method='DQ',episode_return=100+stream),dict(**key,method='DQ_F',episode_return=80+stream)])
            historical.append(dict(**key,method='D',**{'return':70+stream}))
    paired,summary=return_bridge(pd.DataFrame(rows).sample(frac=1,random_state=2),pd.DataFrame(historical))
    assert (paired.critic==20).all() and (paired.predictor==10).all() and (paired.total==30).all()
    assert (summary.adaptation_streams==3).all() and (summary.positive_streams==3).all()
    assert (summary.ci_low==summary.ci_high).all()
    with pytest.raises(ValueError,match='Duplicate'):return_bridge(pd.DataFrame(rows+rows[:1]),pd.DataFrame(historical))
    with pytest.raises(ValueError,match='Unpaired'):return_bridge(pd.DataFrame(rows[1:]),pd.DataFrame(historical))


def test_hierarchical_interval_reproducible():
    frame=pd.DataFrame({'adaptation_seed':[1,1,2,2,3,3],'x':[1,2,3,4,-5,6]})
    assert hierarchical_interval(frame,'x',repetitions=50)==hierarchical_interval(frame,'x',repetitions=50)


def test_historical_relocation_never_uses_filename_alone(tmp_path):
    assert relocate('/old/results/nsgym/a/b.csv',tmp_path)==tmp_path/'a/b.csv'
    with pytest.raises(ValueError):relocate('/other/b.csv',tmp_path)


def test_online_swap_preserves_target_and_source():
    import torch
    from copy import deepcopy
    from experiments.nsgym.analysis.control_q.diagnostics.q_swap import swap_online_critic
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.cfg=SimpleNamespace(num_bins=3,vmin=-1,vmax=1)
            for name in ('_encoder','_dynamics','_reward','_pi','_Qs','_target_Qs_params'):setattr(self,name,torch.nn.Linear(2,2))
    f=Model();dq=deepcopy(f)
    with torch.no_grad():dq._Qs.weight.add_(1);dq._dynamics.weight.add_(1);dq._target_Qs_params.weight.add_(2)
    swapped,hashes=swap_online_critic(dq,f)
    assert hashes['swapped']['q']==hashes['nominal']['q']
    assert hashes['swapped']['target_q']==hashes['adapted']['target_q']
    assert hashes['adapted']['q']!=hashes['swapped']['q']
    with torch.no_grad():swapped._Qs.weight.add_(1)
    assert not torch.equal(swapped._Qs.weight,f._Qs.weight)
