import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_frozen import aggregate


def fixture_rows():
    rows=[]
    for scale in [.7,1.,1.3]:
        for seed in range(10):
            r=dict(seed=seed,mass_scale=scale,episode_return=-10+seed+(scale-1)*10)
            for k in ['torso_mass','forward_velocity','velocity_std','reward_run','reward_control','episode_length','control_effort','action_abs_mean','action_std','action_change_rms','terminated','truncated']:r[k]=0.
            rows.append(r)
    return rows


def test_paired_bootstrap_and_negative_denominator():
    c=dict(mass_scales=[.7,1.,1.3],bootstrap_seed=6,bootstrap_repeats=1000)
    rows=fixture_rows(); a=aggregate(rows,c)
    assert a==aggregate(rows,c)
    assert a[0]['paired_return_difference']==pytest.approx(-3)
    assert a[0]['paired_difference_ci_low']==pytest.approx(-3)
    assert all(r['relative_return_change'] is None for r in a)
    with pytest.raises(ValueError):aggregate(rows[:-1],c)
