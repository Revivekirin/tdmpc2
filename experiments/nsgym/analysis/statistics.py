"""Seed-level statistics with explicit historical policies (never time-step bootstrap)."""
import numpy as np

POLICIES = {'halfcheetah': (20261111, 10000), 'reacher': (20261112, 5000),
            'budgets': (90210, 10000), 'q': (817, 2000)}


def bootstrap(values, policy='halfcheetah', *, seed=None, single=False):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Missing or nonfinite sampling-unit measurements')
    if len(a) == 1 and not single:
        return None, None
    fixed_seed, draws = POLICIES[policy]
    rng = np.random.default_rng(fixed_seed if seed is None else seed)
    means = a[rng.integers(len(a), size=(draws, len(a)))].mean(axis=1)
    return tuple(map(float, np.quantile(means, [.025, .975])))


def summary(values, policy='halfcheetah', *, single=False):
    a = np.asarray(values, dtype=float)
    lo, hi = bootstrap(a, policy, single=single)
    q1, q3 = np.quantile(a, [.25, .75])
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std(ddof=1)) if len(a)>1 else None, ci_low=lo, ci_high=hi,
                outliers=int(((a < q1-1.5*(q3-q1)) | (a > q3+1.5*(q3-q1))).sum()))
