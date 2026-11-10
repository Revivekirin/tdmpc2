"""Descriptive rankings; none of these metrics establish candidate correctness."""
import numpy as np


def ranks(values):
    """Descending midranks (one based); ties receive their average rank."""
    x = np.asarray(values, dtype=float).reshape(-1)
    return np.array([1 + np.sum(x > v) + (np.sum(x == v)-1)/2 for v in x])


def ranking_comparison(a, b, k=10):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if len(a) != len(b) or not len(a) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(status='UNVERIFIED_NONFINITE_OR_EMPTY')
    ra, rb = ranks(a), ranks(b)
    da, db = a[:, None]-a, b[:, None]-b
    i = np.triu_indices(len(a), 1)
    sa, sb = np.sign(da[i]), np.sign(db[i])
    denominator = np.sqrt(np.count_nonzero(sa)*np.count_nonzero(sb))
    top = lambda r: set(np.flatnonzero(r <= sorted(r)[min(k, len(r))-1]).tolist())
    ta, tb = top(ra), top(rb)
    return dict(status='VALID', spearman=float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else None,
                kendall_tau_b=float(np.sum(sa*sb)/denominator) if denominator else None,
                top_k_jaccard=len(ta & tb)/len(ta | tb),
                top1_a=np.flatnonzero(a == a.max()).tolist(), top1_b=np.flatnonzero(b == b.max()).tolist(),
                score_margin_a=float(np.sort(a)[-1]-np.sort(a)[-2]) if len(a)>1 else None,
                tie_policy='midrank; boundary ties included; top1 reports all ties')


def paired_summary(differences):
    """Episode-level percentile bootstrap with private RNG, never global sampling."""
    values = np.asarray(differences, dtype=float)
    if not len(values):
        return dict(status='INSUFFICIENT_SAMPLES', pairs=0)
    rng = np.random.default_rng(0)
    means = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return dict(status='INSUFFICIENT_SAMPLES' if len(values)<5 else 'DESCRIPTIVE_ONLY', pairs=len(values),
                mean=float(values.mean()), confidence_interval_95=np.quantile(means, [.025,.975]).tolist(),
                method='paired episode percentile bootstrap; 2000 replicates; private seed 0')
