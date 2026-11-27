#!/usr/bin/env python3
"""Four read-only follow-up questions on completed Dynamics+Q results. No CUDA."""
from . import schema as _io
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


read = _io.read_required


rows = _io.list_rows


def save(path, data):
    return _io.save_table(path, data, empty='skip')


digest = _io.sha256


def key(r):
    return r['parameter'], float(r['severity']), int(r['adaptation_seed'])


def direction(x, tolerance):
    return 'improved' if x > tolerance else 'worsened' if x < -tolerance else 'neutral'


def evidence(r):
    """Positive x and y both mean improvement; raw differences are retained."""
    result = dict(r)
    for label, mse, ret in [('ttt','ttt_latent_mse','ttt_return'),
                            ('frozen','frozen_latent_mse','frozen_return')]:
        x = r[mse] - r['dq_latent_mse']; y = r['dq_return'] - r[ret]
        mx = max(1e-8, .001*abs(r[mse])); my = max(1e-6, .001*max(1.,abs(r[ret])))
        result.update({f'mse_improvement_vs_{label}':x, f'return_improvement_vs_{label}':y,
            f'mse_neutral_band_vs_{label}':mx, f'return_neutral_band_vs_{label}':my,
            f'prediction_direction_vs_{label}':direction(x,mx),
            f'control_direction_vs_{label}':direction(y,my)})
    result['ttt_above_frozen'] = r['ttt_return'] > r['frozen_return']
    result['dq_above_frozen'] = r['dq_return'] > r['frozen_return']
    result['ttt_gain_lost_below_frozen'] = result['ttt_above_frozen'] and not result['dq_above_frozen']
    result['dq_minus_joint'] = r['dq_return'] - r['joint_return']
    result['worsened_vs_ttt'] = r['dq_return'] < r['ttt_return']
    result['worsened_vs_frozen'] = r['dq_return'] < r['frozen_return']
    return result


METRICS = ['dynamics_loss','q_loss','q_prediction','q_target',
           'dynamics_gradient_norm','q_gradient_norm']


def load_trace(path, identity):
    trace = []; count = 0
    with path.open() as f:
        for line in f:
            r = json.loads(line); count += 1
            assert r['global_step'] == count
            assert r['adaptation_seed'] == identity[2]
            expected = max(0, count//25-2)
            assert r['dynamics_steps'] == r['q_steps'] == expected
            m = r['metrics']
            assert bool(m) == (count >= 75 and count%25 == 0)
            if m:
                assert all(math.isfinite(float(m[k])) for k in METRICS)
                trace.append(dict(global_step=count,update=r['q_steps'],**{k:float(m[k]) for k in METRICS},
                    mean_target_minus_prediction=float(m['q_target'])-float(m['q_prediction'])))
    assert count == 3000 and len(trace) == 118
    return trace


def trace_summary(trace):
    result = {'updates':len(trace),'comparison_window_updates':20}
    for k in METRICS+['mean_target_minus_prediction']:
        result.update({f'{k}_first20_mean':mean(r[k] for r in trace[:20]),
                       f'{k}_last20_mean':mean(r[k] for r in trace[-20:]),
                       f'{k}_last20_median':median(r[k] for r in trace[-20:]),
                       f'{k}_max':max(r[k] for r in trace)})
    result['dynamics_preclip_norm_over_10'] = sum(r['dynamics_gradient_norm']>10 for r in trace)
    result['q_preclip_norm_over_20'] = sum(r['q_gradient_norm']>20 for r in trace)
    return result


def inspect(comparison, runs, output):
    assert read(comparison/'completion.json')['status']=='PASS'
    source = rows(comparison/'per_adaptation_seed_comparison.csv')
    paired = rows(comparison/'four_method_paired_comparison.csv')
    summaries = rows(comparison/'condition_summary.csv')
    fields = ['frozen_return','ttt_return','dq_return','joint_return',
              'frozen_latent_mse','ttt_latent_mse','dq_latent_mse','nominal_parameter']
    expected = {(p,k,s) for p in ('bthigh_damping','torso_mass') for k in (.5,.75,1.,1.25,1.5,2.)
                for s in (42000001,42000002,42000003)}
    assert len(source)==36 and {key(r) for r in source}==expected
    assert len(paired)==720 and len({(*key(r),int(r['evaluation_seed'])) for r in paired})==720
    all_streams=[]
    for raw in source:
        k=key(raw); p=[r for r in paired if key(r)==k]
        assert len(p)==20 and {int(r['evaluation_seed']) for r in p}==set(range(43000001,43000021))
        r=dict(parameter=k[0],severity=k[1],adaptation_seed=k[2],**{f:float(raw[f]) for f in fields})
        assert all(math.isfinite(r[f]) for f in fields)
        for f in fields:
            assert math.isclose(r[f],mean(float(x[f]) for x in p),abs_tol=1e-8,rel_tol=1e-10)
        all_streams.append(evidence(r))
    summary_lookup={(r['parameter'],float(r['severity']),r['metric']):r for r in summaries}
    conditions=[]
    for p,k in sorted({(r['parameter'],r['severity']) for r in all_streams}):
        subset=[r for r in all_streams if (r['parameter'],r['severity'])==(p,k)]
        row=dict(parameter=p,severity=k,nominal_parameter=subset[0]['nominal_parameter'],
            ttt_above_frozen=sum(r['ttt_above_frozen'] for r in subset),
            dq_above_frozen=sum(r['dq_above_frozen'] for r in subset),
            ttt_gain_lost_below_frozen=sum(r['ttt_gain_lost_below_frozen'] for r in subset),
            dq_above_joint=sum(r['dq_minus_joint']>0 for r in subset),
            dq_above_ttt=sum(r['return_improvement_vs_ttt']>0 for r in subset))
        for metric,label,sign in [('dq_minus_frozen','dq_minus_frozen',1),('dq_minus_ttt','dq_minus_ttt',1),
                                  ('joint_minus_dq','dq_minus_joint',-1)]:
            s=summary_lookup[p,k,metric]
            row[label+'_mean']=sign*float(s['mean']);row[label+'_median']=sign*float(s['median'])
            row[label+'_ci_low']=float(s['ci_low']) if sign==1 else -float(s['ci_high'])
            row[label+'_ci_high']=float(s['ci_high']) if sign==1 else -float(s['ci_low'])
        conditions.append(row)
    output.mkdir(parents=True,exist_ok=False)
    figdir=output/'figures';figdir.mkdir()
    save(output/'condition_comparison.csv',conditions)
    save(output/'stream_evidence.csv',all_streams)
    traces={}; diagnostics=[];updates=[];missing=[]
    fingerprints={str(comparison/n):digest(comparison/n) for n in
        ('completion.json','per_adaptation_seed_comparison.csv','four_method_paired_comparison.csv','condition_summary.csv')}
    for r in all_streams:
        label=f"{r['parameter']}-k{r['severity']:g}-dynamics-q-seed{r['adaptation_seed']}"
        path=runs/label/'training.jsonl'
        if not path.exists():
            missing.append(dict(stream=label,path=str(path),required_fields=','.join(METRICS)))
            continue
        trace=load_trace(path,key(r));traces[key(r)]=trace;fingerprints[str(path)]=digest(path)
        diagnostics.append(dict(parameter=r['parameter'],severity=r['severity'],adaptation_seed=r['adaptation_seed'],
            dq_minus_ttt=r['return_improvement_vs_ttt'],dq_minus_frozen=r['return_improvement_vs_frozen'],
            **trace_summary(trace)))
        updates.extend(dict(parameter=r['parameter'],severity=r['severity'],adaptation_seed=r['adaptation_seed'],**x) for x in trace)
    save(output/'training_trace_summary.csv',diagnostics);save(output/'all_update_diagnostics.csv',updates)
    save(output/'missing_training_logs.csv',missing)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    shifted=[r for r in all_streams if r['severity']!=1]
    for ax,reference in zip(axes,('frozen','ttt')):
        for parameter in ('bthigh_damping','torso_mass'):
            sub=[r for r in shifted if r['parameter']==parameter]
            ax.scatter([r[f'mse_improvement_vs_{reference}'] for r in sub],
                       [r[f'return_improvement_vs_{reference}'] for r in sub],label=parameter)
            for r in sub:
                ax.annotate(f"{r['severity']:g}/{r['adaptation_seed']%100}",
                    (r[f'mse_improvement_vs_{reference}'],r[f'return_improvement_vs_{reference}']),fontsize=6)
        ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
        ax.set(xlabel=f'{reference} MSE − D+Q MSE',ylabel=f'D+Q return − {reference} return',
               title='Shifted streams only; labels: severity/seed suffix')
        ax.legend(fontsize=8)
    fig.savefig(figdir/'prediction_control_alignment.png',dpi=180);plt.close(fig)
    # Plot every adaptation replicate of any affected condition, so improved
    # same-condition streams remain visible alongside the worsened ones.
    affected={(r['parameter'],r['severity']) for r in all_streams if r['worsened_vs_ttt'] or r['worsened_vs_frozen']}
    for parameter,severity in sorted(affected):
        subset=[r for r in all_streams if (r['parameter'],r['severity'])==(parameter,severity)]
        available=sum(key(r) in traces for r in subset)
        if not available:continue
        fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
        for r in subset:
            if key(r) not in traces:continue
            trace=traces[key(r)];x=[v['global_step'] for v in trace]
            color='tab:red' if r['worsened_vs_ttt'] else 'tab:blue'
            label=f"seed {r['adaptation_seed']} (D+Q−D={r['return_improvement_vs_ttt']:.1f})"
            for ax,metric in zip(axes.flat,['dynamics_loss','q_loss','q_prediction',
                                          'mean_target_minus_prediction','dynamics_gradient_norm','q_gradient_norm']):
                ax.plot(x,[v[metric] for v in trace],color=color,label=label,alpha=.7)
                if metric=='q_prediction':
                    ax.plot(x,[v['q_target'] for v in trace],color=color,ls='--',alpha=.7,label='target '+str(r['adaptation_seed']))
                ax.set(xlabel='Adaptation transition',ylabel=metric)
            axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=6)
        axes[2,0].axhline(10,color='black',ls=':');axes[2,1].axhline(20,color='black',ls=':')
        fig.suptitle(f'{parameter} ×{severity:g}; nominal={subset[0]["nominal_parameter"]:g}; '
                     f'traces {available}/3\nred: return worsened vs D; blue: other; Q values are pre-update batch means')
        fig.savefig(figdir/f'{parameter}-k{severity:g}-training.png',dpi=150);plt.close(fig)
    lines=['# D+Q follow-up analysis','',
        'All 36 labeled streams retained; overall shifted counts exclude both k=1 labels (30 streams). '
        'These are descriptive counts across a fixed grid, not 30 independent task draws. '
        'Per-condition uncertainty uses three adaptation seeds, each averaged over 20 matched evaluation seeds.','',
        '## 1. Preservation of gains over Frozen','',
        f"D above Frozen: {sum(r['ttt_above_frozen'] for r in shifted)}/30; "
        f"D+Q above Frozen: {sum(r['dq_above_frozen'] for r in shifted)}/30. "
        f"D-positive streams falling to/below Frozen after D+Q: {sum(r['ttt_gain_lost_below_frozen'] for r in shifted)}.",'',
        '## 2. Position relative to Joint','',
        f"D+Q above Joint: {sum(r['dq_minus_joint']>0 for r in shifted)}/30 shifted streams.",'']
    for r in conditions:
        lines.append(f"- {r['parameter']} ×{r['severity']:g}: D+Q−F={r['dq_minus_frozen_mean']:.3f}, "
                     f"D+Q−D={r['dq_minus_ttt_mean']:.3f}, D+Q−Joint={r['dq_minus_joint_mean']:.3f}; "
                     f"above Frozen {r['dq_above_frozen']}/3, above Joint {r['dq_above_joint']}/3.")
    lines+=['','## 3. Prediction/control agreement','',
        'Neutral bands: MSE=max(1e-8, 0.1% × reference MSE), return=max(1e-6, '
        '0.1% × max(1,|reference return|)). Directions are descriptive, not significance tests.']
    for reference in ('frozen','ttt'):
        counts={}
        for r in shifted:
            label=r[f'prediction_direction_vs_{reference}']+'/'+r[f'control_direction_vs_{reference}']
            counts[label]=counts.get(label,0)+1
        lines.append(f'- vs {reference}, prediction/control: {json.dumps(counts,sort_keys=True)}')
    lines+=['','## 4. Training trajectories of worsened streams','',
        f'Traces available: {len(traces)}/36. All update records are retained; summary compares first and last '
        '20 updates. Plots retain improved replicas of the same physical condition as context. '
        'No time step is treated as an independent statistical replicate.','',
        '`q_prediction` is the pre-update mean over the 64 replay samples and five online Q heads. '
        '`q_target` is the sampled batch target mean. Their difference is a **signed difference of means**, '
        'not mean absolute/squared TD error, calibration error, or measured overestimation bias. '
        'Changing replay distributions prevent interpreting a falling training loss as held-out improvement. '
        'Gradient norms are pre-clipping and include loss coefficients (dynamics 1, Q 0.1). '
        'Final-snapshot returns cannot identify the update at which control deteriorated.','',
        'Nominal k=1 adaptation results do not measure retention after nonnominal adaptation. '
        'No new causal claim about Q mismatch follows from these associations.']
    for d in diagnostics:
        if d['dq_minus_ttt'] >= 0 and d['dq_minus_frozen'] >= 0:continue
        lines.append(f"- {d['parameter']} ×{d['severity']:g}, seed {d['adaptation_seed']}: "
            f"Δreturn vs D={d['dq_minus_ttt']:.3f}, vs F={d['dq_minus_frozen']:.3f}; "
            + '; '.join(f"{m} first20→last20 mean: {d[m+'_first20_mean']:.6g}→{d[m+'_last20_mean']:.6g}"
                        for m in METRICS)
            + f"; preclip norm above threshold: D {d['dynamics_preclip_norm_over_10']}/118, "
              f"Q {d['q_preclip_norm_over_20']}/118.")
    report='\n'.join(lines)+'\n';(output/'FOLLOWUP_ANALYSIS.md').write_text(report)
    (output/'manifest.json').write_text(json.dumps(dict(inputs=fingerprints,source_sha256=digest(Path(__file__)),
        comparison=str(comparison.resolve()),runs=str(runs.resolve())),indent=2))
    (output/'completion.json').write_text(json.dumps(dict(status='PASS' if not missing else 'PARTIAL',
        stream_comparison='PASS',streams=36,training_logs=len(traces),missing_logs=missing),indent=2))
    print(report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inspect(a.comparison,a.runs,a.output)


