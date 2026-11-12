"""Read-only four-method comparison; invoked by run_dynamics_q.py aggregate."""
import json
from pathlib import Path

import run_dynamics_q as dq


def key(row):
    return row['parameter'], float(row['severity']), int(row['adaptation_seed']), int(row['evaluation_seed'])


def stage(root, mode, c, method, audit_hash=None):
    root = Path(root)
    done, manifest = dq.read(root/'completion.json'), dq.read(root/'manifest.json')
    assert done['status'] == 'PASS', f'{root}: not complete; failed={done.get("failed")}'
    assert manifest['mode'] == mode and manifest['checkpoint_sha256'] == dq.ttt.SHA
    assert not manifest.get('smoke', False)
    if method != 'joint':
        assert manifest['evaluation_seeds'] == c['evaluation_seeds']
    if method == 'ttt':
        _, reference_config = dq.configuration()
        assert manifest['configuration'] == reference_config
    if method == 'dq':
        assert manifest['configuration'] == c and manifest['audit_sha256'] == audit_hash
    result = {}
    for identity in dq.identities(c):
        label = dq.name(identity)
        if method == 'ttt':
            label = dq.evaluation.label(identity)
        elif method == 'joint':
            label = label.replace('dynamics-q', 'full-joint')
        child = root/label
        completion = dq.read(child/'completion.json')
        assert completion['status'] == 'PASS'
        if method == 'dq':
            assert completion['csv_sha256'] == dq.digest(child/f'{mode}.csv')
        if method == 'ttt':
            cm = dq.read(child/'manifest.json')
            assert cm['checkpoint_sha256'] == dq.ttt.SHA and cm['mode'] == mode
        data = dq.rows(child/f'{mode}.csv')
        assert len(data) == 20
        assert {int(x['evaluation_seed']) for x in data} == set(c['evaluation_seeds'])
        for row in data:
            k = key(row)
            assert k[:3] == (identity['parameter'], identity['severity'], identity['adaptation_seed'])
            assert k not in result
            result[k] = row
    assert len(result) == 720
    return result


def aggregate(a, c):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit = dq.read(a.audit)
    assert audit['status'] == 'PASS' and audit['configuration'] == c and audit['checkpoint_sha256'] == dq.ttt.SHA
    assert len(audit['inventory']) == 36
    h = dq.digest(a.audit)
    prediction = stage(a.prediction,'prediction',c,'dq',h)
    control = stage(a.shifted,'shifted',c,'dq',h)
    ttt_prediction = stage(a.ttt_final/'prediction','prediction',c,'ttt')
    ttt_control = stage(a.ttt_final/'shifted','shifted',c,'ttt')
    joint_control = stage(a.joint_final/'shifted','shifted',c,'joint')
    nominal = stage(a.nominal,'nominal',c,'dq',h) if a.nominal else None
    assert prediction.keys() == control.keys() == ttt_prediction.keys() == ttt_control.keys() == joint_control.keys()
    paired = []
    for k, row in control.items():
        d, j, p, dp = ttt_control[k], joint_control[k], prediction[k], ttt_prediction[k]
        f = float(row['frozen_return'])
        assert abs(f-float(d['frozen_return'])) < 1e-6 and abs(f-float(j['frozen_return'])) < 1e-6
        assert p['bank_sha256'] == dp['bank_sha256'] and p['bank_file'] == dp['bank_file']
        assert abs(float(p['frozen_latent_mse'])-float(dp['frozen_latent_mse'])) < 1e-12
        qret, dret, jret = float(row['episode_return']), float(d['decision_return']), float(j['joint_return'])
        paired.append(dict(parameter=k[0],severity=k[1],adaptation_seed=k[2],evaluation_seed=k[3],
            nominal_label_duplicate=k[1]==1 and k[0]=='torso_mass',
            frozen_return=f,ttt_return=dret,dq_return=qret,joint_return=jret,
            dq_minus_ttt=qret-dret,dq_minus_frozen=qret-f,joint_minus_dq=jret-qret,
            ttt_minus_frozen=dret-f,joint_minus_frozen=jret-f,
            frozen_latent_mse=float(p['frozen_latent_mse']),ttt_latent_mse=float(dp['decision_latent_mse']),
            dq_latent_mse=float(p['latent_mse']),dq_mse_improvement=float(p['frozen_latent_mse'])-float(p['latent_mse']),
            ttt_mse_improvement=float(p['frozen_latent_mse'])-float(dp['decision_latent_mse']),
            dq_forward_velocity=float(row['forward_velocity']),dq_control_cost=float(row['control_cost']),
            dq_action_saturation=float(row['action_saturation']),nominal_parameter=float(row['nominal_parameter'])))
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    figures, reports = out/'figures', out/'reports'
    figures.mkdir(); reports.mkdir()
    dq.table(out/'per_seed_heldout_prediction.csv',list(prediction.values()))
    dq.table(out/'per_seed_heldout_control.csv',list(control.values()))
    dq.table(out/'four_method_paired_comparison.csv',paired)
    dq.table(out/'adaptation_run_inventory.csv',audit['inventory'])
    if nominal:
        dq.table(out/'per_seed_nominal_retention.csv',list(nominal.values()))
    metrics = ['frozen_return','ttt_return','dq_return','joint_return','dq_minus_ttt','dq_minus_frozen',
               'joint_minus_dq','ttt_minus_frozen','joint_minus_frozen','frozen_latent_mse',
               'ttt_latent_mse','dq_latent_mse','dq_mse_improvement','ttt_mse_improvement']
    streams = []
    for i in dq.identities(c):
        rr = [r for r in paired if (r['parameter'],r['severity'],r['adaptation_seed']) ==
              (i['parameter'],i['severity'],i['adaptation_seed'])]
        assert len(rr) == 20
        streams.append(dict(**i,n_evaluation_seeds=20,nominal_parameter=rr[0]['nominal_parameter'],
                            **{m:float(np.mean([r[m] for r in rr])) for m in metrics}))
    dq.table(out/'per_adaptation_seed_comparison.csv',streams)
    summary = []
    for cond in c['conditions']:
        rr = [r for r in streams if r['parameter']==cond['parameter'] and r['severity']==cond['severity']]
        for metric in metrics:
            values = [r[metric] for r in rr]
            sampling_unit = 'adaptation_seed'
            if metric in ('frozen_return','frozen_latent_mse'):
                # The repeated baseline is one set of 20 episodes, not 3 models.
                shared = {r['evaluation_seed']:r[metric] for r in paired
                          if r['parameter']==cond['parameter'] and r['severity']==cond['severity']}
                assert len(shared)==20
                values = list(shared.values())
                sampling_unit = 'evaluation_seed_shared_frozen_baseline'
            stats = dq.evaluation.unit_stats(values)
            if sampling_unit != 'adaptation_seed':
                stats['n_adaptation_seeds'] = 0
            q1,q3 = np.quantile(values,[.25,.75]); iqr=q3-q1
            summary.append(dict(**cond,metric=metric,**stats,sampling_unit=sampling_unit,n_evaluation_seeds=20,
                                positive_streams=sum(v>0 for v in values) if sampling_unit=='adaptation_seed' else None,
                                min=min(values),max=max(values),tukey_outlier_count=sum(v<q1-1.5*iqr or v>q3+1.5*iqr for v in values)))
    dq.table(out/'condition_summary.csv',summary)
    budgets = [dict(method='Frozen',interactions=3000,dynamics_steps=0,q_steps=0,total_optimizer_steps=0,
                    collection='eval_mode=True',reward_supervision=False,replay='control stream; no learning'),
               dict(method='Dynamics-only',interactions=3000,dynamics_steps=118,q_steps=0,total_optimizer_steps=118,
                    collection='eval_mode=True',reward_supervision=False,replay='fresh one-step; capacity=10000'),
               dict(method='Dynamics+Q',interactions=3000,dynamics_steps=118,q_steps=118,total_optimizer_steps=236,
                    collection='eval_mode=True',reward_supervision=True,replay='fresh one-step; final size=3000'),
               dict(method='Full Joint',interactions=3000,dynamics_steps=2001,q_steps=2001,total_optimizer_steps=4002,
                    collection='eval_mode=False',reward_supervision=True,replay='completed episodes; model and Q share optimizer')]
    dq.table(out/'update_interaction_budget.csv',budgets)
    lookup = {(r['parameter'],r['severity'],r['metric']):r for r in summary}
    families = ['bthigh_damping','torso_mass']; grid=sorted({x['severity'] for x in c['conditions']})

    def plot(filename, plot_metrics, ylabel):
        fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for ax,param in zip(axes,families):
            nominal_value = next(r['nominal_parameter'] for r in streams if r['parameter']==param)
            for metric,label in plot_metrics:
                rr = [lookup[param,k,metric] for k in grid]
                means = np.array([r['mean'] for r in rr])
                if metric=='frozen_return':
                    ax.plot(grid,means,'o-',label=label+' (shared baseline)')
                else:
                    lo = np.maximum(0,means-np.array([r['ci_low'] for r in rr]))
                    hi = np.maximum(0,np.array([r['ci_high'] for r in rr])-means)
                    ax.errorbar(grid,means,yerr=[lo,hi],fmt='o-',capsize=3,label=label)
            ax.axvline(1,color='gray',linestyle=':',alpha=.6)
            ax.set(title=f'{param}\nnominal={nominal_value:g}; actual=k × nominal',xlabel='Severity k',ylabel=ylabel)
            ax.set_xticks(grid); ax.legend(fontsize=8)
        fig.savefig(figures/filename,dpi=180); plt.close(fig)

    plot('severity_control_return.png',[(m+'_return',label) for m,label in
        [('frozen','Frozen'),('ttt','Dynamics-only'),('dq','Dynamics+Q'),('joint','Full Joint')]],'Held-out return; seed bootstrap 95% CI')
    plot('severity_frozen_relative.png',[(m+'_minus_frozen',label) for m,label in
        [('ttt','Dynamics-only'),('dq','Dynamics+Q'),('joint','Full Joint')]],'Paired return − Frozen')
    plot('latent_mse_improvement.png',[('ttt_mse_improvement','Dynamics-only'),('dq_mse_improvement','Dynamics+Q')],
         'Frozen MSE − adapted MSE (fixed nominal encoder)')
    plot('dynamics_q_vs_ttt.png',[('dq_minus_ttt','Dynamics+Q − Dynamics-only')],'Paired held-out return difference')
    plot('joint_vs_dynamics_q.png',[('joint_minus_dq','Joint − Dynamics+Q')],'Paired held-out return difference')
    # Log traces are descriptive; never use time steps as independent replicates.
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    training_paths = []
    for item in audit['inventory']:
        log = Path(item['snapshot']).parent/'training.jsonl'
        training_paths.append(log)
        if not log.exists():
            continue
        with log.open() as f:
            data = [json.loads(x) for x in f]
        data = [r for r in data if r['metrics']]
        for ax,metric in zip(axes.flat,['dynamics_loss','q_loss','dynamics_gradient_norm','q_gradient_norm']):
            ax.plot([r['global_step'] for r in data],[r['metrics'][metric] for r in data],alpha=.3)
            ax.set(xlabel='Adaptation transitions',ylabel=metric)
    available = sum(p.exists() for p in training_paths)
    fig.suptitle(f'Training diagnostics: {available}/36 stream logs; lines are not independent samples')
    if not available:
        axes[0,0].text(.1,.5,'NOT_AVAILABLE: original training.jsonl required',transform=axes[0,0].transAxes)
    fig.savefig(figures/'dynamics_q_training_diagnostics.png',dpi=180);plt.close(fig)
    caveats = (
        'Each physical condition has three adaptation seeds; each stream averages 20 matched held-out seeds. '
        'CI is a bootstrap of adaptation-stream means (n=3), not independent time steps or 60 independent models. '
        'Frozen rows are shared references, not three independent Frozen training replicates. '
        'Nominal k=1 family labels are physically equivalent and must not be pooled as independent tasks. '
        'Negative effects are retained; all 36 streams are required. Tukey counts at n=3 are descriptive only.\n\n'
        'Dynamics+Q uses real reward supervision, two fresh Adam optimizers, a one-step actual-latent Q objective, '
        'and 118+118 steps. Dynamics-only uses 118 predictor steps. Online trajectories and Torch RNG usage diverge '
        'after updates even with matched seeds. Joint uses 2001 model plus 2001 actor steps, restored Adam, '
        'imagined 3-step states and exploration noise. This is a procedure-level comparison; improved Dynamics+Q '
        'does not causally prove Q mismatch. Joint self-consistency MSE has a changed encoder and is not placed '
        'on the fixed-encoder MSE comparison plot.\n')
    lines = []
    for cond in c['conditions']:
        r = lookup[cond['parameter'],cond['severity'],'dq_minus_ttt']
        lines.append(f"- {cond['parameter']} ×{cond['severity']:g}: D+Q−D mean {r['mean']:.4f}, median {r['median']:.4f}, "
                     f"95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}], improved {r['positive_streams']}/3.")
    (reports/'DYNAMICS_Q_VS_TTT_JOINT.md').write_text('# Four-method comparison\n\n'+ '\n'.join(lines)+'\n\n'+caveats)
    (reports/'DYNAMICS_Q_HELDOUT_RESULTS.md').write_text(
        '# Held-out results\n\nPrediction: 720 rows; shifted control: 720 rows; 36 final snapshots. '
        f'Nominal retention: {"PASS (720 rows)" if nominal else "NOT_EVALUATED"}. '
        f'Training diagnostic logs: {available}/36.\n\n'+caveats)
    if available != 36:
        (reports/'MISSING_DATA.md').write_text('Missing training diagnostics (global_step, metrics.dynamics_loss, '
            'metrics.q_loss, metrics.dynamics_gradient_norm, metrics.q_gradient_norm):\n\n'+
            '\n'.join(str(p) for p in training_paths if not p.exists()))
    dq.write(out/'manifest.json',dict(audit_sha256=h,configuration=c,checkpoint_sha256=dq.ttt.SHA,
        inputs={str(p):dq.digest(p/'completion.json') for p in
                [a.prediction,a.shifted,a.ttt_final/'prediction',a.ttt_final/'shifted',a.joint_final/'shifted']}))
    dq.write(out/'completion.json',dict(status='PASS',streams=36,paired_rows=720,training_logs=available,
        nominal_retention='PASS' if nominal else 'NOT_EVALUATED'))
