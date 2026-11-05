"""CSV-first Stage 3B aggregation and Matplotlib figures (no simulator imports)."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

METRICS = [('shifted_success', 'Shifted success (%)', 100),
           ('fixed_data_mse', 'Fixed-data one-step latent MSE', 1),
           ('nominal_success', 'Nominal success (%)', 100)]
FIGURES = ['shifted_success_by_adaptation', 'fixed_data_mse_by_adaptation',
           'nominal_retention_by_adaptation']


def write_csv(path, rows, fields=None):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values)>1 else None,
                n_seed_sets=len(values))


def normalized_auc(budgets, values):
    if budgets[0] != 0 or budgets[-1] <= 0 or any(b <= a for a,b in zip(budgets,budgets[1:])):
        raise ValueError('AUC requires increasing budgets starting at zero.')
    return sum((b-a)*(x+y)/2 for a,b,x,y in zip(budgets,budgets[1:],values,values[1:])) / budgets[-1]


def aggregate(raw):
    # First average held-out episodes within each independent seed set.
    by_seed = defaultdict(list)
    for r in raw:
        if r['value'] is not None:
            key = tuple(r[k] for k in ('run_id','task','checkpoint','adaptation','shift_type',
                'shift_magnitude','schedule','budget_native_steps','metric','seed_set'))
            by_seed[key].append(float(r['value']))
    groups = defaultdict(list)
    for key, values in by_seed.items():
        groups[key[:-1]].append(statistics.mean(values))
    keys = ('run_id','task','checkpoint','adaptation','shift_type','shift_magnitude','schedule',
            'budget_native_steps','metric')
    return [dict(zip(keys, key), **mean_sd(values)) for key,values in sorted(groups.items())], by_seed


def analyze(root, output, suite, allow_partial=False, recovered_root=None):
    import matplotlib
    matplotlib.use('Agg')
    raw, costs, issues, seen, references, datasets, seed_controls = [], [], [], {}, {}, {}, {}
    checkpoints, run_ids = set(), set()
    shared_controls, finished = None, set()
    expected_grid = suite['budgets']
    from stage3b_results import selected_results
    paths = (selected_results(root, recovered_root).values() if recovered_root is not None
             else sorted(root.rglob('metadata.json')))
    for path in paths:
        meta = json.loads(path.read_text())
        cfg = meta['config']
        if 'stage3a' not in cfg:
            continue
        settings = cfg['stage3a']
        shift = cfg['dynamics_shift']['parameter']
        method, group, name = meta['active_method'], settings['seed_set'], shift['type']
        key = name, method, group
        if key in seen:
            raise ValueError(f'Duplicate run: {key}')
        seen[key] = path
        if cfg['schedule']['type'] != 'fixed':
            raise ValueError('Stage 3B requires fixed schedules.')
        candidates = {s['type']:s['magnitude'] for s in suite['shifts']}
        if name not in candidates or shift['shifted'] != candidates[name]:
            raise ValueError(f'Suite magnitude mismatch: {name}')
        if method not in suite['methods'] or group not in suite['seed_sets']:
            raise ValueError(f'Unexpected method/seed: {key}')
        if [0]+settings['native_budgets'] != expected_grid:
            raise ValueError('Budget grid differs from suite.')
        controls = {k:v for k,v in cfg.items() if k not in ('stage3a','stage3b','work_dir','dynamics_shift','shift')}
        controls['stage3a'] = {k:v for k,v in settings.items() if k not in (
            'mode','seed_set','adaptation_seeds','evaluation_seeds','fixed_data_seeds','output','fixed_data_path')}
        if shared_controls is not None and shared_controls != controls:
            raise ValueError('Checkpoint/planner/adaptation controls differ across shifts.')
        shared_controls = json.loads(json.dumps(controls))
        # Cross-shift physical configs may differ; compare those within shift.
        controls['dynamics_shift'] = cfg['dynamics_shift']
        if name in references and references[name] != controls:
            raise ValueError('Experimental controls differ within shift.')
        references[name] = controls
        seeds = tuple(tuple(settings[k]) for k in ('adaptation_seeds','evaluation_seeds','fixed_data_seeds'))
        if group in seed_controls and seed_controls[group] != seeds:
            raise ValueError('Methods/shifts must share seed sets.')
        seed_controls[group] = seeds
        dataset_key = name, group
        if dataset_key in datasets and datasets[dataset_key] != meta['fixed_data_sha256']:
            raise ValueError('Fixed dataset differs between methods.')
        datasets[dataset_key] = meta['fixed_data_sha256']
        if not meta['fixed_data_sha256']:
            raise ValueError('Missing shared fixed data.')
        checkpoints.add(meta['checkpoint_sha256'])
        run_id = cfg.get('stage3b', {}).get('run_id', suite['run_id'])
        run_ids.add(run_id)
        evaluation_path = path.parent / 'evaluations.jsonl'
        records = []
        if evaluation_path.exists():
            for line in evaluation_path.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    issues.append(f'{key}: malformed evaluation record')
        grid = [r['native_budget'] for r in records]
        if len(grid) != len(set(grid)) or any(b not in expected_grid for b in grid) or grid != sorted(grid):
            raise ValueError(f'Duplicate/unexpected/out-of-order milestone: {key}')
        complete_path = path.parent / 'complete.json'
        if grid != expected_grid or not complete_path.exists():
            issues.append(f'{key}: incomplete/failed run; observed budgets {grid}')
        elif json.loads(complete_path.read_text())['adaptation_native_steps'] != expected_grid[-1]:
            raise ValueError('Completion budget mismatch.')
        else:
            finished.add(key)
        for r in records:
            if (r['adaptation_native_steps'] != r['native_budget'] or not r['reward_q_parameter_invariance']
                    or not r['evaluation_state_invariance'] or r['fixed_data_sha256'] != meta['fixed_data_sha256']):
                raise ValueError('Budget/state/fixed-data validation failed.')
            if method == 'frozen' and (r['gradient_update_steps'] != 0 or r.get('parameter_drift', 0) != 0):
                raise ValueError('Frozen state changed.')
            base = dict(run_id=run_id, task=cfg['task'], checkpoint=meta['checkpoint_sha256'],
                adaptation=method, shift_type=name, shift_magnitude=shift['shifted'], schedule='fixed',
                budget_native_steps=r['native_budget'], policy_decisions=r['adaptation_policy_decisions'],
                seed_set=group, evaluation_seed='', gradient_updates=r['gradient_update_steps'], replay_size=r['replay_size'])
            def add(metric, value, seed=''):
                if value is not None and not math.isfinite(float(value)):
                    raise ValueError('Non-finite result metric.')
                raw.append(dict(base, evaluation_seed=seed, metric=metric, value=value))
            for condition in ('shifted','nominal'):
                ev = r['evaluation'][condition]
                if [e['seed'] for e in ev['episodes']] != settings['evaluation_seeds']:
                    raise ValueError('Evaluation seeds/count differ from configured protocol.')
                for ep in ev['episodes']:
                    for source, metric in [('success','success'), ('episode_return','return'), ('episode_length','episode_length')]:
                        add(condition+'_'+metric, ep[source], ep['seed'])
                add(condition+'_evaluation_episode_count', len(ev['episodes']))
            prediction = r['fixed_data_prediction']
            for source, metric in [('current','fixed_data_mse'),('initial','fixed_initial_mse')]:
                values = prediction[source]['per_episode_means']
                if len(values) != len(settings['fixed_data_seeds']):
                    raise ValueError('Fixed dataset episode count mismatch.')
                for seed, value in zip(settings['fixed_data_seeds'], values):
                    add(metric, value, seed)
            mapping = dict(adaptation_episodes='adaptation_episodes', replay_samples='replay_samples',
                native_steps='adaptation_native_steps', policy_decisions='adaptation_policy_decisions',
                gradient_updates='gradient_update_steps', replay_size='replay_size',
                consistency_loss='consistency_loss', parameter_drift='parameter_drift',
                gradient_norm='predictor_gradient_norm', update_norm='predictor_update_norm',
                update_compute_time='update_compute_time', nominal_success_delta='nominal_retention_success_delta',
                nominal_return_delta='nominal_retention_return_delta', nominal_length_delta='nominal_retention_length_delta')
            for metric, source in mapping.items():
                add(metric, r.get(source))
            cost = dict(base)
            for field in ('adaptation_episodes','adaptation_transitions','completed_policy_decisions',
                'replay_native_steps','in_flight_decisions','episode_in_progress','replay_samples','update_compute_time',
                'evaluation_native_steps','evaluation_policy_decisions','evaluation_episodes'):
                cost[field] = r.get(field)
            for field in ('fixed_data_collection_native_steps','fixed_data_collection_policy_decisions'):
                cost[field] = meta[field]
            costs.append(cost)
    for shift in suite['shifts']:
        for method in suite['methods']:
            for group in suite['seed_sets']:
                if (shift['type'],method,group) not in seen:
                    issues.append(f'Missing run: {(shift["type"],method,group)}')
    used = set()
    for seeds in seed_controls.values():
        flattened = [s for subset in seeds for s in subset]
        if len(set(flattened)) != len(flattened) or used.intersection(flattened):
            raise ValueError('Independent seed sets overlap.')
        used.update(flattened)
    if not raw or len(checkpoints) != 1 or len(run_ids) != 1:
        raise ValueError('Need nonempty results from one checkpoint and run identifier.')
    if issues and not allow_partial:
        raise ValueError('Incomplete results (use --allow-partial to show gaps): '+ '; '.join(issues))
    summary, by_seed = aggregate(raw)
    curves = defaultdict(dict)
    for key, values in by_seed.items():
        if key[-2] == 'shifted_success':
            curves[(key[4],key[3],key[-1])][key[7]] = statistics.mean(values)
    auc_rows = []
    for (shift, method, seed), points in sorted(curves.items()):
        complete = sorted(points) == expected_grid and (shift,method,seed) in finished
        values = [points[b] for b in expected_grid] if complete else []
        auc_rows.append(dict(run_id=next(iter(run_ids)), shift_type=shift, adaptation=method, seed_set=seed,
            auc=normalized_auc(expected_grid, values) if complete else None, complete=complete,
            initial_to_final=values[-1]-values[0] if complete else None,
            best_observed_success=max(points.values()),
            final_success=values[-1] if complete else None, auc_difference_vs_frozen=None,
            final_success_difference_vs_frozen=None))
    for r in auc_rows:
        ref = next((x for x in auc_rows if x['shift_type']==r['shift_type'] and x['seed_set']==r['seed_set'] and x['adaptation']=='frozen'), None)
        if ref and ref['complete'] and r['complete']:
            r['auc_difference_vs_frozen'] = r['auc']-ref['auc']
            r['final_success_difference_vs_frozen'] = r['final_success']-ref['final_success']
    auc_summary = []
    for shift in suite['shifts']:
        for method in suite['methods']:
            for metric in ('auc','initial_to_final','best_observed_success','auc_difference_vs_frozen','final_success_difference_vs_frozen'):
                values = [r[metric] for r in auc_rows if r['shift_type']==shift['type'] and r['adaptation']==method and r[metric] is not None]
                auc_summary.append(dict(shift_type=shift['type'], adaptation=method, metric=metric,
                    **(mean_sd(values) if values else dict(mean=None,std=None,n_seed_sets=0))))
    output.mkdir(parents=True, exist_ok=False)
    tables = output / 'tables'
    tables.mkdir()
    write_csv(tables/'stage3b_raw_results.csv', raw)
    write_csv(tables/'stage3b_aggregated_results.csv', summary)
    write_csv(tables/'stage3b_interaction_budget.csv', costs)
    write_csv(tables/'stage3b_auc_seed_results.csv', auc_rows)
    write_csv(tables/'stage3b_auc_summary.csv', auc_summary)
    validation = dict(status='PARTIAL' if issues else 'PASS', issues=issues, run_id=next(iter(run_ids)),
        recovered_root=str(recovered_root.resolve()) if recovered_root is not None else None,
        expected_budgets=expected_grid, n_runs=len(seen), input_root=str(root.resolve()),
        aggregation='Equal seed-set weights; mean ± sample SD (not CI); missing SD for n=1.',
        auc='Trapezoid with actual native-step intervals, per seed first; full curves only.')
    (output/'validation.json').write_text(json.dumps(validation, indent=2))
    plot(tables, output/'figures', suite, validation)
    (output/'STAGE3B_RESULTS.md').write_text(
        '# Stage 3B results\n\nRun: '+validation['run_id']+'; status: '+validation['status']+'\n\n'
        'Figures read the aggregated CSV. Success and AUC tables use fractions; success plots use percent. '
        'Bands show sample SD across independent seed sets, not confidence intervals. '
        'Three seed sets provide limited uncertainty estimates. Missing milestones are gaps; incomplete curves have no AUC. '
        'AUC differences are paired within seed sets. No recovery threshold was chosen.\n\n'
        'Evaluation/fixed-data costs are excluded from adaptation. Cumulative costs must not be summed across milestones; '
        'shared fixed data collection is counted once per shift/seed. Physical mass/inertia and friction changes '
        'must not be interpreted as action-delay mechanisms. Frozen fixed-data visitation and reused evaluation seeds '
        'limit generalization claims; Stage 3C method comparisons should reuse this protocol with new final test seeds.\n\n'
        + '\n'.join('- '+issue for issue in issues))
    return validation


def plot(tables, output, suite, validation):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    with (tables/'stage3b_aggregated_results.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    with (tables/'stage3b_auc_summary.csv').open() as stream:
        auc = list(csv.DictReader(stream))
    methods = suite['methods']
    shifts = [s['type'] for s in suite['shifts']]
    colors = dict(zip(shifts, plt.get_cmap('tab10').colors[:len(shifts)]))
    if len(shifts)>10:
        colors = {s:plt.get_cmap('turbo')(i/max(1,len(shifts)-1)) for i,s in enumerate(shifts)}
    budgets = suite['budgets']
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    footer = f"{validation['run_id']} | {stamp} | Mean ± sample SD across seed sets (not CI; small n)"
    if validation['issues']:
        footer += ' | PARTIAL: missing/failed conditions; see validation.json'
    counts = sorted({int(r['n_seed_sets']) for r in summary})
    footer += ' | n=' + ','.join(map(str,counts))
    output.mkdir()
    labels = {s['type']:s['type'].replace('_',' ')+' = '+str(s['magnitude']) for s in suite['shifts']}
    handles = [Line2D([0],[0],color=colors[s],marker='o',label=labels[s]) for s in shifts]
    def panel(ax, method, metric, label, scale):
        for shift in shifts:
            points = {int(r['budget_native_steps']):r for r in summary if r['adaptation']==method and r['shift_type']==shift and r['metric']==metric}
            means = np.array([float(points[b]['mean'])*scale if b in points else np.nan for b in budgets])
            std = np.array([float(points[b]['std'])*scale if b in points and points[b]['std'] else np.nan for b in budgets])
            ax.plot(budgets, means, color=colors[shift], marker='o', linewidth=2.3)
            ax.fill_between(budgets, means-std, means+std, color=colors[shift], alpha=.14)
        ax.set(title=method.replace('_',' '), xlabel='Cumulative native environment steps', ylabel=label)
        ax.set_xticks(budgets)
        ax.tick_params(axis='x', labelrotation=35)
        ax.set_xlim(0,budgets[-1])
        if scale == 100:
            ax.set_ylim(0,100)
        if metric == 'nominal_success':
            ax.axhline(100, color='gray', linestyle=':', linewidth=1)
        ax.grid(alpha=.2)
    def save(fig, name, title):
        if validation['run_id'].startswith('SYNTHETIC'):
            title = 'SYNTHETIC VALIDATION — NOT EXPERIMENTAL RESULTS\n' + title
        fig.suptitle(title, fontsize=16, y=.995)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.93), ncol=min(4,len(shifts)), frameon=False)
        fig.text(.5,.01,footer,ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.065,1,.88])
        fig.savefig(output/(name+'.png'), dpi=220)
        fig.savefig(output/(name+'.pdf'))
        plt.close(fig)
    ncols = min(2,len(methods))
    nrows = math.ceil(len(methods)/ncols)
    for spec, name in zip(METRICS, FIGURES):
        fig, axes = plt.subplots(nrows,ncols,figsize=(7*ncols,5*nrows+1),squeeze=False,sharey=True)
        for ax, method in zip(axes.flat,methods):
            panel(ax,method,*spec)
        for ax in list(axes.flat)[len(methods):]:
            ax.set_visible(False)
        save(fig,name,'Stage 3B — '+spec[1])
    fig,axes=plt.subplots(3,len(methods),figsize=(7*len(methods),14),squeeze=False,sharey='row')
    for row,spec in enumerate(METRICS):
        for column,method in enumerate(methods):
            panel(axes[row,column],method,*spec)
    save(fig,'stage3b_comprehensive_overview','Stage 3B — Dynamics Shift Generalization & Adaptation Horizon')
    fig,ax=plt.subplots(figsize=(max(9,len(shifts)*2),6))
    width=.8/len(methods)
    hatches=['','///','xxx','...','\\\\','+++']
    for j,method in enumerate(methods):
        points={r['shift_type']:r for r in auc if r['adaptation']==method and r['metric']=='auc'}
        vals=[float(points[s]['mean']) if points[s]['mean'] else np.nan for s in shifts]
        errs=[float(points[s]['std']) if points[s]['std'] else 0 for s in shifts]
        ax.bar(np.arange(len(shifts))-.4+width/2+j*width, vals, width, yerr=errs, capsize=3,
               color=[colors[s] for s in shifts], hatch=hatches[j%len(hatches)], edgecolor='black', linewidth=.5)
    ax.set(xticks=np.arange(len(shifts)),xticklabels=[s.replace('_',' ') for s in shifts],ylim=(0,1),
           ylabel='Normalized adaptation AUC (success fraction)',title='Stage 3B — Adaptation AUC')
    ax.legend(handles=[Patch(facecolor='white',edgecolor='black',hatch=hatches[j%len(hatches)],
                             label=m.replace('_',' ')) for j,m in enumerate(methods)])
    if validation['run_id'].startswith('SYNTHETIC'):
        ax.set_title('SYNTHETIC VALIDATION — NOT EXPERIMENTAL RESULTS\nStage 3B — Adaptation AUC')
    fig.text(.5,.02,footer,ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.12,1,1])
    fig.savefig(output/'adaptation_auc_comparison.png',dpi=220)
    fig.savefig(output/'adaptation_auc_comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suite',type=Path,default=Path('benchmarks/stage3b_suite.json'))
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--recovered-root',type=Path,help='Use completed recoveries instead of matching incomplete originals.')
    args=parser.parse_args()
    print(json.dumps(analyze(args.root,args.output,json.loads(args.suite.read_text()),args.allow_partial,args.recovered_root),indent=2))
