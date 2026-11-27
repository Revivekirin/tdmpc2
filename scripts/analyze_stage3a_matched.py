"""Read-only matched-budget aggregation. Outputs always go to a new directory."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def analyze(root, output):
    # Check plotting dependencies before creating any partial report artifacts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows, budgets, datasets, checkpoints, methods = [], {}, {}, set(), {}
    config_reference = None
    seed_sets = {}
    for path in sorted(root.glob('set-*/*/*/evaluations.jsonl')):
        metadata = json.loads((path.parent / 'metadata.json').read_text())
        completion = json.loads((path.parent / 'complete.json').read_text())
        config = metadata['config']
        settings = config['stage3a']
        group, mode = settings['seed_set'], metadata['active_method']
        key = (group, mode)
        if key in budgets:
            raise ValueError(f'Duplicate method/seed set: {key}')
        # Compare experimental controls, including optimizer dose and architecture.
        controls = {k: v for k, v in config.items() if k not in ('stage3a', 'work_dir')}
        controls['stage3a'] = {k: v for k, v in settings.items() if k not in (
            'mode', 'seed_set', 'adaptation_seeds', 'evaluation_seeds', 'fixed_data_seeds',
            'fixed_data_path', 'output')}
        if config_reference is None:
            config_reference = controls
        elif controls != config_reference:
            raise ValueError('Methods/seed sets differ in experimental controls.')
        seeds = tuple(tuple(settings[k]) for k in ('adaptation_seeds', 'evaluation_seeds', 'fixed_data_seeds'))
        if group in seed_sets and seed_sets[group] != seeds:
            raise ValueError('Methods must share seed sets.')
        seed_sets[group] = seeds
        methods.setdefault(group, set()).add(mode)
        checkpoints.add(metadata['checkpoint_sha256'])
        datasets.setdefault(group, set()).add(metadata['fixed_data_sha256'])
        records = [json.loads(line) for line in path.read_text().splitlines()]
        budgets[key] = [r['native_budget'] for r in records]
        if budgets[key] != [0] + settings['native_budgets']:
            raise ValueError('Missing or duplicate budget checkpoints.')
        if completion['adaptation_native_steps'] != settings['native_budgets'][-1]:
            raise ValueError('Incomplete or overshooting run.')
        for record in records:
            if record['adaptation_native_steps'] != record['native_budget']:
                raise ValueError('Native budget mismatch.')
            if not record['reward_q_parameter_invariance'] or not record['evaluation_state_invariance']:
                raise ValueError('Invariance check failed.')
            if record['fixed_data_sha256'] != metadata['fixed_data_sha256']:
                raise ValueError('Fixed data identity mismatch.')
            shifted, nominal = record['evaluation']['shifted'], record['evaluation']['nominal']
            fixed = record['fixed_data_prediction']
            rows.append(dict(seed_set=group, method=mode, native_budget=record['native_budget'],
                completed_episodes=record['adaptation_episodes'],
                policy_decisions_started=record['adaptation_policy_decisions'],
                completed_policy_decisions=record['completed_policy_decisions'],
                replay_native_steps=record['replay_native_steps'],
                in_flight_decisions=record['in_flight_decisions'],
                updates=record['gradient_update_steps'], shifted_success=shifted['success'],
                all_episode_length=shifted['episode_length'],
                successful_completion_length=shifted['successful_completion_length'],
                fixed_initial_mse=fixed['initial']['episode_means']['mean'],
                fixed_current_mse=fixed['current']['episode_means']['mean'],
                fixed_paired_delta=fixed['paired_episode_delta']['mean'],
                fixed_transition_mse=fixed['current']['transition']['mean'],
                nominal_success=nominal['success'], retention_delta=record['nominal_retention_success_delta'],
                evaluation_native_steps=record['evaluation_native_steps'],
                evaluation_policy_decisions=record['evaluation_policy_decisions'],
                dataset_native_steps=metadata['fixed_data_collection_native_steps'],
                dataset_policy_decisions=metadata['fixed_data_collection_policy_decisions']))
    if not rows or len(checkpoints) != 1:
        raise ValueError('Need paired runs from one pretrained checkpoint.')
    if any(v != {'frozen', 'dynamics_only'} for v in methods.values()):
        raise ValueError('Each seed set requires both methods.')
    if any(len(v) != 1 or None in v for v in datasets.values()):
        raise ValueError('Both methods must use the same fixed dataset per seed set.')
    if any(b != next(iter(budgets.values())) for b in budgets.values()):
        raise ValueError('Methods and seed sets must have identical budget grids.')
    used = set()
    for sets in seed_sets.values():
        combined = [s for seeds in sets for s in seeds]
        if len(combined) != len(set(combined)) or used & set(combined):
            raise ValueError('Seed sets must be disjoint within and across replications.')
        used.update(combined)
    output.mkdir(parents=True, exist_ok=False)
    def write_csv(name, entries):
        with (output / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    write_csv('adaptation_curve.csv', rows)
    write_csv('fixed_data_comparison.csv', [{k:r[k] for k in ('seed_set','method','native_budget',
        'fixed_initial_mse','fixed_current_mse','fixed_paired_delta','fixed_transition_mse')} for r in rows])
    write_csv('interaction_budget_report.csv', [{k:r[k] for k in ('seed_set','method','native_budget',
        'completed_episodes','policy_decisions_started','completed_policy_decisions','replay_native_steps',
        'in_flight_decisions','updates','evaluation_native_steps','evaluation_policy_decisions',
        'dataset_native_steps','dataset_policy_decisions')} for r in rows])
    summary = []
    for mode in ('frozen', 'dynamics_only'):
        for budget in next(iter(budgets.values())):
            subset = [r for r in rows if r['method'] == mode and r['native_budget'] == budget]
            entry = dict(method=mode, native_budget=budget, seed_sets=len(subset))
            for metric in ('shifted_success','fixed_current_mse','fixed_paired_delta','nominal_success',
                           'successful_completion_length','updates'):
                values = [r[metric] for r in subset if r[metric] is not None]
                entry[metric+'_mean'] = statistics.mean(values) if values else None
                entry[metric+'_std'] = statistics.stdev(values) if len(values) > 1 else None
                entry[metric+'_count'] = len(values)
            summary.append(entry)
    write_csv('across_seed_sets.csv', summary)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric, label in zip(axes.flat,
        ('shifted_success','fixed_current_mse','nominal_success','successful_completion_length'),
        ('Shifted success','Fixed-data latent MSE (episode mean)','Nominal success','Successful completion decisions')):
        for mode in ('frozen', 'dynamics_only'):
            points = [r for r in summary if r['method'] == mode]
            ax.errorbar([r['native_budget'] for r in points],
                [r[metric+'_mean'] if r[metric+'_mean'] is not None else float('nan') for r in points],
                yerr=[r[metric+'_std'] or 0 for r in points], marker='o', label=mode, capsize=3)
        ax.set(xlabel='Adaptation native steps', ylabel=label)
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle('Mean ± sample SD across independent seed sets (not confidence intervals)')
    fig.tight_layout()
    fig.savefig(output / 'adaptation_curve.png', dpi=160)
    fig.savefig(output / 'adaptation_curve.pdf')
    plt.close(fig)
    limitations = '''# Interpretation limits

- Exact native-step budgets are matched; completed episodes and update counts need not match.
- Checkpoints precede the in-flight decision commit and any terminal episode update. Final stopping does not fabricate a truncated transition or perform an extra update.
- Fixed-data collection and online evaluation costs are excluded from adaptation and reported separately. A shared dataset is collected once per seed set, not once per method.
- Fixed data comes from pretrained Frozen MPC in the shifted environment: it tests this visitation distribution, not every possible state/action or the adapted policy distribution.
- Paired fixed-data deltas compare both predictors on identical transitions. They do not establish a causal link between prediction accuracy and task success.
- Successful completion length excludes failures; all-episode length is also recorded. Check the number of successes before interpreting completion speed.
- Online held-out MSE still has policy-dependent visitation effects.
- Error bars are sample SD across independent seed sets, not confidence intervals. Repeated checkpoints reuse seeds and are not independent test samples.
- Selecting settings/checkpoints using these evaluations requires a new final test set.
- Existing Stage 2 PlannerTrace is reused; no new planning loss or value adaptation is introduced.
'''
    (output / 'interpretation_limits.md').write_text(limitations)
    (output / 'validation.json').write_text(json.dumps(dict(seed_sets=len(methods),
        checkpoint_sha256=next(iter(checkpoints)), matched_budgets=True,
        shared_fixed_data=True, input_root=str(root.resolve())), indent=2))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output)
