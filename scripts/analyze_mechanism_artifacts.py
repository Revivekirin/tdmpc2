"""Reproduce compact descriptive aggregates from the two Stage 2 ZIP archives.

Reads CSV/JSON data only; never imports or runs the model or simulator. Episode
weighting prevents long failed episodes from dominating the ranking summaries.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile


RANK_FIELDS = ('actual_total_vs_reward', 'actual_total_vs_q', 'actual_vs_diagnostic')


def csv_rows(archive, base, filename):
    with archive.open(f'{base}/mechanism/{filename}') as stream:
        yield from csv.DictReader(io.TextIOWrapper(stream))


def episode_mean(pairs):
    groups = defaultdict(list)
    for episode, value in pairs:
        groups[int(episode)].append(float(value))
    return mean(mean(values) for values in groups.values()) if groups else None


def aggregate_run(summaries, diagnostics, base):
    read_json = lambda name: json.loads(summaries.read(f'{base}/{name}'))
    summary = read_json('summary.json')
    final_iteration = str(summary['config']['iterations'] - 1)
    policy_prior_candidates = summary['config']['num_pi_trajs']
    manifest = read_json('mechanism/manifest.json')
    validity = read_json('mechanism/validity_report.json')
    episode_summary = read_json('mechanism/mechanism_summary.json')['episodes']
    result = dict(
        path=base, checkpoint=Path(manifest['checkpoint']).name,
        task=manifest['task'], shift_type=summary['shift_type'],
        schedule=summary['schedule_type'], status=summary['status'],
        episodes=len(episode_summary), success=summary['summary_metrics']['success_mean'],
        mean_episode_length=summary['summary_metrics']['episode_length_mean'],
        exposure_fraction=summary['exposure_fraction'],
        episode_mean_latent_mse=mean(row['latent_prediction_mse'] for row in episode_summary),
        episode_mean_reward_absolute_error=mean(row['reward_absolute_error'] for row in episode_summary),
        checkpoint_unchanged=summary['checkpoint_unchanged'],
        model_unchanged=validity['model_parameters_unchanged'],
        model_hash_equal=manifest['model_sha256_before'] == validity['model_sha256_after'],
        simulator_validation=validity['simulator_validation'],
    )
    one = list(csv_rows(diagnostics, base, 'one_step.csv'))
    valid = [row for row in one if row['executed_prediction_validity'].startswith('VALID')]
    result['one_step'] = dict(
        decisions=len(one), executed_substitution_valid=len(valid),
        commanded_mse_valid=mean(float(row['latent_prediction_mse']) for row in valid) if valid else None,
        executed_mse_valid=mean(float(row['executed_latent_prediction_mse']) for row in valid) if valid else None,
        episode_mean_command_minus_executed_mse=episode_mean(
            (row['episode_id'], float(row['latent_prediction_mse'])-
             float(row['executed_latent_prediction_mse'])) for row in valid),
        mean_command_executed_difference=mean(float(row['commanded_executed_action_difference']) for row in one),
    )
    multi = {(row['episode_id'], row['policy_decision_step'], row['horizon']): float(row['latent_prediction_mse'])
             for row in csv_rows(diagnostics, base, 'multistep.csv')}
    starts = [(episode, step) for episode, step, horizon in multi if horizon == '5']
    result['multi_step_same_starts'] = dict(
        starts=len(starts), **{f'horizon_{h}': mean(multi[(episode, step, h)] for episode, step in starts)
                             if starts else None for h in ('1', '3', '5')})
    result['q_head_std_episode_mean'] = episode_mean(
        (row['episode_id'], row['Q_std']) for row in csv_rows(diagnostics, base, 'reward_q.csv'))
    ranks = defaultdict(list)
    iteration_status = Counter()
    comparison_status = Counter()
    final_rows = 0
    for row in csv_rows(diagnostics, base, 'planner_summary.csv'):
        iteration_status[row['status']] += 1
        if row['iteration_id'] != final_iteration:
            continue
        final_rows += 1
        ranks['selected_policy_prior'].append(
            (row['episode_id'], float(int(row['selected_candidate_id']) < policy_prior_candidates)))
        for field in RANK_FIELDS:
            comparison = json.loads(row[field])
            comparison_status[comparison['status']] += 1
            agreement = bool(set(comparison['top1_a']) & set(comparison['top1_b']))
            for metric, value in (('top1_agreement', float(agreement)),
                                  ('top_k_jaccard', comparison['top_k_jaccard']),
                                  ('spearman', comparison['spearman']),
                                  ('kendall_tau_b', comparison['kendall_tau_b'])):
                if value is not None:
                    ranks[f'{field}_{metric}'].append((row['episode_id'], value))
    result['planner_final_iteration'] = dict(
        decisions=final_rows, selected_policy_prior_episode_mean=episode_mean(ranks['selected_policy_prior']),
        **{f'{field}_{metric}_episode_mean': episode_mean(ranks[f'{field}_{metric}'])
           for field in RANK_FIELDS for metric in
           ('top1_agreement', 'top_k_jaccard', 'spearman', 'kendall_tau_b')})
    result['planner_iteration_status_counts'] = dict(iteration_status)
    result['planner_final_comparison_status_counts'] = dict(comparison_status)
    paired_name = f'{base}/paired.json'
    if paired_name in summaries.namelist():
        paired = read_json('paired.json')
        result['paired'] = dict(
            full_state_pairing=paired['full_state_pairing'],
            exposed_episodes=sum(row['shifted_exposure'] > 0 for row in paired['pairs']),
            success_transitions=dict(Counter(row['success_transition'] for row in paired['pairs'])),
            mechanism_pairing=paired['mechanism_pairing'],
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', type=Path, default=Path('results/mechanism-full-summaries.zip'))
    parser.add_argument('--diagnostics', type=Path, default=Path('results/mechanism-full-diagnostics.zip'))
    parser.add_argument('--out', type=Path, default=Path('results/mechanism-aggregates.json'))
    args = parser.parse_args()
    with ZipFile(args.summaries) as summaries, ZipFile(args.diagnostics) as diagnostics:
        bases = sorted({name.rsplit('/mechanism/', 1)[0] for name in diagnostics.namelist()
                        if name.endswith('/mechanism/planner_summary.csv')})
        runs = [aggregate_run(summaries, diagnostics, base) for base in bases]
        suite = json.loads(summaries.read('mechanism-full-status.json'))
    result = dict(
        summary_zip_sha256=hashlib.sha256(args.summaries.read_bytes()).hexdigest(),
        diagnostics_zip_sha256=hashlib.sha256(args.diagnostics.read_bytes()).hexdigest(),
        suite_status=suite['analysis_status'], suite_commands=len(suite['runs']),
        runs=runs, weighting='episode means for ranking/Q; same valid decisions for command/execution MSE',
        interpretation='descriptive frozen-model associations; no true candidate returns or exact state branching',
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {args.out} with {len(runs)} runs')


if __name__ == '__main__':
    main()
