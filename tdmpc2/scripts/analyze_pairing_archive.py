"""Summarize pre-shift evidence available in the uploaded Stage 2 archives.

The archives omit native observations, controller state and policy RNG traces,
so this script never upgrades full pairing or pre-shift equivalence to PASS.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile


def csv_rows(archive, base, name):
    with archive.open(f'{base}/mechanism/{name}.csv') as stream:
        return list(csv.DictReader(io.TextIOWrapper(stream)))


def by_episode(rows):
    result = defaultdict(list)
    for row in rows:
        result[int(row['episode_id'])].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', type=Path, default=Path('results/mechanism-full-summaries.zip'))
    parser.add_argument('--diagnostics', type=Path, default=Path('results/mechanism-full-diagnostics.zip'))
    parser.add_argument('--out', type=Path, default=Path('results/pairing-validation-archive.json'))
    args = parser.parse_args()
    with ZipFile(args.summaries) as summaries, ZipFile(args.diagnostics) as diagnostics:
        pair_name = next(name for name in summaries.namelist()
                         if 'action_delay-abrupt' in name and name.endswith('/paired.json'))
        paired = json.loads(summaries.read(pair_name))
        paths = {name.rsplit('/mechanism/', 1)[0] for name in diagnostics.namelist()
                 if name.endswith('/mechanism/action_history.csv')}
        nominal = next(path for path in paths if path.endswith(Path(paired['nominal_run']).name))
        shifted = next(path for path in paths if path.endswith(Path(paired['shifted_run']).name))
        histories = [by_episode(csv_rows(diagnostics, path, 'action_history')) for path in (nominal, shifted)]
        planner = [by_episode(row for row in csv_rows(diagnostics, path, 'planner_summary')
                              if row['iteration_id'] == '5') for path in (nominal, shifted)]
        one_step = [by_episode(csv_rows(diagnostics, path, 'one_step')) for path in (nominal, shifted)]
    episodes = []
    for pair in paired['pairs']:
        episode = pair['episode_id']
        a, b = (history[episode][:10] for history in histories)
        first_native = None
        first_action = None
        for step, (left, right) in enumerate(zip(a, b)):
            fields = [field for field in ('requested_action', 'native_executed_action',
                                         'delay_queue_state_if_available', 'current_shift')
                      if left[field] != right[field]]
            if fields:
                if first_native is None:
                    first_native = dict(step=step, fields=fields)
                if first_action is None and any(field in fields for field in ('requested_action', 'native_executed_action', 'current_shift')):
                    first_action = dict(step=step, fields=[field for field in fields if field in
                                                          ('requested_action', 'native_executed_action', 'current_shift')])
        actions = []
        for left, right in zip(planner[0][episode], planner[1][episode]):
            if int(left['policy_decision_step']) >= 5:
                break
            actions.append(left['selected_first_action'] == right['selected_first_action'])
        errors = []
        for left, right in zip(one_step[0][episode], one_step[1][episode]):
            if int(left['policy_decision_step']) >= 5:
                break
            errors.append(left['latent_prediction_mse'] == right['latent_prediction_mse'])
        episodes.append(dict(episode_id=episode, seed=pair['initial_seed'],
                             initial_exposed_components=pair['initial_comparison'],
                             first_logged_native_divergence=first_native,
                             first_logged_action_divergence=first_action,
                             first_selected_action_difference=next((i for i, same in enumerate(actions) if not same), None),
                             first_one_step_mse_difference=next((i for i, same in enumerate(errors) if not same), None),
                             pre_shift_equivalence='UNVERIFIED', first_divergence_step=None,
                             pairing_validity='UNVERIFIED'))
    result = dict(schema_version=1, source='uploaded Stage 2 summary and diagnostic archives',
                  summaries_sha256=hashlib.sha256(args.summaries.read_bytes()).hexdigest(),
                  diagnostics_sha256=hashlib.sha256(args.diagnostics.read_bytes()).hexdigest(),
                  nominal_run=nominal, shifted_run=shifted, episodes=episodes,
                  counts=dict(initial_comparisons_pass=sum(all(value == 'PASS' for key, value in e['initial_exposed_components'].items()
                                                               if key in ('observation', 'simulator', 'controller', 'rng')) for e in episodes),
                              first_selected_action_match=sum(e['first_selected_action_difference'] != 0 for e in episodes),
                              any_pre_shift_selected_action_difference=sum(e['first_selected_action_difference'] is not None for e in episodes),
                              any_pre_shift_logged_action_difference=sum(e['first_logged_action_divergence'] is not None for e in episodes),
                              any_first_five_one_step_mse_difference=sum(e['first_one_step_mse_difference'] is not None for e in episodes)),
                  pre_shift_equivalence='UNVERIFIED', first_divergence_step=None,
                  pairing_validity='UNVERIFIED', simulator_full_state='UNVERIFIED',
                  limitation='No post-step native observation, controller state or global policy RNG states in these archives.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(args.out.resolve())


if __name__ == '__main__':
    main()
