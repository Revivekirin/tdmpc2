"""Evaluate a frozen nominal Reacher checkpoint on independent paired seeds."""
from pathlib import Path

import numpy as np

from inspection_common import collect_inspection, parse_inspection_args
from research_utils import bootstrap, digest, write_json
from training_support import evaluate, load_checkpoint


def inspect(checkpoint, output, episodes=10):
    out, meta, seeds, policy, random, zero = collect_inspection(
        checkpoint, output, episodes, 'nsgym-reacher', load_checkpoint, evaluate)
    gains = np.array([p['episode_return'] - r['episode_return']
                      for p, r in zip(policy, random)])
    def average(rows, key):
        return float(np.mean([row[key] for row in rows]))
    summary = dict(checkpoint=str(checkpoint),
                   checkpoint_sha256=digest(Path(checkpoint) / 'model.pt'),
                   environment_transitions=meta['environment_transitions'],
                   episodes=episodes, evaluation_seeds=seeds,
                   policy_return_mean=average(policy, 'episode_return'),
                   random_return_mean=average(random, 'episode_return'),
                   zero_action_return_mean=average(zero, 'episode_return'),
                   paired_gain_mean=float(np.mean(gains)),
                   paired_gain_95pct_bootstrap=bootstrap(gains),
                   policy_final_goal_distance_mean=average(policy, 'goal_distance_final'),
                   random_final_goal_distance_mean=average(random, 'goal_distance_final'),
                   zero_action_final_goal_distance_mean=average(zero, 'goal_distance_final'),
                   policy_goal_progress_mean=average(policy, 'goal_progress'),
                   random_goal_progress_mean=average(random, 'goal_progress'),
                   zero_action_goal_progress_mean=average(zero, 'goal_progress'),
                   interpretation='Nominal validation only; return gain alone does not establish target reaching.')
    write_json(out / 'summary.json', summary)
    return out


def main():
    args = parse_inspection_args(__doc__)
    print(inspect(args.checkpoint, args.output, args.episodes))


if __name__ == '__main__':
    main()
