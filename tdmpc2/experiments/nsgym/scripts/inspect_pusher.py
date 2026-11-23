"""Evaluate a saved nominal Pusher checkpoint against paired random actions."""
from pathlib import Path
import numpy as np

from inspection_common import collect_inspection, parse_inspection_args
from research_utils import bootstrap, digest, write_json
from training_support import evaluate, load_checkpoint


def inspect(checkpoint, output, episodes=10):
    out, meta, seeds, policy, random, zero = collect_inspection(
        checkpoint, output, episodes, 'nsgym-pusher', load_checkpoint, evaluate)
    gains = np.array([p['episode_return'] - r['episode_return']
                      for p, r in zip(policy, random)])
    summary = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(Path(checkpoint) / 'model.pt'),
                   environment_transitions=meta['environment_transitions'], episodes=episodes,
                   evaluation_seeds=seeds,
                   policy_return_mean=float(np.mean([r['episode_return'] for r in policy])),
                   random_return_mean=float(np.mean([r['episode_return'] for r in random])),
                   zero_action_return_mean=float(np.mean([r['episode_return'] for r in zero])),
                   paired_gain_mean=float(np.mean(gains)),
                   paired_gain_95pct_bootstrap=bootstrap(gains),
                   policy_final_goal_distance_mean=float(np.mean([r['goal_distance_final'] for r in policy])),
                   random_final_goal_distance_mean=float(np.mean([r['goal_distance_final'] for r in random])),
                   zero_action_final_goal_distance_mean=float(np.mean([r['goal_distance_final'] for r in zero])),
                   policy_goal_progress_mean=float(np.mean([r['goal_progress'] for r in policy])),
                   random_goal_progress_mean=float(np.mean([r['goal_progress'] for r in random])),
                   zero_action_goal_progress_mean=float(np.mean([r['goal_progress'] for r in zero])),
                   interpretation='Interim nominal checkpoint inspection; no official Pusher success rate or online adaptation claim.')
    write_json(out / 'summary.json', summary)
    return out


def main():
    args = parse_inspection_args(__doc__)
    print(inspect(args.checkpoint, args.output, args.episodes))


if __name__ == '__main__':
    main()
