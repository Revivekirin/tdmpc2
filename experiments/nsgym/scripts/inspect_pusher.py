"""Evaluate a saved nominal Pusher checkpoint against paired random actions."""
import argparse
from pathlib import Path
import numpy as np

from research_utils import ROOT, bootstrap, csv_rows, digest, output_dir, write_json
from training_support import evaluate, load_checkpoint


def inspect(checkpoint, output, episodes=10):
    if episodes < 2:
        raise ValueError('At least two evaluation seeds are required')
    agent, meta = load_checkpoint(checkpoint)
    if meta['config']['task'] != 'nsgym-pusher':
        raise ValueError('Pusher checkpoint required')
    seeds = list(range(20000001, 20000001 + episodes))
    policy, _ = evaluate(agent, meta['config'], seeds)
    random, _ = evaluate(None, meta['config'], seeds, random_actions=True)
    zero, _ = evaluate(None, meta['config'], seeds, zero_actions=True)
    out = output_dir(output)
    csv_rows(out / 'policy.csv', policy)
    csv_rows(out / 'random.csv', random)
    csv_rows(out / 'zero_action.csv', zero)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()
    print(inspect(args.checkpoint, args.output, args.episodes))


if __name__ == '__main__':
    main()
