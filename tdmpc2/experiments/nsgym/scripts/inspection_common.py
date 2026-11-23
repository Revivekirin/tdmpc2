"""Shared checkpoint inspection flow; task-specific summaries stay in callers."""

import argparse
from pathlib import Path

from research_utils import csv_rows, output_dir


def collect_inspection(checkpoint, output, episodes, task, load_checkpoint, evaluate):
    if episodes < 2:
        raise ValueError('At least two evaluation seeds are required')
    agent, meta = load_checkpoint(checkpoint)
    if meta['config']['task'] != task:
        raise ValueError(f'{task.removeprefix("nsgym-").capitalize()} checkpoint required')
    seeds = list(range(20000001, 20000001 + episodes))
    policy, _ = evaluate(agent, meta['config'], seeds)
    random, _ = evaluate(None, meta['config'], seeds, random_actions=True)
    zero, _ = evaluate(None, meta['config'], seeds, zero_actions=True)
    out = output_dir(output)
    csv_rows(out / 'policy.csv', policy)
    csv_rows(out / 'random.csv', random)
    csv_rows(out / 'zero_action.csv', zero)
    return out, meta, seeds, policy, random, zero


def parse_inspection_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    return parser.parse_args()
