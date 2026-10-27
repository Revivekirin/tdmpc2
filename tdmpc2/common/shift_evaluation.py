"""Streaming evaluation artifacts; no model or optimizer mutation."""
import csv
import dataclasses
import hashlib
import importlib.metadata
import platform
import json
from pathlib import Path
import subprocess
import uuid
from datetime import datetime, timezone
import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def shift_env(env):
    from envs.wrappers.dynamics_shift import DynamicsShiftWrapper
    while env is not None:
        if isinstance(env, DynamicsShiftWrapper):
            return env
        env = getattr(env, 'env', None)
    return None


class EvaluationLog:
    def __init__(self, cfg, run_tag=None, output_root=None):
        tag = run_tag or f'{cfg.shift_mode}-{cfg.shift_clock}-g{cfg.shift_gain}'
        self.path = (Path(output_root) if output_root is not None else Path(cfg.work_dir) / 'evaluations') / (tag + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:10])
        self.path.mkdir(parents=True, exist_ok=False)
        self.checkpoint = Path(cfg.checkpoint).resolve()
        self.digest = self.checksum()
        root = Path(__file__).resolve().parents[2]
        def git(*args):
            return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True).stdout.strip()
        versions = {}
        for package in ['numpy', 'torch', 'gym', 'gymnasium', 'mani-skill2', 'sapien', 'hydra-core']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (root / 'tdmpc2').rglob('*.py')}
        self.metadata = dict(python=platform.python_version(), packages=versions, source_sha256=source_hashes, config=dataclasses.asdict(cfg), checkpoint=str(self.checkpoint),
                             checkpoint_sha256=self.digest, code_version=git('rev-parse', 'HEAD'),
                             git_status=git('status', '--short'), clock_unit='native step',
                             step_index='zero-based, before action', std_ddof=0,
                             output_directory=str(self.path.resolve()))
        (self.path / 'code.diff').write_text(git('diff', 'HEAD'))
        self.write_json('metadata.json', self.metadata)
        self.episodes = []
        self.steps_file = (self.path / 'timesteps.jsonl').open('w')

    def checksum(self):
        with self.checkpoint.open('rb') as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
            return digest.hexdigest()

    def write_json(self, name, data):
        (self.path / name).write_text(json.dumps(data, default=json_value, indent=2) + '\n')

    def step(self, task, episode, decision_step, action, reward, done, info):
        records = info.get('shift_native_steps')
        if records is None:
            records = [dict(shift_commanded_action=action.detach().cpu().numpy(),
                            shift_executed_action=None, shift_gain=None,
                            reward=float(reward), done=bool(done))]
        for record in records:
            row = dict(task=task, episode=episode, decision_step=decision_step,
                       decision_done=bool(done), **record)
            self.steps_file.write(json.dumps(row, default=json_value) + '\n')
        self.steps_file.flush()

    def episode(self, row):
        self.episodes.append(row)
        path = self.path / 'episodes.csv'
        with path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if len(self.episodes) == 1:
                writer.writeheader()
            writer.writerow(row)

    def finish(self):
        self.steps_file.close()
        unchanged = self.checksum() == self.digest
        summary = {}
        for task in dict.fromkeys(row['task'] for row in self.episodes):
            rows = [r for r in self.episodes if r['task'] == task]
            summary[task] = dict(episodes=len(rows))
            for field in ['episode_return', 'success', 'episode_length']:
                values = [r[field] for r in rows]
                summary[task][field + '_mean'] = float(np.mean(values))
                summary[task][field + '_std'] = float(np.std(values))
        self.write_json('summary.json', dict(tasks=summary, checkpoint_unchanged=unchanged,
                                             **self.metadata))
        if not unchanged:
            raise RuntimeError('Checkpoint changed during evaluation.')
        return summary
