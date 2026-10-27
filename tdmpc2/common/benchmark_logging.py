"""Three-level benchmark artifacts and per-episode trajectory arrays."""
from pathlib import Path
import json
import hashlib
import re
import numpy as np
from common.shift_evaluation import EvaluationLog, json_value
from common.shift_metrics import summary_metrics


class BenchmarkLog(EvaluationLog):
    def __init__(self, cfg, seeds):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', cfg.benchmark_id):
            raise ValueError('benchmark_id may contain only letters, digits, underscores and hyphens.')
        tag = f"{cfg.benchmark_id}-{cfg.dynamics_shift['parameter']['type']}-{cfg.dynamics_shift['schedule']['type']}"
        super().__init__(cfg, run_tag=tag, output_root=Path(cfg.work_dir) / 'benchmark' / cfg.benchmark_split)
        self.options = cfg.dynamics_shift['logging']
        self.events_file = (self.path / 'shift_events.jsonl').open('w')
        self.metadata.update(experiment_id=self.path.name, evaluation_protocol=cfg.adaptation_mode,
                             split=cfg.benchmark_split, episode_seeds=seeds, shift_config=cfg.dynamics_shift,
                             environment_version=self.metadata['packages'], status='running')
        self.write_json('metadata.json', self.metadata)
        self.save_trajectory = cfg.benchmark_save_trajectory
        self.trajectory = []
        self.native_trajectory = []

    def native_steps(self, records, decision_done):
        if self.save_trajectory:
            self.native_trajectory.extend(records)
        if self.options['save_step_trace']:
            for record in records:
                row = dict(record, decision_done=bool(decision_done))
                self.steps_file.write(json.dumps(row, default=json_value) + '\n')
            self.steps_file.flush()

    def events(self, events):
        if self.options['save_shift_events']:
            for event in events:
                self.events_file.write(json.dumps(event, default=json_value) + '\n')
            self.events_file.flush()

    def transition(self, transition):
        if self.save_trajectory:
            self.trajectory.append((transition.observation.detach().cpu().numpy().copy(),
                                    transition.commanded_action.detach().cpu().numpy().copy(),
                                    transition.next_observation.detach().cpu().numpy().copy(),
                                    transition.reward, transition.done))

    def episode(self, row):
        row = dict(row, trajectory_path=None, trajectory_sha256=None)
        if self.trajectory:
            columns = list(zip(*self.trajectory))
            path = self.path / f"episode-{row['episode_id']:04d}.npz"
            np.savez_compressed(path,
                                observations=np.asarray(columns[0]), commanded_actions=np.asarray(columns[1]),
                                next_observations=np.asarray(columns[2]), rewards=np.asarray(columns[3]),
                                dones=np.asarray(columns[4]), seed=row['seed'],
                                diagnostic_native_commanded_actions=np.asarray([r['commanded_action'] for r in self.native_trajectory]),
                                diagnostic_native_executed_actions=np.asarray([r['executed_action'] for r in self.native_trajectory]),
                                diagnostic_native_parameters=np.asarray([r['applied_parameter'] for r in self.native_trajectory]),
                                diagnostic_native_steps=np.asarray([r['native_step'] for r in self.native_trajectory]))
            row.update(trajectory_path=str(path.resolve()), trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if self.options['save_episode_summary']:
            super().episode(row)
        else:
            self.episodes.append(row)
        self.trajectory.clear()
        self.native_trajectory.clear()

    def finish(self, error=None):
        self.steps_file.close()
        self.events_file.close()
        unchanged = self.checksum() == self.digest
        result = dict(self.metadata, status='failed' if error or not unchanged else 'complete',
                      error=error or (None if unchanged else 'Checkpoint hash changed'),
                      checkpoint_unchanged=unchanged, summary_metrics=summary_metrics(self.episodes))
        self.write_json('summary.json', result)
        if not unchanged:
            raise RuntimeError('Checkpoint changed during benchmark evaluation.')
        return result
