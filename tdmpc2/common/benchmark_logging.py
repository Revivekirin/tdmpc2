"""Three-level benchmark artifacts and per-episode trajectory arrays."""
from pathlib import Path
import json
import hashlib
import re
import dataclasses
from datetime import datetime, timezone
from common.benchmark_metrics import exposure
import numpy as np
from common.shift_evaluation import EvaluationLog, json_value
from common.shift_metrics import summary_metrics


class BenchmarkLog(EvaluationLog):
    def __init__(self, cfg, seeds):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', cfg.benchmark_id):
            raise ValueError('benchmark_id may contain only letters, digits, underscores and hyphens.')
        v2 = getattr(cfg, 'protocol_version', None) == 2
        shift_config = cfg.dynamics_shift
        if v2:
            from common.benchmark_config import runtime_shift
            shift_config = runtime_shift(dict(dynamics_shift=cfg.dynamics_shift, schedule=cfg.schedule))
        tag = f"{cfg.benchmark_id}-{shift_config['parameter']['type']}-{shift_config['schedule']['type']}"
        super().__init__(cfg, run_tag=tag, output_root=Path(cfg.work_dir) / 'benchmark' / cfg.benchmark_split)
        self.options = dict(cfg.dynamics_shift['logging'])
        self.v2 = v2
        if v2:
            self.options.update(save_step_trace=True, save_episode_summary=True)
        self.exposure_records = []
        self.constraints = getattr(cfg, 'evaluation', {})
        self.events_file = (self.path / 'shift_events.jsonl').open('w')
        self.metadata.update(experiment_id=self.path.name, evaluation_protocol=cfg.adaptation_mode,
                             split=cfg.benchmark_split, episode_seeds=seeds, shift_config=shift_config,
                             environment_version=self.metadata['packages'], status='running')
        if v2:
            from omegaconf import OmegaConf
            from common.benchmark_config import config_hash
            config = json.loads(json.dumps(dataclasses.asdict(cfg), default=json_value))
            resolved = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
            (self.path / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(OmegaConf.create(resolved)))
            self.metadata.update(protocol_version=2, schema_version=2, task=cfg.task,
                                 checkpoint_path=str(self.checkpoint), source_revision=self.metadata['code_version'],
                                 source_dirty_state=self.metadata['git_status'], resolved_config_hash=config_hash(resolved),
                                 seed=cfg.seed, adaptation_mode=cfg.adaptation_mode,
                                 shift_type=shift_config['parameter']['type'], schedule_type=cfg.schedule['type'],
                                 clock=cfg.schedule['clock'], start_timestamp=datetime.now(timezone.utc).isoformat())
        if v2 and cfg.schedule['type'] == 'sinusoidal':
            self.metadata['sinusoidal'] = dict(period=cfg.schedule['interval'],
                amplitude=(cfg.schedule['amplitude'] * np.abs(np.asarray(shift_config['parameter']['shifted']) -
                                                             np.asarray(shift_config['parameter']['nominal']))).tolist(),
                crossing_definition='sign change across nominal, ignoring samples within atol=1e-8')
        self.write_json('metadata.json', self.metadata)
        self.save_trajectory = cfg.benchmark_save_trajectory
        self.trajectory = []
        self.native_trajectory = []

    def native_steps(self, records, decision_done):
        self.exposure_records.extend(records)
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
        if self.v2:
            row = dict(row, initial_seed=row['seed'], decision_length=row['episode_length'],
                       **exposure(self.exposure_records))
            row['return'] = row['episode_return']
            row['exposure_constraint_met'] = (row['exposure_fraction'] >= self.constraints['min_exposure_fraction']
                and (not self.constraints['require_shift_exposure'] or row['shifted_native_steps'] > 0))
        self.exposure_records.clear()
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
        if self.v2:
            total = sum(r['native_length'] for r in self.episodes)
            shifted = sum(r['shifted_native_steps'] for r in self.episodes)
            result.update(actual_shift_observed=shifted > 0, shifted_native_steps=shifted,
                          native_steps=total, exposure_fraction=shifted/total if total else 0,
                          episodes_with_exposure=sum(r['shifted_native_steps'] > 0 for r in self.episodes),
                          shift_transition_count=sum(r['shift_transition_count'] for r in self.episodes),
                          exposure_constraints_met=all(r['exposure_constraint_met'] for r in self.episodes))
            if not result['exposure_constraints_met']:
                result.update(status='failed', error='Exposure constraints not met')
        self.write_json('summary.json', result)
        if not unchanged:
            raise RuntimeError('Checkpoint changed during benchmark evaluation.')
        return result
