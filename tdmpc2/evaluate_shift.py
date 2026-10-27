"""Frozen benchmark entrypoint; no adaptation implementation or optimizer calls."""
import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
from pathlib import Path
import json
import hydra
import numpy as np
import torch
from common.parser import parse_cfg
from common.seed import set_seed
from common.shift_protocol import make_protocol, evaluation_seeds, validate_probe_holdout, Transition
from common.benchmark_logging import BenchmarkLog
from envs import make_env
from envs.wrappers.dynamics_shift.schedule import Schedule
from envs.wrappers.dynamics_shift.logging import find_shift, capture_environment, encode_state
from tdmpc2 import TDMPC2


@hydra.main(config_path='.', config_name='config', version_base=None)
def evaluate_shift(cfg):
    cfg.dynamics_shift.enabled = True
    protocol = make_protocol(cfg.adaptation_mode)
    seeds = evaluation_seeds(cfg.seed, cfg.eval_episodes, cfg.eval_seeds)
    schedule = Schedule(dict(cfg.dynamics_shift.schedule), cfg.dynamics_shift.parameter.nominal,
                        cfg.dynamics_shift.parameter.shifted)
    validate_probe_holdout(schedule, cfg.benchmark_split, cfg.benchmark_probe_values)
    checkpoint = Path(hydra.utils.to_absolute_path(cfg.checkpoint))
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    cfg.checkpoint = str(checkpoint)
    cfg = parse_cfg(cfg)
    if cfg.multitask:
        raise ValueError('Benchmark evaluation requires a single ManiSkill task.')
    if not torch.cuda.is_available():
        raise RuntimeError('The unchanged TD-MPC2 agent requires CUDA for checkpoint evaluation.')
    set_seed(cfg.seed)
    env = make_env(cfg)
    log = None
    try:
        shift = find_shift(env)
        agent = TDMPC2(cfg)
        agent.load(cfg.checkpoint)
        log = BenchmarkLog(cfg, seeds)
        print(f'Benchmark artifacts: {log.path.resolve()}')
        for episode_id, seed in enumerate(seeds):
            set_seed(seed)
            observation = env.reset(seed=seed)
            done, total, length, event_count = False, 0., 0, 0
            native_start = shift.global_step
            frames = [env.render()] if cfg.save_video else None
            while not done:
                # Match evaluate.py inference arguments and MPC configuration exactly.
                action = agent.act(observation, t0=length == 0, task=None)
                next_observation, reward, done, info = env.step(action)
                records = info['shift_native_steps']
                events = shift.drain_events()
                event_count += sum(e['previous_parameter'] is not None for e in events)
                log.native_steps(records, done)
                log.events(events)
                transition = Transition(observation, action, float(reward), next_observation, bool(done),
                                        {'native_steps': records, 'shift_events': events})
                protocol.observe(transition)
                log.transition(transition)
                if events and cfg.benchmark_save_segment_state:
                    snapshot = capture_environment(env)
                    log.write_json(f'environment-{shift.global_step:09d}.json',
                                   dict(captured_after_decision=True, triggering_events=events,
                                        state=encode_state(snapshot)))
                total += float(reward)
                length += 1
                observation = next_observation
                if frames is not None:
                    frames.append(env.render())
            log.episode(dict(checkpoint=str(checkpoint), seed=seed, episode_id=episode_id,
                             shift_id=log.path.name, episode_return=total, success=float(info['success']),
                             episode_length=length, shift_events=event_count,
                             native_start=native_start, native_end=shift.global_step,
                             native_episode_length=shift.episode_step, decision_end=shift.decision_step))
            if frames is not None:
                import imageio
                imageio.mimsave(log.path / f'episode-{episode_id:04d}.mp4', frames, fps=15)
        result = log.finish()
        log = None
        print(json.dumps(result['summary_metrics'], indent=2))
    except Exception as exc:
        if log is not None:
            log.finish(error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        env.close()


if __name__ == '__main__':
    evaluate_shift()
