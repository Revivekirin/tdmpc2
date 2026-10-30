"""Frozen benchmark entrypoint; no adaptation implementation or optimizer calls."""
import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
from pathlib import Path
import json
import hydra
from common.parser import parse_cfg
from common.shift_protocol import make_protocol, Transition
from common.benchmark_logging import BenchmarkLog
from common.benchmark_config import cli_aliases, normalize
from common.benchmark_metrics import initial_state, compare_initial


def run_condition(cfg, seeds):
    from common.seed import set_seed
    from tdmpc2 import TDMPC2
    from envs import make_env
    from envs.wrappers.dynamics_shift.logging import find_shift, capture_environment, encode_state
    protocol = make_protocol(cfg.adaptation_mode)
    checkpoint = Path(cfg.checkpoint)
    set_seed(cfg.seed)
    env = make_env(cfg)
    # Constructing the wrapper validates action dimensions and schedule values
    # before loading the frozen model.
    log = None
    starts = []
    try:
        shift = find_shift(env)
        agent = TDMPC2(cfg)
        agent.load(cfg.checkpoint)
        log = BenchmarkLog(cfg, seeds)
        log.metadata['native_action_bounds'] = dict(low=shift.action_space.low.tolist(), high=shift.action_space.high.tolist())
        log.write_json('metadata.json', log.metadata)
        print(f'Benchmark artifacts: {log.path.resolve()}')
        for episode_id, seed in enumerate(seeds):
            set_seed(seed)
            observation = env.reset(seed=seed)
            state = initial_state(env, observation)
            starts.append(state)
            log.write_json(f'initial-{episode_id:04d}.json', state)
            done, total, length, event_count = False, 0., 0, 0
            native_start = shift.global_step
            frames = [env.render()] if cfg.save_video else None
            while not done:
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
        output = (log.path, list(log.episodes), starts, result)
        log = None
        if result['status'] != 'complete':
            raise RuntimeError(result['error'])
        print(json.dumps(result['summary_metrics'], indent=2))
        return output
    except Exception as exc:
        if log is not None:
            log.finish(error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        env.close()


@hydra.main(config_path='.', config_name='config', version_base=None)
def evaluate_shift(cfg):
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    from copy import deepcopy
    cfg = normalize(cfg, HydraConfig.get().overrides.task)
    # Schedule validation is NumPy-only, loaded independently of optional env stacks.
    from common.benchmark_validation import validate
    validate(OmegaConf.to_container(cfg, resolve=True))
    seeds = list(cfg.eval_seeds)
    checkpoint = Path(hydra.utils.to_absolute_path(cfg.checkpoint))
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    cfg.checkpoint = str(checkpoint)
    cfg = parse_cfg(cfg)
    if cfg.multitask:
        raise ValueError('Benchmark evaluation requires a single ManiSkill task.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('The unchanged TD-MPC2 agent requires CUDA for checkpoint evaluation.')
    if cfg.evaluation['mode'] == 'standard':
        run_condition(cfg, seeds)
        return
    reference = deepcopy(cfg)
    reference.schedule['type'] = 'nominal'
    reference.benchmark_id += '-nominal'
    reference.evaluation.update(require_shift_exposure=False, min_exposure_fraction=0.)
    nominal = run_condition(reference, seeds)
    shifted = run_condition(cfg, seeds)
    pairs = []
    for a, b, sa, sb in zip(nominal[1], shifted[1], nominal[2], shifted[2]):
        pairs.append(dict(episode_id=b['episode_id'], initial_seed=b['seed'],
                          initial_comparison=compare_initial(sa, sb, cfg.evaluation['pairing_atol']),
                          success_transition=f"{int(a['success'])}->{int(b['success'])}",
                          return_delta=b['episode_return']-a['episode_return'],
                          decision_length_delta=b['episode_length']-a['episode_length'],
                          native_length_delta=b['native_length']-a['native_length'],
                          nominal_exposure=a['exposure_fraction'], shifted_exposure=b['exposure_fraction']))
    report = dict(protocol_version=2, nominal_run=str(nominal[0].resolve()),
                  shifted_run=str(shifted[0].resolve()), full_state_pairing='UNVERIFIED', pairs=pairs)
    (shifted[0] / 'paired.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    import sys
    sys.argv[1:] = cli_aliases(sys.argv[1:])
    evaluate_shift()
