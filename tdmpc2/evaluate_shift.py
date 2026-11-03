"""Frozen benchmark entrypoint; no adaptation implementation or optimizer calls."""
import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
from pathlib import Path
import json
import hydra
from common.parser import parse_cfg
from common.shift_protocol import make_protocol, Transition
from common.benchmark_logging import BenchmarkLog
from common.benchmark_config import cli_aliases, normalize, validate_checkpoint_episodic
from common.benchmark_metrics import initial_state, compare_initial


def run_condition(cfg, seeds, capture_prefix=False):
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
    analysis = None
    pairing_capture = (capture_prefix and cfg.evaluation['mode'] == 'paired'
                       and cfg.dynamics_shift['parameter']['type'] == 'action_delay'
                       and cfg.schedule['clock'] == 'episode_native_step')
    try:
        shift = find_shift(env)
        agent = TDMPC2(cfg)
        agent.load(cfg.checkpoint)
        log = BenchmarkLog(cfg, seeds)
        log.metadata['native_action_bounds'] = dict(low=shift.action_space.low.tolist(), high=shift.action_space.high.tolist())
        log.write_json('metadata.json', log.metadata)
        if getattr(cfg, 'analysis', {}).get('enabled', False):
            from common.mechanism_analysis import MechanismAnalysis
            analysis = MechanismAnalysis(cfg, agent, log.path, shift.action_space)
            shift.analysis_capture_history = True
        print(f'Benchmark artifacts: {log.path.resolve()}')
        for episode_id, seed in enumerate(seeds):
            prefix = []
            if pairing_capture:
                from common.pairing_validation import native_record
                switch_step = cfg.schedule['switch_step']
                def observe_native(obs, record, queue_before):
                    if record['episode_native_step'] < switch_step:
                        prefix.append(native_record(shift, obs, record, queue_before))
                shift.pairing_observer = observe_native
            if analysis is not None:
                analysis.begin_episode(episode_id, seed)
            set_seed(seed)
            observation = env.reset(seed=seed)
            state = initial_state(env, observation)
            starts.append(state)
            log.write_json(f'initial-{episode_id:04d}.json', state)
            done, total, length, event_count = False, 0., 0, 0
            native_start = shift.global_step
            frames = [env.render()] if cfg.save_video else None
            while not done:
                if analysis is not None:
                    analysis.before_action(length)
                action = agent.act(observation, t0=length == 0, task=None)
                next_observation, reward, done, info = env.step(action)
                records = info['shift_native_steps']
                events = shift.drain_events()
                event_count += sum(e['previous_parameter'] is not None for e in events)
                log.native_steps(records, done)
                log.events(events)
                transition = Transition(observation, action, float(reward), next_observation, bool(done),
                                        {'native_steps': records, 'shift_events': events})
                if analysis is not None:
                    analysis.observe(observation, action, next_observation, reward, done, info, events)
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
            if analysis is not None:
                analysis.end_episode(info['success'])
            if pairing_capture:
                log.write_json(f'pairing-trace-{episode_id:04d}.json', prefix)
            log.episode(dict(checkpoint=str(checkpoint), seed=seed, episode_id=episode_id,
                             shift_id=log.path.name, episode_return=total, success=float(info['success']),
                             episode_length=length, shift_events=event_count,
                             native_start=native_start, native_end=shift.global_step,
                             native_episode_length=shift.episode_step, decision_end=shift.decision_step))
            if frames is not None:
                import imageio
                imageio.mimsave(log.path / f'episode-{episode_id:04d}.mp4', frames, fps=15)
        if analysis is not None:
            analysis.finish()
        result = log.finish()
        output = (log.path, list(log.episodes), starts, result)
        log = None
        if result['status'] != 'complete':
            raise RuntimeError(result['error'])
        print(json.dumps(result['summary_metrics'], indent=2))
        return output
    except Exception as exc:
        if analysis is not None:
            analysis.finish(error=f'{type(exc).__name__}: {exc}')
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
    validate_checkpoint_episodic(checkpoint, cfg.episodic, torch)
    if cfg.evaluation['mode'] == 'standard':
        run_condition(cfg, seeds)
        return
    reference = deepcopy(cfg)
    reference.schedule['type'] = 'nominal'
    reference.benchmark_id += '-nominal'
    reference.evaluation.update(require_shift_exposure=False, min_exposure_fraction=0.)
    capture_prefix = (cfg.dynamics_shift['parameter']['type'] == 'action_delay'
                      and cfg.schedule['type'] == 'abrupt'
                      and cfg.schedule['clock'] == 'episode_native_step')
    nominal = run_condition(reference, seeds, capture_prefix=capture_prefix)
    shifted = run_condition(cfg, seeds, capture_prefix=capture_prefix)
    pairs = []
    compare_prefix = None
    if capture_prefix:
        from common.pairing_validation import compare_prefix
    for a, b, sa, sb in zip(nominal[1], shifted[1], nominal[2], shifted[2]):
        validation = dict(pre_shift_equivalence='UNVERIFIED', first_divergence_step=None,
                          pairing_validity='UNVERIFIED', simulator_full_state='UNVERIFIED')
        if compare_prefix is not None:
            name = f"pairing-trace-{b['episode_id']:04d}.json"
            validation = compare_prefix(json.loads((nominal[0] / name).read_text()),
                                        json.loads((shifted[0] / name).read_text()),
                                        cfg.schedule['switch_step'], cfg.evaluation['pairing_atol'])
        pairs.append(dict(episode_id=b['episode_id'], initial_seed=b['seed'],
                          initial_comparison=compare_initial(sa, sb, cfg.evaluation['pairing_atol']),
                          success_transition=f"{int(a['success'])}->{int(b['success'])}",
                          return_delta=b['episode_return']-a['episode_return'],
                          decision_length_delta=b['episode_length']-a['episode_length'],
                          native_length_delta=b['native_length']-a['native_length'],
                          nominal_exposure=a['exposure_fraction'], shifted_exposure=b['exposure_fraction'],
                          **validation))
    report = dict(protocol_version=2, nominal_run=str(nominal[0].resolve()),
                  shifted_run=str(shifted[0].resolve()), full_state_pairing='UNVERIFIED',
                  pre_shift_equivalence='FAIL' if any(p['pre_shift_equivalence'] == 'FAIL' for p in pairs)
                  else ('PASS_OBSERVED_PREFIX' if compare_prefix is not None and all(p['pre_shift_equivalence'] == 'PASS_OBSERVED_PREFIX' for p in pairs) else 'UNVERIFIED'),
                  first_divergence_step=min((p['first_divergence_step'] for p in pairs if p['first_divergence_step'] is not None), default=None),
                  first_behavioral_divergence_step=min((p['first_behavioral_divergence_step'] for p in pairs
                                                       if p.get('first_behavioral_divergence_step') is not None), default=None),
                  pairing_validity='INVALID_PRE_SHIFT' if any(p['pairing_validity'] == 'INVALID_PRE_SHIFT' for p in pairs) else 'UNVERIFIED',
                  pairs=pairs)
    if getattr(cfg, 'analysis', {}).get('enabled', False):
        from common.mechanism_metrics import paired_summary
        mechanism_nominal = json.loads((nominal[0] / 'mechanism/mechanism_summary.json').read_text())['episodes']
        mechanism_shifted = json.loads((shifted[0] / 'mechanism/mechanism_summary.json').read_text())['episodes']
        a_by_seed = {r['seed']: r for r in mechanism_nominal}
        comparable = [(a_by_seed[r['seed']], r) for r in mechanism_shifted if r['seed'] in a_by_seed]
        metric_differences = {key: paired_summary([b[key]-a[key] for a,b in comparable])
                              for key in ('latent_prediction_mse', 'reward_absolute_error')}
        report['mechanism_pairing'] = dict(status='approximate_pairing',
            mechanism_episode_differences=metric_differences,
            common_action_branch=False, return_difference=paired_summary([p['return_delta'] for p in pairs]),
            interpretation='Seed-paired closed-loop outcomes; no exact state or candidate-return claim')
    (shifted[0] / 'paired.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    import sys
    sys.argv[1:] = cli_aliases(sys.argv[1:])
    evaluate_shift()
