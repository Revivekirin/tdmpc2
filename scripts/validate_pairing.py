"""Checkpoint-backed pre-shift A/A validation using the benchmark Hydra config.

Example (server):
  python scripts/validate_pairing.py --seed 7 --out results/pairing-validation.json \
    task=pick-cube checkpoint=/nfs4/jhkim/repos/tdmpc2/checkpoints/maniskill2/pick-cube-1.pt \
    episodic=false compile=false model_size=5 dynamics_shift=action_delay \
    schedule=abrupt dynamics_shift.parameter.shifted=2 evaluation=paired \
    evaluation.episodes=1 seed=1 mpc=true analysis=frozen_mechanism
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tdmpc2'))


def run_prefix(cfg, seed, analysis_enabled, out, action_sequence=None, wrapper_enabled=True,
               warmup_seeds=(), warmup_action_sequences=None):
    cfg = deepcopy(cfg)  # TDMPC2.__init__ may adjust planner iteration count.
    cfg.analysis['enabled'] = analysis_enabled
    import numpy as np
    import torch
    from common.seed import set_seed
    from common.benchmark_metrics import initial_state
    from common.pairing_validation import native_record, rng_digest
    from envs import make_env
    from envs.maniskill import MANISKILL_TASKS, ManiSkillWrapper, legacy_gym
    from envs.wrappers.timeout import Timeout
    from envs.wrappers.tensor import TensorWrapper
    from envs.wrappers.dynamics_shift.logging import encode_state
    from envs.wrappers.dynamics_shift.logging import find_shift
    from tdmpc2 import TDMPC2
    from common.mechanism_analysis import MechanismAnalysis

    set_seed(cfg.seed)
    trace = []
    if wrapper_enabled:
        env = make_env(cfg)
        shift = find_shift(env)
    else:
        if analysis_enabled:
            raise ValueError('Disabled-wrapper calibration does not run mechanism analysis.')
        task = MANISKILL_TASKS[cfg.task]
        raw = legacy_gym.make(task['env'], obs_mode='state', control_mode=task['control_mode'],
                              render_camera_cfgs=dict(width=384, height=384))
        class PassiveCapture:
            def __init__(self, native):
                self.env = native
                self.action_space, self.observation_space = native.action_space, native.observation_space
                self.episode_step = 0
            @property
            def unwrapped(self):
                return self.env.unwrapped
            def reset(self, **kwargs):
                self.episode_step = 0
                return self.env.reset(**kwargs)
            def step(self, action):
                obs, reward, done, info = self.env.step(action)
                if self.episode_step < int(cfg.schedule['switch_step']):
                    native = self.unwrapped
                    record = dict(native_step=self.episode_step, parameter=0.,
                                  commanded_action=encode_state(np.asarray(action).copy()),
                                  executed_action=encode_state(np.asarray(action).copy()),
                                  delay_queue_before=None, delay_queue_after=None,
                                  observation=encode_state(np.asarray(obs).copy()),
                                  simulator=encode_state(native.get_state()),
                                  controller=encode_state(native.agent.controller.get_state()),
                                  rng=encode_state({k: getattr(native, k).get_state() for k in ('_main_rng', '_episode_rng')}),
                                  unavailable={}, policy_rng_after_step=rng_digest())
                    trace.append(record)
                self.episode_step += 1
                return obs, reward, done, info
            def close(self):
                return self.env.close()
        shift = PassiveCapture(raw)
        env = TensorWrapper(Timeout(ManiSkillWrapper(shift, cfg), max_episode_steps=100))
        cfg.obs_shape = {cfg.obs: env.observation_space.shape}
        cfg.action_dim = env.action_space.shape[0]
        cfg.episode_length = env.max_episode_steps
        cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    agent = TDMPC2(cfg)
    agent.load(cfg.checkpoint)
    analysis = None
    decisions = []
    warmup_actions = []
    switch = int(cfg.schedule['switch_step'])
    def observe_native(obs, record, queue_before):
        if record['episode_native_step'] < switch:
            trace.append(native_record(shift, obs, record, queue_before))
    try:
        if analysis_enabled:
            out.mkdir(parents=True, exist_ok=True)
            analysis = MechanismAnalysis(cfg, agent, out, shift.action_space)
            shift.analysis_capture_history = True
        for episode_id, previous_seed in enumerate(warmup_seeds):
            if analysis is not None:
                analysis.begin_episode(episode_id, previous_seed)
            set_seed(previous_seed)
            previous_observation = env.reset(seed=previous_seed)
            previous_done = False
            previous_actions = []
            while not previous_done:
                step = len(previous_actions)
                if analysis is not None:
                    analysis.before_action(step)
                if warmup_action_sequences is None:
                    previous_action = agent.act(previous_observation, t0=step == 0, task=None)
                else:
                    if step >= len(warmup_action_sequences[episode_id]):
                        raise RuntimeError(f'Warmup episode {episode_id} exceeded its supplied action sequence.')
                    previous_action = torch.as_tensor(warmup_action_sequences[episode_id][step])
                following_observation, reward, previous_done, info = env.step(previous_action)
                previous_actions.append(previous_action.detach().cpu().numpy().tolist())
                events = shift.drain_events() if wrapper_enabled else []
                if analysis is not None:
                    analysis.observe(previous_observation, previous_action, following_observation,
                                     reward, previous_done, info, events)
                previous_observation = following_observation
            if analysis is not None:
                analysis.end_episode(info['success'])
            warmup_actions.append(previous_actions)
        trace.clear()
        if wrapper_enabled:
            shift.pairing_observer = observe_native
        if analysis is not None:
            analysis.begin_episode(len(warmup_seeds), seed)
        set_seed(seed)
        observation = env.reset(seed=seed)
        if wrapper_enabled:
            initial = initial_state(env, observation)
        else:
            native = shift.unwrapped
            initial = dict(observation=encode_state(observation.detach().cpu().numpy()),
                           simulator=encode_state(native.get_state()),
                           controller=encode_state(native.agent.controller.get_state()),
                           rng=encode_state({k: getattr(native, k).get_state() for k in ('_main_rng', '_episode_rng')}),
                           unavailable={})
        done = False
        while shift.episode_step < switch and not done and (action_sequence is None or len(decisions) < len(action_sequence)):
            decision = len(decisions)
            if analysis is not None:
                analysis.before_action(decision)
            rng_before = rng_digest()
            action = (agent.act(observation, t0=decision == 0, task=None) if action_sequence is None
                      else torch.as_tensor(action_sequence[decision]['action']))
            rng_after = rng_digest()
            planner = getattr(agent, '_mechanism_trace', None)
            candidate_hash = None
            if planner is not None and planner.iterations:
                candidate_hash = hashlib.sha256(np.ascontiguousarray(planner.iterations[0]['actions']).tobytes()).hexdigest()
            next_observation, reward, done, info = env.step(action)
            decisions.append(dict(decision=decision, action=action.detach().cpu().numpy().tolist(),
                                  rng_before=rng_before, rng_after=rng_after,
                                  first_iteration_candidates_sha256=candidate_hash))
            events = shift.drain_events() if wrapper_enabled else []
            if analysis is not None:
                analysis.observe(observation, action, next_observation, reward, done, info, events)
            observation = next_observation
        if analysis is not None:
            analysis.end_episode(info['success'] if decisions else False)
            analysis.finish()
        return dict(initial=initial, native=trace, decisions=decisions, warmup_actions=warmup_actions,
                    reached_switch=shift.episode_step >= switch, ended=bool(done))
    finally:
        env.close()


def compare_decisions(a, b, atol, fields=('action', 'rng_before', 'rng_after', 'first_iteration_candidates_sha256')):
    from common.pairing_validation import equal
    n = min(len(a), len(b))
    for i in range(n):
        differences = [field for field in fields
                       if a[i][field] != b[i][field] and not (field == 'action' and equal(a[i][field], b[i][field], atol))]
        if differences:
            return dict(status='FAIL', first_decision=i, fields=differences, compared_decisions=i + 1)
    return dict(status='PASS_OBSERVED_PREFIX' if len(a) == len(b) else 'UNVERIFIED',
                first_decision=None, fields=[], compared_decisions=n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=Path, default=Path('results/pairing-validation.json'))
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--warmup-seeds', default='',
                        help='Comma-separated prior episode seeds, e.g. 1,2,3,4,5,6 for target seed 7.')
    args, overrides = parser.parse_known_args()
    if args.atol < 0:
        parser.error('--atol must be nonnegative')
    warmup_seeds = [int(value) for value in args.warmup_seeds.split(',') if value]
    if args.seed in warmup_seeds or len(set(warmup_seeds)) != len(warmup_seeds):
        parser.error('Warmup seeds must be distinct and exclude --seed.')
    import hydra
    import torch
    from omegaconf import OmegaConf
    from common.benchmark_config import cli_aliases, normalize, validate_checkpoint_episodic
    from common.benchmark_validation import validate
    from common.parser import parse_cfg
    from common.benchmark_metrics import compare_initial
    from common.pairing_validation import compare_prefix
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA and the checkpoint are required for this validation.')
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'tdmpc2'), version_base=None):
        raw = hydra.compose(config_name='config', overrides=cli_aliases(overrides))
    cfg = normalize(raw, overrides)
    validate(OmegaConf.to_container(cfg, resolve=True))
    cfg.checkpoint = str(Path(cfg.checkpoint).resolve())
    validate_checkpoint_episodic(Path(cfg.checkpoint), cfg.episodic, torch)
    with patch('hydra.utils.get_original_cwd', return_value=str(ROOT)):
        cfg = parse_cfg(cfg)
    if cfg.dynamics_shift['parameter']['type'] != 'action_delay' or cfg.schedule['type'] != 'abrupt' or cfg.schedule['clock'] != 'episode_native_step':
        raise ValueError('Use action_delay with an abrupt episode_native_step schedule.')
    nominal = deepcopy(cfg)
    nominal.schedule['type'] = 'nominal'
    # Both runs use the benchmark's enabled action-delay wrapper. The nominal
    # schedule has max_delay=0; abrupt keeps the inert queue until switch_step.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        aa1 = run_prefix(nominal, args.seed, True, root / 'aa1', warmup_seeds=warmup_seeds)
        aa2 = run_prefix(nominal, args.seed, True, root / 'aa2', warmup_seeds=warmup_seeds)
        replay = run_prefix(nominal, args.seed, False, root / 'replay', aa1['decisions'],
                            warmup_seeds=warmup_seeds, warmup_action_sequences=aa1['warmup_actions'])
        disabled = run_prefix(nominal, args.seed, False, root / 'disabled', aa1['decisions'],
                              wrapper_enabled=False, warmup_seeds=warmup_seeds,
                              warmup_action_sequences=aa1['warmup_actions'])
        abrupt = run_prefix(cfg, args.seed, True, root / 'abrupt', warmup_seeds=warmup_seeds)
        no_analysis = run_prefix(nominal, args.seed, False, root / 'no_analysis', warmup_seeds=warmup_seeds)
    initial = compare_initial(aa1['initial'], abrupt['initial'], args.atol)
    report = dict(schema_version=1, seed=args.seed, warmup_seeds=warmup_seeds, checkpoint=cfg.checkpoint,
                  task=cfg.task, switch_step=cfg.schedule['switch_step'], atol=args.atol,
                  nominal_nominal_open_loop=compare_prefix(aa1['native'], replay['native'], cfg.schedule['switch_step'], args.atol, compare_policy_rng=False),
                  disabled_vs_enabled_nominal_wrapper=compare_prefix(disabled['native'], replay['native'], cfg.schedule['switch_step'], args.atol),
                  nominal_enabled_nominal_wrapper=compare_prefix(aa1['native'], abrupt['native'], cfg.schedule['switch_step'], args.atol),
                  closed_loop_mpc_rng=compare_decisions(aa1['decisions'], aa2['decisions'], args.atol),
                  analysis_on_off_action=compare_decisions(aa1['decisions'], no_analysis['decisions'], args.atol, ('action',)),
                  initial_comparisons=dict(nominal_nominal=compare_initial(aa1['initial'], aa2['initial'], args.atol),
                                           disabled_enabled=compare_initial(disabled['initial'], replay['initial'], args.atol),
                                           nominal_abrupt=initial),
                  first_divergence_evidence=compare_prefix(aa1['native'], abrupt['native'], cfg.schedule['switch_step'], args.atol),
                  reached_switch={name: run['reached_switch'] for name, run in
                                  (('nominal_a', aa1), ('nominal_b', aa2), ('nominal_replay', replay), ('disabled_wrapper', disabled), ('abrupt', abrupt), ('analysis_off', no_analysis))},
                  simulator_full_state='UNVERIFIED',
                  limitation='Exposed state and RNG only; hidden SAPIEN/PhysX state is not certified.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(args.out.resolve())


if __name__ == '__main__':
    main()
