"""Stage 3A Hydra runner. Stage 2 analysis/evaluation stays unchanged."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from common.benchmark_config import cli_aliases, normalize, validate_checkpoint_episodic
from common.parser import parse_cfg
from common.seed import set_seed
from common.stage3a import EpisodeAdapter, validate


def json_default(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return str(x)


def episode(agent, env, seed, adapter=None, diagnostic=None):
    set_seed(seed)
    obs = env.reset(seed=seed)
    done, count, native, total = False, 0, 0, 0.
    while not done:
        action = agent.act(obs, t0=count == 0, eval_mode=True, task=None)
        next_obs, reward, done, info = env.step(action)
        records = info['shift_native_steps']
        if not records:
            raise RuntimeError('Native-step records required for interaction accounting.')
        if adapter is not None:
            adapter.observe(obs, action, next_obs, len(records))
        if diagnostic is not None:
            diagnostic(obs, action, next_obs, records)
        obs = next_obs
        total += float(reward)
        count += 1
        native += len(records)
    return dict(seed=seed, success=float(info['success']), episode_return=total,
                episode_length=count, native_steps=native, policy_decisions=count)


def evaluate(agent, adapter, cfg, make_env, baseline_predictor):
    # Evaluation must not perturb the adaptation planner or random streams either.
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    previous_mean = agent._prev_mean.clone()
    outputs = {}
    try:
        with adapter.held_out():
            for condition in ('shifted', 'nominal'):
                evaluation_cfg = deepcopy(cfg)
                if condition == 'nominal':
                    evaluation_cfg.schedule['type'] = 'nominal'
                env = make_env(evaluation_cfg)
                q_sum = q_square_sum = None
                q_count = 0
                def q_input_hook(module, args):
                    nonlocal q_sum, q_square_sum, q_count
                    x = args[0].detach().reshape(-1, args[0].shape[-1]).double()
                    total, squares = x.sum(0), x.square().sum(0)
                    q_sum = total if q_sum is None else q_sum + total
                    q_square_sum = squares if q_square_sum is None else q_square_sum + squares
                    q_count += x.shape[0]
                handle = agent.model._Qs.register_forward_pre_hook(q_input_hook)
                try:
                    errors, current_inputs, pretrained_inputs = [], [], []
                    def diagnostic(obs, action, next_obs, records):
                        z = agent.model.encode(obs.to(agent.device).unsqueeze(0), None)
                        a = action.to(agent.device).unsqueeze(0)
                        target = agent.model.encode(next_obs.to(agent.device).unsqueeze(0), None)
                        prediction = agent.model.next(z, a, None)
                        errors.append((prediction-target).square().mean().item())
                        # Matched commanded-action rollouts isolate predictor-induced latent drift.
                        original, current = z, z
                        for _ in range(cfg.horizon):
                            original = baseline_predictor(torch.cat([original, a], -1))
                            current = agent.model.next(current, a, None)
                            pretrained_inputs.append(original.cpu())
                            current_inputs.append(current.cpu())
                    episodes = [episode(agent, env, seed, diagnostic=diagnostic)
                                for seed in cfg.stage3a['evaluation_seeds']]
                    current, original = torch.cat(current_inputs), torch.cat(pretrained_inputs)
                    if q_count == 0:
                        raise RuntimeError('MPC did not produce Q inputs.')
                    q_mean = q_sum / q_count
                    q_std = (q_square_sum / q_count - q_mean.square()).clamp_min(0).sqrt()
                    outputs[condition] = dict(episodes=episodes,
                        planner_q_input_moments=dict(count=q_count, mean=q_mean.cpu().tolist(), std=q_std.cpu().tolist()),
                        success=float(np.mean([e['success'] for e in episodes])),
                        episode_length=float(np.mean([e['episode_length'] for e in episodes])),
                        held_out_latent_prediction_error=float(np.mean(errors)),
                        predictor_to_q_input_drift=dict(
                            estimator='matched held-out observations; repeated commanded action; planning horizon latent marginal',
                            paired_mse=(current-original).square().mean().item(),
                            mean_l2=(current.mean(0)-original.mean(0)).norm().item(),
                            std_l2=(current.std(0, unbiased=False)-original.std(0, unbiased=False)).norm().item()),
                        native_steps=sum(e['native_steps'] for e in episodes),
                        policy_decisions=sum(e['policy_decisions'] for e in episodes))
                finally:
                    handle.remove()
                    env.close()
    finally:
        agent._prev_mean.copy_(previous_mean)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
    return outputs


@hydra.main(config_path='.', config_name='stage3a', version_base=None)
def main(cfg):
    from hydra.core.hydra_config import HydraConfig
    settings = OmegaConf.to_container(cfg.stage3a, resolve=True)
    validate(settings)
    if cfg.adaptation.mode != 'frozen' or cfg.adaptation_mode != 'frozen':
        raise ValueError('Select methods with stage3a.mode; legacy adaptation groups belong to Stage 2.')
    cfg = normalize(cfg, HydraConfig.get().overrides.task)
    from common.benchmark_validation import validate as validate_shift
    validate_shift(OmegaConf.to_container(cfg, resolve=True))
    if cfg.schedule.type != 'fixed':
        raise ValueError('Stage 3A first experiments require a fixed shifted schedule.')
    if np.array_equal(cfg.dynamics_shift.parameter.nominal, cfg.dynamics_shift.parameter.shifted):
        raise ValueError('A non-identity shifted parameter is required.')
    if cfg.compile:
        raise ValueError('Stage 3A requires compile=false for auditable Q-input instrumentation.')
    if cfg.analysis.enabled:
        raise ValueError('Stage 3A does not run Stage 2 analysis.')
    checkpoint = Path(hydra.utils.to_absolute_path(cfg.checkpoint))
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Pretrained checkpoint not found: {checkpoint}')
    if not torch.cuda.is_available():
        raise RuntimeError('Real TD-MPC2 MPC runs require CUDA; CPU tests only validate the adaptation protocol.')
    validate_checkpoint_episodic(checkpoint, cfg.episodic, torch)
    cfg.checkpoint = str(checkpoint)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    cfg = parse_cfg(cfg)
    if cfg.multitask or cfg.obs != 'state' or not cfg.mpc:
        raise ValueError('Stage 3A requires single-task state observations and MPC.')
    from envs import make_env
    from tdmpc2 import TDMPC2
    from envs.wrappers.dynamics_shift.logging import find_shift
    set_seed(cfg.seed)
    env = make_env(cfg)
    try:
        agent = TDMPC2(cfg)
        agent.load(str(checkpoint))
        baseline = deepcopy(agent.model._dynamics).requires_grad_(False).eval()
        adapter = EpisodeAdapter(agent.model, settings, cfg.consistency_coef, cfg.grad_clip_norm, cfg.seed)
        shift = find_shift(env)
        shift.analysis_capture_history = True
        output = Path(hydra.utils.to_absolute_path(settings['output'])) / settings['mode'] / str(cfg.seed)
        output.mkdir(parents=True, exist_ok=False)
        digest = hashlib.sha256()
        with checkpoint.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                digest.update(chunk)
        (output / 'metadata.json').write_text(json.dumps(dict(config=resolved,
            checkpoint_sha256=digest.hexdigest(), active_method=settings['mode'], replay_protocol=settings,
            evaluation_budget_excluded=True, policy_eval_mode=True), indent=2, default=json_default))
        eval_native = eval_decisions = eval_episodes = 0
        initial_nominal = None
        initial_q_moments = {}
        with (output / 'evaluations.jsonl').open('w') as evaluations, (output / 'diagnostics.jsonl').open('w') as diagnostics:
            def checkpoint_evaluation():
                nonlocal eval_native, eval_decisions, eval_episodes, initial_nominal
                result = evaluate(agent, adapter, cfg, make_env, baseline)
                for value in result.values():
                    eval_native += value['native_steps']
                    eval_decisions += value['policy_decisions']
                    eval_episodes += len(value['episodes'])
                if initial_nominal is None:
                    initial_nominal = result['nominal']['success']
                    initial_q_moments.update({k: deepcopy(v['planner_q_input_moments']) for k, v in result.items()})
                for condition, value in result.items():
                    moments, reference = value['planner_q_input_moments'], initial_q_moments[condition]
                    value['planner_q_input_distribution_drift'] = dict(
                        reference='same method at pretrained evaluation checkpoint, including visitation effects',
                        mean_l2=float(np.linalg.norm(np.array(moments['mean'])-reference['mean'])),
                        std_l2=float(np.linalg.norm(np.array(moments['std'])-reference['std'])))
                row = dict(**adapter.metrics(), evaluation=result,
                    evaluation_native_steps=eval_native, evaluation_policy_decisions=eval_decisions,
                    evaluation_episodes=eval_episodes, evaluation_state_invariance=True,
                    nominal_retention_success_delta=result['nominal']['success']-initial_nominal)
                evaluations.write(json.dumps(row, default=json_default, allow_nan=False)+'\n')
                evaluations.flush()
                print(json.dumps(row, default=json_default, allow_nan=False))
            checkpoint_evaluation()
            for index, seed in enumerate(settings['adaptation_seeds']):
                adapter.begin_episode()
                def diagnostic(obs, action, next_obs, records):
                    diagnostics.write(json.dumps(dict(episode=index, seed=seed, native_records=records), default=json_default)+'\n')
                    shift.drain_events()
                episode(agent, env, seed, adapter=adapter, diagnostic=diagnostic)
                adapter.end_episode()
                diagnostics.flush()
                if (index+1) % settings['eval_every'] == 0 or index+1 == settings['episodes']:
                    checkpoint_evaluation()
        torch.save(dict(predictor=agent.model._dynamics.state_dict(),
                        optimizer=adapter.optimizer.state_dict() if adapter.optimizer else None,
                        metrics=adapter.metrics()), output / 'adapted.pt')
        (output / 'complete.json').write_text(json.dumps(adapter.metrics(), indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    import sys
    sys.argv[1:] = cli_aliases(sys.argv[1:])
    main()
