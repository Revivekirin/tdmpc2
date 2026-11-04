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
from common.stage3a import EpisodeAdapter, validate, NativeBudgetMonitor, NativeBudgetReached


def json_default(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return str(x)


def episode(agent, env, seed, adapter=None, diagnostic=None, planner_metrics=None):
    set_seed(seed)
    obs = env.reset(seed=seed)
    done, count, native, total = False, 0, 0, 0.
    while not done:
        if planner_metrics is not None and count == 0:
            from common.planner_trace import PlannerTrace
            trace = PlannerTrace()
            old_trace = getattr(agent, '_mechanism_trace', None)
            agent._mechanism_trace = trace
            try:
                action = agent.act(obs, t0=count == 0, eval_mode=True, task=None)
                planner_metrics.extend(dict(seed=seed, policy_decision=count, **row)
                                       for row in trace.summaries())
            finally:
                agent._mechanism_trace = old_trace
        else:
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


def mse_statistics(values):
    """Descriptive population statistics, not an uncertainty estimate (ddof=0)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError('MSE statistics require nonempty finite scalar errors.')
    return dict(count=int(values.size), mean=float(values.mean()),
                variance=float(values.var(ddof=0)), std=float(values.std(ddof=0)),
                p95=float(np.percentile(values, 95, method='linear')), ddof=0)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_data_identity(cfg, checkpoint_hash):
    return dict(checkpoint_sha256=checkpoint_hash, task=cfg.task, model_size=cfg.model_size,
                episodic=cfg.episodic, obs=cfg.obs, dynamics_shift=cfg.dynamics_shift,
                schedule=cfg.schedule, seeds=cfg.stage3a['fixed_data_seeds'],
                collector='pretrained Frozen MPC', policy_eval_mode=True,
                planner={k: getattr(cfg, k) for k in ('horizon', 'iterations', 'num_samples',
                    'num_elites', 'num_pi_trajs', 'temperature', 'min_std', 'max_std')})


def collect_fixed_data(agent, adapter, cfg, make_env, path, identity):
    """Explicit one-time collection, outside adaptation; never overwrite an artifact."""
    if path.exists():
        raise FileExistsError(path)
    env = make_env(deepcopy(cfg))
    transitions, ids, episodes = [], [], []
    try:
        with adapter.held_out():
            for index, seed in enumerate(cfg.stage3a['fixed_data_seeds']):
                def capture(obs, action, next_obs, records):
                    transitions.append(tuple(x.detach().cpu().clone() for x in (obs, action, next_obs)))
                    ids.append(index)
                episodes.append(episode(agent, env, seed, diagnostic=capture))
        payload = dict(version=1, identity=identity, episodes=episodes,
                       observation=torch.stack([t[0] for t in transitions]),
                       action=torch.stack([t[1] for t in transitions]),
                       next_observation=torch.stack([t[2] for t in transitions]),
                       episode_ids=torch.tensor(ids),
                       collection_native_steps=sum(e['native_steps'] for e in episodes),
                       collection_policy_decisions=len(transitions))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            torch.save(payload, stream)
        print(json.dumps(dict(fixed_data_path=str(path), sha256=file_sha256(path),
                              native_steps=payload['collection_native_steps'],
                              policy_decisions=len(transitions))))
    finally:
        env.close()


def fixed_prediction_metrics(agent, baseline, dataset):
    errors = {'initial': [], 'current': []}
    with torch.no_grad():
        for start in range(0, len(dataset['action']), 256):
            obs, action, next_obs = [dataset[k][start:start+256].to(agent.device)
                                     for k in ('observation', 'action', 'next_observation')]
            z, target = agent.model.encode(obs, None), agent.model.encode(next_obs, None)
            predictions = dict(initial=baseline(torch.cat([z, action], -1)),
                               current=agent.model.next(z, action, None))
            for name, prediction in predictions.items():
                errors[name].extend((prediction-target).square().mean(-1).cpu().tolist())
    ids = dataset['episode_ids'].numpy()
    output = {}
    for name, values in errors.items():
        values = np.asarray(values)
        means = [float(values[ids == i].mean()) for i in range(len(dataset['episodes']))]
        output[name] = dict(transition=mse_statistics(values), episode_means=mse_statistics(means),
                           per_episode_means=means)
    output['paired_transition_delta'] = mse_statistics(np.array(errors['current'])-errors['initial'])
    output['paired_episode_delta'] = mse_statistics(
        np.array(output['current']['per_episode_means'])-output['initial']['per_episode_means'])
    return output


def evaluate(agent, adapter, cfg, make_env, baseline_predictor):
    # Evaluation must not perturb the adaptation planner or random streams either.
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    previous_mean = agent._prev_mean.clone()
    outputs = {}
    try:
        with adapter.held_out(allow_in_episode=bool(cfg.stage3a.get('native_budgets'))):
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
                    episodes, planner_metrics = [], []
                    for seed in cfg.stage3a['evaluation_seeds']:
                        start = len(errors)
                        result = episode(agent, env, seed, diagnostic=diagnostic,
                                         planner_metrics=planner_metrics if cfg.stage3a.get('planner_trace') else None)
                        result['latent_prediction_mse'] = mse_statistics(errors[start:])
                        episodes.append(result)
                    mse_stats = dict(
                        transition=mse_statistics(errors),
                        episode_means=mse_statistics([
                            e['latent_prediction_mse']['mean'] for e in episodes]))
                    current, original = torch.cat(current_inputs), torch.cat(pretrained_inputs)
                    if q_count == 0:
                        raise RuntimeError('MPC did not produce Q inputs.')
                    q_mean = q_sum / q_count
                    q_std = (q_square_sum / q_count - q_mean.square()).clamp_min(0).sqrt()
                    successful = [e for e in episodes if e['success']]
                    outputs[condition] = dict(episodes=episodes,
                        successful_completion_length=(float(np.mean([e['episode_length'] for e in successful])) if successful else None),
                        successful_completion_native_steps=(float(np.mean([e['native_steps'] for e in successful])) if successful else None),
                        planner_trace_summaries=planner_metrics,
                        planner_q_input_moments=dict(count=q_count, mean=q_mean.cpu().tolist(), std=q_std.cpu().tolist()),
                        success=float(np.mean([e['success'] for e in episodes])),
                        episode_length=float(np.mean([e['episode_length'] for e in episodes])),
                        held_out_latent_prediction_error=mse_stats['transition']['mean'],
                        held_out_latent_prediction_stats=mse_stats,
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
    if type(settings['seed_set']) is not int or settings['seed_set'] < 0:
        raise ValueError('seed_set must be a nonnegative integer.')
    offset = settings['seed_set'] * settings['seed_set_stride']
    if settings['adaptation_seed_start'] is not None:
        settings['adaptation_seeds'] = list(range(settings['adaptation_seed_start'],
            settings['adaptation_seed_start'] + settings['episodes']))
    for key in ('adaptation_seeds', 'evaluation_seeds', 'fixed_data_seeds'):
        settings[key] = [s + offset for s in settings[key]]
    validate(settings)
    cfg.stage3a = OmegaConf.create(settings)
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
        checkpoint_hash = file_sha256(checkpoint)
        identity = fixed_data_identity(cfg, checkpoint_hash)
        fixed_path = (Path(hydra.utils.to_absolute_path(settings['fixed_data_path']))
                      if settings['fixed_data_path'] else None)
        if settings['fixed_data_collect']:
            if fixed_path is None or not settings['fixed_data_seeds']:
                raise ValueError('Collection needs fixed_data_path and fixed_data_seeds.')
            collect_fixed_data(agent, adapter, cfg, make_env, fixed_path, identity)
            return
        fixed_dataset = None
        fixed_hash = None
        if fixed_path is not None:
            fixed_hash = file_sha256(fixed_path)
            fixed_dataset = torch.load(fixed_path, map_location='cpu', weights_only=False)
            if fixed_dataset['version'] != 1 or fixed_dataset['identity'] != identity:
                raise ValueError('Fixed-data checkpoint/shift/planner/seeds provenance mismatch.')
        output = Path(hydra.utils.to_absolute_path(settings['output']))
        if settings['native_budgets']:
            output = output / f"set-{settings['seed_set']}"
        output = output / settings['mode'] / str(cfg.seed)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'metadata.json').write_text(json.dumps(dict(config=resolved,
            checkpoint_sha256=checkpoint_hash, active_method=settings['mode'], replay_protocol=settings,
            fixed_data_sha256=fixed_hash, fixed_data_path=str(fixed_path),
            fixed_data_collection_native_steps=fixed_dataset['collection_native_steps'] if fixed_dataset else 0,
            fixed_data_collection_policy_decisions=fixed_dataset['collection_policy_decisions'] if fixed_dataset else 0,
            evaluation_budget_excluded=True, policy_eval_mode=True,
            budget_timing='exact native step, before decision commit and episode-end update'),
            indent=2, default=json_default))
        eval_native = eval_decisions = eval_episodes = 0
        initial_nominal = None
        initial_q_moments = {}
        monitor = None
        with (output / 'evaluations.jsonl').open('w') as evaluations, (output / 'diagnostics.jsonl').open('w') as diagnostics, (output / 'adaptation-episodes.jsonl').open('w') as episode_log:
            def checkpoint_evaluation(native_budget=None):
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
                fixed_metrics = None
                if fixed_dataset is not None:
                    if file_sha256(fixed_path) != fixed_hash:
                        raise RuntimeError('Fixed held-out dataset changed during the run.')
                    with adapter.held_out(allow_in_episode=bool(settings['native_budgets'])):
                        fixed_metrics = fixed_prediction_metrics(agent, baseline, fixed_dataset)
                metrics = adapter.metrics()
                if native_budget is not None:
                    metrics.update(replay_native_steps=adapter.native_steps,
                        adaptation_native_steps=native_budget,
                        adaptation_policy_decisions=monitor.decisions_started if monitor else 0,
                        completed_policy_decisions=adapter.transitions,
                        in_flight_decisions=(monitor.decisions_started-adapter.transitions) if monitor else 0,
                        episode_in_progress=adapter.in_episode)
                row = dict(**metrics, evaluation=result,
                    native_budget=native_budget, fixed_data_sha256=fixed_hash,
                    fixed_data_prediction=fixed_metrics,
                    evaluation_native_steps=eval_native, evaluation_policy_decisions=eval_decisions,
                    evaluation_episodes=eval_episodes, evaluation_state_invariance=True,
                    nominal_retention_success_delta=result['nominal']['success']-initial_nominal)
                evaluations.write(json.dumps(row, default=json_default, allow_nan=False)+'\n')
                evaluations.flush()
                if settings['native_budgets']:
                    with (output / f'predictor-budget-{native_budget}.pt').open('xb') as stream:
                        torch.save(dict(predictor=agent.model._dynamics.state_dict(),
                            checkpoint_sha256=checkpoint_hash, native_budget=native_budget,
                            metrics=metrics), stream)
                print(json.dumps(row, default=json_default, allow_nan=False))
            checkpoint_evaluation(0 if settings['native_budgets'] else None)
            if settings['native_budgets']:
                monitor = NativeBudgetMonitor(settings['native_budgets'], checkpoint_evaluation,
                    lambda record: diagnostics.write(json.dumps(record, default=json_default)+'\n'))
                if getattr(shift, 'pairing_observer', None) is not None:
                    raise RuntimeError('Native observer is already in use.')
                shift.pairing_observer = monitor
            try:
                for index, seed in enumerate(settings['adaptation_seeds']):
                    adapter.begin_episode()
                    def diagnostic(obs, action, next_obs, records):
                        if monitor is None:
                            diagnostics.write(json.dumps(dict(episode=index, seed=seed, native_records=records), default=json_default)+'\n')
                        shift.drain_events()
                    episode_result = episode(agent, env, seed, adapter=adapter, diagnostic=diagnostic)
                    adapter.end_episode()
                    episode_log.write(json.dumps(dict(**episode_result, **adapter.metrics()))+'\n')
                    episode_log.flush()
                    diagnostics.flush()
                    if monitor is None and ((index+1) % settings['eval_every'] == 0 or index+1 == settings['episodes']):
                        checkpoint_evaluation()
            except NativeBudgetReached:
                pass  # Do not end/update an incomplete episode or insert a partial decision.
            finally:
                if monitor is not None:
                    shift.pairing_observer = None
            if monitor is not None and monitor.index != len(monitor.budgets):
                raise RuntimeError('Episode safety limit exhausted before native budget; increase episodes.')
        completion = adapter.metrics()
        if monitor is not None:
            completion.update(replay_native_steps=adapter.native_steps,
                adaptation_native_steps=monitor.native_steps,
                adaptation_policy_decisions=monitor.decisions_started,
                completed_policy_decisions=adapter.transitions,
                in_flight_decisions=monitor.decisions_started-adapter.transitions,
                episode_in_progress=adapter.in_episode)
        torch.save(dict(predictor=agent.model._dynamics.state_dict(),
                        optimizer=adapter.optimizer.state_dict() if adapter.optimizer else None,
                        checkpoint_sha256=checkpoint_hash, fixed_data_sha256=fixed_hash,
                        metrics=completion), output / 'adapted.pt')
        (output / 'complete.json').write_text(json.dumps(completion, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    import sys
    sys.argv[1:] = cli_aliases(sys.argv[1:])
    main()
