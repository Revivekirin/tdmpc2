"""Stage 3B orchestration; Stage 3A optimizer, loss and held-out evaluator are reused."""
from copy import deepcopy
import json
from pathlib import Path
import random
import numpy as np
import torch


class BudgetPause(Exception):
    """Suspend a repeat without discarding its pending native results."""


def resumable_repeat(wrapper, action):
    """Same two native calls/reward sum/termination as ManiSkillWrapper.step.

    A checkpoint runs after the native wrapper has advanced its counters, but
    before the decision returns. The pending result survives even at repeat=2.
    """
    pending = wrapper.__dict__.get('_stage3b_pending')
    if pending is None:
        wrapper.env.begin_decision()
        pending = dict(action=np.asarray(action).copy(), records=[], reward=0., done=False)
        wrapper._stage3b_pending = pending
    elif not np.array_equal(action, pending['action']):
        raise ValueError('Resume must finish the saved commanded action.')
    while len(pending['records']) < 2 and not pending['done']:
        obs, reward, done, info = wrapper.env.step(action)
        pending.update(obs=obs, reward=pending['reward'] + reward, done=done, info=info)
        pending['records'].append(info['shift_record'])
        wrapper.native_checkpoint(info['shift_record'])
    wrapper.env.end_decision()
    info = dict(pending['info'], terminated=pending['done'], shift_native_steps=pending['records'])
    result = pending['obs'], pending['reward'], pending['done'], info
    wrapper._stage3b_pending = None
    return result


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def run(agent, adapter, env, cfg, make_env, baseline, checkpoint_hash,
        fixed_dataset, fixed_hash, fixed_path, output, resolved):
    from adapt_shift import evaluate, fixed_prediction_metrics, file_sha256, json_default
    from common.seed import set_seed
    from envs.wrappers.dynamics_shift.logging import capture_environment, restore_environment, find_shift
    import hydra

    settings = cfg.stage3a
    decision_ttt = settings['mode'] == 'decision_ttt'
    if not settings['native_budgets'] or fixed_dataset is None:
        raise ValueError('Stage 3B requires native budgets and shared fixed data.')
    shift = find_shift(env)
    wrapper = env
    while wrapper is not None and wrapper.__class__.__name__ != 'ManiSkillWrapper':
        wrapper = wrapper.__dict__.get('env')
    if wrapper is None:
        raise ValueError('Stage 3B resumable native budgets require ManiSkillWrapper.')
    budgets = [0] + list(settings['native_budgets'])
    identity = dict(checkpoint_sha256=checkpoint_hash, fixed_data_sha256=fixed_hash,
        controls={k: v for k, v in resolved.items() if k not in ('stage3a', 'stage3b', 'work_dir')})
    output = (Path(hydra.utils.to_absolute_path(cfg.stage3b['output_dir']))
              if cfg.stage3b.get('output_dir') else
              output / f"set-{settings['seed_set']}" / settings['mode'] / str(cfg.seed))
    rows, episodes = [], []
    state = dict(native_steps=0, decisions=0, episode_index=0, obs=None, action=None,
                 episode_decisions=0, episode_native=0, episode_return=0.)
    if cfg.stage3b['resume']:
        saved = torch.load(hydra.utils.to_absolute_path(cfg.stage3b['resume']), weights_only=False,
                           map_location='cpu')
        if saved['version'] != 1 or saved['identity'] != identity:
            raise ValueError('Resume checkpoint/data/config identity mismatch.')
        if saved['budgets'] != budgets[:len(saved['budgets'])]:
            raise ValueError('Resume must preserve all evaluated milestones.')
        state, rows, episodes = saved['rollout'], saved['rows'], saved['episodes']
        adapter.load_state_dict(saved['adapter'])
        # Replay and RNG state use CPU tensors even when model tensors are CUDA.
        adapter.replay = type(adapter.replay)([tuple(t.cpu() for t in tr) for tr in adapter.replay],
                                             maxlen=settings['capacity'])
        if saved['environment'] is not None:
            env.reset(seed=settings['adaptation_seeds'][state['episode_index']])
            restore_environment(env, saved['environment'])
        agent._prev_mean.copy_(saved['planner'].to(agent.device))
        random_state = saved['rng']
        random_state = (random_state[0], random_state[1], random_state[2].cpu(),
                        [s.cpu() for s in random_state[3]] if random_state[3] is not None else None)
        restore_rng(random_state)
        if state['obs'] is not None:
            state['obs'] = state['obs'].cpu()
        if state['action'] is not None:
            state['action'] = state['action'].cpu()
    if decision_ttt and output.exists():
        from uuid import uuid4
        output = output / 'versions' / uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(config=resolved, **{k:v for k,v in identity.items() if k != 'controls'},
        active_method=settings['mode'], fixed_data_path=str(fixed_path),
        fixed_data_collection_native_steps=fixed_dataset['collection_native_steps'],
        fixed_data_collection_policy_decisions=fixed_dataset['collection_policy_decisions'],
        evaluation_budget_excluded=True, policy_eval_mode=True,
        resumed_from=cfg.stage3b['resume'], optimizer_class='torch.optim.Adam' if adapter.optimizer else None,
        update_granularity=settings['update_granularity'],
        effective_warmup_transitions=max(settings['warmup'], settings['batch_size']),
        batch_size=settings['batch_size'], warmup_transitions=settings['warmup'],
        timing_units={'update_compute_time': 'seconds', 'update_wall_time_ms': 'milliseconds'},
        fixed_data_nominal_mse_status='unavailable: no nominal fixed dataset configured',
        resume_kind='simulator_controller_planner_rng_replay_optimizer',
        budget_timing=('after completed decision update and terminal bookkeeping; mid-repeat native milestones preserve pending repeat'
                       if decision_ttt else 'after native step, before decision commit and episode update; pending repeat preserved'))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=json_default))
    def write_logs():
        for name, records in [('evaluations', rows), ('adaptation-episodes', episodes)]:
            (output / (name + '.jsonl')).write_text(''.join(json.dumps(r, default=json_default, allow_nan=False)+'\n' for r in records))
    write_logs()

    def metrics():
        return dict(adapter.metrics(), replay_native_steps=adapter.native_steps,
            completed_policy_decisions=adapter.transitions,
            in_flight_decisions=state['decisions']-adapter.transitions,
            episode_in_progress=adapter.in_episode) | dict(
            adaptation_native_steps=state['native_steps'], adaptation_policy_decisions=state['decisions'])

    def checkpoint():
        result = evaluate(agent, adapter, cfg, make_env, baseline)
        if file_sha256(fixed_path) != fixed_hash:
            raise RuntimeError('Shared fixed dataset changed.')
        with adapter.held_out(allow_in_episode=True):
            fixed = fixed_prediction_metrics(agent, baseline, fixed_dataset)
        first = rows[0]['evaluation']['nominal'] if rows else result['nominal']
        prior = rows[-1] if rows else {}
        row = dict(**metrics(), native_budget=state['native_steps'], evaluation=result,
            fixed_data_sha256=fixed_hash, fixed_data_prediction=fixed,
            evaluation_state_invariance=True,
            nominal_retention_success_delta=result['nominal']['success']-first['success'],
            nominal_retention_return_delta=result['nominal']['episode_return']-first['episode_return'],
            nominal_retention_length_delta=result['nominal']['episode_length']-first['episode_length'])
        for name, key in [('evaluation_native_steps','native_steps'), ('evaluation_policy_decisions','policy_decisions')]:
            row[name] = prior.get(name, 0) + sum(v[key] for v in result.values())
        row['evaluation_episodes'] = prior.get('evaluation_episodes', 0) + sum(len(v['episodes']) for v in result.values())
        rows.append(row)
        # Save the complete continuation, including an action repeat interrupted
        # before replay commit. No extra simulator interaction or update is used.
        payload = dict(version=1, identity=identity, budgets=[r['native_budget'] for r in rows],
            adapter=adapter.state_dict(), environment=capture_environment(env, allow_pending=True) if adapter.in_episode else None,
            rollout=deepcopy(state), planner=agent._prev_mean.clone(), rng=rng_state(), rows=rows, episodes=episodes)
        with (output / f"resume-budget-{state['native_steps']}.pt").open('xb') as stream:
            torch.save(payload, stream)
        write_logs()
        print(json.dumps(row, default=json_default, allow_nan=False))

    with (output / 'diagnostics.jsonl').open('x') as diagnostics:
        def native_checkpoint(record):
            state['native_steps'] += 1
            diagnostics.write(json.dumps(record, default=json_default)+'\n')
            pending = wrapper.__dict__.get('_stage3b_pending')
            if decision_ttt and pending and (pending['done'] or len(pending['records']) == 2):
                # Commit the completed transition/update before measuring this budget.
                # Mid-repeat budgets still use the existing exact snapshot path.
                return
            if state['native_steps'] in budgets:
                checkpoint()
                diagnostics.flush()
            if state['native_steps'] == budgets[-1]:
                raise BudgetPause()
        wrapper.native_checkpoint = native_checkpoint
        try:
            if not rows:
                checkpoint()
            while state['native_steps'] < budgets[-1]:
                if not adapter.in_episode:
                    if state['episode_index'] >= len(settings['adaptation_seeds']):
                        raise RuntimeError('Episode safety limit exhausted; increase episodes.')
                    seed = settings['adaptation_seeds'][state['episode_index']]
                    set_seed(seed)
                    state.update(obs=env.reset(seed=seed), action=None,
                                 episode_decisions=0, episode_native=0, episode_return=0.)
                    if state['native_steps'] == 0:
                        capture_environment(env)  # Fail before interaction if exact resume is unsupported.
                    adapter.begin_episode()
                if state['action'] is None:
                    state['action'] = agent.act(state['obs'], t0=state['episode_decisions'] == 0, eval_mode=True, task=None)
                    state['decisions'] += 1
                obs, reward, done, info = env.step(state['action'])
                update_record = adapter.observe(state['obs'], state['action'], obs, len(info['shift_native_steps']))
                if decision_ttt:
                    diagnostics.write(json.dumps(update_record, allow_nan=False)+'\n')
                state.update(obs=obs, action=None, episode_decisions=state['episode_decisions']+1,
                    episode_native=state['episode_native']+len(info['shift_native_steps']),
                    episode_return=state['episode_return']+float(reward))
                shift.drain_events()
                if done:
                    adapter.end_episode()
                    episodes.append(dict(seed=settings['adaptation_seeds'][state['episode_index']],
                        success=float(info['success']), episode_return=state['episode_return'],
                        episode_length=state['episode_decisions'], native_steps=state['episode_native'], **adapter.metrics()))
                    state['episode_index'] += 1
                if (decision_ttt and state['native_steps'] in budgets
                        and rows[-1]['native_budget'] != state['native_steps']):
                    checkpoint()
                    diagnostics.flush()
            # Resuming an already completed budget is a no-op, without committing
            # the pending terminal decision or creating an extra optimizer update.
        except BudgetPause:
            pass
        finally:
            wrapper.native_checkpoint = None
    write_logs()
    torch.save(dict(predictor=agent.model._dynamics.state_dict(),
        optimizer=adapter.optimizer.state_dict() if adapter.optimizer else None,
        checkpoint_sha256=checkpoint_hash, fixed_data_sha256=fixed_hash, metrics=metrics()), output / 'adapted.pt')
    (output / 'complete.json').write_text(json.dumps(metrics(), indent=2))
