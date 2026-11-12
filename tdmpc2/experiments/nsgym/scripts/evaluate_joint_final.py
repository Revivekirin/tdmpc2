#!/usr/bin/env python3
"""Read-only final-snapshot HalfCheetah Joint RL evaluation and three-way comparison."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from uuid import uuid4

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_ttt_final as common
import run_dynamics_only_ttt as ttt
import run_full_joint_rl as joint


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def table(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'No rows for {path}')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def rows(path):
    with Path(path).open(newline='') as f:
        yield from csv.DictReader(f)


def key(identity):
    return identity['parameter'], float(identity['severity']), int(identity['adaptation_seed'])


def name(identity):
    p, k, s = key(identity)
    return f'{p}-k{k:g}-full-joint-seed{s}'


def expected_keys(config):
    return {(x['parameter'], float(x['severity']), int(s))
            for x in config['conditions'] for s in config['adaptation_seeds']}


def check_training_log(path):
    count = updates = 0
    with (Path(path) / 'training.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            count += 1
            assert row['transition'] == count
            expected = max(0, count - 999)
            assert row['updates'] == expected
            assert row['episode'] == count // 1000
            assert row['replay_episodes'] == count // 1000
            metrics = row['metrics']
            assert bool(metrics) == (count >= 1000)
            assert all(math.isfinite(float(value)) for value in metrics.values())
            updates = row['updates']
    assert count == 3000 and updates == 2001


def audit(a, config):
    protocol = joint.protocol()
    root = Path(a.runs)
    top = read(root / 'manifest.json')
    done = read(root / 'completion.json')
    assert top['command'] == 'run' and done['status'] == 'PASS'
    assert top['protocol'] == protocol and top['checkpoint_sha256'] == ttt.SHA
    assert common.digest(Path(a.checkpoint) / 'model.pt') == ttt.SHA
    assert top['heldout_evaluation'] == done['heldout_evaluation'] == 'NOT_RUN'
    assert protocol['conditions'] == config['conditions']
    assert protocol['adaptation_seeds'] == config['adaptation_seeds']
    assert protocol['evaluation_seeds'] == config['evaluation_seeds']
    assert protocol['transitions'] == 3000 and protocol['expected_updates'] == 2001
    assert protocol['horizon'] == config['horizon'] == 1000
    assert protocol['automatic_heldout_evaluation'] is False
    assert set(done['completed']) == {name(dict(parameter=p, severity=k, adaptation_seed=s))
                                      for p, k, s in expected_keys(config)}
    frozen = common.baseline(a.reference, ttt.SHA, config)
    ref_config = frozen['frozen_integrity']['checkpoint_metadata']['config']
    ref_optim = frozen['frozen_integrity']['file_sha256']['optimizers.pt']
    assert common.digest(Path(a.checkpoint) / 'optimizers.pt') == ref_optim
    # The on-disk TTT evaluation, rather than a copied analysis CSV, supplies
    # the paired third method and its evaluation seed contract.
    td = read(Path(a.ttt_shifted) / 'completion.json')
    tm = read(Path(a.ttt_shifted) / 'manifest.json')
    assert td['status'] == 'PASS' and tm['mode'] == 'shifted'
    assert tm['checkpoint_sha256'] == ttt.SHA and tm['evaluation_seeds'] == config['evaluation_seeds']
    assert tm['configuration'] == config
    assert len(td['completed']) + len(td.get('skipped', [])) == 36
    checks = []
    children = [p for p in root.iterdir() if p.is_dir()]
    assert len(children) == 36
    for path in sorted(children):
        manifest = read(path / 'manifest.json')
        complete = read(path / 'completion.json')
        identity = manifest['identity']
        assert path.name == name(identity)
        assert key(identity) in expected_keys(config)
        assert identity['method'] == 'full_joint_rl' and identity['schedule'] == 'fixed'
        assert manifest['protocol'] == protocol and manifest['checkpoint_sha256'] == ttt.SHA
        assert manifest['optimizer_sha256'] == ref_optim
        assert manifest['checkpoint_config'] == ref_config
        assert manifest['evaluation_seed_replay_access'] is False
        assert complete['status'] == 'PASS' and complete['heldout_evaluation'] == 'NOT_RUN'
        assert complete['transitions'] == 3000 and complete['updates'] == 2001
        assert complete['optimizer_step_calls'] == 4002
        assert all(complete['components_changed'].get(part) for part in joint.COMPONENTS)
        for before, after in zip(complete['optimizer_initial_steps'], complete['optimizer_final_steps']):
            assert after == sorted({n + 2001 for n in before})
        check_training_log(path)
        try:
            common.static_sources(frozen['frozen_metadata'], manifest)
        except common.ReferenceCheckNeeded as exc:
            raise ValueError(f'Frozen/Joint simulator or model source differs: {exc}') from exc
        snapshot = path / 'snapshot-0003000.pt'
        assert snapshot.is_file()
        assert common.digest(snapshot) == complete['snapshot_sha256']
        checks.append(dict(parameter=identity['parameter'], severity=float(identity['severity']),
                           adaptation_seed=int(identity['adaptation_seed']), path=str(path.resolve()),
                           snapshot_sha256=complete['snapshot_sha256'],
                           checkpoint_sha256=ttt.SHA, interactions=3000,
                           joint_updates=2001, optimizer_step_calls=4002,
                           training_wall_seconds=complete['wall_seconds'],
                           training_update_seconds=complete['update_seconds']))
    assert {key(x) for x in checks} == expected_keys(config)
    return checks, frozen


def load_agent(a, check, protocol):
    import torch
    from training_support import tree_hash
    snapshot = Path(check['path']) / 'snapshot-0003000.pt'
    assert common.digest(snapshot) == check['snapshot_sha256']
    payload = torch.load(snapshot, map_location='cpu', weights_only=False)
    manifest = read(Path(check['path']) / 'manifest.json')
    complete = read(Path(check['path']) / 'completion.json')
    assert payload['identity'] == manifest['identity']
    assert payload['checkpoint_sha256'] == ttt.SHA
    assert payload['protocol'] == protocol
    assert payload['configuration'] == manifest['checkpoint_config']
    assert payload['transitions'] == 3000 and payload['updates'] == 2001
    assert payload['resume'] == 'evaluation_only_no_replay_or_environment_state'
    agent, meta = ttt.load_agent(argparse.Namespace(checkpoint=a.checkpoint, synthetic=False))
    assert meta['config'] == payload['configuration']
    agent.model.load_state_dict(payload['model'], strict=True)
    agent.optim.load_state_dict(payload['optim'])
    agent.pi_optim.load_state_dict(payload['pi_optim'])
    agent.scale.load_state_dict(payload['scale'])
    for p in agent.model.parameters():
        p.requires_grad_(False)
    agent.model.eval()
    def forbidden(*args, **kwargs):
        raise RuntimeError('Held-out evaluation cannot update or insert replay')
    agent.update = forbidden
    agent.optim.step = forbidden
    agent.pi_optim.step = forbidden
    state_hash = tree_hash(dict(model=agent.model.state_dict(), optim=agent.optim.state_dict(),
                                pi_optim=agent.pi_optim.state_dict(), scale=agent.scale.state_dict()))
    assert tree_hash(payload['model']) == tree_hash(agent.model.state_dict())
    for component in ('optim', 'pi_optim', 'scale'):
        assert tree_hash(payload[component]) == tree_hash(getattr(agent, component).state_dict())
    assert complete['snapshot_sha256'] == check['snapshot_sha256']
    return agent, state_hash


def assert_unchanged(agent, before):
    from training_support import tree_hash
    now = tree_hash(dict(model=agent.model.state_dict(), optim=agent.optim.state_dict(),
                         pi_optim=agent.pi_optim.state_dict(), scale=agent.scale.state_dict()))
    assert now == before, 'Evaluation mutated model, optimizer or scale'


def evaluate_one(a, config, frozen, check, mode, limit=None, max_steps=None):
    import numpy as np
    import torch
    agent, before = load_agent(a, check, joint.protocol())
    parameter, severity, adaptation_seed = key(check)
    seeds = config['evaluation_seeds'][:limit] if limit else config['evaluation_seeds']
    result = []
    for seed in seeds:
        ttt.seeded(seed)  # MPPI samples policy noise from torch RNG.
        if mode == 'prediction':
            bank_key = (parameter, severity, seed)
            filename = frozen['files'][bank_key]
            bank = Path(a.reference) / 'frozen' / filename
            assert common.digest(bank) == frozen['hashes'][filename]
            with np.load(bank) as raw:
                obs = np.asarray(raw['observations'], dtype=np.float32)
                actions = np.asarray(raw['actions'], dtype=np.float32)
            assert obs.shape == (1001, 17) and actions.shape == (1000, 6)
            total = 0.
            with torch.no_grad():
                for start in range(0, 1000, 256):
                    end = min(start + 256, 1000)
                    o = torch.as_tensor(obs[start:end], device=agent.device)
                    n = torch.as_tensor(obs[start+1:end+1], device=agent.device)
                    action = torch.as_tensor(actions[start:end], device=agent.device)
                    z = agent.model.encode(o, None)
                    target = agent.model.encode(n, None)
                    total += float((agent.model.next(z, action, None) - target).square().mean(-1).sum().item())
            mse = total / 1000
            assert math.isfinite(mse)
            result.append(dict(parameter=parameter, severity=severity, adaptation_seed=adaptation_seed,
                               evaluation_seed=seed, joint_self_consistency_mse=mse, transitions=1000,
                               bank_file=filename, bank_sha256=frozen['hashes'][filename]))
        else:
            condition = dict(parameter=parameter, severity=severity)
            env = ttt.environment(config, condition, 'fixed')
            try:
                obs, _ = env.reset(seed=seed)
                rewards = []; velocity = []; cost = []; action_abs = []; action_sat = []
                force_rms = []; force_sat = []; actual = []
                terminated = truncated = False
                with torch.no_grad():
                    for step in range(max_steps or config['horizon']):
                        action = agent.act(torch.as_tensor(obs, dtype=torch.float32),
                                           t0=step == 0, eval_mode=True).detach().cpu().numpy()
                        assert action.shape == (6,) and np.isfinite(action).all()
                        obs, reward, terminated, truncated, info = env.step(action)
                        assert obs.shape == (17,) and np.isfinite(obs).all() and math.isfinite(float(reward))
                        force = env.unwrapped.data.actuator_force.copy()
                        model = env.unwrapped.model
                        limited = model.actuator_forcelimited.astype(bool)
                        bounds = model.actuator_forcerange
                        rewards.append(float(reward)); velocity.append(float(info['x_velocity']))
                        cost.append(-float(info['reward_ctrl']))
                        action_abs.append(float(np.abs(action).mean()))
                        action_sat.append(float((np.abs(action) >= .99).mean()))
                        force_rms.append(float(np.sqrt(np.mean(force**2))))
                        if limited.any():
                            force_sat.append(float(np.mean((force[limited] <= bounds[limited, 0] + 1e-6) |
                                                           (force[limited] >= bounds[limited, 1] - 1e-6))))
                        actual.append(float(env.readback()))
                        assert abs(actual[-1] - env.nominal_value * severity) <= 1e-10
                        if terminated or truncated:
                            break
                assert len(rewards) == (max_steps or config['horizon'])
                if max_steps is None:
                    assert truncated and not terminated
                ref = frozen['episodes'][(parameter, severity, seed)]
                result.append(dict(parameter=parameter, severity=severity, adaptation_seed=adaptation_seed,
                                   evaluation_seed=seed, joint_return=sum(rewards),
                                   frozen_return=float(ref['episode_return']) if max_steps is None else None,
                                   partial_smoke=max_steps is not None,
                                   joint_forward_velocity=float(np.mean(velocity)),
                                   frozen_forward_velocity=float(ref['forward_velocity']),
                                   joint_control_cost=sum(cost), frozen_control_cost=float(ref['control_cost']),
                                   joint_reward_run=sum(rewards) + sum(cost), joint_action_abs=float(np.mean(action_abs)),
                                   joint_action_component_saturation=float(np.mean(action_sat)),
                                   joint_actuator_force_rms=float(np.mean(force_rms)),
                                   joint_actuator_force_saturation=float(np.mean(force_sat)) if force_sat else None,
                                   force_limited_count=int(limited.sum()), episode_length=len(rewards),
                                   terminated=bool(terminated), truncated=bool(truncated),
                                   nominal_parameter=float(env.nominal_value), actual_parameter_min=min(actual),
                                   actual_parameter_max=max(actual)))
            finally:
                env.close()
        assert_unchanged(agent, before)
    return result


def evaluate(a, config, checks, frozen, smoke=False):
    ttt.dependencies()
    mode = 'shifted' if smoke else a.mode
    selected = [next(x for x in checks if key(x) == ('bthigh_damping', 2., config['adaptation_seeds'][0]))] if smoke else checks
    root = Path(a.output)
    if a.resume:
        assert root.is_dir() and not smoke
        manifest = read(root / 'manifest.json')
        assert manifest['mode'] == mode and manifest['checkpoint_sha256'] == ttt.SHA
        assert manifest['audit_sha256'] == common.digest(a.audit)
    else:
        root.mkdir(parents=True, exist_ok=False)
        write(root / 'manifest.json', dict(mode=mode, smoke=smoke, checkpoint_sha256=ttt.SHA,
                                           audit_sha256=common.digest(a.audit),
                                           policy_rng='seeded_per_evaluation_seed_before_rollout'))
    if smoke:
        first, first_hash = load_agent(a, selected[0], joint.protocol())
        second, second_hash = load_agent(a, selected[0], joint.protocol())
        assert first_hash == second_hash, 'Joint snapshot reload changed model or optimizer state'
        del first, second
    passed = []; failed = []
    for check in selected:
        dest = root / name(check)
        if (dest / 'completion.json').is_file():
            if read(dest / 'completion.json')['status'] == 'PASS':
                passed.append(dest.name)
                continue
        if dest.exists():
            archived = root / f'interrupted-{dest.name}-{uuid4().hex[:8]}'
            dest.rename(archived)
            dest.mkdir()
        else:
            dest.mkdir()
        started = time.perf_counter()
        try:
            data = evaluate_one(a, config, frozen, check, mode,
                                limit=1 if smoke else None, max_steps=24 if smoke else None)
            assert len(data) == (1 if smoke else len(config['evaluation_seeds']))
            table(dest / f'{mode}.csv', data)
            write(dest / 'completion.json', dict(status='PASS',rows=len(data),wall_seconds=time.perf_counter()-started,
                                                 snapshot_sha256=check['snapshot_sha256'],model_unchanged=True,
                                                 optimizer_updates=0))
            passed.append(dest.name)
            print('PASS', dest.name, flush=True)
        except Exception as exc:
            write(dest / 'failure.json', dict(error=type(exc).__name__, message=str(exc)))
            failed.append(dict(identity=dest.name, error=str(exc)))
            print('FAIL', dest.name, exc, flush=True)
    status = 'PASS' if not failed and len(passed) == len(selected) else 'PARTIAL'
    # Completion is regenerated on resume, while per-stream files remain immutable.
    completion = root / 'completion.json'
    if completion.exists():
        completion.rename(root / f'completion-attempt-{uuid4().hex[:8]}.json')
    write(completion, dict(status=status, mode=mode, completed=passed, failed=failed, expected=len(selected)))
    return status


def collect(root, mode, expected_count):
    root = Path(root)
    assert read(root / 'completion.json')['status'] == 'PASS'
    assert read(root / 'manifest.json')['mode'] == mode
    result = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith('interrupted-'):
            continue
        assert read(path / 'completion.json')['status'] == 'PASS'
        data = list(rows(path / f'{mode}.csv'))
        assert len(data) == expected_count
        result.extend(data)
    assert len(result) == 36 * expected_count
    return result


def aggregate(a, config, checks, frozen):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    joint_rows = collect(a.shifted, 'shifted', 20)
    prediction_rows = collect(a.prediction, 'prediction', 20)
    ttt_rows, failures = common.collect_stage(a.ttt_shifted, 'shifted')
    assert not failures and len(ttt_rows) == 720
    identity = lambda x: (x['parameter'], float(x['severity']), int(x['adaptation_seed']), int(x['evaluation_seed']))
    J = {identity(x): x for x in joint_rows}
    D = {identity(x): x for x in ttt_rows}
    P = {identity(x): x for x in prediction_rows}
    assert len(J) == len(D) == len(P) == 720 and set(J) == set(D) == set(P)
    comparison = []
    for ident in sorted(J):
        j = J[ident]; d = D[ident]
        frozen_return = float(j['frozen_return'])
        assert abs(frozen_return - float(d['frozen_return'])) < 1e-6
        joint_return = float(j['joint_return'])
        ttt_return = float(d['decision_return'])
        comparison.append(dict(parameter=ident[0], severity=ident[1], adaptation_seed=ident[2], evaluation_seed=ident[3],
                               frozen_return=frozen_return, ttt_return=ttt_return, joint_return=joint_return,
                               ttt_minus_frozen=ttt_return-frozen_return,
                               joint_minus_frozen=joint_return-frozen_return,
                               joint_minus_ttt=joint_return-ttt_return,
                               frozen_velocity=float(j['frozen_forward_velocity']),
                               ttt_velocity=float(d['decision_velocity']), joint_velocity=float(j['joint_forward_velocity']),
                               frozen_control_cost=float(j['frozen_control_cost']),
                               ttt_control_cost=float(d['decision_control_cost']), joint_control_cost=float(j['joint_control_cost'])))
    stream = []
    for check in checks:
        subset = [x for x in comparison if key(x) == key(check)]
        assert len(subset) == 20
        stream.append(dict(parameter=check['parameter'],severity=check['severity'],adaptation_seed=check['adaptation_seed'],
                           frozen_return=float(np.mean([x['frozen_return'] for x in subset])),
                           ttt_return=float(np.mean([x['ttt_return'] for x in subset])),
                           joint_return=float(np.mean([x['joint_return'] for x in subset])),
                           joint_return_eval_seed_std=float(np.std([x['joint_return'] for x in subset],ddof=1)),
                           joint_minus_frozen=float(np.mean([x['joint_minus_frozen'] for x in subset])),
                           joint_minus_ttt=float(np.mean([x['joint_minus_ttt'] for x in subset])),
                           joint_self_consistency_mse=float(np.mean([float(P[identity(x)]['joint_self_consistency_mse']) for x in subset])),
                           adaptation_interactions=3000,joint_updates=2001,ttt_updates=118,
                           optimizer_step_calls=4002,snapshot_sha256=check['snapshot_sha256']))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    table(out/'joint_episodes.csv', joint_rows)
    table(out/'joint_per_stream.csv', stream)
    table(out/'frozen_ttt_joint_comparison.csv', comparison)
    table(out/'joint_prediction_self_consistency.csv', prediction_rows)
    conditions = [(p,k) for p,k,_ in sorted(expected_keys(config))]
    conditions = list(dict.fromkeys(conditions))
    summary = []
    for p,k in conditions:
        rr = [x for x in stream if x['parameter']==p and float(x['severity'])==k]
        assert len(rr)==3
        summary.append(dict(parameter=p,severity=k,adaptation_streams=3,evaluation_seeds_per_stream=20,
                            frozen_return=float(np.mean([x['frozen_return'] for x in rr])),
                            frozen_return_adaptation_seed_std=float(np.std([x['frozen_return'] for x in rr],ddof=1)),
                            ttt_return=float(np.mean([x['ttt_return'] for x in rr])),
                            ttt_return_adaptation_seed_std=float(np.std([x['ttt_return'] for x in rr],ddof=1)),
                            joint_return=float(np.mean([x['joint_return'] for x in rr])),
                            joint_return_adaptation_seed_std=float(np.std([x['joint_return'] for x in rr],ddof=1)),
                            joint_minus_frozen_mean=float(np.mean([x['joint_minus_frozen'] for x in rr])),
                            joint_minus_frozen_std=float(np.std([x['joint_minus_frozen'] for x in rr],ddof=1)),
                            joint_minus_frozen_ci_low=common.unit_stats([x['joint_minus_frozen'] for x in rr])['ci_low'],
                            joint_minus_frozen_ci_high=common.unit_stats([x['joint_minus_frozen'] for x in rr])['ci_high'],
                            joint_minus_ttt_mean=float(np.mean([x['joint_minus_ttt'] for x in rr])),
                            joint_minus_ttt_std=float(np.std([x['joint_minus_ttt'] for x in rr],ddof=1)),
                            joint_minus_ttt_ci_low=common.unit_stats([x['joint_minus_ttt'] for x in rr])['ci_low'],
                            joint_minus_ttt_ci_high=common.unit_stats([x['joint_minus_ttt'] for x in rr])['ci_high'],
                            improving_streams_vs_frozen=sum(x['joint_minus_frozen']>0 for x in rr),
                            improving_streams_vs_ttt=sum(x['joint_minus_ttt']>0 for x in rr)))
    table(out/'joint_condition_summary.csv',summary)
    table(out/'nominal_k1_control.csv',[x for x in summary if x['severity']==1.])
    for param in ('bthigh_damping','torso_mass'):
        subset = [x for x in summary if x['parameter']==param]
        x = [x['severity'] for x in subset]
        fig,ax=plt.subplots(figsize=(7,4))
        for method,label in [('frozen_return','Frozen'),('ttt_return','Dynamics-only TTT'),('joint_return','Full Joint RL')]:
            ax.plot(x,[v[method] for v in subset],marker='o',label=label)
        ax.set(xlabel='Severity multiplier',ylabel='Held-out shifted return',title=param)
        ax.legend();fig.tight_layout();fig.savefig(out/f'{param}_three_way_return.png',dpi=160);plt.close(fig)
    for metric,title in [('joint_minus_frozen_mean','Joint − Frozen'),('joint_minus_ttt_mean','Joint − TTT')]:
        fig,ax=plt.subplots(figsize=(8,3))
        image=np.array([[next(x[metric] for x in summary if x['parameter']==p and x['severity']==k)
                         for k in (0.5,0.75,1.,1.25,1.5,2.)] for p in ('bthigh_damping','torso_mass')])
        bound=max(1e-9,float(np.max(abs(image))))
        m=ax.imshow(image,aspect='auto',cmap='RdBu',vmin=-bound,vmax=bound)
        ax.set(xticks=range(6),xticklabels=['0.5','0.75','1','1.25','1.5','2'],
               yticks=[0,1],yticklabels=['Damping','Torso mass'],xlabel='Severity',title=title)
        fig.colorbar(m,ax=ax,label='Paired return difference');fig.tight_layout()
        fig.savefig(out/f'{metric}.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,label in zip(axs,('velocity','control_cost'),('Forward velocity','Control cost')):
        for method in ('frozen','ttt','joint'):
            ax.plot(range(len(conditions)),[np.mean([float(x[f'{method}_{metric}']) for x in comparison
                                              if x['parameter']==p and x['severity']==k]) for p,k in conditions],label=method)
        ax.set(xlabel='Condition index',ylabel=label);ax.legend()
    fig.tight_layout();fig.savefig(out/'velocity_control_cost.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for p in ('bthigh_damping','torso_mass'):
        rr=[x for x in stream if x['parameter']==p]
        ax.scatter([x['severity'] for x in rr],[x['joint_minus_frozen'] for x in rr],label=p)
    ax.axhline(0,color='gray');ax.set(xlabel='Severity',ylabel='Stream mean Joint − Frozen return');ax.legend()
    fig.tight_layout();fig.savefig(out/'adaptation_seed_variability.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for p in ('bthigh_damping','torso_mass'):
        rr=[x for x in stream if x['parameter']==p]
        ax.scatter([x['severity'] for x in rr],[x['joint_self_consistency_mse'] for x in rr],label=p)
    ax.set(xlabel='Severity',ylabel='Joint latent self-consistency MSE (own encoder)');ax.legend()
    fig.tight_layout();fig.savefig(out/'joint_self_consistency.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(['Frozen','TTT','Joint'],[0,118,2001]);ax.set(ylabel='Model update calls per 3000 interactions')
    fig.tight_layout();fig.savefig(out/'interaction_optimizer_cost.png',dpi=160);plt.close(fig)
    write(out/'completeness.json',dict(status='PASS',joint_streams=36,ttt_streams=36,
                                        evaluation_seeds_per_stream=20,paired_episode_rows=720,
                                        adaptation_interactions_per_stream=3000,
                                        optimizer_updates=dict(frozen=0,ttt=118,joint=2001)))
    (out/'JOINT_FINAL_EVALUATION_AUDIT.md').write_text(
        '# Joint final-snapshot artifact audit\n\n36/36 Joint runs verified: original 750k checkpoint, 3000 transitions, 2001 update calls, 4002 optimizer step calls, final snapshot hashes, fixed conditions and held-out seed split.\n')
    (out/'JOINT_HELDOUT_EVALUATION.md').write_text(
        '# Joint held-out evaluation\n\n720 fixed-shift episodes, one final snapshot per independent adaptation stream. Policy RNG is reset to each evaluation seed before its rollout. No optimizer or replay mutation was allowed. Joint one-step MSE uses the Joint encoder and is not directly comparable to Frozen or TTT latent MSE.\n')
    (out/'THREE_WAY_ADAPTATION_COMPARISON.md').write_text(
        '# Frozen vs Dynamics-only TTT vs Full Joint RL\n\nAll return differences pair the same held-out evaluation seed. Each condition has 3 independent adaptation streams, each summarized over 20 evaluation episodes. k=1 is nominal physics; inspect it separately for retention. Methods share 3000 online interactions but use 0, 118 and 2001 model update calls respectively.\n\n'
        + '\n'.join(f"- {x['parameter']} ×{x['severity']:g}: Joint−Frozen {x['joint_minus_frozen_mean']:.3f}; Joint−TTT {x['joint_minus_ttt_mean']:.3f}; improved streams {x['improving_streams_vs_frozen']}/3 and {x['improving_streams_vs_ttt']}/3" for x in summary)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','smoke','evaluate','aggregate'))
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--runs',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--ttt-shifted',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--audit')
    p.add_argument('--mode',choices=('prediction','shifted'))
    p.add_argument('--prediction')
    p.add_argument('--shifted')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.command in ('smoke','evaluate','aggregate') and not a.audit:
        p.error('--audit is required')
    if a.command=='evaluate' and not a.mode:
        p.error('--mode is required for evaluate')
    if a.command=='aggregate' and (not a.prediction or not a.shifted):
        p.error('--prediction and --shifted are required for aggregate')
    a.checkpoint=str(Path(a.checkpoint).resolve())
    a.runs=str(Path(a.runs).resolve())
    a.reference=str(Path(a.reference).resolve())
    a.ttt_shifted=str(Path(a.ttt_shifted).resolve())
    config=ttt.configuration(argparse.Namespace(config=ttt.DEFAULT,command='run'))
    assert common.digest(Path(a.checkpoint)/'model.pt')==ttt.SHA
    checks,frozen=audit(a,config)
    if a.command=='audit':
        out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
        table(out/'joint_run_integrity.csv',checks)
        write(out/'audit.json',dict(status='PASS',checkpoint_sha256=ttt.SHA,
                                     checks=checks,configuration=config,
                                     ttt_shifted_sha256=common.digest(Path(a.ttt_shifted)/'completion.json')))
        write(out/'completion.json',dict(status='PASS',streams=36))
        (out/'JOINT_FINAL_EVALUATION_AUDIT.md').write_text(
            '# Joint final-snapshot artifact audit\n\n'
            'PASS: 36 fixed Joint streams; each has 3,000 logged transitions, 2,001 actual TD-MPC2 update calls, '
            '4,002 optimizer step calls, all model components changed, and an intact final snapshot. '
            'The original 750k model and optimizer hashes, checkpoint/MPPI configuration, Frozen environment source, '
            'physical grid, adaptation seed split and completed TTT shifted evaluation are compatible. '
            'Held-out evaluation was not run during training.\n')
        print('PASS: 36 Joint streams, Frozen and TTT reference compatible')
        return
    prior=read(a.audit)
    assert prior['status']=='PASS' and prior['checkpoint_sha256']==ttt.SHA
    assert prior['checks']==checks and prior['configuration']==config
    assert prior['ttt_shifted_sha256']==common.digest(Path(a.ttt_shifted)/'completion.json')
    if a.command in ('smoke','evaluate'):
        status=evaluate(a,config,checks,frozen,smoke=a.command=='smoke')
        print('BATCH',status)
        if status!='PASS':
            raise SystemExit(2)
    else:
        aggregate(a,config,checks,frozen)
        print('PASS: three-way aggregation')


if __name__=='__main__':
    main()
