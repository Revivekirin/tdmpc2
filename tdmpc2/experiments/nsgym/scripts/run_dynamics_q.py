#!/usr/bin/env python3
"""Isolated Dynamics+Q adaptation, audit and held-out evaluation. No automatic jobs."""
import argparse
from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import run_dynamics_only_ttt as ttt
import evaluate_ttt_final as evaluation

DEFAULT = ttt.ROOT / 'experiments/nsgym/configs/dynamics_q.json'
read, write, digest = ttt.read, ttt.js, ttt.digest


def table(path, data):
    if not data:
        return
    keys = list(dict.fromkeys(k for row in data for k in row))
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(data)


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def configuration(path=DEFAULT):
    c = read(path)
    reference = ttt.configuration(argparse.Namespace(config=ttt.DEFAULT, command='run'))
    c.update(onset=reference['onset'], conditions=reference['conditions'], adaptation_seeds=reference['adaptation_seeds'],
             evaluation_seeds=reference['evaluation_seeds'])
    assert c['method'] == 'dynamics_q_118' and c['transitions'] == 3000 and c['horizon'] == 1000
    assert (c['batch_size'], c['warmup'], c['update_every']) == (64, 64, 25)
    assert c['capacity'] >= 3000 and c['collection_eval_mode'] is True
    assert c['adaptation_seeds'] == [42000001, 42000002, 42000003]
    assert c['evaluation_seeds'] == list(range(43000001, 43000021))
    assert len(c['conditions']) == 12
    for k in ('dynamics_lr', 'q_lr', 'eps', 'consistency_coef', 'value_coef', 'dynamics_grad_clip', 'q_grad_clip'):
        assert math.isfinite(c[k]) and c[k] > 0
    assert c['betas'] == [0.9, 0.999]
    return c, reference


def identities(c):
    return [dict(**condition, adaptation_seed=seed, method=c['method'], schedule='fixed')
            for condition in c['conditions'] for seed in c['adaptation_seeds']]


def name(i):
    return f"{i['parameter']}-k{float(i['severity']):g}-dynamics-q-seed{i['adaptation_seed']}"


def sources():
    paths = [Path(__file__), Path(__file__).with_name('analyze_dynamics_q.py'), DEFAULT, ttt.ROOT/'tdmpc2/common/dynamics_q.py',
             ttt.ROOT/'tdmpc2/common/math.py', ttt.ROOT/'tdmpc2/common/world_model.py',
             ttt.ROOT/'tdmpc2/tdmpc2.py', ttt.ROOT/'tdmpc2/common/layers.py',
             ttt.ROOT/'tdmpc2/envs/nsgym/severity.py', ttt.ROOT/'tdmpc2/envs/nsgym/halfcheetah.py',
             Path(ttt.__file__), Path(evaluation.__file__),
             ttt.ROOT/'experiments/nsgym/scripts/training_support.py', ttt.DEFAULT,
             ttt.ROOT/'experiments/nsgym/configs/physical_severity_grid.yaml',
             ttt.ROOT/'experiments/nsgym/configs/evaluation_seed_splits.yaml']
    return {str(p.relative_to(ttt.ROOT)): digest(p) for p in paths}


def open_stage(output, manifest, resume=False):
    out = Path(output)
    if resume:
        assert read(out/'manifest.json') == manifest, 'Resume configuration/source/input mismatch'
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out/'manifest.json', manifest)
    return out


def finish(out, data):
    p = out/'completion.json'
    if p.exists():
        p.rename(out/f'completion-attempt-{uuid4().hex[:8]}.json')
    write(p, data)


def destination(out, label):
    p = out/label
    if (p/'completion.json').is_file() and read(p/'completion.json')['status'] == 'PASS':
        return p, True
    if p.exists():
        p.rename(out/f'interrupted-{label}-{uuid4().hex[:8]}')
    p.mkdir()
    return p, False


def preflight(a, reference):
    # Reuse the actual completed Frozen/Decision manifest, update-log and bank audit.
    checks, baseline = evaluation.audit(argparse.Namespace(runs=a.ttt_runs, reference=a.reference,
                                                          checkpoint=a.checkpoint), reference)
    assert len(checks) == 36
    evaluation.static_sources(baseline['frozen_metadata'], {'source_hashes': sources()})
    return baseline


def cpu_agent():
    """Real WorldModel/Q implementation with small CPU dimensions, no research results."""
    from types import SimpleNamespace, MethodType
    import torch
    from common.world_model import WorldModel
    from tdmpc2 import TDMPC2
    cfg = SimpleNamespace(multitask=False, episodic=False, obs='state', obs_shape={'state': (17,)},
                          num_enc_layers=2, enc_dim=16, latent_dim=16, mlp_dim=16, action_dim=6,
                          task_dim=0, simnorm_dim=8, num_q=5, num_bins=101, vmin=-10., vmax=10.,
                          bin_size=.2, dropout=.01, log_std_min=-10., log_std_max=2., tau=.01)
    agent = SimpleNamespace(cfg=cfg, model=WorldModel(cfg), discount=.995, device=torch.device('cpu'))
    agent._td_target = MethodType(TDMPC2._td_target, agent)
    return agent


def optimizer_record(adapter):
    result = {}
    for label, opt in [('dynamics', adapter.dynamics_optimizer), ('q', adapter.q_optimizer)]:
        g = opt.param_groups[0]
        result[label] = dict(lr=g['lr'], betas=list(g['betas']), eps=g['eps'], weight_decay=g['weight_decay'],
                             parameter_count=sum(p.numel() for p in g['params']), initialization='fresh')
    return result


def train_one(a, c, identity, out, budget):
    import numpy as np
    import torch
    from common.dynamics_q import DynamicsQAdapter, expected_updates
    from common.stage3a import equal_state
    from training_support import tree_hash
    started = time.perf_counter()
    agent, metadata = ttt.load_agent(argparse.Namespace(checkpoint=a.checkpoint, synthetic=False))
    assert metadata['config']['lr'] == c['q_lr'] == .0003
    assert metadata['config']['value_coef'] == c['value_coef'] == .1
    assert metadata['config']['grad_clip_norm'] == c['q_grad_clip'] == 20
    assert metadata['config']['tau'] == .01 and agent.discount == .995
    ttt.seeded(identity['adaptation_seed'])
    initial_model_hash = tree_hash(agent.model.state_dict())
    auxiliary = deepcopy(ttt.aux_state(agent))
    adapter = DynamicsQAdapter(agent, c, identity)
    env = ttt.environment(c, identity_condition(identity), 'fixed')
    write(out/'manifest.json', dict(identity=identity, configuration=c, checkpoint_sha256=ttt.SHA,
        checkpoint_metadata=metadata, initial_model_hash=initial_model_hash,
        initial_fixed_hash=tree_hash(adapter.initial_fixed), original_optimizer_hash=tree_hash(auxiliary),
        optimizers=optimizer_record(adapter), source_hashes=sources(), source_revision=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ttt.ROOT, text=True).strip(),
        source_dirty=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ttt.ROOT, text=True),
        nominal_parameter=env.nominal_value, requested_transitions=budget))
    episodes = []
    try:
        o, _ = env.reset(seed=identity['adaptation_seed'])
        episode_return = 0.
        with (out/'training.jsonl').open('x') as log:
            for step in range(budget):
                ep, t = divmod(step, c['horizon'])
                action = agent.act(torch.as_tensor(o, dtype=torch.float32), t0=t == 0, eval_mode=True).numpy()
                n, reward, term, trunc, info = env.step(action)
                metrics = adapter.observe(o, action, n, reward, term, trunc, ep, t,
                                          env.readback(), env.nominal_value)
                episode_return += float(reward)
                record = dict(global_step=step+1, episode=ep, episode_step=t, adaptation_seed=identity['adaptation_seed'],
                    actual_parameter=env.readback(), nominal_parameter=env.nominal_value, reward=float(reward),
                    episode_return_so_far=episode_return, forward_velocity=float(info['x_velocity']),
                    control_cost=-float(info['reward_ctrl']), action_saturation=float((np.abs(action) >= .99).mean()),
                    terminated=bool(term), truncated=bool(trunc), discount=float(agent.discount),
                    dynamics_steps=adapter.dynamics_steps, q_steps=adapter.q_steps, target_steps=adapter.target_steps,
                    replay_size=len(adapter.replay), wall_seconds=time.perf_counter()-started,
                    update_seconds=adapter.update_seconds, metrics=metrics)
                if metrics:
                    record['parameter_drift'] = adapter.parameter_audit()
                log.write(json.dumps(record, allow_nan=False)+'\n')
                if (step+1) % 100 == 0:
                    log.flush()
                    print(f'{name(identity)}: {step+1}/{budget}; dynamics={adapter.dynamics_steps}, Q={adapter.q_steps}', flush=True)
                o = n
                if term or trunc:
                    episodes.append(dict(episode=ep, steps=t+1, episode_return=episode_return))
                    if step+1 < budget:
                        o, _ = env.reset()
                        episode_return = 0.
        count = expected_updates(budget)
        assert adapter.dynamics_steps == adapter.q_steps == adapter.target_steps == count
        invariance = adapter.parameter_audit()
        assert all(row['changed'] == (row['component'] in ('dynamics', 'q', 'target_q')) for row in invariance)
        assert equal_state(auxiliary, ttt.aux_state(agent)), 'Original optimizer/scale changed'
        assert all(torch.isfinite(p).all() for p in agent.model.parameters())
        payload = dict(identity=identity, configuration=c, checkpoint_sha256=ttt.SHA,
            checkpoint_config=metadata['config'], model=agent.model.state_dict(), adapter=adapter.state_dict(),
            fixed_hash=tree_hash(adapter.fixed_state()), initial_model_hash=initial_model_hash,
            original_optimizer_hash=tree_hash(auxiliary), source_hashes=sources(),
            resume='evaluation_only_no_environment_or_planner_state')
        path = out/f'snapshot-{budget:07d}.pt'
        torch.save(payload, path)
        restored = torch.load(path, map_location=agent.device, weights_only=False)
        assert tree_hash(restored) == tree_hash(payload)
        assert digest(Path(a.checkpoint)/'model.pt') == ttt.SHA
        assert digest(Path(a.checkpoint)/'optimizers.pt') == metadata['optimizer_sha256']
        table(out/'parameter_invariance.csv', invariance)
        table(out/'online_episodes.csv', episodes)
        write(out/'completion.json', dict(status='PASS', identity=identity, transitions=budget,
            dynamics_steps=count, q_steps=count, total_optimizer_steps=count*2, target_steps=count,
            snapshot_sha256=digest(path), initial_model_hash=initial_model_hash,
            final_model_hash=tree_hash(agent.model.state_dict()), fixed_hash=payload['fixed_hash'],
            parameter_invariance=invariance, replay_size=len(adapter.replay),
            wall_seconds=time.perf_counter()-started, update_seconds=adapter.update_seconds,
            heldout_evaluation='NOT_RUN'))
    finally:
        env.close()


def identity_condition(identity):
    return {k: identity[k] for k in ('parameter', 'severity')}


def train(a, c):
    selected = identities(c)
    if a.command == 'smoke':
        selected = [x for x in selected if x['parameter'] == 'bthigh_damping' and x['severity'] == 2.][:1]
    elif any(x is not None for x in (a.parameter, a.severity, a.seed)):
        assert all(x is not None for x in (a.parameter, a.severity, a.seed)), 'Specify parameter, severity and seed together'
        selected = [x for x in selected if (x['parameter'], x['severity'], x['adaptation_seed']) == (a.parameter, a.severity, a.seed)]
        assert len(selected) == 1, 'Condition outside predefined grid'
    if a.command == 'run':
        assert a.smoke_result, 'Explicit successful real checkpoint smoke required'
        s = Path(a.smoke_result)
        assert read(s/'completion.json')['status'] == 'PASS'
        m = read(s/'manifest.json')
        assert m['command'] == 'smoke' and not m['cpu_fixture']
        assert m['configuration'] == c and m['source_hashes'] == sources() and m['checkpoint_sha256'] == ttt.SHA
    manifest = dict(command=a.command, cpu_fixture=False, configuration=c, identities=selected,
                    source_hashes=sources(), checkpoint_sha256=ttt.SHA,
                    reference=str(Path(a.reference).resolve()), ttt_runs=str(Path(a.ttt_runs).resolve()))
    out = open_stage(a.output, manifest, a.resume)
    passed, failed = [], []
    for i in selected:
        path, skip = destination(out, name(i))
        if skip:
            previous = read(path/'manifest.json')
            assert previous['configuration'] == c and previous['identity'] == i
            assert previous['source_hashes'] == sources() and previous['checkpoint_sha256'] == ttt.SHA
            done = read(path/'completion.json')
            snapshot = path/f"snapshot-{done['transitions']:07d}.pt"
            assert digest(snapshot) == done['snapshot_sha256']
            passed.append(path.name)
            continue
        try:
            train_one(a, c, i, path, 100 if a.command == 'smoke' else 3000)
            passed.append(path.name)
        except Exception as exc:
            write(path/'failure.json', dict(error=repr(exc)))
            failed.append(dict(identity=path.name, error=repr(exc)))
            print('FAIL', path.name, repr(exc), flush=True)
        else:
            print('PASS', path.name, flush=True)
    status = 'PASS' if len(passed) == len(selected) and not failed else 'PARTIAL'
    finish(out, dict(status=status, completed=passed, failed=failed, expected=len(selected)))
    (out/'DYNAMICS_Q_COMPLETION.md').write_text(
        f'# Dynamics+Q adaptation\n\n{status}: {len(passed)}/{len(selected)} streams. '
        f'Held-out evaluation: NOT_RUN. Failed streams: {json.dumps(failed)}\n')
    return status


def check_log(path, c):
    from common.dynamics_q import expected_updates
    count = 0
    with (path/'training.jsonl').open() as f:
        log = [json.loads(line) for line in f]
    for row in log:
        count += 1
        assert row['global_step'] == count
        assert row['episode'] == (count-1)//1000 and row['episode_step'] == (count-1)%1000
        assert row['dynamics_steps'] == row['q_steps'] == row['target_steps'] == expected_updates(count)
        assert row['adaptation_seed'] in c['adaptation_seeds'] and row['adaptation_seed'] not in c['evaluation_seeds']
        assert row['truncated'] == (count % 1000 == 0) and not row['terminated']
        assert bool(row['metrics']) == (count >= 75 and count % 25 == 0)
        assert all(math.isfinite(v) for v in row['metrics'].values())
        assert row['replay_size'] == count
    assert count == 3000


def audit_runs(a, c):
    import torch
    from common.stage3a import equal_state
    from common.dynamics_q import DynamicsQAdapter
    from training_support import tree_hash
    root = Path(a.runs)
    top = read(root/'manifest.json')
    assert top['command'] == 'run' and top['configuration'] == c and top['source_hashes'] == sources()
    assert read(root/'completion.json')['status'] == 'PASS'
    expected = {name(i) for i in identities(c)}
    assert set(read(root/'completion.json')['completed']) == expected
    assert {name(i) for i in top['identities']} == expected
    agent, meta = ttt.load_agent(argparse.Namespace(checkpoint=a.checkpoint, synthetic=False))
    nominal = deepcopy(agent.model.state_dict())
    initial_hash = tree_hash(nominal)
    inventory, parameter_rows = [], []
    for identity in identities(c):
        path = root/name(identity)
        m, d = read(path/'manifest.json'), read(path/'completion.json')
        snapshot = path/'snapshot-0003000.pt'
        assert m['identity'] == d['identity'] == identity
        assert m['configuration'] == c and m['checkpoint_metadata']['config'] == meta['config']
        assert m['source_hashes'] == sources() and m['checkpoint_sha256'] == ttt.SHA
        assert d['status'] == 'PASS' and d['transitions'] == 3000
        assert d['dynamics_steps'] == d['q_steps'] == d['target_steps'] == 118 and d['total_optimizer_steps'] == 236
        assert m['initial_model_hash'] == d['initial_model_hash'] == initial_hash
        assert d['snapshot_sha256'] == digest(snapshot)
        check_log(path, c)
        p = torch.load(snapshot, map_location='cpu', weights_only=False)
        assert p['identity'] == identity and p['configuration'] == c and p['checkpoint_sha256'] == ttt.SHA
        assert p['source_hashes'] == sources() and p['checkpoint_config'] == meta['config']
        assert p['initial_model_hash'] == initial_hash
        assert p['original_optimizer_hash'] == m['original_optimizer_hash'] == tree_hash(ttt.aux_state(agent))
        agent.model.load_state_dict(nominal)
        adapter = DynamicsQAdapter(agent, c, identity)
        agent.model.load_state_dict(p['model'])
        adapter.assert_fixed()
        assert tree_hash(adapter.fixed_state()) == p['fixed_hash'] == d['fixed_hash'] == m['initial_fixed_hash']
        assert tree_hash(agent.model.state_dict()) == d['final_model_hash']
        actual_audit = adapter.parameter_audit()
        assert all(r['changed'] == (r['component'] in ('dynamics','q','target_q')) for r in actual_audit)
        state = p['adapter']
        assert state['transitions'] == 3000 and state['dynamics_steps'] == state['q_steps'] == state['target_steps'] == 118
        assert len(state['replay']) == 3000
        for j, record in enumerate(state['replay']):
            assert record['global_step'] == j+1 and record['adaptation_seed'] == identity['adaptation_seed']
            assert record['episode'] == j//1000 and record['episode_step'] == j%1000
            assert record['parameter'] == identity['parameter'] and record['severity'] == identity['severity']
            assert record['discount'] == .995 and not record['terminated']
            assert record['truncated'] == ((j+1)%1000 == 0)
            assert abs(record['actual_parameter']-record['nominal_parameter']*identity['severity']) < 1e-10
            assert math.isfinite(record['reward'])
            if j and j%1000:
                assert equal_state(state['replay'][j-1]['next_observation'], record['observation'])
        for field, opt in [('dynamics_optimizer',adapter.dynamics_optimizer), ('q_optimizer',adapter.q_optimizer)]:
            saved = state[field]
            group = saved['param_groups'][0]
            own = opt.param_groups[0]
            for key in ('lr','betas','eps','weight_decay'):
                assert group[key] == own[key]
            assert saved['state'] and all(int(s['step']) == 118 for s in saved['state'].values())
        inventory.append(dict(**identity, snapshot=str(snapshot.resolve()), snapshot_sha256=digest(snapshot),
                              interactions=3000, dynamics_steps=118, q_steps=118, total_optimizer_steps=236,
                              wall_seconds=d['wall_seconds'], update_seconds=d['update_seconds'], replay_size=3000))
        parameter_rows.extend(dict(**identity, **row) for row in actual_audit)
        del adapter, p
    return inventory, parameter_rows


def load_final(a, c, item):
    import torch
    from common.dynamics_q import DynamicsQAdapter
    from training_support import tree_hash
    assert digest(item['snapshot']) == item['snapshot_sha256']
    p = torch.load(item['snapshot'], map_location='cpu', weights_only=False)
    identity = {k: item[k] for k in ('parameter','severity','adaptation_seed','method','schedule')}
    assert p['identity'] == identity and p['configuration'] == c and p['checkpoint_sha256'] == ttt.SHA
    agent, meta = ttt.load_agent(argparse.Namespace(checkpoint=a.checkpoint, synthetic=False))
    assert p['checkpoint_config'] == meta['config']
    adapter = DynamicsQAdapter(agent, c, identity)
    agent.model.load_state_dict(p['model'])
    adapter.load_state_dict(p['adapter'])
    adapter.assert_fixed()
    assert tree_hash(adapter.fixed_state()) == p['fixed_hash']
    agent.model.requires_grad_(False)
    agent.model.eval()
    return agent, adapter


def evaluate_one(a, c, reference, baseline, item):
    import numpy as np
    import torch
    from training_support import tree_hash
    agent, adapter = load_final(a, c, item)
    before = tree_hash(ttt.aux_state(agent))
    identity = {k:item[k] for k in ('parameter','severity','adaptation_seed','method','schedule')}
    result = []
    with adapter.held_out():
        for seed in c['evaluation_seeds']:
            ttt.seeded(seed)
            k = (item['parameter'], float(item['severity']), seed)
            row = dict(**identity, evaluation_seed=seed, mode=a.mode)
            if a.mode == 'prediction':
                filename = baseline['files'][k]
                bank = Path(a.reference)/'frozen'/filename
                assert digest(bank) == baseline['hashes'][filename]
                with np.load(bank) as raw:
                    obs, actions = raw['observations'].astype('float32'), raw['actions'].astype('float32')
                assert obs.shape == (1001,17) and actions.shape == (1000,6)
                total = 0.
                for start in range(0,1000,256):
                    end = min(start+256,1000)
                    o, n, ac = [torch.as_tensor(v, device=agent.device) for v in
                                (obs[start:end], obs[start+1:end+1], actions[start:end])]
                    prediction = agent.model.next(agent.model.encode(o,None), ac,None)
                    total += float((prediction-agent.model.encode(n,None)).square().mean(-1).sum())
                mse = total/1000
                f = baseline['mse'][k]
                row.update(latent_mse=mse, frozen_latent_mse=f, mse_change=mse-f,
                           mse_relative_change=(mse-f)/f if f > 1e-10 else None,
                           bank_file=filename, bank_sha256=baseline['hashes'][filename], transitions=1000)
            else:
                cond = identity_condition(identity)
                schedule = 'nominal' if a.mode == 'nominal' else 'fixed'
                env = ttt.environment(reference, cond, schedule)
                try:
                    data = [x[0] for x in ttt.rollout(agent, adapter, reference, cond, schedule, seed, env, 0, False)]
                    assert len(data) == 1000 and data[-1]['truncated'] and not data[-1]['terminated']
                    actual = [r['actual_parameter'] for r in data]
                    expected = env.nominal_value*(1 if a.mode == 'nominal' else identity['severity'])
                    assert all(abs(x-expected) < 1e-10 for x in actual)
                    frozen = baseline['episodes'][(identity['parameter'], 1. if a.mode == 'nominal' else identity['severity'], seed)]
                    ret = sum(r['reward'] for r in data)
                    row.update(episode_return=ret, frozen_return=float(frozen['episode_return']),
                        dq_minus_frozen=ret-float(frozen['episode_return']), return_per_step=ret/len(data),
                        forward_velocity=float(np.mean([r['velocity'] for r in data])),
                        control_cost=sum(r['control_cost'] for r in data),
                        action_saturation=float(np.mean([r['action_component_saturation'] for r in data])),
                        action_abs=float(np.mean([r['action_abs'] for r in data])),
                        actuator_force_rms=float(np.mean([r['actuator_force_rms'] for r in data])),
                        actuator_force_saturation=float(np.mean([r['actuator_force_saturation'] for r in data])) if data[0]['force_limited_count'] else None,
                        episode_length=len(data), terminated=False, truncated=True,
                        nominal_parameter=env.nominal_value, actual_parameter_min=min(actual), actual_parameter_max=max(actual))
                finally:
                    env.close()
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
            result.append(row)
    assert tree_hash(ttt.aux_state(agent)) == before
    assert digest(item['snapshot']) == item['snapshot_sha256']
    return result


def evaluate(a, c, reference, baseline):
    audit = read(a.audit)
    assert audit['status'] == 'PASS' and audit['configuration'] == c and audit['source_hashes'] == sources()
    assert audit['checkpoint_sha256'] == ttt.SHA and audit['reference'] == str(Path(a.reference).resolve())
    manifest = dict(mode=a.mode, audit_sha256=digest(a.audit), checkpoint_sha256=ttt.SHA,
                    configuration=c, source_hashes=sources(), evaluation_seeds=c['evaluation_seeds'])
    out = open_stage(a.output, manifest, a.resume)
    passed, failed = [], []
    for item in audit['inventory']:
        path, skip = destination(out, name(item))
        if skip:
            d = read(path/'completion.json')
            assert d['snapshot_sha256'] == item['snapshot_sha256'] and d['csv_sha256'] == digest(path/f'{a.mode}.csv')
            passed.append(path.name)
            continue
        started = time.perf_counter()
        try:
            data = evaluate_one(a,c,reference,baseline,item)
            assert len(data) == 20
            table(path/f'{a.mode}.csv',data)
            write(path/'completion.json', dict(status='PASS',rows=20,optimizer_steps=0,replay_writes=0,
                snapshot_sha256=item['snapshot_sha256'],csv_sha256=digest(path/f'{a.mode}.csv'),
                wall_seconds=time.perf_counter()-started))
            passed.append(path.name)
            print('PASS',path.name,flush=True)
        except Exception as exc:
            failed.append(dict(identity=path.name,error=repr(exc)))
            write(path/'failure.json',dict(error=repr(exc)))
            print('FAIL',path.name,repr(exc),flush=True)
    status = 'PASS' if len(passed)==36 and not failed else 'PARTIAL'
    finish(out,dict(status=status,mode=a.mode,completed=passed,failed=failed,expected=36))
    return status


def fixture_smoke(a,c):
    import torch
    from common.dynamics_q import DynamicsQAdapter
    torch.set_num_threads(1)
    torch.manual_seed(17)
    agent = cpu_agent()
    identity = dict(parameter='bthigh_damping', severity=2., adaptation_seed=c['adaptation_seeds'][0],
                    method=c['method'], schedule='fixed')
    adapter = DynamicsQAdapter(agent,c,identity)
    o = torch.randn(17)
    for t in range(100):
        action = torch.randn(6).tanh()
        n = o+.01*torch.randn(17)
        adapter.observe(o,action,n,float(n[0]),False,False,0,t,12.,6.)
        o = n
    invariance = adapter.parameter_audit()
    assert all(r['changed'] == (r['component'] in ('dynamics','q','target_q')) for r in invariance)
    assert adapter.dynamics_steps == adapter.q_steps == 2
    with adapter.held_out():
        agent.model.Q(agent.model.encode(o[None],None),action[None],None)
    out = open_stage(a.output,dict(command='smoke',cpu_fixture=True,configuration=c,source_hashes=sources()))
    table(out/'parameter_invariance.csv',invariance)
    finish(out,dict(status='PASS',cpu_fixture=True,transitions=100,dynamics_steps=2,q_steps=2,
                    real_checkpoint_physics='NOT_RUN',metrics=adapter.last_metrics))
    return 'PASS'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['smoke','run','audit','evaluate','aggregate','status'])
    p.add_argument('--config',type=Path,default=DEFAULT)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--reference',type=Path,help='Existing physical-severity-750k-v1')
    p.add_argument('--ttt-runs',type=Path,help='Existing fixed-frozen/fixed-decision parent')
    p.add_argument('--runs',type=Path,help='Completed Dynamics+Q matrix')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke-result',type=Path)
    p.add_argument('--audit',type=Path)
    p.add_argument('--mode',choices=['prediction','shifted','nominal'])
    p.add_argument('--prediction',type=Path)
    p.add_argument('--shifted',type=Path)
    p.add_argument('--nominal',type=Path)
    p.add_argument('--ttt-final',type=Path)
    p.add_argument('--joint-final',type=Path)
    p.add_argument('--parameter',choices=['bthigh_damping','torso_mass'])
    p.add_argument('--severity',type=float)
    p.add_argument('--seed',type=int)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--allow-shared-gpu',action='store_true')
    p.add_argument('--resume',action='store_true',help='Skip verified PASS children; archive incomplete children; no mid-stream resume')
    p.add_argument('--cpu-fixture',action='store_true',help='Smoke only: real Q math on synthetic CPU data, not physical/CUDA evidence')
    a = p.parse_args()
    c, reference = configuration(a.config)
    if a.command == 'status':
        print(json.dumps(read(a.output/'completion.json') if (a.output/'completion.json').exists() else
                         dict(status='INCOMPLETE',completed_children=len(list(a.output.glob('*/completion.json')))),indent=2))
        return
    if a.cpu_fixture and a.command != 'smoke':
        p.error('--cpu-fixture is only valid for smoke')
    sys.path.insert(0,str(ttt.ROOT/'tdmpc2'))
    if a.cpu_fixture:
        print(fixture_smoke(a,c))
        return
    needed = ['audit','prediction','shifted','ttt_final','joint_final'] if a.command=='aggregate' else ['checkpoint','reference','ttt_runs']
    if a.command=='audit': needed += ['runs']
    if a.command=='evaluate': needed += ['audit','mode']
    missing = [x for x in needed if getattr(a,x) is None]
    if missing: p.error('Required: '+', '.join('--'+x.replace('_','-') for x in missing))
    if a.command=='aggregate':
        from analyze_dynamics_q import aggregate
        aggregate(a,c)
        return
    from run_full_joint_rl import gpu_status
    gpu = gpu_status(a.gpu,8192,a.allow_shared_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
    ttt.dependencies()
    baseline = preflight(a,reference)
    if a.command in ('smoke','run'):
        status = train(a,c)
    elif a.command=='audit':
        inventory, parameters = audit_runs(a,c)
        a.output.mkdir(parents=True,exist_ok=False)
        write(a.output/'audit.json',dict(status='PASS',configuration=c,source_hashes=sources(),
            checkpoint_sha256=ttt.SHA,reference=str(a.reference.resolve()),inventory=inventory))
        table(a.output/'adaptation_run_inventory.csv',inventory)
        table(a.output/'parameter_invariance.csv',parameters)
        write(a.output/'completion.json',dict(status='PASS',streams=len(inventory)))
        status = 'PASS'
    else:
        status = evaluate(a,c,reference,baseline)
    print(status,flush=True)
    if status!='PASS': raise SystemExit(1)


if __name__ == '__main__':
    main()
