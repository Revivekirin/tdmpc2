"""Read-only nominal convergence audit. Run on the original CUDA evaluation host."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

METRICS = ('consistency_loss', 'reward_loss', 'value_loss', 'total_loss',
           'grad_norm', 'pi_entropy', 'pi_scaled_entropy', 'pi_scale', 'pi_grad_norm')
ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def write_csv(path, rows):
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stream_training(path, seed_steps, width=50000):
    """One bin in memory, exact medians, population SD, entire file in one pass."""
    bins, values = [], {k: [] for k in METRICS}
    current = 0
    state = dict(rows=0, first_update_transition=None, final_updates=0,
                 update_count_mismatches=0, missing_metric_rows=0)
    h = hashlib.sha256()

    def flush():
        for k, v in values.items():
            bins.append(dict(bin_start=current * width + 1, bin_end=(current + 1) * width,
                             metric=k, count=len(v), mean=statistics.mean(v) if v else None,
                             median=statistics.median(v) if v else None,
                             std=statistics.pstdev(v) if v else None))

    with Path(path).open('rb') as f:
        for line_no, raw in enumerate(f, 1):
            h.update(raw)
            try:
                row = json.loads(raw)
            except Exception as exc:
                raise ValueError(f'Invalid JSON at line {line_no}; use a complete stable log') from exc
            t = row['environment_transitions']
            if t != state['rows'] + 1:
                raise ValueError(f'Non-contiguous transitions at line {line_no}: {t}')
            b = (t - 1) // width
            if b != current:
                flush()
                current, values = b, {k: [] for k in METRICS}
            expected = 0 if t <= seed_steps else t - 1
            u = row['gradient_updates']
            state['update_count_mismatches'] += int(u != expected)
            if u and state['first_update_transition'] is None:
                state['first_update_transition'] = t
            if t > seed_steps:
                state['missing_metric_rows'] += int(any(k not in row for k in METRICS))
            for k in METRICS:
                if k in row:
                    v = float(row[k])
                    if not math.isfinite(v):
                        raise ValueError(f'Nonfinite {k} at transition {t}')
                    values[k].append(v)
            state.update(rows=t, final_updates=u)
    if not state['rows']:
        raise ValueError('Empty training log')
    flush()
    state['sha256'] = h.hexdigest()
    return bins, state


def training_curve(run):
    path = run / 'evaluation.csv'
    groups = {}
    if path.exists():
        with path.open() as f:
            for r in csv.DictReader(f):
                if str(r.get('random_actions', '')).lower() != 'false':
                    continue
                if float(r.get('mass_scale', 1)) != 1:
                    raise ValueError('Non-nominal training evaluation encountered')
                groups.setdefault(int(r['environment_transitions']), []).append(float(r['episode_return']))
    return {t: statistics.mean(v) for t, v in groups.items()}


def plots(out, bins, summaries, rows, curve):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, keys, shape in [('nominal_losses.png', METRICS[:5], (3, 2)),
                              ('nominal_policy_diagnostics.png', METRICS[5:], (2, 2))]:
        fig, axes = plt.subplots(*shape, figsize=(12, 9), squeeze=False)
        for ax, key in zip(axes.flat, keys):
            data = [b for b in bins if b['metric'] == key and b['count']]
            x = np.array([b['bin_end'] for b in data]) / 1000
            m = np.array([b['mean'] for b in data]); sd = np.array([b['std'] for b in data])
            ax.plot(x, m, label='Mean'); ax.plot(x, [b['median'] for b in data], label='Median')
            ax.fill_between(x, m-sd, m+sd, alpha=.2, label='Mean ± population SD')
            ax.set(title=key, xlabel='Environment transitions (k)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(keys):]:
            ax.set_visible(False)
        fig.suptitle('Exact 50k transition bins; last bin may be partial')
        fig.tight_layout(); fig.savefig(out/name, dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = [s['transitions']/1000 for s in summaries]
    axes[0].errorbar(x, [s['mean_return'] for s in summaries],
                    yerr=[s['std_return'] for s in summaries], fmt='o-', capsize=4, label='Validation mean ± sample SD')
    for seed in sorted({r['seed'] for r in rows}):
        data = sorted([r for r in rows if r['seed'] == seed], key=lambda r:r['transitions'])
        axes[0].plot([r['transitions']/1000 for r in data], [r['episode_return'] for r in data], alpha=.15)
    if curve:
        t = sorted(curve)
        axes[1].plot([v/1000 for v in t], [curve[v] for v in t], 'o-', label='Repeated training evaluation seeds')
    axes[1].plot(x, [s['mean_return'] for s in summaries], 'o-', label='Independent validation seeds')
    for ax in axes:
        ax.set(xlabel='Checkpoint transitions (k)', ylabel='Episode return'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'nominal_checkpoint_comparison.png', dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Must not exist; outside the training run')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed-start', type=int, default=30000001)
    args = p.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    if out == run or run in out.parents:
        p.error('Output must be outside the original training run')
    if args.episodes < 10:
        p.error('Use at least 10 independent validation seeds')
    if not (run/'completion.json').exists():
        p.error('completion.json required: audit a completed, stable run')
    provenance = json.loads((run/'metadata.json').read_text())
    config = provenance['config']
    if config['task'] != 'nsgym-halfcheetah' or config['episodic'] or config['nsgym']['mode'] != 'nominal' or config['nsgym']['mass_scale'] != 1:
        p.error('Expected continuing-task nominal HalfCheetah configuration')
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    if set(seeds) & set(config['eval_seeds']) or any(config['seed'] <= s <= config['seed'] + config['steps'] for s in seeds):
        p.error('Validation seeds overlap training/reset/evaluation seeds')
    available = {int(d.name.split('-')[-1]): d for d in run.glob('checkpoint-*') if d.is_dir()}
    candidates = {500000, 750000, 950000}
    if not candidates <= available.keys():
        p.error(f'Missing required checkpoints: {sorted(candidates - available.keys())}')
    curve = training_curve(run)
    if not curve:
        p.error('evaluation.csv must contain the full training evaluation curve')
    peak = max(curve, key=curve.get)
    if peak not in available:
        p.error(f'Training curve peak checkpoint {peak} is missing')
    candidates.update([peak, max(available)])
    from training_support import load_checkpoint, evaluate, learned_state, tree_hash
    from research_utils import metadata, bootstrap
    import numpy as np
    import torch
    out.mkdir(parents=True, exist_ok=False)
    bins, log = stream_training(run/'training.jsonl', config['seed_steps'])
    write_csv(out/'nominal_loss_statistics.csv', bins)
    write_json(out/'training_log_audit.json', log)
    completion = json.loads((run/'completion.json').read_text())
    if log['rows'] != config['steps'] or completion['environment_transitions'] != log['rows'] or completion['gradient_updates'] != log['final_updates'] or log['update_count_mismatches'] or log['missing_metric_rows']:
        raise ValueError('Training log/count validation failed; inspect training_log_audit.json')
    audit_meta = metadata(dict(run=str(run), seeds=seeds, candidates=sorted(candidates), nominal_only=True))
    write_json(out/'audit_provenance.json', audit_meta)
    rows, summaries, checkpoint_meta, integrity = [], [], {}, []
    for t in sorted(candidates):
        cp = available[t]
        print(f'Validating {cp.name} on {len(seeds)} nominal seeds', flush=True)
        hashes = {name: sha(cp/name) for name in ('model.pt','optimizers.pt','metadata.json')}
        agent, meta = load_checkpoint(cp)
        if meta['config'].get('smoke') or meta['environment_transitions'] != t or meta['optimizer_updates'] != t-1:
            raise ValueError('Checkpoint metadata/count mismatch')
        for key, value in config.items():
            if meta['config'].get(key) != value:
                raise ValueError(f'Checkpoint config differs from training config: {key}')
        checkpoint_meta[t] = meta
        before = tree_hash(learned_state(agent))
        original_act = agent.act
        captured = []

        def measured_act(*a, **kw):
            action = original_act(*a, **kw)
            captured.append(action.detach().cpu().numpy().copy())
            return action

        def forbidden_update(*a, **kw):
            raise RuntimeError('Updates forbidden during audit')

        agent.act = measured_act
        agent.update = forbidden_update
        for seed in seeds:
            captured.clear()
            episode, elapsed = evaluate(agent, meta['config'], [seed], scale=1., fixed=False)
            r = episode[0]
            a = np.asarray(captured)
            if len(a) != r['episode_length'] or not np.isfinite(a).all() or np.abs(a).max() > 1.00001:
                raise ValueError('Invalid action trace')
            r.update(transitions=t, checkpoint_sha256=hashes['model.pt'],
                     return_per_environment_step=r['episode_return']/r['episode_length'],
                     control_cost=-r['reward_control'], control_cost_per_step=-r['reward_control']/r['episode_length'],
                     action_abs_max=float(np.abs(a).max()), action_rms=float(np.sqrt(np.square(a).mean())),
                     action_saturation_fraction=float((np.abs(a)>=.99).mean()),
                     forward_reward_reconstructed=r['episode_return']-r['reward_control'],
                     evaluation_wall_seconds=elapsed)
            rows.append(r)
            with (out/'validation_progress.jsonl').open('a') as f:
                f.write(json.dumps(r, allow_nan=False)+'\n')
            print(f'  seed={seed} return={r["episode_return"]:.2f}', flush=True)
        after = tree_hash(learned_state(agent))
        if before != after or any(sha(cp/name) != value for name,value in hashes.items()):
            raise ValueError('Checkpoint or learned state changed')
        integrity.append(dict(transitions=t, learned_state_before=before, learned_state_after=after,
                              file_sha256=hashes, optimizer_updates_during_evaluation=0))
        group = [r for r in rows if r['transitions']==t]
        returns = [r['episode_return'] for r in group]
        summary = dict(transitions=t, episodes=len(group), mean_return=statistics.mean(returns),
                       std_return=statistics.stdev(returns), return_ci95=bootstrap(returns))
        for key in ('return_per_environment_step','forward_velocity','control_cost','control_cost_per_step',
                    'episode_length','terminated','truncated','action_abs_mean','action_abs_max','action_rms','action_saturation_fraction'):
            summary[key] = statistics.mean(float(r[key]) for r in group)
        summaries.append(summary)
        del agent, original_act
        torch.cuda.empty_cache()
    if sha(run/'training.jsonl') != log['sha256']:
        raise ValueError('Training log changed during audit')
    write_json(out/'nominal_integrity.json', integrity)
    write_csv(out/'nominal_checkpoint_validation.csv', rows)
    write_json(out/'nominal_checkpoint_summary.json', summaries)
    selected = max(summaries, key=lambda s:(s['mean_return'], -s['transitions']))
    t = selected['transitions']; meta = checkpoint_meta[t]
    comparisons = []
    winner = {r['seed']:r['episode_return'] for r in rows if r['transitions']==t}
    for s in summaries:
        if s['transitions'] == t:
            continue
        difference = [winner[r['seed']] - r['episode_return'] for r in rows if r['transitions']==s['transitions']]
        comparisons.append(dict(other_transitions=s['transitions'], paired_gain=statistics.mean(difference),
                                exploratory_ci95=bootstrap(difference)))
    write_json(out/'selected_nominal_checkpoint.json', dict(
        status='SELECTED_ON_NOMINAL_VALIDATION', checkpoint=str(available[t]), sha256=sha(available[t]/'model.pt'),
        configuration=meta['config'], source_revision=meta['provenance'].get('git_sha'),
        training_provenance=meta['provenance'], evaluation_source_revision=audit_meta['git_sha'],
        evaluation_source_hashes=audit_meta['source_hashes'], training_evaluation_seeds=config['eval_seeds'],
        validation_seeds=seeds, candidates=sorted(candidates), validation_summary=selected,
        selection_rule='Maximum nominal validation mean episode return; exact tie selects earlier checkpoint',
        paired_comparisons=comparisons,
        limitation='Selection-set performance, not independent final-test performance; no universal convergence threshold.'))
    plots(out, bins, summaries, rows, curve)
    lines = ['# Nominal convergence audit', '',
        f'Run: `{run}`. Completed log: {log["rows"]:,} transitions; SHA256 `{log["sha256"]}`.',
        f'Selected checkpoint: **{t:,}**, nominal validation return **{selected["mean_return"]:.2f} ± {selected["std_return"]:.2f}** (sample SD, {len(seeds)} seeds).',
        'Selection uses nominal return only. No training, adaptation, or dynamics-shift evaluation was executed.', '',
        '## Evidence and limits',
        'A high return alone does not prove convergence. No absolute acceptable-return threshold was supplied. '
        'Compare the late checkpoints and seed variability below; validation selection introduces optimism. '
        'Paired bootstrap intervals are exploratory, not corrected for checkpoint selection or multiple comparisons. '
        'An independent final test would be needed for an unbiased selected-controller performance estimate.',
        f'Training evaluation seeds: {config["eval_seeds"]}. Validation seeds: {seeds}.',
        f'Candidates: {sorted(candidates)}; includes requested checkpoints, training-curve peak ({peak:,}), and latest available checkpoint.', '',
        '| Transitions | Return mean | Sample SD | Return/step | Velocity | Cost/step | Length | Terminated fraction | Truncated fraction | Saturation fraction |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append('| ' + ' | '.join(f'{s[k]:.4f}' for k in ('transitions','mean_return','std_return','return_per_environment_step',
                     'forward_velocity','control_cost_per_step','episode_length','terminated','truncated','action_saturation_fraction')) + ' |')
    lines += ['', 'Control cost is minus summed reward_ctrl; saturation is the fraction of action components with |a| ≥ 0.99. '
              'Per-seed CSV includes action mean magnitude, RMS, maximum, and termination/truncation. '
              'Forward reward is also reconstructed as return minus control reward to avoid historical reward_run reporting defects.',
              '', '## Paired selected-minus-other comparisons']
    for c in comparisons:
        lines.append(f'- Against {c["other_transitions"]:,}: mean gain {c["paired_gain"]:.2f}; exploratory 95% CI {c["exploratory_ci95"]}.')
    lines += ['', '## Training log and code audit',
        f'All {log["rows"]:,} JSONL records streamed. Exact mean/median/population SD use bins [1, 50000], [50001, 100000], etc.; '
        'only one bin is retained in memory. Missing pre-update losses are not zero-filled. Full statistics: nominal_loss_statistics.csv.',
        f'First gradient update transition: {log["first_update_transition"]}; final cumulative updates: {log["final_updates"]}; count mismatches: {log["update_count_mismatches"]}.',
        'Local source contract: train_nominal.train uses random actions while zero-based step <= seed_steps. '
        'At seed_steps=10000, random exploration lasts 10001 transitions; the first policy action is transition 10002. '
        'After collecting transition 10001, 10000 pretraining updates run; subsequent transitions each add one update. '
        'Thus cumulative updates U(T)=0 for T<=10000, otherwise T-1. Each update has model and policy optimizer steps.',
        'Only the LAST of the initial 10000 update metrics is logged. The loss statistics describe logged transition-level samples, '
        'not all optimizer steps; the initial 9999 optimizer metrics cannot be reconstructed.',
        'NS-Gym adapter resets on terminated OR truncated. Replay stores only true terminated. '
        'The trainer rejects true termination when episodic=false. The TD target is reward + discount*(1-terminated)*target_Q; '
        'time-limit truncation retains bootstrapping from the final observation. The termination head/loss is disabled for episodic=false.',
        'pi_grad_norm and grad_norm are pre-clipping norms returned by clip_grad_norm_. '
        'pi_scale is the running Q-value percentile-range normalization scale, not action standard deviation. '
        'pi_entropy is sampled negative squashed log probability; pi_scaled_entropy uses the implementation’s scaled log probability. '
        'Neither alone certifies good control or convergence.', '', '### First versus last nonempty 50k bins']
    for key in METRICS:
        data = [b for b in bins if b['metric']==key and b['count']]
        lines.append(f'- {key}: mean {data[0]["mean"]:.6g} → {data[-1]["mean"]:.6g}; '
                     f'median {data[0]["median"]:.6g} → {data[-1]["median"]:.6g}; SD {data[0]["std"]:.6g} → {data[-1]["std"]:.6g}.')
    lines += ['', '## Evaluator and provenance',
        'evaluate_frozen.py supports --checkpoint, --config, --output, --quality-report, --integration-only, '
        '--nominal-only, --episodes, --expanded, --summarize. Scientific mode requires a quality report matching the checkpoint hash; '
        'that gate tests superiority to random, not convergence. This audit directly reuses training_support.load_checkpoint/evaluate '
        'for checkpoint selection without fabricating a quality report. It calls scale=1, fixed=False only.',
        'The loader verifies model/optimizer SHA256, requires CUDA, validates dimensions, restores model/optimizer/RunningScale, '
        'and disables parameter gradients. Evaluation uses eval_mode=True, no_grad, per-seed RNG reset, and restores RNG/planner state. '
        'The audit forbids agent.update and verifies learned-state and checkpoint-file hashes before/after each candidate.',
        f'Training revision: `{meta["provenance"].get("git_sha")}`. Evaluation revision: `{audit_meta["git_sha"]}`. '
        'See selected_nominal_checkpoint.json and audit_provenance.json for configuration, dirty status, versions and source hashes. '
        'Code descriptions refer to the executing checkout; saved provenance should be compared if training/evaluation revisions differ.',
        '', '![Losses](nominal_losses.png)', '![Policy diagnostics](nominal_policy_diagnostics.png)',
        '![Checkpoint comparison](nominal_checkpoint_comparison.png)']
    (out/'NOMINAL_CONVERGENCE_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(f'Audit complete: {out}', flush=True)


if __name__ == '__main__':
    main()
