"""Reanalyse completed 750k severity data without running an environment or agent.

All uncertainty intervals resample independent evaluation/calibration seeds, never
individual time steps or correlated probe windows. Outputs go to a new directory.
"""
from . import schema as _io
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


_CONTRACT = _io.read_required(Path(__file__).with_name('config.json'))['historical_adapters']['physical']
PARAMS = tuple(_CONTRACT['parameters'])
LABELS = _CONTRACT['labels']
SEVERITIES = tuple(_CONTRACT['severities'])
COLORS = dict(zip(SEVERITIES, plt.get_cmap('viridis')(np.linspace(.08, .9, len(SEVERITIES)))))
PRIMARY_SHA256 = _CONTRACT['checkpoint_sha256']


read_csv = _io.list_rows


def write_csv(path, rows):
    return _io.save_table(path, rows, list(rows[0]) if rows else None, mode='w', empty='error')


sha256 = _io.sha256


def stats(values, seed=20261111):
    from .statistics import bootstrap
    a = np.asarray(values, dtype=float)
    lo, hi = bootstrap(a, seed=seed, single=True)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), ci_low=lo, ci_high=hi)


def key(row):
    return row['parameter'], float(row['severity']), int(row['seed'])


def verify(run):
    required = ('physical', 'physical-latent', 'frozen', 'heldout-latent',
                'nonstationary-nominal', 'nonstationary-fixed',
                'nonstationary-abrupt', 'nonstationary-abrupt-latent', 'planning')
    manifests = {}
    for stage in required:
        folder = run / stage
        done = json.loads((folder / 'completion.json').read_text())
        if done['status'] != 'PASS':
            raise ValueError(f'{stage} is not PASS')
        manifests[stage] = json.loads((folder / 'manifest.json').read_text())
    if len({m['protocol_sha256'] for m in manifests.values()}) != 1:
        raise ValueError('Mixed protocol hashes')
    if len({m['expected_checkpoint_sha256'] for m in manifests.values()}) != 1:
        raise ValueError('Mixed checkpoints')
    if (manifests['physical']['expected_checkpoint_sha256'] != PRIMARY_SHA256 or
            any(m['checkpoint_role'] != 'primary' for m in manifests.values())):
        raise ValueError('This analysis requires the fixed primary 750k checkpoint')
    for stage in ('frozen', 'planning', 'nonstationary-nominal',
                  'nonstationary-fixed', 'nonstationary-abrupt'):
        integrity = json.loads((run / stage / 'model_integrity.json').read_text())
        if integrity['optimizer_updates'] != 0 or integrity['before'] != integrity['after']:
            raise ValueError(f'Frozen model integrity failed for {stage}')
    expected = {(p, k) for p in PARAMS for k in SEVERITIES}
    for stage in ('physical', 'frozen', 'nonstationary-abrupt'):
        m = manifests[stage]
        if stage == 'nonstationary-abrupt' and m['schedule_timing']['onset'] != 250:
            raise ValueError('Unexpected abrupt onset')
        actual = {(x['parameter'], float(x['severity'])) for x in m['conditions']}
        if actual != expected:
            raise ValueError(f'{stage} does not contain the complete preregistered grid')
    return manifests['physical']['protocol_sha256'], manifests['physical']['expected_checkpoint_sha256']


def episodes(run, stage):
    rows = read_csv(run / stage / 'episodes.csv')
    return {key(r): r for r in rows}


def check_duplicates(run, frozen, abrupt):
    nominal = episodes(run, 'nonstationary-nominal')
    fixed = episodes(run, 'nonstationary-fixed')
    if set(fixed) != set(frozen):
        raise ValueError('S1 and Frozen condition/seed sets differ')
    if any(float(fixed[k]['episode_return']) != float(frozen[k]['episode_return']) for k in frozen):
        raise ValueError('S1 fixed differs from Frozen')
    for (par, sev, seed), row in nominal.items():
        if sev != 1.0 or float(row['episode_return']) != float(frozen[par, 1.0, seed]['episode_return']):
            raise ValueError('S0 differs from Frozen nominal')
    if set(abrupt) != set(frozen):
        raise ValueError('S2 and Frozen condition/seed sets differ')


def physical_seed_data(run):
    groups = defaultdict(list)
    values = {}
    source = run / 'physical/physical_response_pairs.csv'
    for row in read_csv(source):
        if row['numerically_valid'] != 'True':
            raise ValueError('Invalid physical probe included')
        par, sev, seed = key(row)
        measure = 'joint_velocity_l2' if par == 'bthigh_damping' else 'root_velocity_l2'
        groups[par, sev, seed].append(float(row[measure]))
        pair = float(row['actual_parameter']), float(row['nominal_parameter'])
        if (par, sev) in values and not np.allclose(values[par, sev], pair, atol=1e-8, rtol=0):
            raise ValueError(f'Parameter readback inconsistency for {par}, {sev}')
        values[par, sev] = pair
    means = {k: float(np.mean(v)) for k, v in groups.items()}
    if len(means) != 36 or any(len(v) != 16 for v in groups.values()):
        raise ValueError('Expected 3 seeds x 12 conditions x 16 probe windows')
    return means, values


def latent_seed_data(path, onset=None, bins=False):
    sums = defaultdict(float)
    counts = defaultdict(int)
    regimes = defaultdict(int)
    binned = defaultdict(lambda: [0., 0])
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if int(row['horizon']) != 1:
                continue
            par, sev, seed = key(row)
            t = int(row['episode_step'])
            if onset is not None and t < onset:
                if bins:
                    bin_key = (par, sev, seed, t // 25)
                    binned[bin_key][0] += float(row['latent_mse'])
                    binned[bin_key][1] += 1
                continue
            k = (par, sev, seed)
            value = float(row['latent_mse'])
            sums[k] += value
            counts[k] += 1
            regimes[par, sev, row['action_regime']] += 1
            if bins:
                bin_key = (par, sev, seed, t // 25)
                binned[bin_key][0] += value
                binned[bin_key][1] += 1
    means = {k: sums[k] / counts[k] for k in counts}
    bin_means = {k: v[0] / v[1] for k, v in binned.items()}
    return means, counts, regimes, bin_means


def physical_fixed_latent(run):
    return latent_seed_data(run / 'physical-latent/latent_errors.csv')[0]


def velocity_bins(run):
    folder = run / 'nonstationary-abrupt'
    entries = json.loads((folder / 'dataset_index.json').read_text())
    result = {}
    for entry in entries:
        par = entry['condition']['parameter']
        sev = float(entry['condition']['severity'])
        seed = int(entry['seed'])
        bins = defaultdict(list)
        with (folder / entry['steps']).open(newline='') as stream:
            rows = csv.DictReader(stream)
            for row in rows:
                t = int(row['episode_step'])
                if t in (249, 250) and sev != 1.0:
                    expected = float(row['nominal_parameter']) * (1.0 if t == 249 else sev)
                    if not np.isclose(float(row['actual_parameter']), expected, atol=1e-8, rtol=0):
                        raise ValueError(f'S2 readback mismatch: {par}, k={sev}, seed={seed}, t={t}')
                bins[t // 25].append(float(row['forward_velocity']))
        if len(bins) != 40 or any(len(v) != 25 for v in bins.values()):
            raise ValueError('Expected 40 complete 25-step bins per episode')
        result[par, sev, seed] = {b: float(np.mean(v)) for b, v in bins.items()}
    return result


def condition_rows(physical, applied, latent, frozen):
    rows = []
    seeds = sorted({k[2] for k in frozen})
    calibration = sorted({k[2] for k in physical})
    for par in PARAMS:
        for sev in SEVERITIES:
            response = stats([physical[par, sev, s] for s in calibration])
            latent_ratio = stats([latent[par, sev, s] / latent[par, 1., s] for s in calibration])
            returns = stats([float(frozen[par, sev, s]['episode_return']) for s in seeds])
            baseline = [float(frozen[par, 1., s]['episode_return']) for s in seeds]
            diffs = stats([float(frozen[par, sev, s]['episode_return']) - b for s, b in zip(seeds, baseline)])
            actual, nominal = applied[par, sev]
            rows.append(dict(parameter=par, severity=sev, actual_parameter=actual, nominal_parameter=nominal,
                             physical_metric='joint_velocity_l2_rad_per_s' if par == 'bthigh_damping' else 'root_velocity_l2_m_per_s',
                             physical_seed_n=response['n'], physical_probe_windows_per_seed=16,
                             **{f'physical_{k}': v for k, v in response.items() if k != 'n'},
                             latent_calibration_seed_n=latent_ratio['n'],
                             **{f'latent_ratio_{k}': v for k, v in latent_ratio.items() if k != 'n'},
                             return_seed_n=returns['n'],
                             **{f'return_{k}': v for k, v in returns.items() if k != 'n'},
                             **{f'paired_return_{k}': v for k, v in diffs.items() if k != 'n'}))
    return rows


def applied_text(applied, par):
    nominal = applied[par, 1.][1]
    units = 'kg' if par == 'torso_mass' else 'MuJoCo damping units'
    mapping = ', '.join(f'{k:g}→{applied[par, k][0]:.3f}' for k in SEVERITIES)
    return f'{LABELS[par]}: nominal {nominal:.3f} {units}; applied k→value: {mapping}'


def panel_label(ax, title, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.22)


def errbar(ax, x, summaries, prefix, color):
    y = [r[f'{prefix}_mean'] for r in summaries]
    lo = [r[f'{prefix}_ci_low'] for r in summaries]
    hi = [r[f'{prefix}_ci_high'] for r in summaries]
    ax.errorbar(x, y, yerr=[np.maximum(0, np.array(y) - lo), np.maximum(0, np.array(hi) - y)],
                color=color, marker='o', capsize=3, lw=1.8)


def physical_latent_control_plot(out, rows, applied):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex='row')
    for row_i, par in enumerate(PARAMS):
        data = [r for r in rows if r['parameter'] == par]
        x = np.array(SEVERITIES)
        color = '#204c70' if row_i == 0 else '#a95d1a'
        axes[row_i, 0].plot(x, [r['actual_parameter'] for r in data], 'o-', color=color)
        axes[row_i, 0].axhline(applied[par, 1.][1], color='black', ls=':', lw=1)
        panel_label(axes[row_i, 0], 'Applied physical parameter',
                    'Damping (MuJoCo units)' if row_i == 0 else 'Torso mass (kg)')
        errbar(axes[row_i, 1], x, data, 'physical', color)
        panel_label(axes[row_i, 1], 'Controlled one-step response\n3 calibration seeds; 16 paired windows/seed',
                    'Joint velocity L2 (rad/s)' if row_i == 0 else 'Root velocity L2 (m/s)')
        errbar(axes[row_i, 2], x, data, 'latent_ratio', color)
        axes[row_i, 2].axhline(1, color='black', ls=':', lw=1)
        panel_label(axes[row_i, 2], 'Fixed-data one-step latent error\n3 calibration seeds', 'MSE ratio to same-seed nominal')
        errbar(axes[row_i, 3], x, data, 'return', color)
        axes[row_i, 3].axhline(next(r['return_mean'] for r in data if r['severity'] == 1.), color='black', ls=':', lw=1)
        panel_label(axes[row_i, 3], 'Frozen fixed-shift return\n20 held-out seeds', 'Episode return')
        for ax in axes[row_i]:
            ax.set_xticks(x)
            ax.set_xlabel('Predefined severity multiplier k')
        fig.text(.03, .91 if row_i == 0 else .475, applied_text(applied, par), fontsize=9, weight='bold')
    fig.suptitle('750k severity: applied physics → controlled response → latent error → Frozen control', fontsize=15)
    fig.text(.5, .015, 'Points are seed means; bars are 95% bootstrap CIs over seeds. Probe windows and time steps are not independent samples.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .92), h_pad=3)
    fig.savefig(out / 'physical_latent_control_severity.png', dpi=170)
    plt.close(fig)


def time_table(latent_bins, velocities, applied):
    rows = []
    seeds = sorted({k[2] for k in velocities})
    for par in PARAMS:
        for sev in SEVERITIES:
            for bin_id in range(40):
                errors = [latent_bins[par, sev, seed, bin_id] for seed in seeds]
                velocity_delta = [velocities[par, sev, seed][bin_id] - velocities[par, 1., seed][bin_id]
                                  for seed in seeds]
                a = stats(errors)
                v = stats(velocity_delta)
                rows.append(dict(parameter=par, severity=sev, actual_parameter=applied[par, sev][0],
                                 nominal_parameter=applied[par, sev][1], shift_relative_bin_start=bin_id * 25 - 250,
                                 shift_relative_bin_end=bin_id * 25 - 226, seed_n=len(seeds),
                                 **{f'latent_mse_{k}': value for k, value in a.items() if k != 'n'},
                                 **{f'velocity_delta_{k}': value for k, value in v.items() if k != 'n'}))
    return rows


def shift_plot(out, rows, applied):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for row_i, par in enumerate(PARAMS):
        for sev in SEVERITIES:
            data = [r for r in rows if r['parameter'] == par and r['severity'] == sev]
            x = [r['shift_relative_bin_start'] + 12 for r in data]
            label = f'k={sev:g} (applied {applied[par, sev][0]:.3f})'
            color = 'black' if sev == 1. else COLORS[sev]
            axes[row_i, 0].plot(x, [r['latent_mse_mean'] for r in data], label=label, color=color, lw=1.6)
            axes[row_i, 1].plot(x, [r['velocity_delta_mean'] for r in data], label=label, color=color, lw=1.6)
            if sev == 2.:
                for col, prefix in ((0, 'latent_mse'), (1, 'velocity_delta')):
                    axes[row_i, col].fill_between(x, [r[f'{prefix}_ci_low'] for r in data],
                                                  [r[f'{prefix}_ci_high'] for r in data], color=color, alpha=.16)
        for ax in axes[row_i]:
            ax.axvline(0, ls='--', color='#7d2c2c', lw=1)
            ax.grid(alpha=.2)
            ax.set_xlabel('Control steps relative to abrupt onset (t=250)')
        axes[row_i, 0].set_yscale('log')
        panel_label(axes[row_i, 0], 'One-step latent prediction error', 'Latent MSE (log scale)')
        panel_label(axes[row_i, 1], 'Forward velocity minus same-seed nominal', 'Velocity difference (m/s)')
        axes[row_i, 1].axhline(0, color='black', lw=.8)
        axes[row_i, 0].legend(fontsize=7, loc='upper left', ncol=2)
        fig.text(.05, .91 if row_i == 0 else .47, applied_text(applied, par), fontsize=9, weight='bold')
    fig.suptitle('S2 abrupt shift: prediction error and paired velocity on the same onset clock', fontsize=15)
    fig.text(.5, .012, '25-step bins; 20 seed means per point; shaded band is the k=2 seed-bootstrap 95% CI. Time bins are not independent samples.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .92), h_pad=3)
    fig.savefig(out / 'shift_aligned_latent_vs_velocity.png', dpi=170)
    plt.close(fig)


def scatter_data(frozen, abrupt, fixed_error, abrupt_error, regime_fixed, regime_abrupt, applied):
    all_rows = []
    summaries = []
    counts = []
    seeds = sorted({k[2] for k in frozen})
    for schedule, eps, errors, regimes in (('fixed', frozen, fixed_error, regime_fixed),
                                           ('abrupt_post_onset', abrupt, abrupt_error, regime_abrupt)):
        for par in PARAMS:
            for sev in SEVERITIES:
                group = []
                for seed in seeds:
                    shifted = float(eps[par, sev, seed]['episode_return'])
                    nominal = float(eps[par, 1., seed]['episode_return'])
                    group.append(dict(schedule=schedule, parameter=par, severity=sev, seed=seed,
                                      actual_parameter=applied[par, sev][0], nominal_parameter=applied[par, sev][1],
                                      latent_mse=errors[par, sev, seed], nominal_latent_mse=errors[par, 1., seed],
                                      latent_mse_difference=errors[par, sev, seed] - errors[par, 1., seed],
                                      paired_return_difference=shifted - nominal, outlier_iqr=False))
                y = np.array([r['paired_return_difference'] for r in group])
                q1, q3 = np.quantile(y, (.25, .75))
                lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                for r in group:
                    r['outlier_iqr'] = bool(r['paired_return_difference'] < lower or r['paired_return_difference'] > upper)
                all_rows.extend(group)
                xstat = stats([r['latent_mse_difference'] for r in group])
                ystat = stats(y)
                rho = float(spearmanr([r['latent_mse_difference'] for r in group], y).statistic) if sev != 1. else None
                summaries.append(dict(schedule=schedule, parameter=par, severity=sev, seed_n=len(group),
                                      actual_parameter=applied[par, sev][0], nominal_parameter=applied[par, sev][1],
                                      **{f'latent_mse_difference_{k}': v for k, v in xstat.items() if k != 'n'},
                                      **{f'paired_return_difference_{k}': v for k, v in ystat.items() if k != 'n'},
                                      spearman_within_severity=rho,
                                      return_outlier_iqr_count=sum(r['outlier_iqr'] for r in group)))
                counts.append(dict(schedule=schedule, parameter=par, severity=sev, episode_seed_n=len(group),
                                   saturated_one_step_windows=regimes[par, sev, 'saturated'],
                                   unsaturated_one_step_windows=regimes[par, sev, 'unsaturated'],
                                   phase='all episode steps' if schedule == 'fixed' else 'steps 250–999 only'))
    return all_rows, summaries, counts


def scatter_plot(out, rows, applied):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row_i, par in enumerate(PARAMS):
        for col_i, schedule in enumerate(('fixed', 'abrupt_post_onset')):
            ax = axes[row_i, col_i]
            panel = [r for r in rows if r['parameter'] == par and r['schedule'] == schedule]
            for sev in SEVERITIES:
                data = [r for r in panel if r['severity'] == sev]
                x = [r['latent_mse_difference'] for r in data]
                y = [r['paired_return_difference'] for r in data]
                ax.scatter(x, y, s=25, alpha=.55, color=COLORS[sev],
                           label=f'k={sev:g}, applied {applied[par, sev][0]:.3f}')
                outliers = [r for r in data if r['outlier_iqr']]
                if outliers:
                    ax.scatter([r['latent_mse_difference'] for r in outliers],
                               [r['paired_return_difference'] for r in outliers], s=66,
                               facecolors='none', edgecolors='black', linewidths=1.2)
            ax.axhline(0, color='black', lw=.8)
            ax.axvline(0, color='black', lw=.8)
            panel_label(ax, 'Fixed shift' if col_i == 0 else 'Abrupt shift (error after onset)',
                        'Paired episode return difference')
            ax.set_xlabel('One-step latent MSE minus same-seed nominal')
            ax.legend(fontsize=6.8, loc='lower left', ncol=2)
        fig.text(.06, .91 if row_i == 0 else .47, applied_text(applied, par), fontsize=9, weight='bold')
    fig.suptitle('Seed-level model error vs paired Frozen return (20 evaluation seeds/condition)', fontsize=15)
    fig.text(.5, .012, 'Each dot is one episode/seed. Black rings flag 1.5×IQR return outliers within a severity. Associations are descriptive, not causal.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .92), h_pad=3)
    fig.savefig(out / 'per_seed_error_vs_return.png', dpi=170)
    plt.close(fig)


def effort_data(run, frozen, applied):
    rows = []
    folder = run / 'frozen'
    for (par, sev, seed), episode in sorted(frozen.items()):
        with np.load(folder / episode['data_file']) as d:
            action = d['actions']
            actual_force = d['actuator_force'][1:]
            if actual_force.shape != action.shape:
                raise ValueError('Actuator force/action alignment mismatch')
            max_difference = float(np.max(np.abs(actual_force - action)))
            force_rms = float(np.sqrt(np.mean(np.square(actual_force))))
        rows.append(dict(parameter=par, severity=sev, seed=seed,
                         actual_parameter=applied[par, sev][0], nominal_parameter=applied[par, sev][1],
                         action_component_saturation=float(episode['action_saturation']),
                         actuator_force_rms=force_rms,
                         control_cost_per_step=float(episode['control_cost']) / float(episode['episode_length']),
                         action_smoothness_rms=float(episode['action_smoothness_rms']),
                         max_force_minus_action=max_difference))
    return rows


def effort_plot(out, rows, applied):
    metrics = [('action_component_saturation', 'Action-component saturation', 'Fraction |action| ≥ 0.99'),
               ('actuator_force_rms', 'Recorded actuator-force RMS', 'MuJoCo actuator_force units'),
               ('control_cost_per_step', 'Control cost', 'Reward penalty per step'),
               ('action_smoothness_rms', 'Action change', 'RMS successive-action difference')]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), sharex=True)
    summary = []
    for row_i, par in enumerate(PARAMS):
        for col_i, (metric, title, ylabel) in enumerate(metrics):
            ax = axes[row_i, col_i]
            vals = []
            for sev in SEVERITIES:
                data = [r[metric] for r in rows if r['parameter'] == par and r['severity'] == sev]
                s = stats(data)
                vals.append(s)
                summary.append(dict(parameter=par, severity=sev, metric=metric, seed_n=s['n'],
                                    actual_parameter=applied[par, sev][0], nominal_parameter=applied[par, sev][1],
                                    mean=s['mean'], median=s['median'], ci_low=s['ci_low'], ci_high=s['ci_high']))
            y = [v['mean'] for v in vals]
            ax.errorbar(SEVERITIES, y,
                        yerr=[np.array(y) - [v['ci_low'] for v in vals],
                              [v['ci_high'] for v in vals] - np.array(y)],
                        color='#204c70' if row_i == 0 else '#a95d1a', marker='o', capsize=3)
            panel_label(ax, title, ylabel)
            ax.set_xticks(SEVERITIES)
            ax.set_xlabel('Severity multiplier k')
        fig.text(.03, .91 if row_i == 0 else .475, applied_text(applied, par), fontsize=9, weight='bold')
    fig.suptitle('Frozen fixed-shift action effort; 20 seed-level means per condition', fontsize=15)
    fig.text(.5, .012, 'Bars: 95% seed-bootstrap CI. Action-component saturation is not actuator-force saturation; force-limit flags/ranges are absent from saved results.',
             ha='center', fontsize=8.8)
    fig.tight_layout(rect=(0, .04, 1, .92), h_pad=3)
    fig.savefig(out / 'action_effort_diagnostics.png', dpi=170)
    plt.close(fig)
    return summary


def report(out, run, applied, integrated, seed_summary, counts, effort, provenance):
    def match(rows, par, sev, schedule=None):
        return next(r for r in rows if r['parameter'] == par and r['severity'] == sev and
                    (schedule is None or r['schedule'] == schedule))
    lines = [
        '# NS-Gym 750k severity analysis refinement', '',
        'This is an offline reanalysis of completed results. No adaptation, new severity selection, environment rollout, or checkpoint update was performed.', '',
        '## Sources and statistical unit', '',
        f'- Input: `{run}`. Primary checkpoint SHA256: `{provenance["checkpoint_sha256"]}`; protocol SHA256: `{provenance["protocol_sha256"]}`.',
        '- Calibration: 3 seeds × 16 paired physical windows per condition. Physical and fixed-data latent CIs resample the **3 seed means**, not 48 windows or time steps.',
        '- Frozen/S2: 20 held-out episode seeds. Return and action CIs resample **20 seed-level values**; onset curves average 25 control steps within seed before resampling seeds.',
        '- S0 nominal exactly duplicates nominal Frozen episodes; S1 fixed exactly duplicates Frozen episodes. They are validation checks and are not pooled as extra independent episodes.',
        '- CI = deterministic 10,000-draw percentile bootstrap of a seed-level mean. Exploratory intervals are not corrected for multiple severity comparisons.', '',
        '## Applied parameters and all predefined conditions', '',
        '| Family | Nominal | k=0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for par in PARAMS:
        values = [applied[par, k][0] for k in SEVERITIES]
        lines.append(f'| {LABELS[par]} | {applied[par, 1.][1]:.4f} | ' +
                     ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', 'Damping values are MuJoCo parameter units; torso mass is kg. The same multiplier does not imply equal physical difficulty across families.', '',
              '## Main findings', '',
              '| Family | k | Fixed return mean [95% CI] | Return median | Paired return change mean [95% CI] | Controlled one-step response mean [95% CI] | Fixed-data one-step latent MSE ratio mean [95% CI] |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for par in PARAMS:
        for sev in SEVERITIES:
            r = match(integrated, par, sev)
            lines.append(f'| {LABELS[par]} | {sev:g} | {r["return_mean"]:.1f} [{r["return_ci_low"]:.1f}, {r["return_ci_high"]:.1f}] | {r["return_median"]:.1f} | '
                         f'{r["paired_return_mean"]:.1f} [{r["paired_return_ci_low"]:.1f}, {r["paired_return_ci_high"]:.1f}] | '
                         f'{r["physical_mean"]:.4g} [{r["physical_ci_low"]:.4g}, {r["physical_ci_high"]:.4g}] | '
                         f'{r["latent_ratio_mean"]:.2f} [{r["latent_ratio_ci_low"]:.2f}, {r["latent_ratio_ci_high"]:.2f}] |')
    lines += ['', 'The physical response column is **joint velocity L2 (rad/s)** for damping and **root velocity L2 (m/s)** for mass; they are intentionally separate units/panels. `physical_latent_control_severity.png` compares all four quantities on the same predefined grid.', '',
              '## Abrupt onset and seed-level error/return relation', '',
              '- `shift_aligned_latent_vs_velocity.png` uses the same step axis relative to action 250. Its latent panel shows absolute one-step MSE; its velocity panel shows the same-seed deviation from nominal. Each point is a 25-step within-seed mean followed by a mean over 20 seeds. The shaded k=2 band resamples seeds.',
              '- `per_seed_error_vs_return.png` uses **one point per episode/seed**. Its x-value is mean one-step latent MSE minus the same-seed nominal mean (whole episode for fixed, steps 250–999 for abrupt); y is episode return minus the same-seed nominal return. The table below reports mean, median, seed-bootstrap CI and within-severity descriptive Spearman rho. It does not use time steps as independent replicates.', '',
              '| Schedule | Family | k | Δlatent MSE mean [95% CI] | Δlatent MSE median | Paired return Δ mean [95% CI] | Paired return Δ median | IQR outliers / 20 | Spearman ρ |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for schedule in ('fixed', 'abrupt_post_onset'):
        for par in PARAMS:
            for sev in SEVERITIES:
                r = match(seed_summary, par, sev, schedule)
                rho = 'N/A (nominal)' if r['spearman_within_severity'] is None else f'{r["spearman_within_severity"]:.2f}'
                lines.append(f'| {schedule} | {LABELS[par]} | {sev:g} | '
                             f'{r["latent_mse_difference_mean"]:.6f} [{r["latent_mse_difference_ci_low"]:.6f}, {r["latent_mse_difference_ci_high"]:.6f}] | '
                             f'{r["latent_mse_difference_median"]:.6f} | '
                             f'{r["paired_return_difference_mean"]:.1f} [{r["paired_return_difference_ci_low"]:.1f}, {r["paired_return_difference_ci_high"]:.1f}] | '
                             f'{r["paired_return_difference_median"]:.1f} | '
                             f'{r["return_outlier_iqr_count"]} | {rho} |')
    lines += ['', 'Outlier rule: outside Q1−1.5×IQR or Q3+1.5×IQR of the 20 paired returns within one schedule/family/severity. All flagged points remain in the plots and statistics. Correlations are descriptive and can reflect severity, visitation, reward components or other confounding; they do not establish model-error causation.', '',
              '### Action-regime sample counts (one-step prediction windows)', '',
              '| Schedule | Family | k | Saturated windows | Unsaturated windows | Episode seeds |',
              '|---|---|---:|---:|---:|---:|']
    for r in counts:
        lines.append(f'| {r["schedule"]} | {LABELS[r["parameter"]]} | {r["severity"]:g} | '
                     f'{r["saturated_one_step_windows"]} | {r["unsaturated_one_step_windows"]} | {r["episode_seed_n"]} |')
    lines += ['', 'The window count is descriptive coverage, **not the inferential sample size**. `saturated` means at least one action component had |a|≥0.99 in that one-step window. The CI and association analyses use episode seeds.', '',
              '## Action effort and actuator-force limit', '',
              '- `action_effort_diagnostics.png` separates action-component saturation, recorded `actuator_force` RMS, reward control cost per step, and successive-action RMS. The 240 Frozen raw `.npz` trajectories contain `actions` and `actuator_force`; their largest componentwise difference was '
              f'{max(r["max_force_minus_action"] for r in effort):.3g} in this dataset. These values are force effort, **not force saturation**.',
              '- The saved `inventory/environment_contract.json` records `actuator_ctrlrange` but not `actuator_forcelimited` or `actuator_forcerange`. Therefore actual force-saturation fraction is **NOT_MEASURED** from this run. The action threshold |a|≥0.99 is not a measured actuator-force limit.',
              '- To assert server-side actuator-force saturation status, add `actuator_forcelimited` and `actuator_forcerange` to a new read-only environment contract; if any actuator is force-limited, pair those bounds with `frozen/condition-XX-seed-YYYYYYYY.npz` → `actuator_force[1:]`. Do not infer a bound from action saturation or gear.', '',
              '## Planning and limits', '',
              '- `planning/planning_comparisons.csv` contains six examples. It has no candidate simulator returns (`candidate_actual_returns_measured=False`), so ranking correctness and regret remain **NOT_MEASURED**.',
              '- All 12 valid grid conditions remain in the integrated, effort, time and seed-level tables. S0/S1 duplicates are never added to sample counts. No full controllability Gramian or oracle MPC is attempted.',
              '- A causal claim that latent error explains return loss would require a controlled intervention or additional diagnostic design. The association figure alone does not justify adaptation necessity.', '',
              '## Derived outputs and exact source fields', '',
              '- `physical_latent_control_severity.png` and `integrated_seed_summary.csv`: `physical/physical_response_pairs.csv` (`seed`, `actual_parameter`, `nominal_parameter`, `joint_velocity_l2`, `root_velocity_l2`), `physical-latent/latent_errors.csv` (`seed`, `horizon`, `latent_mse`), `frozen/episodes.csv` (`seed`, `episode_return`).',
              '- `shift_aligned_latent_vs_velocity.png` and `shift_aligned_seed_bins.csv`: `nonstationary-abrupt-latent/latent_errors.csv` (`seed`, `episode_step`, `horizon`, `latent_mse`) and `nonstationary-abrupt/*-steps.csv` (`seed`, `episode_step`, `forward_velocity`, `actual_parameter`, `nominal_parameter`).',
              '- `per_seed_error_vs_return.png`, `per_seed_error_vs_return.csv`, `per_seed_error_return_summary.csv`, `action_regime_counts.csv`: the fixed and abrupt raw latent error files (`seed`, `horizon`, `action_regime`, `latent_mse`) plus Frozen and S2 `episodes.csv` (`seed`, `episode_return`).',
              '- `action_effort_diagnostics.png` and `action_effort_seed_summary.csv`: Frozen `episodes.csv` (`action_saturation`, `control_cost`, `episode_length`, `action_smoothness_rms`) and the corresponding raw `.npz` (`actions`, `actuator_force`).',
              '- A compact summary-only copy is insufficient for the per-seed error/return scatter: it needs `heldout-latent/latent_errors.csv` and `nonstationary-abrupt-latent/latent_errors.csv` with the fields above. These raw files were available locally and were used here; no missing values were imputed.', '',
              'The separate `DYNAMICS_ONLY_ADAPTATION_PROTOCOL.md` is a preparation document only; no adaptation experiment has been run.', '']
    (out / 'SEVERITY_ANALYSIS_REFINEMENT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/nsgym/physical-severity-750k-v1'))
    parser.add_argument('--output', type=Path, default=Path('results/nsgym/physical-severity-750k-refinement-01'))
    args = parser.parse_args()
    run = args.run.resolve()
    out = args.output.resolve()
    protocol_hash, checkpoint_hash = verify(run)
    frozen = episodes(run, 'frozen')
    abrupt = episodes(run, 'nonstationary-abrupt')
    check_duplicates(run, frozen, abrupt)
    physical, applied = physical_seed_data(run)
    fixed_latent = physical_fixed_latent(run)
    integrated = condition_rows(physical, applied, fixed_latent, frozen)
    fixed_error, fixed_n, fixed_regime, _ = latent_seed_data(run / 'heldout-latent/latent_errors.csv')
    abrupt_error, abrupt_n, abrupt_regime, abrupt_bins = latent_seed_data(
        run / 'nonstationary-abrupt-latent/latent_errors.csv', onset=250, bins=True)
    if any(n != 1000 for n in fixed_n.values()) or any(n != 750 for n in abrupt_n.values()):
        raise ValueError('Unexpected episode/window length for per-seed latent comparison')
    velocities = velocity_bins(run)
    timeline = time_table(abrupt_bins, velocities, applied)
    seed_rows, seed_summary, regime_counts = scatter_data(
        frozen, abrupt, fixed_error, abrupt_error, fixed_regime, abrupt_regime, applied)
    effort = effort_data(run, frozen, applied)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'integrated_seed_summary.csv', integrated)
    write_csv(out / 'shift_aligned_seed_bins.csv', timeline)
    write_csv(out / 'per_seed_error_vs_return.csv', seed_rows)
    write_csv(out / 'per_seed_error_return_summary.csv', seed_summary)
    write_csv(out / 'action_regime_counts.csv', regime_counts)
    write_csv(out / 'action_effort_seed_summary.csv', effort)
    physical_latent_control_plot(out, integrated, applied)
    shift_plot(out, timeline, applied)
    scatter_plot(out, seed_rows, applied)
    effort_summary = effort_plot(out, effort, applied)
    write_csv(out / 'action_effort_condition_summary.csv', effort_summary)
    provenance = dict(source_run=str(run), protocol_sha256=protocol_hash, checkpoint_sha256=checkpoint_hash,
                      analysis_script_sha256=sha256(Path(__file__)), bootstrap_draws=10000,
                      inferential_unit='seed', frozen_seed_count=20, physical_calibration_seed_count=3,
                      S0_duplicate_of_nominal_Frozen=True, S1_duplicate_of_Frozen=True,
                      candidate_returns_measured=False, force_saturation_measured=False)
    (out / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    report(out, run, applied, integrated, seed_summary, regime_counts, effort, provenance)
    print(f'PASS: {out}; conditions={len(integrated)}, seed points={len(seed_rows)}, onset bins={len(timeline)}')


