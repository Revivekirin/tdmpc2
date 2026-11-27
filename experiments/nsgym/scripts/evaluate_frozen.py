"""Frozen paired robustness evaluation; software smoke is not scientific evidence."""
import argparse
import csv
import json
from pathlib import Path

from research_utils import ROOT, metadata, output_dir, write_json, csv_rows, digest, bootstrap
from training_support import load_checkpoint, evaluate, tree_hash, learned_state
import numpy as np
from omegaconf import OmegaConf


def aggregate(rows, config):
    nominal={r['seed']:r['episode_return'] for r in rows if r['mass_scale']==1.0}
    if not nominal:raise ValueError('Nominal condition required')
    result=[]
    for scale in config['mass_scales']:
        group=[r for r in rows if r['mass_scale']==scale]
        if len(group)!=len(nominal) or {r['seed'] for r in group}!=set(nominal):raise ValueError('Unpaired or unequal severity budgets')
        returns=np.array([r['episode_return'] for r in group]);diff=returns-np.array([nominal[r['seed']] for r in group])
        ci=bootstrap(returns,config['bootstrap_seed'],config['bootstrap_repeats'])
        dc=bootstrap(diff,config['bootstrap_seed'],config['bootstrap_repeats'])
        baseline=np.array(list(nominal.values()));bc=bootstrap(baseline,config['bootstrap_seed'],config['bootstrap_repeats'])
        row=dict(mass_scale=scale,episodes=len(group),mean_return=float(returns.mean()),std_return=float(returns.std(ddof=1)),
                 return_ci_low=ci[0],return_ci_high=ci[1],paired_return_difference=float(diff.mean()),
                 paired_difference_ci_low=dc[0],paired_difference_ci_high=dc[1],
                 relative_return_change=float(diff.mean()/baseline.mean()) if baseline.mean()>1e-6 and bc[0]>0 else None)
        for key in ['torso_mass','forward_velocity','velocity_std','reward_run','reward_control','episode_length',
                    'control_effort','action_abs_mean','action_std','action_change_rms','terminated','truncated']:
            row[key+'_mean']=float(np.mean([r[key] for r in group]))
        result.append(row)
    return result


def summarize(directory):
    out=Path(directory)
    for name in ['severity_summary.csv','frozen_profile.png','REPORT.md']:
        if (out/name).exists():raise FileExistsError(out/name)
    meta=json.loads((out/'metadata.json').read_text());config=meta['config']
    rows=[json.loads(s) for s in (out/'episodes.jsonl').read_text().splitlines()]
    table=aggregate(rows,config);csv_rows(out/'severity_summary.csv',table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(8,9),sharex=True)
    x=[r['mass_scale'] for r in table];y=np.array([r['mean_return'] for r in table])
    axes[0].errorbar(x,y,yerr=[y-np.array([r['return_ci_low'] for r in table]),np.array([r['return_ci_high'] for r in table])-y],fmt='o-')
    axes[0].set_ylabel('Return (95% bootstrap CI)')
    axes[1].plot(x,[r['forward_velocity_mean'] for r in table],'o-');axes[1].set_ylabel('Mean forward velocity (m/s)')
    axes[2].plot(x,[r['control_effort_mean'] for r in table],'o-');axes[2].set_ylabel('Sum squared actions');axes[2].set_xlabel('Torso mass scale')
    fig.tight_layout();fig.savefig(out/'frozen_profile.png');plt.close(fig)
    status='INTEGRATION ONLY; scientific interpretation blocked' if config['integration_only'] else 'Preliminary frozen robustness evaluation'
    (out/'REPORT.md').write_text(f'# {status}\n\nSame checkpoint and paired episode seeds across all conditions. Fixed torso mass applies before transition zero; policy actions can differ. Confidence intervals use percentile bootstrap resampling of episode seeds (10,000 draws by default); difference intervals resample paired seed-level differences. Negative/uncertain nominal returns suppress percentage ratios. A ten-episode sweep is preliminary. No adaptation is performed and no model prediction causality is asserted. See integrity.json for immutable model/optimizer/file checks.\n')


def run(checkpoint_path, config, destination, quality_path=None):
    seeds=config['evaluation_seeds']
    if len(seeds)<2 or len(seeds)!=len(set(seeds)):raise ValueError('At least two distinct seeds required')
    if not config['integration_only'] and len(seeds)<10:raise ValueError('Scientific preliminary sweep requires at least 10 paired episodes')
    if 1.0 not in config['mass_scales'] or len(set(config['mass_scales']))!=len(config['mass_scales']):raise ValueError('Unique severity grid including nominal required')
    checkpoint_path=Path(checkpoint_path)
    model_hash=digest(checkpoint_path/'model.pt')
    if not config['integration_only']:
        if quality_path is None:raise ValueError('--quality-report required for scientific evaluation')
        quality=json.loads(Path(quality_path).read_text())
        if not quality['scientific_sweep_eligible'] or quality['checkpoint_sha256']!=model_hash:
            raise ValueError('Nominal controller quality is unestablished or refers to another checkpoint')
    agent,meta=load_checkpoint(checkpoint_path)
    if meta['config']['smoke'] and not config['integration_only']:raise ValueError('Smoke checkpoint cannot support scientific conclusions')
    if any(meta['config']['seed']<=s<=meta['config']['seed']+meta['environment_transitions'] for s in seeds):
        raise ValueError('Frozen seeds overlap possible training seeds')
    out=output_dir(destination);prov=metadata(config,checkpoint_path/'model.pt')
    prov.update(checkpoint_metadata=meta,quality_report=json.loads(Path(quality_path).read_text()) if quality_path else None)
    write_json(out/'metadata.json',prov)
    before=tree_hash(learned_state(agent)); rows=[]
    for scale in config['mass_scales']:
        condition,elapsed=evaluate(agent,meta['config'],seeds,scale=scale,fixed=True)
        for row in condition:row.update(checkpoint_sha256=model_hash,evaluation_wall_seconds=elapsed)
        rows.extend(condition)
    assert tree_hash(learned_state(agent))==before
    assert digest(checkpoint_path/'model.pt')==model_hash
    assert digest(checkpoint_path/'optimizers.pt')==meta['optimizer_sha256']
    with (out/'episodes.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    csv_rows(out/'episodes.csv',rows)
    write_json(out/'integrity.json',dict(status='PASS',learned_state_before=before,learned_state_after=tree_hash(learned_state(agent)),
        checkpoint_sha256=model_hash,optimizer_file_sha256=meta['optimizer_sha256'],optimizer_updates_during_evaluation=0))
    summarize(out);return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/nsgym/configs/frozen_sweep.yaml')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--quality-report',type=Path)
    parser.add_argument('--integration-only',action='store_true')
    parser.add_argument('--nominal-only',action='store_true')
    parser.add_argument('--expanded',action='store_true')
    parser.add_argument('--episodes',type=int)
    parser.add_argument('--summarize',type=Path)
    args=parser.parse_args()
    if args.summarize:summarize(args.summarize);return
    if args.checkpoint is None:parser.error('--checkpoint is required')
    if args.expanded and args.nominal_only:parser.error('--expanded and --nominal-only are mutually exclusive')
    config=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    config['integration_only']=args.integration_only
    if args.expanded:config['mass_scales']=config['expanded_mass_scales']
    if args.nominal_only:config['mass_scales']=[1.0]
    if args.episodes is not None:config['evaluation_seeds']=list(range(20000001,20000001+args.episodes))
    print(run(args.checkpoint,config,args.output or config['output'],args.quality_report))

if __name__=='__main__':main()
