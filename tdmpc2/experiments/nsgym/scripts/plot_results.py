"""Regenerate plots/tables into a NEW directory; leave source results untouched."""
import argparse
from pathlib import Path
import shutil
from research_utils import output_dir,write_json,metadata,digest
import numpy as np
import reacher_pipeline as r
from analyze_reacher_pipeline import aggregate as aggregate_reacher, plotting


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['physical','training','frozen','reacher-comparison'],required=True)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--inputs',type=Path,nargs='+',help='Reacher: probe, Frozen, Dynamics-only prediction/control, Joint evaluation')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.kind=='reacher-comparison':
        if args.input or not args.inputs or len(args.inputs)!=5:
            parser.error('Reacher comparison requires exactly five --inputs and no --input')
        inputs=args.inputs
        for path in inputs:
            if r.read(path/'completion.json')['status']!='PASS':
                raise ValueError(f'Incomplete input: {path}')
        one=lambda predicate:next((x for x in inputs if predicate(x)),None)
        probe=one(lambda p:(p/'reacher_physical_response.csv').is_file())
        frozen=one(lambda p:(p/'reacher_frozen_severity.csv').is_file())
        prediction=one(lambda p:any(p.glob('*/predict.csv')))
        control=one(lambda p:any(p.glob('*/control.csv')) and not (p/'reacher_joint_control.csv').is_file())
        joint=one(lambda p:(p/'reacher_joint_control.csv').is_file())
        if None in (probe,frozen,prediction,control,joint) or len({probe,frozen,prediction,control,joint})!=5:
            raise ValueError('Expected one unique probe, Frozen, Dynamics-only prediction, Dynamics-only control and Joint root')
        checkpoint_sha=r.read(joint/'manifest.json')['checkpoint_sha256']
        for path in inputs:
            source=r.read(path/'manifest.json')
            if source.get('checkpoint_sha256') not in (None,checkpoint_sha):
                raise ValueError(f'Checkpoint mismatch: {path}')
        c=r.config(r.DEFAULT)
        out=output_dir(args.output)
        aggregate_reacher([probe,frozen,prediction,control],out,c)
        controls=[row for file in control.glob('*/control.csv') for row in r.csvread(file)]
        predictions=[row for file in prediction.glob('*/predict.csv') for row in r.csvread(file)]
        compare_methods(frozen,controls,predictions,joint,out,c)
        write_json(out/'analysis_manifest.json',dict(kind=args.kind,checkpoint_sha256=checkpoint_sha,
            config_fingerprint=r.fingerprint(c),analysis_script_sha256=digest(__file__),
            reacher_analysis_sha256=digest(Path(__file__).with_name('analyze_reacher_pipeline.py')),
            inputs={str(p.resolve()):digest(p/'completion.json') for p in inputs}))
        write_json(out/'completion.json',dict(status='PASS',conditions=len(r.conditions(c)),
            dynamics_control_rows=len(controls),dynamics_prediction_rows=len(predictions),
            joint_control_rows=len(r.csvread(joint/'reacher_joint_control.csv')),
            joint_prediction_rows=len(r.csvread(joint/'reacher_joint_prediction.csv'))))
        print(out)
        return
    if args.input is None or args.inputs:parser.error('This kind requires --input and does not accept --inputs')
    files={'physical':['raw.npz','checks.json','metadata.json'],
           'training':['training.jsonl','evaluation.csv'],
           'frozen':['metadata.json','episodes.jsonl','integrity.json']}[args.kind]
    for name in files:
        if not (args.input/name).is_file():raise FileNotFoundError(args.input/name)
    out=output_dir(args.output)
    for name in files:shutil.copy2(args.input/name,out/name)
    write_json(out/'plot_provenance.json',metadata(dict(kind=args.kind,input=str(args.input),
               source_hashes={n:digest(args.input/n) for n in files})))
    if args.kind=='physical':
        from validate_physics import summarize
    elif args.kind=='training':
        from train_nominal import plot_training as summarize
    else:
        from evaluate_frozen import summarize
    summarize(out)
    print(out)


def compare_methods(frozen_root,control,prediction,joint_root,out,c):
    """Compare complete final-budget evaluations by matched adaptation/evaluation seeds."""
    if r.read(joint_root/'completion.json')['status']!='PASS':
        raise ValueError('Joint held-out evaluation is incomplete')
    frozen=r.csvread(frozen_root/'reacher_frozen_severity.csv')
    joint=r.csvread(joint_root/'reacher_joint_control.csv')
    joint_prediction=r.csvread(joint_root/'reacher_joint_prediction.csv')
    conditions=r.conditions(c);seeds=c['adaptation_seeds'];heldout=c['evaluation_seeds']
    expected=len(conditions)*len(seeds)*len(heldout)
    marker=r.read(joint_root/'completion.json')
    if (marker.get('snapshots')!=len(conditions)*len(seeds) or
            marker.get('control_rows')!=2*expected or
            marker.get('prediction_rows')!=expected):
        raise ValueError('Joint completion marker does not cover the full matrix')
    def indexed(rows,keys):
        result={}
        for row in rows:
            key=tuple(float(row[k]) if k=='severity' else int(row[k]) if k in ('adaptation_seed','seed','budget') else row[k] for k in keys)
            if key in result:raise ValueError(f'Duplicate evaluation identity: {key}')
            result[key]=row
        return result
    condkey=lambda x:(x['family'],float(x['severity']))
    frozen_index=indexed(frozen,('family','severity','seed'))
    if len(frozen_index)!=len(conditions)*len(heldout):
        raise ValueError('Frozen reference is not a complete 24-condition held-out grid')
    dyn=[x for x in control if x['method']=='decision' and int(x['budget'])==10000]
    dyn_index=indexed(dyn,('family','severity','adaptation_seed','seed','kind'))
    joint_index=indexed(joint,('family','severity','adaptation_seed','seed','kind'))
    if len(dyn_index)!=2*expected or len(joint_index)!=2*expected:
        raise ValueError('Final Dynamics-only or Joint control matrix is incomplete')
    pred_dyn=[x for x in prediction if x['method']=='decision' and int(x['budget'])==10000]
    expected_prediction={(x['family'],float(x['severity']),a,s)
                         for x in conditions for a in seeds for s in heldout}
    if set(indexed(pred_dyn,('family','severity','adaptation_seed','seed')))!=expected_prediction:
        raise ValueError('Final Dynamics-only prediction matrix is incomplete')
    joint_prediction_index=indexed(joint_prediction,('family','severity','adaptation_seed','seed'))
    if set(joint_prediction_index)!=expected_prediction:
        raise ValueError('Joint prediction matrix is incomplete')
    nominal_frozen={(int(x['seed'])):x for x in frozen if x['family']=='D0' and float(x['severity'])==1.}
    if len(nominal_frozen)!=len(heldout):raise ValueError('Canonical nominal Frozen reference is incomplete')
    paired=[]
    for cond in conditions:
        family,severity=condkey(cond)
        for adaptation_seed in seeds:
            for seed in heldout:
                for kind in ('shifted','nominal'):
                    key=(family,severity,adaptation_seed,seed,kind)
                    d=dyn_index[key];j=joint_index[key]
                    f=frozen_index[(family,severity,seed)] if kind=='shifted' else nominal_frozen[seed]
                    baseline=float(f['episode_return']);dynamic=float(d['episode_return']);joint_return=float(j['episode_return'])
                    if (abs(float(j['frozen_episode_return'])-baseline)>1e-6 or
                            abs(float(j['paired_return_difference'])-(joint_return-baseline))>1e-6):
                        raise ValueError(f'Joint Frozen pairing differs for {key}')
                    paired.append(dict(**cond,adaptation_seed=adaptation_seed,seed=seed,kind=kind,
                        frozen_return=baseline,dynamics_only_return=dynamic,joint_rl_return=joint_return,
                        dynamics_minus_frozen=dynamic-baseline,joint_minus_frozen=joint_return-baseline,
                        joint_minus_dynamics=joint_return-dynamic,
                        frozen_distance_final=float(f['distance_final']),
                        dynamics_distance_final=float(d['distance_final']),joint_distance_final=float(j['distance_final']),
                        frozen_action_abs=float(f['action_abs']),dynamics_action_abs=float(d['action_abs']),
                        joint_action_abs=float(j['action_abs'])))
    r.table(out/'reacher_method_paired_seeds.csv',paired)
    metrics=('frozen_return','dynamics_only_return','joint_rl_return','dynamics_minus_frozen',
             'joint_minus_frozen','joint_minus_dynamics','frozen_distance_final',
             'dynamics_distance_final','joint_distance_final','frozen_action_abs',
             'dynamics_action_abs','joint_action_abs')
    summaries=[]
    for cond in conditions:
        for kind in ('shifted','nominal'):
            subset=[x for x in paired if x['family']==cond['family'] and x['severity']==cond['severity'] and x['kind']==kind]
            for metric in metrics:
                if metric.startswith('frozen_'):
                    values=[float(x[metric]) for x in subset if x['adaptation_seed']==seeds[0]]
                    unit='evaluation_seed'
                else:
                    values=[float(np.mean([float(x[metric]) for x in subset if x['adaptation_seed']==seed])) for seed in seeds]
                    unit='adaptation_seed'
                summaries.append(dict(**cond,kind=kind,metric=metric,unit=unit,**r.stats(values)))
    r.table(out/'reacher_method_comparison_summary.csv',summaries)
    joint_prediction_summary=[]
    for cond in conditions:
        values=[float(np.mean([float(joint_prediction_index[(cond['family'],float(cond['severity']),a,s)]['latent_mse'])
                               for s in heldout])) for a in seeds]
        joint_prediction_summary.append(dict(**cond,metric='joint_latent_mse',unit='adaptation_seed',
                                             **r.stats(values)))
    r.table(out/'reacher_joint_prediction_summary.csv',joint_prediction_summary)
    plt=plotting()
    figures=[('shifted','return','Episode return'),('nominal','return','Episode return'),
             ('shifted','distance_final','Final tip-target distance [m]'),
             ('shifted','action_abs','Mean |normalized action|')]
    for kind,measure,ylabel in figures:
        fig,axes=plt.subplots(4,6,figsize=(24,14),squeeze=False)
        for row,(family,param) in enumerate(c['families'].items()):
            for col,severity in enumerate(c['severities']):
                ax=axes[row,col];sub=[x for x in summaries if x['family']==family and x['severity']==severity and x['kind']==kind]
                by={x['metric']:x for x in sub};names=['Frozen','Dynamics-only','Joint RL']
                keys=['frozen_return','dynamics_only_return','joint_rl_return'] if measure=='return' else [f'frozen_{measure}',f'dynamics_{measure}',f'joint_{measure}']
                means=[by[k]['mean'] for k in keys];low=[by[k]['ci_low'] for k in keys];high=[by[k]['ci_high'] for k in keys]
                ax.bar(range(3),means,color=['#999999','#3b8bc2','#de8b36'])
                ax.errorbar(range(3),means,
                    yerr=[[max(0.,v-l) for v,l in zip(means,low)],
                          [max(0.,h-v) for h,v in zip(means,high)]],
                    fmt='none',color='black',capsize=3)
                nominal=next(float(x['nominal_parameter']) for x in frozen if x['family']==family)
                unit='kg' if param.endswith('mass') else 'N m s/rad'
                actual=nominal*severity if kind=='shifted' else nominal
                ax.set(title=f'{param} × {severity:g}\nnominal {nominal:.5g}; evaluation {actual:.5g} {unit}',
                       xticks=range(3),xticklabels=names,ylabel=ylabel)
                ax.tick_params(axis='x',labelrotation=30)
        fig.suptitle(f'Reacher {kind} control: {ylabel}; final 10k snapshot, paired held-out seeds')
        fig.tight_layout();fig.savefig(out/f'reacher_{kind}_method_{measure}.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(4,6,figsize=(24,14),squeeze=False)
    for row,(family,param) in enumerate(c['families'].items()):
        for col,severity in enumerate(c['severities']):
            ax=axes[row,col];sub=[x for x in summaries if x['family']==family and x['severity']==severity and x['kind']=='shifted']
            stat=next(x for x in sub if x['metric']=='joint_minus_dynamics')
            ax.errorbar([0],[stat['mean']],
                yerr=[[max(0.,stat['mean']-stat['ci_low'])],
                      [max(0.,stat['ci_high']-stat['mean'])]],fmt='o',capsize=4)
            ax.axhline(0,color='gray',lw=.8);ax.set(title=f'{param} × {severity:g}',xlim=(-1,1),xticks=[],ylabel='Joint RL − Dynamics-only return')

    fig.suptitle('Paired final-snapshot difference; adaptation streams n=3, 95% bootstrap CI')
    fig.tight_layout();fig.savefig(out/'reacher_joint_minus_dynamics.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(4,6,figsize=(24,14),squeeze=False)
    for row,(family,param) in enumerate(c['families'].items()):
        for col,severity in enumerate(c['severities']):
            ax=axes[row,col]
            stat=next(x for x in joint_prediction_summary if x['family']==family and x['severity']==severity)
            ax.errorbar([0],[stat['mean']],
                yerr=[[max(0.,stat['mean']-stat['ci_low'])],
                      [max(0.,stat['ci_high']-stat['mean'])]],fmt='o',capsize=4)
            ax.set(title=f'{param} × {severity:g}',xlim=(-1,1),xticks=[],ylabel='Joint latent MSE')
    fig.suptitle('Joint RL fixed-bank prediction, final snapshot; own latent space')
    fig.tight_layout();fig.savefig(out/'reacher_joint_prediction_descriptive.png',dpi=140);plt.close(fig)
    table_lines=['| Family | Parameter | Multiplier | Frozen return | Dynamics-only return | Joint RL return | Joint − Dynamics-only (95% CI) |',
                 '|---|---|---:|---:|---:|---:|---:|']
    for cond in conditions:
        by={x['metric']:x for x in summaries if x['family']==cond['family'] and
            x['severity']==cond['severity'] and x['kind']=='shifted'}
        delta=by['joint_minus_dynamics']
        table_lines.append(f"| {cond['family']} | {cond['parameter']} | {cond['severity']:g} | "
            f"{by['frozen_return']['mean']:.3f} | {by['dynamics_only_return']['mean']:.3f} | "
            f"{by['joint_rl_return']['mean']:.3f} | {delta['mean']:+.3f} "
            f"[{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}] |")
    (out/'REACHER_METHOD_COMPARISON.md').write_text(
        '# Reacher method comparison\n\nComplete 24-condition final 10k grids are paired by condition, '
        'adaptation seed, held-out seed and evaluation physics. Frozen is a single policy reference '
        'per held-out seed; its copies across adaptation seeds are not independent evidence. '
        'Adapted-method means first average 20 held-out seeds within each of three adaptation streams; '
        'sample SD and bootstrap CI use those three streams. Frozen CI uses 20 evaluation seeds and '
        'therefore has a different sampling interpretation. Four k=1 labels represent one nominal '
        'physical setting and must not be pooled as four independent tasks. All outliers are retained. '
        'Joint RL also changes encoder, reward and policy, uses a different collection policy and many '
        'more updates, so its difference from Dynamics-only does not isolate a Q effect. Joint latent '
        'MSE is available in the original Joint evaluation CSV but is not plotted against Dynamics-only '
        'fixed-encoder MSE because their latent representations differ. These figures describe held-out '
        'performance; no new training or environment evaluation was run.\n\n'
        '## Shifted held-out return by condition\n\n'+'\n'.join(table_lines)+'\n')

if __name__=='__main__':main()
