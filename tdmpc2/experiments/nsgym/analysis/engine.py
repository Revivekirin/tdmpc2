"""Authoritative raw-to-report execution. All inputs come from discovered metadata."""
from __future__ import annotations
import json
import platform
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .discovery import Catalog,first
from .sources import collect
from .schema import write_csv,sha256
from .metrics import summarize_records,paired_improvements,prediction_improvements,prediction_control_pairs,method_contrasts,normalized_comparisons

PROJECT=Path(__file__).resolve().parents[3]
DEFAULT_RESULTS=PROJECT.parent/'results/nsgym' if (PROJECT.parent/'results/nsgym').is_dir() else PROJECT/'results/nsgym'


def table(out,name,data):
    data=list(data);fields=list(dict.fromkeys(k for r in data for k in r)) or ['status']
    write_csv(out/name,data,fields)


def figures(out,summaries,comparisons,coupling):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outputs=[]
    nominal_values={(r['task'],r['cohort_id'],r['shift_parameter']):r['nominal_parameter'] for r in summaries if r.get('nominal_parameter') not in ('',None)}
    def draw(name,records,value='mean',ylabel='Mean ± 95% bootstrap CI'):
        if not records:return
        # Every subplot has a single task/checkpoint/schedule/physical parameter/metric/budget.
        panels=defaultdict(list)
        for r in records:panels[tuple(str(r.get(k,'')) for k in ('task','cohort_id','schedule','shift_parameter','metric','adaptation_budget','kind'))].append(r)
        pages=[list(panels.items())[i:i+8] for i in range(0,len(panels),8)]
        for page,items in enumerate(pages):
            fig,axes=plt.subplots(len(items),1,figsize=(10,3.2*len(items)),squeeze=False)
            for ax,(key,rr) in zip(axes[:,0],items):
                for method in sorted({r['method'] for r in rr}):
                    vals=sorted([r for r in rr if r['method']==method],key=lambda r:float(r['severity_multiplier']))
                    x=[float(r['severity_multiplier']) for r in vals];y=[r[value] for r in vals]
                    # Percentile intervals need not contain the mean at n=3. Draw endpoints without clipping them.
                    line,=ax.plot(x,y,'o-',label=method)
                    for xx,r in zip(x,vals):
                        if r.get('ci_low') is not None:ax.vlines(xx,r['ci_low'],r['ci_high'],alpha=.5,color=line.get_color())
                ax.axhline(0,color='gray',lw=.7);ax.set_xlabel('Physical multiplier (k=1: nominal)');ax.set_ylabel({'root_velocity_l2':'Root velocity difference [m/s]','joint_velocity_l2':'Joint velocity difference [rad/s]','delta_qvel_rad_s':'Joint velocity difference [rad/s]','delta_tip_m':'Tip displacement [m]'}.get(key[4],ylabel))
                nominal=next((r.get('nominal_parameter') for r in rr if r.get('nominal_parameter') not in ('',None)),nominal_values.get((key[0],key[1],key[3]),'not recorded'))
                budget='saved diagnostic dataset' if key[6].startswith('q_') else f'budget={key[5]}'
                evidence='; DERIVED_ONLY' if any(r.get('evidence')=='DERIVED_ONLY' for r in rr) else ''
                ax.set_title(f'{key[0]} | {key[3]} | {key[2]} | {key[4] or "return − Frozen"}{evidence}\nnominal parameter={nominal}; {budget}; checkpoint={key[1][:12]}')
                ax.legend(fontsize=8);ax.grid(alpha=.2)
            fig.tight_layout();filename=f'{name}{"-"+str(page+1) if len(pages)>1 else ""}.png';fig.savefig(out/filename,dpi=140);plt.close(fig)
            outputs.append(dict(figure=filename,rows=[r for _,rr in items for r in rr],value_column=value))
    for task in sorted({r['task'] for r in summaries}):
        nominal=[r for r in summaries if r['task']==task and r['kind']=='nominal_validation' and r['metric']=='episode_return']
        if nominal:
            fig,ax=plt.subplots(figsize=(8,4))
            nominal.sort(key=lambda r:float(r['checkpoint_id']) if str(r['checkpoint_id']).isdigit() else 0)
            ax.plot(range(len(nominal)),[r['mean'] for r in nominal],'o-')
            ax.set_xticks(range(len(nominal)),[r['checkpoint_id'][:14] for r in nominal],rotation=30)
            for i,r in enumerate(nominal):
                if r['ci_low'] is not None:ax.vlines(i,r['ci_low'],r['ci_high'])
            ax.set(xlabel='Nominal checkpoint transitions',ylabel='Held-out episode return',title=task+' nominal validation')
            fig.tight_layout();name=task+'_nominal_validation.png';fig.savefig(out/name,dpi=140);plt.close(fig)
            outputs.append(dict(figure=name,rows=nominal,value_column='mean'))
        draw(f'{task}_adaptation', [r for r in comparisons if r['task']==task],ylabel='Paired return improvement over Frozen')
        draw(f'{task}_frozen', [r for r in summaries if r['task']==task and r['method']=='frozen' and r['kind']=='shifted' and r['metric']=='episode_return'],ylabel='Episode return')
        draw(f'{task}_physical', [r for r in summaries if r['task']==task and r['kind']=='physical' and r['metric'] in ('root_velocity_l2','joint_velocity_l2','delta_qvel_rad_s','delta_tip_m')],ylabel='Recorded physical response (metric units)')
        draw(f'{task}_prediction', [r for r in summaries if r['task']==task and r['kind']=='fixed_bank' and r['metric']=='latent_mse'],ylabel='Held-out latent MSE')
        draw(f'{task}_action_effort', [r for r in summaries if r['task']==task and r['kind']=='shifted' and r['metric'] in ('action_saturation','actuator_force_saturation','actuator_force_rms','action_abs','control_cost')])
        draw(f'{task}_nominal_retention', [r for r in summaries if r['task']==task and r['kind']=='nominal' and r['metric']=='episode_return'],ylabel='Nominal retention episode return')
    draw('q_distribution',[r for r in summaries if r['kind']=='q_immediate' and r['metric'] in ('q','entropy','bellman_residual')])
    draw('planning',[r for r in summaries if r['kind'] in ('q_planning','q_ranking','q_terminal','q_counterfactual')])
    if coupling:
        fig,ax=plt.subplots(figsize=(9,6))
        for task in sorted({r['task'] for r in coupling}):
            rr=[r for r in coupling if r['task']==task];ax.scatter([r['mse_improvement'] for r in rr],[r['mean_return_improvement'] for r in rr],label=task,alpha=.65)
        ax.axhline(0,color='gray');ax.axvline(0,color='gray');ax.set(xlabel='Frozen MSE − adapted MSE',ylabel='Adapted return − Frozen return',title='Matched adaptation streams; descriptive association');ax.legend();fig.tight_layout();fig.savefig(out/'prediction_vs_control.png',dpi=140);plt.close(fig)
        outputs.append(dict(figure='prediction_vs_control.png',rows=coupling,value_column='mean_return_improvement'))
    (out/'figure_data.json').write_text(json.dumps(outputs,indent=2))
    return outputs


def coverage(catalog,records):
    grouped=defaultdict(list)
    for r in records:grouped[(r['task'],r['method'],r['kind'],r['metric'],r['evidence'])].append(r)
    rows=[dict(zip(('task','method','analysis','metric','evidence'),k),canonical_records=len(rr),sources=len({r['source_path'] for r in rr})) for k,rr in sorted(grouped.items())]
    for task in catalog.registry:
        for method in ('frozen','dynamics_only','dynamics_q','joint'):
            trained=[x for x in catalog.training.values() if x['task']==task and x['method']==method and x['status']=='PASS']
            rows.append(dict(task=task,method=method,analysis='adaptation_training',metric='completed_streams',evidence='RAW_REPRODUCED' if trained else 'MISSING',canonical_records=len(trained),sources=len(trained)))
            for kind in ('shifted','fixed_bank'):
                if method=='joint' and kind=='fixed_bank':kind='joint_self_consistency'
                if not any(r['task']==task and r['method']==method and r['kind']==kind for r in records):rows.append(dict(task=task,method=method,analysis=kind,metric='',evidence='MISSING',canonical_records=0,sources=0))
        for kind in ('nominal_validation','physical','fixed_bank','shifted','nominal','q_immediate','q_planning'):
            if task=='reacher' and kind.startswith('q_'):continue
            if not any(r['task']==task and r['kind']==kind for r in records):rows.append(dict(task=task,method='',analysis=kind,metric='',evidence='MISSING',canonical_records=0,sources=0))
    return rows


def report(out,catalog,rec,summaries,comparisons,cov,regression):
    lines=['# Unified NS-Gym research report','',
        'This report was rebuilt from automatically discovered completed source records. No training, model inference or environment rollout was run.',
        '',f'Source root: `{catalog.root}`. Canonical measurements: {len(rec.rows)}. Paired return conditions: {len(comparisons)}.',
        '', 'Each adaptation seed is one statistical sampling unit; held-out episodes are paired by seed. Frozen replicas are shared references. k=1 family labels are nominal aliases, not independent tasks. Joint encoder-changing MSE is reported separately. Checkpoints, schedules and adaptation budgets are kept separate.',
        '', '## Paired held-out return results','']
    for r in comparisons:
        lines.append(f'- {r["task"]} / {r["shift_parameter"]} ×{r["severity_multiplier"]} / {r["schedule"]} / {r["method"]}, budget {r["adaptation_budget"]}: Δreturn={r["mean"]:.6g}, median={r["median"]:.6g}, CI=[{r["ci_low"]}, {r["ci_high"]}], improved {r["improved_streams"]}/{r["n_adaptation_streams"]}. Sources: ' + ' '.join(f'[{Path(s.rsplit(":",1)[0]).name}]({catalog.root/s.rsplit(":",1)[0]})' for s in sorted(set(r['source_records'].split(';')))[:4]))
    lines += ['', 'Every complete lineage is retained in `comparisons.csv:source_records`; each canonical row contains the exact source path, row range and SHA256.',
        '', '## Q and planning evidence','',
        'Q decoding uses the existing categorical TD-MPC2 implementation: symexp(E[symlog-support]), not E[symexp(support)]. Stored logits are decoded on CPU. Q arrays and per-observation measurements are distinguished from derived-only diagnostics. Missing Bellman targets prevent independent residual reconstruction. Candidate rank overlap describes shared model-score proposals; actual candidate returns/regret were not measured.',
        '', '## Coverage','',f'Regression: {regression["status"]}; counts: {regression["counts"]}.',
        'GPU float32 measurements remain the scientific values after independent CPU decoding checks. Cancellation-sensitive differences beyond direct tolerance are listed as NUMERICAL_DIFFERENCE in RAW_TO_REPORT_CHECKS.csv, with measured error and propagated float32 bound. This status is not an exact regression match.',
        'Missing actuator-force saturation is never replaced with zero or inferred from action-component saturation. Adaptation costs and optimization budgets are listed in ADAPTATION_COSTS.csv; method contrasts are procedure comparisons, not isolated causal interventions.',
        'See SOURCE_COVERAGE.csv and UNRESOLVED_DATA_GAPS.md for measured, derived-only, missing and invalid evidence.']
    (out/'UNIFIED_RESEARCH_REPORT.md').write_text('\n'.join(lines)+'\n')
    gaps=['# Unresolved data gaps','']
    for r in cov:
        if r['evidence'] in ('MISSING','DERIVED_ONLY'):gaps.append(f'- {r["task"]}/{r["method"]}/{r["analysis"]}/{r["metric"]}: {r["evidence"]}.')
    for r in catalog.issues:gaps.append(f'- {r["status"]}: {r["path"]}: {r["reason"]}')
    gaps += ['', 'Incomplete/smoke sources are listed separately in EXCLUDED_SOURCES.csv. They are not scientific failures and do not enter aggregates.',
             'The original server process was not accessed; execution used the user-provided server results copy. Missing raw targets or unsupported schemas are not inferred.']
    (out/'UNRESOLVED_DATA_GAPS.md').write_text('\n'.join(gaps)+'\n')


def run(args):
    config=json.loads(args.config.read_text());root=(args.results_root or DEFAULT_RESULTS).resolve();out=args.output.resolve()
    if not root.is_dir():raise ValueError(f'Results root absent: {root}')
    if out.exists():raise ValueError(f'Output exists; choose a new directory: {out}')
    code_hashes={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    catalog=Catalog(root,config['tasks'],out)
    filters={k:set(str(getattr(args,k)).split(',')) for k in ('task','shift_family','severity','schedule') if getattr(args,k,None) not in ('all',None,'')}
    if args.methods!='all':filters['method']=set(args.methods.split(','))|{'frozen','physical_probe'}
    if 'severity' in filters:filters['severity']={format(float(x),'.12g') for x in filters['severity']}
    filters['analysis']=args.analysis if args.operation in ('full','validate') else args.operation
    rec=collect(catalog,filters) if args.operation not in ('discover','inventory') else None
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps(dict(pipeline='nsgym_raw_to_report',no_training_or_simulator=True,results_root=str(root),operation=args.operation,config_version=config['version'],python=platform.python_version()),indent=2))
    table(out,'RAW_SOURCE_INVENTORY.csv',catalog.inventory)
    table(out,'DISCOVERED_EXPERIMENTS.csv',catalog.experiments)
    table(out,'EXCLUDED_SOURCES.csv',catalog.excluded)
    if rec is None:
        (out/'completion.json').write_text(json.dumps(dict(status='PASS',files=len(catalog.files),experiments=len(catalog.experiments)),indent=2));return
    # Analysis selection filters measurements without changing discovery/completion validation.
    requested=getattr(args,'analysis','full') if args.operation in ('full','validate') else args.operation
    kinds={'nominal':{'nominal_validation'},'physical':{'physical'},'q_distribution':{'q_immediate','q_divergence'},'planning':{'q_terminal','q_planning','q_ranking','q_counterfactual'},'adaptation':{'shifted','fixed_bank','nominal','joint_self_consistency'},'frozen':{'shifted','fixed_bank'},'cross_task':{'shifted','fixed_bank'}}
    if requested in kinds:rec.rows=[r for r in rec.rows if r['kind'] in kinds[requested] and (requested!='frozen' or r['method']=='frozen')]
    summaries=summarize_records(rec.rows);streams,comparisons=paired_improvements(rec.rows)
    contrasts=method_contrasts(rec.rows)
    nominal_streams,nominal_comparisons=paired_improvements(rec.rows,kind='nominal')
    pred=prediction_improvements(rec.rows);coupling=prediction_control_pairs(streams,pred)
    cov=coverage(catalog,rec.rows)
    costs=[]
    for training in catalog.training.values():
        if training['status']!='PASS':continue
        d=training['done']
        costs.append(dict(task=training['task'],method=training['method'],**{k:v for k,v in training['identity'].items() if k!='method'},
            environment_transitions=first(d,'transitions','adaptation_transitions'),model_update_calls=first(d,'updates','gradient_update_steps'),
            dynamics_steps=d.get('dynamics_steps',''),q_steps=d.get('q_steps',''),optimizer_step_calls=first(d,'optimizer_step_calls','total_optimizer_steps'),
            wall_seconds=first(d,'wall_seconds','training_wall_seconds'),source_path=training['path']))
    table(out,'ADAPTATION_COSTS.csv',costs)
    for name,data in [('cross_task_normalized.csv',normalized_comparisons(rec.rows,comparisons)),('adapted_method_contrasts.csv',contrasts),('canonical_records.csv',rec.rows),('statistical_summary.csv',summaries),('comparisons.csv',comparisons),('per_adaptation_stream.csv',streams),('nominal_retention.csv',nominal_comparisons),('prediction_improvements.csv',pred),('prediction_control_pairs.csv',coupling),('TRAINING_DIAGNOSTICS.csv',rec.training),('SOURCE_COVERAGE.csv',cov),('DUPLICATE_RECORDS.csv',rec.duplicates),('INVALID_SOURCES.csv',catalog.issues)]:table(out,name,data)
    if not args.no_plots:figures(out,summaries,comparisons,coupling)
    from .regression import validate
    result=validate(catalog,rec,out,summaries,comparisons,args.unified_output)
    result['missing_metric_families']=sum(r['evidence']=='MISSING' for r in cov)
    for metadata_path in catalog.json:catalog.digest(metadata_path)
    # A live run may append to an unrelated log while analysis is running.
    # Only files actually consumed or used as provenance affect this report.
    changed=[catalog.rel(p) for p in catalog.hashes if not p.is_file() or (p.stat().st_size,p.stat().st_mtime_ns)!=catalog.initial_stats[p]]
    for p in changed:catalog.issues.append(dict(path=p,status='INVALID',reason='Source changed during read-only analysis'))
    provenance=dict(source_root=str(root),analysis_config_sha256=sha256(args.config),analysis_files=code_hashes,files={catalog.rel(p):h for p,h in sorted(catalog.hashes.items())},
        lineage='canonical_records.csv:source_record_id and source_sha256; statistical_summary.csv/comparisons.csv:source_records',
        duplicate_records=len(rec.duplicates),raw_files_unchanged=not changed,changed_sources=changed)
    (out/'SOURCE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
    report(out,catalog,rec,summaries,comparisons,cov,result)
    status='PARTIAL' if catalog.issues or result['counts'].get('FAIL',0) else 'PASS_WITH_COVERAGE_GAPS' if any(r['evidence'] in ('MISSING','DERIVED_ONLY') for r in cov) else 'PASS'
    done=dict(status=status,records=len(rec.rows),paired_conditions=len(comparisons),regression=result,invalid_sources=len(catalog.issues),excluded_sources=len(catalog.excluded),duplicates=len(rec.duplicates))
    (out/'completion.json').write_text(json.dumps(done,indent=2))
    print(json.dumps(done,indent=2),flush=True)
    if result['counts'].get('FAIL',0):raise SystemExit(1)
