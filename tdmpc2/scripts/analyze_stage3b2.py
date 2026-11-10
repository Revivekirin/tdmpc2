"""Manifest-selected Stage 3B aggregation; no simulation, copying of .pt, or source mutation."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics
import tempfile
from stage3b2_audit import read, BUDGETS, value_label
from analyze_stage3b import analyze, write_csv


def reference_controls(cfg):
    # Exclude adaptation instrumentation/settings; compare all environment/model/MPC controls.
    return {k:v for k,v in cfg.items() if k not in
            ('stage3a','stage3b','severity','work_dir','checkpoint')}


def frozen_references(root,entry,checkpoint_hash):
    from analyze_stage3b_severity import inspect
    cfg=entry['requested_config'];references=[]
    for path in sorted((Path(root)/'runs').rglob('severity-metadata.json')):
        try:
            meta=read(path)
            if (meta['shift_type']!=entry['shift_type'] or meta['shift_value']!=entry['shift_value'] or meta['set_id']!=0):continue
            record,_=inspect(dict(path=str(path.parent),shift_type=meta['shift_type'],shift_value=meta['shift_value'],
                                  set_id=0,seeds=meta['evaluation_seeds']),meta)
            reasons=[]
            if record['status']!='COMPLETE':reasons.append(record['reason'] or record['status'])
            if meta['checkpoint_sha256']!=checkpoint_hash:reasons.append('pretrained checkpoint differs')
            if meta['evaluation_seeds']!=cfg['stage3a']['evaluation_seeds']:reasons.append('held-out seeds differ')
            if reference_controls(meta['config'])!=reference_controls(cfg):reasons.append('environment/model/MPC protocol controls differ')
            references.append(dict(path=str(path.parent),compatible=not reasons,reason='; '.join(reasons),
                                   success=record['success'],episode_length=record['episode_length'],
                                   episode_return=record['episode_return'],nominal_success=record['nominal_success'],
                                   evaluation_seeds=meta['evaluation_seeds'],code_revision=meta.get('code_revision'),
                                   source_sha256=meta.get('source_sha256')))
        except (ValueError,KeyError,TypeError,OSError) as exc:
            references.append(dict(path=str(path.parent),compatible=False,reason=str(exc),
                                   success='',episode_length='',episode_return='',nominal_success=''))
    return references


def report(manifest_path,output):
    manifest=read(manifest_path);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    comparisons=[];completions=[];all_tables={};references=[];figures=[]
    for entry in manifest['entries']:
        kind,value=entry['shift_type'],entry['shift_value'];label=kind+'-'+value_label(value)
        source=entry.get('result_path');complete=bool(source)
        # Partial logs may contain a half-written line. Plot only the verified checkpoint's rows.
        partial=next((x for x in entry['candidates'] if x['status']=='RESUME_PARTIAL' and x['path']==entry.get('source_path')),None)
        result=dict(shift_type=kind,shift_value=value,set_id=0,initial_status=entry['status'],
                    status='COMPLETE' if complete else 'MISSING',path=source or entry.get('target',''),reason='')
        row=dict(shift_type=kind,shift_value=value,set_id=0,status=result['status'],
                 frozen_path='',frozen_compatible=False,frozen_success='',frozen_episode_length='',frozen_return='',
                 ttt_initial_success='',ttt_final_success='',success_delta_vs_frozen='',ttt_final_episode_length='',
                 ttt_final_return='',nominal_initial_success='',nominal_final_success='',nominal_retention_delta='',
                 fixed_initial_mse='',fixed_final_mse='',gradient_updates='',replay_size='',adaptation_native_steps='',
                 policy_decisions='',completed_episodes='',adaptation_auc='',ttt_path=source or '',reason='')
        records=[]
        if source or partial:
            with tempfile.TemporaryDirectory(prefix='stage3b2-report-') as temporary:
                view=Path(temporary)/'input';view.mkdir()
                origin=Path(source or partial['path'])
                meta=read(origin/'metadata.json')
                (view/'metadata.json').write_text(json.dumps(meta))
                if complete:
                    shutil.copyfile(origin/'evaluations.jsonl',view/'evaluations.jsonl')
                    shutil.copyfile(origin/'complete.json',view/'complete.json')
                    records=[json.loads(line) for line in (view/'evaluations.jsonl').read_text().splitlines()]
                else:
                    from stage3b2_audit import load_pt
                    records=load_pt(partial['resume'])['rows']
                    (view/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                    result['status']=row['status']='PARTIAL'
                    result['path']=row['ttt_path']=str(origin)
                suite=dict(run_id=meta['config']['stage3b']['run_id'],budgets=BUDGETS,methods=['dynamics_only'],
                           seed_sets=[0],shifts=[dict(type=kind,magnitude=value)])
                analyze(view,output/label,suite,allow_partial=not complete)
                validation_path=output/label/'validation.json'
                validation=read(validation_path);validation['input_root']=str(origin)
                validation['selection_manifest']=str(Path(manifest_path).resolve())
                validation_path.write_text(json.dumps(validation,indent=2))
            for table in (output/label/'tables').glob('*.csv'):
                with table.open() as stream:data=list(csv.DictReader(stream))
                for item in data:item['shift_value']=value
                all_tables.setdefault(table.name,[]).extend(data)
            figures.extend(str(p) for p in (output/label/'figures').glob('*'))
            first,last=records[0],records[-1]
            row.update(ttt_initial_success=first['evaluation']['shifted']['success'],
                       nominal_initial_success=first['evaluation']['nominal']['success'],
                       gradient_updates=last['gradient_update_steps'],replay_size=last['replay_size'],
                       adaptation_native_steps=last['adaptation_native_steps'],policy_decisions=last['adaptation_policy_decisions'],
                       completed_episodes=last['adaptation_episodes'],
                       fixed_initial_mse=statistics.mean(first['fixed_data_prediction']['initial']['per_episode_means']))
            if complete:
                row.update(ttt_final_success=last['evaluation']['shifted']['success'],
                           ttt_final_episode_length=last['evaluation']['shifted']['episode_length'],
                           ttt_final_return=last['evaluation']['shifted']['episode_return'],
                           nominal_final_success=last['evaluation']['nominal']['success'],
                           nominal_retention_delta=last['evaluation']['nominal']['success']-first['evaluation']['nominal']['success'],
                           fixed_final_mse=statistics.mean(last['fixed_data_prediction']['current']['per_episode_means']))
                from analyze_stage3b import normalized_auc
                row['adaptation_auc']=normalized_auc(BUDGETS,[r['evaluation']['shifted']['success'] for r in records])
        refs=frozen_references(manifest['root'],entry,manifest['checkpoint_sha256'])
        for ref in refs:references.append(dict(shift_type=kind,shift_value=value,**ref))
        matched=[r for r in refs if r['compatible']]
        if matched:
            ref=matched[-1]
            row.update(frozen_path=ref['path'],frozen_compatible=True,frozen_success=ref['success'],
                       frozen_episode_length=ref['episode_length'],frozen_return=ref['episode_return'])
            if complete:row['success_delta_vs_frozen']=row['ttt_final_success']-ref['success']
        else:row['reason']='No verified protocol-compatible Frozen reference. See frozen_references.json for separate comparisons.'
        completions.append(result);comparisons.append(row)
    for name,rows in all_tables.items():
        if rows:write_csv(output/name,rows)
    write_csv(output/'baseline_comparison.csv',comparisons)
    write_csv(output/'completion.csv',completions)
    (output/'frozen_references.json').write_text(json.dumps(references,indent=2))
    (output/'completion.json').write_text(json.dumps(completions,indent=2))
    # Length is exported by the existing pipeline; render an additional curve from its CSV.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=all_tables.get('stage3b_aggregated_results.csv',[])
    for metric,ylabel in [('shifted_episode_length','Shifted episode length (policy decisions)'),
                          ('shifted_return','Shifted episode return')]:
        if not rows:continue
        fig,ax=plt.subplots(figsize=(10,6))
        for entry in manifest['entries']:
            points=[r for r in rows if r['shift_type']==entry['shift_type'] and r['shift_value']==entry['shift_value'] and r['metric']==metric]
            if points:ax.plot([int(r['budget_native_steps']) for r in points],[float(r['mean']) for r in points],marker='o',label=f"{entry['shift_type']}={entry['shift_value']}")
        title='Stage 3B-2 | set 0 | '+manifest['run_id']
        if manifest['run_id'].startswith('SYNTHETIC'):title='SYNTHETIC VALIDATION — NOT EXPERIMENTAL RESULTS'
        ax.set(title=title,xlabel='Cumulative native adaptation steps',ylabel=ylabel);ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout()
        for ext in ('png','pdf'):
            target=output/(metric+'.'+ext);fig.savefig(target);figures.append(str(target))
        plt.close(fig)
    (output/'figures.json').write_text(json.dumps(figures,indent=2))
    reused=[e['result_path'] for e in manifest['entries'] if e['action']=='REUSE']
    executed=[e.get('result_path',e.get('target')) for e in manifest['entries'] if e.get('execution_status')=='COMPLETE']
    missing=[f"{r['shift_type']}={r['shift_value']} ({r['status']})" for r in completions if r['status']!='COMPLETE']
    import shlex
    commands='\n'.join(shlex.join(cmd) for e in manifest['entries'] for cmd in e['commands'])
    (output/'STAGE3B2_RESULTS.md').write_text(
        '# Stage 3B-2 results\n\n'
        +f"Manifest: {Path(manifest_path).resolve()}\n\n"
        +'Reused runs:\n\n'+'\n'.join('- '+p for p in reused)
        +'\n\nNewly completed runs:\n\n'+'\n'.join('- '+p for p in executed)
        +'\n\nOutstanding conditions:\n\n'+('\n'.join('- '+x for x in missing) or 'None.')
        +'\n\nExact planned commands (execution status is recorded in the manifest):\n\n```bash\n'+commands+'\n```\n\n'
        +'`baseline_comparison.csv` contains shifted success/return/length, nominal retention, fixed-data MSE, '
        +'updates, replay size, interaction counts and normalized adaptation AUC. `completion.csv` lists all eight conditions. '
        +'Per-severity tables and PNG/PDF curves use the existing Stage 3B aggregation pipeline; `figures.json` lists paths.\n\n'
        +'Only set 0 is included (10 held-out episodes). No across-seed uncertainty is available. '
        +'Fixed MSE is an equal-episode mean on Frozen-collected data; it is not a task-success metric. '
        +'Frozen references with different seeds, checkpoint or protocol are retained separately in `frozen_references.json`; '
        +'no paired difference is computed for them. Frozen final severity performance is not an adaptation curve; '
        +'no Frozen AUC is invented from it. Legacy artifacts do not record dependency/source hashes, so '
        +'config/artifact compatibility cannot establish historical binary-environment equivalence. '
        +'Resumed runs depend on their preserved ancestor checkpoints. No new simulator validation was performed by this report.\n')
    return completions


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.manifest,args.output)
