"""Summarize only protocol-compatible complete Frozen severity evaluations."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tdmpc2'))
from common.severity_protocol import PROTOCOL
from common.severity_evaluation import same_initial_kinematics


def inspect(entry, manifest):
    path=Path(entry['path'])
    row=dict(run_id=manifest['run_id'],shift_type=entry['shift_type'],shift_value=entry['shift_value'],
             set_id=entry['set_id'],status='MISSING',reason='',episode_count=0,success='',
             episode_return='',episode_length='',nominal_success='',initial_state_set_sha256='',
             shift_definition_sha256='',source_sha256='',path=str(path))
    meta_path=path/'severity-metadata.json';done_path=path/'severity-complete.json'
    if not meta_path.exists():
        if path.exists(): row.update(status='PARTIAL',reason='Directory exists without metadata')
        return row,None
    try:
        meta=json.loads(meta_path.read_text())
        checks=dict(protocol=PROTOCOL,run_id=manifest['run_id'],shift_type=entry['shift_type'],
                    shift_value=entry['shift_value'],set_id=entry['set_id'],evaluation_seeds=entry['seeds'],
                    checkpoint_sha256=manifest['checkpoint_sha256'],task=manifest['task'],
                    model_size=manifest['model_size'],episodic=manifest['episodic'],log_eef=manifest['log_eef'])
        if any(meta.get(k)!=v for k,v in checks.items()):
            row.update(status='PROTOCOL_INCOMPATIBLE',reason='Metadata/manifest mismatch')
            return row,None
        identity=(meta['checkpoint_sha256'],meta['protocol_controls_sha256'],meta['code_revision'],
                  meta.get('source_sha256'),json.dumps(meta['environment_versions'],sort_keys=True),len(entry['seeds']))
        row['shift_definition_sha256']=meta['shift_definition_sha256']
        row['source_sha256']=meta.get('source_sha256','')
        if not done_path.exists():
            row.update(status='PARTIAL',reason='No completion record')
            return row,identity
        result=json.loads(done_path.read_text())
        if any(result[k]!=0 for k in ('parameter_drift','gradient_updates','replay_updates')) or not result['nominal_restored']:
            row.update(status='PROTOCOL_INCOMPATIBLE',reason='Frozen/isolation invariance failed')
            return row,identity
        for condition in ('shifted','nominal'):
            episodes=result['evaluation'][condition]
            if [e['seed'] for e in episodes]!=entry['seeds']:
                row.update(status='PARTIAL',reason='Episode seed/count mismatch')
                return row,identity
            if len(result['effective_physical_values'][condition])!=len(episodes):
                row.update(status='PARTIAL',reason='Missing effective parameter evidence')
                return row,identity
            for seed in entry['seeds']:
                if not all((path/f'{condition}-seed-{seed}.{ext}').is_file() for ext in ('npz','json')):
                    row.update(status='PARTIAL',reason='Missing trajectory files')
                    return row,identity
                import numpy as np
                ep = next(e for e in episodes if e['seed']==seed)
                trace_meta=json.loads((path/f'{condition}-seed-{seed}.json').read_text())
                if (trace_meta['evaluation_seed']!=seed or trace_meta['shift_type']!=entry['shift_type']
                        or trace_meta['shift_value']!=entry['shift_value']
                        or trace_meta['checkpoint_sha256']!=manifest['checkpoint_sha256']):
                    row.update(status='PROTOCOL_INCOMPATIBLE',reason='Trajectory provenance mismatch')
                    return row,identity
                with np.load(path/f'{condition}-seed-{seed}.npz',allow_pickle=False) as trace:
                    n=ep['native_steps']
                    required=['timestep','commanded_action','executed_action','observation','reward','episode_success']
                    if manifest['log_eef']: required.append('eef_xyz')
                    if (any(key not in trace or len(trace[key])!=n for key in required)
                            or not np.array_equal(trace['timestep'],np.arange(n))
                            or any(not np.isfinite(trace[key]).all() for key in required)):
                        row.update(status='PARTIAL',reason='Invalid native trajectory arrays/count')
                        return row,identity
            if any(not math.isfinite(float(e[k])) for e in episodes for k in ('success','episode_return','episode_length')):
                row.update(status='PROTOCOL_INCOMPATIBLE',reason='Non-finite evaluation metrics')
                return row,identity
        shifted_kin=result.get('initial_kinematic_states',{}).get('shifted')
        nominal_kin=result.get('initial_kinematic_states',{}).get('nominal')
        if shifted_kin is not None and nominal_kin is not None:
            if (len(shifted_kin)!=len(entry['seeds']) or len(nominal_kin)!=len(entry['seeds'])
                    or any(not same_initial_kinematics(a,b) for a,b in zip(shifted_kin,nominal_kin))):
                row.update(status='PROTOCOL_INCOMPATIBLE',reason='Initial kinematics mismatch')
                return row,identity
        elif result['initial_state_ids']['shifted']!=result['initial_state_ids']['nominal']:
            row.update(status='PROTOCOL_INCOMPATIBLE',reason='Initial-state mismatch; no kinematic evidence')
            return row,identity
        episodes=result['evaluation']['shifted']
        row.update(status='COMPLETE',episode_count=len(episodes),
                   initial_state_set_sha256=hashlib.sha256(json.dumps(result['initial_state_ids']['nominal']).encode()).hexdigest(),
                   success=statistics.mean(e['success'] for e in episodes),
                   episode_return=statistics.mean(e['episode_return'] for e in episodes),
                   episode_length=statistics.mean(e['episode_length'] for e in episodes),
                   nominal_success=statistics.mean(e['success'] for e in result['evaluation']['nominal']))
        return row,identity
    except (ValueError,KeyError,TypeError,OSError,EOFError,zipfile.BadZipFile) as exc:
        row.update(status='PARTIAL',reason=f'Unreadable/incomplete result: {exc}')
        return row,None


def summarize(manifests, output, latest=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]; identities=[]
    for manifest_path in manifests:
        manifest=json.loads(Path(manifest_path).read_text())
        for entry in manifest['entries']:
            row,identity=inspect(entry,manifest);rows.append(row);identities.append(identity)
    # A figure is drawn per shift. Require shared checkpoint/evaluation controls
    # globally, and identical source within each shift, where severities are
    # compared. Distinct shifts may have distinct validation-only source builds;
    # their source hashes remain visible in the summary for provenance.
    complete_pairs=[(row,identity) for row,identity in zip(rows,identities) if row['status']=='COMPLETE']
    scientific_ids={(identity[0],identity[1],identity[4],identity[5])
                    for _,identity in complete_pairs}
    if len(scientific_ids)>1:
        for row in rows:
            if row['status']=='COMPLETE': row.update(status='PROTOCOL_INCOMPATIBLE',reason='Mixed checkpoint/evaluation controls')
    for kind in {row['shift_type'] for row,_ in complete_pairs}:
        subset=[(row,identity) for row,identity in complete_pairs if row['shift_type']==kind]
        if len({identity for _,identity in subset})>1:
            for row,_ in subset:
                if row['status']=='COMPLETE':
                    row.update(status='PROTOCOL_INCOMPATIBLE',reason='Mixed source revisions within shift')
    groups={}
    for row in rows:
        if row['status']=='COMPLETE':
            key=(row['shift_type'],row['shift_value'],row['set_id'])
            groups.setdefault(key,[]).append(row)
    for duplicates in groups.values():
        if len(duplicates)>1:
            for row in duplicates: row.update(status='PROTOCOL_INCOMPATIBLE',reason='Duplicate evaluation set; select one run')
    for kind,seed_set in {(r['shift_type'],r['set_id']) for r in rows}:
        subset=[r for r in rows if r['shift_type']==kind and r['set_id']==seed_set
                and r['status']=='COMPLETE']
        if len({r['initial_state_set_sha256'] for r in subset})>1:
            for row in subset:
                row.update(status='PROTOCOL_INCOMPATIBLE',reason='Nominal initial states differ across severities')
    for kind in {r['shift_type'] for r in rows}:
        subset=[r for r in rows if r['shift_type']==kind and r['status']=='COMPLETE']
        if len({r['shift_definition_sha256'] for r in subset})>1:
            for row in subset:
                row.update(status='PROTOCOL_INCOMPATIBLE',reason='Shift definition differs beyond severity')
    for row in rows:
        if row['status']!='COMPLETE':
            for key in ('success','episode_return','episode_length','nominal_success'):row[key]=''
    def save(path):
        temporary=path.with_suffix('.tmp')
        with temporary.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        temporary.replace(path)
    save(output/'summary.csv')
    if latest is not None: save(Path(latest))
    plot(rows,output)
    return rows


def plot(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in sorted({r['shift_type'] for r in rows}):
        selected=[r for r in rows if r['shift_type']==kind and r['status']=='COMPLETE']
        fig,ax=plt.subplots(figsize=(8,5))
        values=sorted({r['shift_value'] for r in rows if r['shift_type']==kind})
        for value in values:
            group=[r for r in selected if r['shift_value']==value]
            if not group: continue
            rates=[100*r['success'] for r in group]
            mean=statistics.mean(rates);sd=statistics.stdev(rates) if len(rates)>1 else 0
            ax.errorbar(value,mean,yerr=sd,fmt='o',color='tab:blue',capsize=4)
            ax.plot(value,statistics.mean(100*r['nominal_success'] for r in group),'x',color='gray')
            ax.annotate(f"n={sum(r['episode_count'] for r in group)}; sets={len(group)}",(value,mean),
                        xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.plot([],[],'o',color='tab:blue',label='Frozen shifted')
        ax.plot([],[],'x',color='gray',label='Matched nominal reference')
        ax.set(xlabel='Delay (native steps)' if kind=='action_delay' else 'Multiplicative factor',
               ylabel='Success (%)',title=kind.replace('_',' '),ylim=(-3,118),xticks=values)
        ax.grid(alpha=.2);ax.legend(loc='lower right')
        excluded=sum(r['status']!='COMPLETE' for r in rows if r['shift_type']==kind)
        fig.text(.5,.015,f'SD across independent sets only when n_sets > 1 (not CI). Excluded conditions: {excluded}',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(output/f'{kind}.png',dpi=180);fig.savefig(output/f'{kind}.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',nargs='+',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    summarize(args.manifest,args.output)
