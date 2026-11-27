"""Read-only benchmark audit. Writes only to a newly created audit directory."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import uuid
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np


def output_directory(root):
    path = Path(root) / ('audit-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def check(status, evidence):
    return dict(status=status, evidence=evidence)


def close(a, b, atol=1e-7):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.allclose(a,b,atol=atol,rtol=0,equal_nan=False))


def intended(config, t, random_values):
    """Independent implementation for audit, not the production Schedule class."""
    p, s = config['parameter'], config['schedule']
    a, b = np.asarray(p['nominal']), np.asarray(p['shifted'])
    kind = s['type']
    interval = s.get('interval',200)
    if kind == 'nominal': return a
    if kind == 'fixed': return b
    if kind == 'abrupt': return a if t < s.get('switch_step',100) else b
    if kind == 'repeated': return a if (t//interval)%2 == 0 else b
    if kind == 'sinusoidal': return a + s.get('amplitude',1.)*(b-a)*math.sin(2*math.pi*t/interval)
    if kind == 'linear_drift': return a + np.clip((t-s.get('switch_step',100))/s.get('duration',200),0,1)*(b-a)
    if kind == 'random_piecewise': return np.asarray(random_values[t//interval])
    if kind == 'sequence':
        elapsed = 0
        for segment in s['segments']:
            elapsed += segment['duration']
            if t < elapsed: return np.asarray(segment['value'])
        return np.asarray(s['segments'][-1]['value'])
    raise ValueError(f'Unknown schedule: {kind}')


def timestamp(row, clock):
    return {'global_native_step':row['global_step'], 'episode_native_step':row['episode_step'],
            'global_decision_step':row['decision_step'], 'episode_decision_step':row['episode_decision_step'],
            'episode_index':row['episode_id']}[clock]


def audit_run(path):
    path = Path(path)
    findings, episodes, observations = {}, [], {}
    metadata = json.loads((path/'metadata.json').read_text())
    summary = json.loads((path/'summary.json').read_text()) if (path/'summary.json').exists() else {}
    cfg = metadata.get('shift_config', metadata.get('config',{}).get('dynamics_shift'))
    identity = metadata.get('config',{}).get('benchmark_id',path.name)
    result = dict(run=str(path.resolve()), condition=identity, checks=findings, episodes=episodes,
                  checkpoint_hash=metadata.get('checkpoint_sha256'), code_revision=metadata.get('code_version'),
                  source_sha256=metadata.get('source_sha256'), shift_config=cfg, environment_versions=metadata.get('environment_version',metadata.get('packages')))
    findings['completion'] = check('PASS' if summary.get('status')=='complete' else ('FAIL' if summary else 'UNVERIFIED'), summary.get('status','missing summary'))
    integrity = summary.get('checkpoint_unchanged')
    findings['checkpoint_integrity'] = check('PASS' if integrity is True else ('FAIL' if integrity is False else 'UNVERIFIED'), 'Recorded end-of-run hash check; file not reloaded by audit.')
    findings['live_physics_readback'] = check('UNVERIFIED','Trace readback is cached at reset; it is not an independent live property measurement.')
    findings['paired_full_initial_state'] = check('UNVERIFIED','Original logs do not save initial simulator+controller+RNG state.')
    csv_path, trace_path = path/'episodes.csv', path/'timesteps.jsonl'
    if not cfg or not csv_path.exists() or not trace_path.exists():
        findings['trace_audit'] = check('UNVERIFIED','Missing shift config, episodes.csv or timesteps.jsonl.')
        return result, observations
    with csv_path.open() as file: rows=list(csv.DictReader(file))
    by_episode=defaultdict(list)
    with trace_path.open() as file:
        for line in file:
            if line.strip():
                row=json.loads(line); by_episode[int(row['episode_id'])].append(row)
    if not by_episode:
        findings['trace_audit']=check('UNVERIFIED','Empty native trace; exposure cannot be inferred from event counts.')
        return result, observations
    all_records=[r for group in by_episode.values() for r in group]
    rng=np.random.default_rng(cfg['schedule'].get('seed',42))
    max_t=max(timestamp(r,cfg['schedule'].get('clock','global_native_step')) for r in all_records)
    choices=cfg['schedule'].get('values') or [0]
    random_values=[choices[int(rng.integers(len(choices)))] for _ in range(max_t//cfg['schedule'].get('interval',200)+1)]
    failures=[]; readback_missing=readback_count=0
    expected_seeds=metadata.get('episode_seeds')
    if expected_seeds is not None and [int(r['seed']) for r in rows]!=expected_seeds:
        failures.append('Episode seeds/order differ from recorded evaluation protocol.')
    if summary.get('summary_metrics',{}).get('episodes',len(rows))!=len(rows):
        failures.append('Summary episode count differs from CSV.')
    p=cfg['parameter']; clock=cfg['schedule'].get('clock','global_native_step')
    previous_global=-1; previous_decision=-1; history=[]; formula_unknown=False
    bounds=metadata.get('native_action_bounds')  # Absent from the original logger; do not assume [-1,1].
    for row in rows:
        eid=int(row['episode_id']); records=by_episode.pop(eid,[])
        if not records:
            failures.append(f'episode {eid}: missing trace'); continue
        if p.get('reset_queue_on_episode',True): history=[]
        selected_time=timestamp(records[0],clock)
        altered=effective=0; first_shift=None; repeat_counts=defaultdict(int)
        native_reward=0.; parameter_errors=[]
        for i,r in enumerate(records):
            native_reward+=r['reward']
            if r['global_step']!=previous_global+1 or r['native_step']!=r['global_step'] or r['episode_step']!=i:
                failures.append(f'episode {eid}, native {r["native_step"]}: counter discontinuity')
            previous_global=r['global_step']
            repeat_counts[r['decision_step']]+=1
            if r['decision_step'] not in {previous_decision,previous_decision+1}:
                failures.append(f'episode {eid}: decision counter discontinuity')
            previous_decision=r['decision_step']
            t=selected_time if cfg.get('application')=='reset' else timestamp(r,clock)
            expected=intended(cfg,t,random_values)
            if not close(expected,r['applied_parameter']): parameter_errors.append(r['native_step'])
            if not close(p['nominal'],r['applied_parameter']):
                altered+=1
                if first_shift is None: first_shift=r['native_step']
            command=np.asarray(r['commanded_action'],dtype=float); executed=np.asarray(r['executed_action'])
            if not close(command,executed): effective+=1
            if p.get('target','arm')=='arm' and p['type'].startswith('action_') and not close(command[-1],executed[-1]):
                failures.append(f'episode {eid}: gripper changed')
            history.append(command)
            prediction=command.copy(); arm=slice(None,-1) if p.get('target','arm')=='arm' else slice(None)
            value=np.asarray(r['applied_parameter'])
            if p['type']=='action_gain': prediction[arm]*=value
            elif p['type']=='action_bias': prediction[arm]+=value
            elif p['type']=='action_coupling': prediction[arm]=value@command[arm]
            elif p['type']=='action_delay':
                delay=int(value)
                delayed=history[-delay-1] if len(history)>delay else np.broadcast_to(p.get('initial_action',0.),command.shape)
                prediction[arm]=delayed[arm]
            if bounds: prediction=np.clip(prediction,bounds['low'],bounds['high'])
            if not close(prediction,executed):
                if bounds or p['type'] in {'object_mass','contact_friction'}: failures.append(f'episode {eid}: action transform mismatch')
                else: formula_unknown=True
            if p['type'] in {'object_mass','contact_friction'}:
                rb=r.get('physics_readback')
                if not rb: readback_missing+=1; continue
                readback_count+=1
                if p['type']=='object_mass':
                    if not close(rb['mass'],rb['nominal_mass']*float(value),atol=1e-5): failures.append(f'episode {eid}: mass readback mismatch')
                elif 'nominal_materials' in rb:
                    expected_materials=np.asarray(rb['nominal_materials'])*[float(value),float(value),1.]
                    if not close(expected_materials,rb['materials'],atol=1e-5): failures.append(f'episode {eid}: friction readback mismatch')
                else: readback_missing+=1
        if parameter_errors: failures.append(f'episode {eid}: schedule mismatch first native={parameter_errors[0]}')
        if float(row['success']) not in {0.,1.}: failures.append(f'episode {eid}: success is not binary')
        if len(records)!=int(row['native_episode_length']) or len(repeat_counts)!=int(row['episode_length']): failures.append(f'episode {eid}: length mismatch')
        if records[0]['global_step']!=int(row['native_start']) or records[-1]['global_step']+1!=int(row['native_end']): failures.append(f'episode {eid}: global boundary mismatch')
        if any(count not in {1,2} for count in repeat_counts.values()): failures.append(f'episode {eid}: invalid repeat count')
        for decision_index,decision_id in enumerate(repeat_counts):
            group=[r for r in records if r['decision_step']==decision_id]
            if len(group)==1 and not group[0]['done']: failures.append(f'episode {eid}: repeat ended without native done')
            if any(r['episode_decision_step']!=decision_index for r in group): failures.append(f'episode {eid}: episode decision counter mismatch')
            if any(not close(r['commanded_action'],group[0]['commanded_action']) for r in group): failures.append(f'episode {eid}: command changed inside repeat')
        for j,r in enumerate(records):
            if r['done'] and j != len(records)-1: failures.append(f'episode {eid}: native steps after done')
        if not records[-1].get('decision_done',False): failures.append(f'episode {eid}: episode lacks decision termination')
        if not math.isclose(native_reward,float(row['episode_return']),abs_tol=1e-4,rel_tol=1e-6): failures.append(f'episode {eid}: return mismatch')
        episodes.append(dict(episode_id=eid,seed=int(row['seed']),success=float(row['success']),episode_return=float(row['episode_return']),
                             decision_length=int(row['episode_length']),native_length=len(records),shifted_native_steps=altered,
                             exposure_fraction=altered/len(records),first_shift_native_step=first_shift,
                             command_execution_difference_steps=effective,shift_events=int(row['shift_events']),
                             intended_parameter_first=json.dumps(np.asarray(intended(cfg,selected_time,random_values)).tolist()),
                             applied_parameter_first=json.dumps(records[0]['applied_parameter']),
                             applied_parameter_last=json.dumps(records[-1]['applied_parameter']),
                             applied_parameter_min=float(min(float(r['applied_parameter']) for r in records)) if np.asarray(records[0]['applied_parameter']).ndim==0 else None,
                             applied_parameter_max=float(max(float(r['applied_parameter']) for r in records)) if np.asarray(records[0]['applied_parameter']).ndim==0 else None))
        trajectory=path/f'episode-{eid:04d}.npz'
        if trajectory.exists():
            if row.get('trajectory_sha256') and hashlib.sha256(trajectory.read_bytes()).hexdigest()!=row['trajectory_sha256']:
                failures.append(f'episode {eid}: trajectory hash mismatch')
            with np.load(trajectory,allow_pickle=False) as data:
                if int(data['seed'])!=int(row['seed']): failures.append(f'episode {eid}: dataset seed mismatch')
                observations[(eid,int(row['seed']))]=data['observations'][0].copy()
    if by_episode: failures.append('Trace contains episodes absent from episodes.csv.')
    findings['trace_contract']=check('FAIL' if failures else 'PASS',failures[:50] or 'Counters, schedule application, rewards and available readbacks agree with saved configuration.')
    findings['action_formula']=check('UNVERIFIED' if formula_unknown else 'PASS', 'Clipping bounds absent; differing saturated transforms cannot be certified.' if formula_unknown else 'Logged actions agree with the formula on the observed inputs; unexercised clipping remains unverified.')
    findings['physics_logged_readback']=check('UNVERIFIED' if readback_missing or not readback_count else 'PASS',dict(available_rows=readback_count,missing_rows=readback_missing))
    findings['actual_exposure']=check('PASS',dict(episodes_with_exposure=sum(e['shifted_native_steps']>0 for e in episodes),episodes=len(episodes),
                                                       shifted_native_steps=sum(e['shifted_native_steps'] for e in episodes),native_steps=sum(e['native_length'] for e in episodes)))
    for field,csv_field in [('episode_return','episode_return'),('success','success'),('episode_length','decision_length')]:
        observed=[e[csv_field] for e in episodes]
        for suffix,fn in [('mean',np.mean),('std',np.std)]:
            key=field+'_'+suffix; value=summary.get('summary_metrics',{}).get(key)
            if value is not None and not math.isclose(float(fn(observed)),value,abs_tol=1e-6,rel_tol=1e-6): failures.append(f'Summary {key} disagrees with CSV.')
    findings['trace_contract']=check('FAIL' if failures else 'PASS',failures[:50] or 'Saved records are internally consistent; this is not a simulator validation.')
    result['research_status']='UNVERIFIED' if not failures else 'FAIL'
    return result,observations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs',type=Path,required=True)
    parser.add_argument('--out-root',type=Path,default=Path('audits'))
    args=parser.parse_args(); output=output_directory(args.out_root)
    paths=sorted(p.parent for p in args.logs.rglob('metadata.json'))
    results=[]; observations={}
    for path in paths:
        try:
            result,initial=audit_run(path); results.append(result); observations[str(path.resolve())]=initial
        except Exception as exc:
            results.append(dict(run=str(path.resolve()),research_status='UNVERIFIED',error=f'{type(exc).__name__}: {exc}'))
    references=[r for r in results if r.get('condition')=='E0']
    for run in results:
        if 'checks' not in run: continue
        candidates=[r for r in references if r.get('checkpoint_hash') and r['checkpoint_hash']==run.get('checkpoint_hash') and r['code_revision']==run.get('code_revision') and r.get('source_sha256')==run.get('source_sha256') and r.get('environment_versions')==run.get('environment_versions')]
        if len(candidates)!=1:
            run['checks']['paired_initial_observation']=check('UNVERIFIED','Need exactly one compatible E0 run; do not select a reference arbitrarily.')
            continue
        ref=candidates[0]; a=observations[ref['run']]; b=observations[run['run']]
        shared=set(a)&set(b); different=[key for key in shared if not close(a[key],b[key],1e-6)]
        if different: run['research_status']='FAIL'
        complete=bool(shared) and len(shared)==len(run['episodes'])==len(ref['episodes'])
        run['checks']['paired_initial_observation']=check('FAIL' if different else ('PASS' if complete else 'UNVERIFIED'),
            dict(reference=ref['run'],matched=len(shared),different=different,full_simulator_identity=False))
    report=dict(runs=results,status='FAIL' if any(r.get('research_status')=='FAIL' for r in results) else 'UNVERIFIED', scope='saved_log_audit_only',
                note='PASS applies to each named check only; historical physical validity requires independent simulator tests.')
    (output/'log_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'exposure.csv').open('w',newline='') as file:
        writer=None
        for run in results:
            for episode in run.get('episodes',[]):
                row=dict(condition=run['condition'],run=run['run'],**episode)
                if writer is None: writer=csv.DictWriter(file,fieldnames=row); writer.writeheader()
                writer.writerow(row)
    print(output.resolve())


if __name__=='__main__': main()
