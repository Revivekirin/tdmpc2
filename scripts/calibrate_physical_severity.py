"""Prepare user-run calibration commands and export measured severity evidence.

Never launches training or experiments. Empty exports explicitly remain pending.
Final selection requires complete measurements AND an explicit evidence review.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tdmpc2-calibration-mpl')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tdmpc2'))
from common.severity_protocol import PILOT

NOMINAL = dict(action_gain=1., action_delay=0, object_mass=1., contact_friction=1.)
METRICS = {
    'action_gain': ('free_space_eef_rmse_m', 'target_rmse_m'),
    'action_delay': ('eef_rmse_m', 'eef_motion_onset_delta_s'),
    'object_mass': ('object_acceleration_rmse_m_s2', 'max_lift_delta_m'),
    'contact_friction': ('finger_relative_path_delta_m', 'plane_relative_path_delta_m'),
}


def read(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(v): return f'{v:g}'
def dump(p, value): Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def validate_manifest(m):
    cal, final = m['calibration_seeds'], m['final_benchmark_seeds']
    if not cal or not final or len(set(cal))!=len(cal) or len(set(final))!=len(final) or set(cal)&set(final):
        raise ValueError('Nonempty, unique, disjoint calibration/final seeds required')
    if len(cal)>10: raise ValueError('This bounded calibration supports at most ten seeds')
    if any(type(s)!=int or s<0 for s in cal+final): raise ValueError('Invalid seed')
    expected=list(range(4000+10000*m['calibration_set_id'],4000+10000*m['calibration_set_id']+len(cal)))
    if cal!=expected: raise ValueError('Calibration seeds must match existing Frozen runner seed-set convention')
    if m['motion_detection_m']<=0 or m['max_native_steps'] not in range(1,81): raise ValueError('Invalid probe bound')
    for kind,values in m['candidates'].items():
        if kind not in NOMINAL or NOMINAL[kind] not in values or len(set(values))!=len(values): raise ValueError('Invalid candidates')
        if any(not np.isfinite(v) or (v<0 or v!=int(v) or v>3 if kind=='action_delay' else v<=0) for v in values):
            raise ValueError('Invalid candidate values')


def frozen_dir(base, m, kind, value):
    return base/'frozen/runs'/kind/f"set-{m['calibration_set_id']}"/'frozen'/label(value)/'severity-eval'/f'calibration-{kind}'


def prepare(args):
    if args.calibration_set==args.final_set: raise ValueError('Calibration and final seed sets overlap')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    candidates={k:list(dict.fromkeys([NOMINAL[k]]+list(v))) for k,v in PILOT.items()}
    manifest=dict(protocol='physical-severity-calibration-v1',candidate_class='exploratory_engineering_heuristic',
                  candidates=candidates,nominal=NOMINAL,calibration_set_id=args.calibration_set,
                  calibration_seeds=list(range(4000+10000*args.calibration_set,4000+10000*args.calibration_set+args.seeds)),
                  final_benchmark_seeds=list(range(4000+10000*args.final_set,4010+10000*args.final_set)),
                  checkpoint=args.checkpoint,task='pick-cube',max_native_steps=40,motion_detection_m=1e-5,
                  detection_note='Motion onset is first measured displacement beyond 10 micrometers; diagnostic threshold, not severity acceptance or causal latency.',
                  minimum_calibration_seeds=3,selection_reviews={})
    validate_manifest(manifest);dump(out/'calibration_manifest.json',manifest)
    # Relative output paths are resolved on the server from the script directory.
    header=['#!/usr/bin/env bash','set -euo pipefail','CALIBRATION_DIR="$(cd "$(dirname "$0")" && pwd)"',
            ': "${REPO:?Set REPO to the server repository}"','cd "$REPO"']
    frozen=header+[': "${CHECKPOINT:?Set CHECKPOINT to the unchanged pretrained checkpoint}"']
    for kind,values in candidates.items():
        frozen += ['python scripts/run_stage3b.py --phase severity --method frozen --checkpoint "$CHECKPOINT" '+
                   f'--shift-type {kind} --shift-values '+ ' '.join(map(label,values))+
                   f" --set-id {args.calibration_set} --episodes {args.seeds} --log-eef --run-id calibration-{kind}"+
                   ' --output "$CALIBRATION_DIR/frozen" --execute']
    (out/'run_frozen.sh').write_text('\n'.join(frozen)+'\n')
    probes=header.copy()
    for seed in manifest['calibration_seeds']:
        tape=f'"$CALIBRATION_DIR/frozen/runs/action_gain/set-{args.calibration_set}/frozen/1/severity-eval/calibration-action_gain/nominal-seed-{seed}.npz"'
        for kind,values in candidates.items():
            probes += [f'python scripts/probe_physical_response.py --repo "$REPO" --actions {tape} --seed {seed} --shift {kind} '+
                       '--values '+' '.join(map(label,values))+' --max-native-steps 40 '+
                       f'--output "$CALIBRATION_DIR/probes/{kind}/seed-{seed}"']
    (out/'run_probes.sh').write_text('\n'.join(probes)+'\n')
    export(out/'calibration_manifest.json',out/'pending')
    print(out)


def active_pairs(step):
    pairs=set()
    for c in step['contacts']:
        for p in c['points']:
            for key in ['position','normal','impulse']:
                a=np.asarray(p[key])
                if a.shape!=(3,) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all(): raise ValueError('Non-numeric contact vector')
            if np.linalg.norm(p['impulse'])>0:
                if c['kind']=='object-finger': pairs.update(n for n in c['actors'] if n in step['state']['object_in_fingers'])
                elif c['kind']=='object-plane': pairs.add('@plane')
    return pairs


def trajectory(run, detection):
    rows=run['rows'];inv=run['inventory']
    if not rows: raise ValueError('Empty trajectory')
    steps=[s for r in rows for s in r['substeps']]
    if any(len(r['substeps'])!=inv['sim_freq']//inv['control_freq'] for r in rows): raise ValueError('Missing physics substeps')
    pairs=[active_pairs(s) for s in steps]
    finger_path=plane_path=0.;finger_intervals=plane_intervals=0
    for i in range(1,len(steps)):
        for pair in pairs[i]&pairs[i-1]:
            if pair=='@plane':
                # Horizontal travel while plane contact persists, not automatically sliding (rolling possible).
                plane_path+=float(np.linalg.norm(np.array(steps[i]['state']['object_pose'][:2])-steps[i-1]['state']['object_pose'][:2]));plane_intervals+=1
            else:
                finger_path+=float(np.linalg.norm(np.array(steps[i]['state']['object_in_fingers'][pair][:3])-steps[i-1]['state']['object_in_fingers'][pair][:3]));finger_intervals+=1
    eef_sub=np.array([s['state']['eef_pose'][:3] for s in steps])
    moved=np.where(np.linalg.norm(eef_sub-np.array(run['initial']['eef_pose'][:3]),axis=1)>detection)[0]
    target_sub=np.array([s['target_pose_world'][:3] for s in steps])
    targeted=np.where(np.linalg.norm(target_sub-np.array(run['initial']['eef_pose'][:3]),axis=1)>detection)[0]
    simultaneous=sum(len(p-{'@plane'})>=2 for p in pairs)
    obj=np.array([r['after']['object_pose'][:3] for r in rows])
    return dict(eef=np.array([r['after']['eef_pose'][:3] for r in rows]),
                target=np.array([r['target_pose_world'][:3] for r in rows]),obj=obj,
                acceleration=np.array([r['average_object_acceleration'] for r in rows]),
                finger_path=finger_path if finger_intervals else None,plane_path=plane_path if plane_intervals else None,
                finger_intervals=finger_intervals,plane_intervals=plane_intervals,
                simultaneous_fraction=simultaneous/len(steps),
                motion_onset=(int(moved[0])+1)/inv['sim_freq'] if len(moved) else None,
                target_onset=(int(targeted[0])+1)/inv['sim_freq'] if len(targeted) else None,
                lift=float(max(obj[:,2])-run['initial']['object_pose'][2]),
                free=np.array([not any(c['kind']!='object-plane' for s in r['substeps'] for c in s['contacts']) for r in rows]))


def paired_response(nom, shifted, meta, kind, value, detection):
    for key in ['qpos','qvel','eef_pose','object_pose','object_velocity','object_angular_velocity']:
        np.testing.assert_allclose(nom['initial'][key],shifted['initial'][key],atol=1e-6,rtol=0)
    if nom['inventory']['control_freq']!=shifted['inventory']['control_freq']:raise ValueError('Timing mismatch')
    n=min(len(nom['rows']),len(shifted['rows']))
    a=trajectory(dict(nom,rows=nom['rows'][:n]),detection)
    b=trajectory(dict(shifted,rows=shifted['rows'][:n]),detection)
    issued=np.array([r['commanded'] for r in shifted['rows']]);exe=np.array([r['executed'] for r in shifted['rows']])
    np.testing.assert_allclose(issued[:n],[r['commanded'] for r in nom['rows'][:n]],atol=0,rtol=0)
    if kind=='action_gain':np.testing.assert_allclose(exe[:,:3],np.clip(issued[:,:3]*value,-1,1),atol=1e-6)
    if kind=='action_delay':
        d=int(value);expected=np.zeros_like(issued[:,:3])
        if d==0:expected=issued[:,:3]
        elif len(issued)>d:expected[d:]=issued[:-d,:3]
        np.testing.assert_allclose(exe[:,:3],expected,atol=1e-6)
    np.testing.assert_allclose(exe[:,-1],issued[:,-1],atol=1e-6)
    def rmse(key,mask=None):
        delta=np.sum((a[key][:n]-b[key][:n])**2,axis=1)
        if mask is not None:delta=delta[mask]
        return float(np.sqrt(delta.mean())) if len(delta) else None
    free=a['free'][:n]&b['free'][:n]
    rb=shifted['inventory'].get('physics_readback')
    if kind=='object_mass':
        np.testing.assert_allclose(shifted['inventory']['mass'],nom['inventory']['mass']*value,rtol=1e-5)
        np.testing.assert_allclose(shifted['inventory']['inertia'],np.array(nom['inventory']['inertia'])*value,rtol=1e-5)
    if kind=='contact_friction':
        if not rb:raise ValueError('Missing physical material readback')
        np.testing.assert_allclose(rb['materials'],np.array(rb['nominal_materials'])*[value,value,1],rtol=1e-5)
    result=dict(common_native_steps=n,nominal_native_steps=len(nom['rows']),shifted_native_steps=len(shifted['rows']),
                free_space_eef_rmse_m=rmse('eef',free),free_space_candidate_native_steps=int(free.sum()),
                eef_rmse_m=rmse('eef'),target_rmse_m=rmse('target'),
                effective_native_latency_steps=int(value) if kind=='action_delay' else None,
                effective_native_latency_s=value/shifted['inventory']['control_freq'] if kind=='action_delay' else None,
                nominal_eef_motion_onset_s=a['motion_onset'],shifted_eef_motion_onset_s=b['motion_onset'],
                nominal_target_onset_observed_s=a['target_onset'],shifted_target_onset_observed_s=b['target_onset'],
                eef_motion_onset_delta_s=b['motion_onset']-a['motion_onset'] if None not in (a['motion_onset'],b['motion_onset']) else None,
                object_acceleration_rmse_m_s2=rmse('acceleration'),max_lift_delta_m=b['lift']-a['lift'],
                nominal_max_lift_m=a['lift'],shifted_max_lift_m=b['lift'],
                nominal_dual_finger_contact_fraction=a['simultaneous_fraction'],shifted_dual_finger_contact_fraction=b['simultaneous_fraction'],
                nominal_finger_intervals=a['finger_intervals'],shifted_finger_intervals=b['finger_intervals'],
                nominal_plane_intervals=a['plane_intervals'],shifted_plane_intervals=b['plane_intervals'],
                finger_relative_path_delta_m=b['finger_path']-a['finger_path'] if None not in (a['finger_path'],b['finger_path']) else None,
                plane_relative_path_delta_m=b['plane_path']-a['plane_path'] if None not in (a['plane_path'],b['plane_path']) else None)
    covered = bool(free.any()) if kind in ['action_gain','action_delay'] else a['simultaneous_fraction']>0 and b['simultaneous_fraction']>0 and a['lift']>detection and b['lift']>detection
    if kind=='contact_friction':covered=covered and all(result[k]>0 for k in ['nominal_finger_intervals','shifted_finger_intervals','nominal_plane_intervals','shifted_plane_intervals'])
    return result,covered,rb


def control_evidence(directory, seed):
    sidecar=directory/f'shifted-seed-{seed}.json';tape=sidecar.with_suffix('.npz')
    if not (directory/'severity-complete.json').exists():return None
    metadata=read(directory/'severity-metadata.json');complete=read(directory/'severity-complete.json')
    if metadata['config']['stage3a']['mode']!='frozen' or complete['gradient_updates']!=0 or complete['parameter_drift']!=0:
        raise ValueError('Not an unchanged Frozen evaluation')
    data=read(sidecar)
    if data['evaluation_seed']!=seed:raise ValueError('Frozen seed mismatch')
    with np.load(tape,allow_pickle=False) as arrays:
        if 'object_goal_distance_m' not in arrays:return None
        distances=arrays['object_goal_distance_m']
        initial=data['initial_object_goal_distance_m']
        if not np.isfinite(distances).all() or not np.isfinite(initial):raise ValueError('Invalid task progress')
    return dict(success=float(data['result']['success']),episode_length=data['result']['episode_length'],
                native_steps=data['result']['native_steps'],initial_goal_distance_m=initial,
                final_goal_distance_m=float(distances[-1]),best_goal_distance_m=float(min(distances)),
                goal_distance_reduction_m=initial-float(distances[-1]),
                checkpoint_sha256=metadata['checkpoint_sha256'],environment_versions=metadata['environment_versions'],
                source_sha256=metadata['source_sha256'],
                code_revision=metadata['code_revision'],path=str(directory),sidecar_sha256=digest(sidecar),tape_sha256=digest(tape))


def eligible(rows, review, min_seeds):
    if not review:return False,'PENDING_REVIEW'
    if len(rows)<min_seeds or any(r['evidence_status']!='MEASURED' for r in rows):return False,'INSUFFICIENT_MEASUREMENTS_OR_CONTACT'
    metric=review.get('response_metric'); threshold=review.get('minimum_absolute_response'); rate=review.get('minimum_frozen_success_rate')
    allowed = METRICS[rows[0]['shift_type']]
    if rows[0]['shift_type']=='action_gain': allowed=('free_space_eef_rmse_m',)
    if metric not in allowed or not isinstance(threshold,(int,float)) or not np.isfinite(threshold) or threshold<=0:
        return False,'INVALID_PHYSICAL_CRITERION'
    if not isinstance(rate,(int,float)) or not 0<rate<=1:return False,'INVALID_CONTROL_CRITERION'
    values=[r['measured_physical_response'].get(metric) for r in rows]
    if any(v is None or abs(v)<threshold for v in values):return False,'PHYSICAL_CRITERION_NOT_MET_ON_ALL_SEEDS'
    success=np.mean([r['control_performance']['success'] for r in rows])
    if success<rate:return False,'CONTROL_CRITERION_NOT_MET'
    if set(review.get('feasibility_seed_set',[])) != {r['calibration_seed'] for r in rows}:
        return False,'FEASIBILITY_SEED_SET_MISMATCH'
    if not review.get('rationale') or not review.get('task_feasibility_rationale') or not review.get('threshold_rationale'):return False,'MISSING_RATIONALE'
    evidence=Path(review.get('task_feasibility_evidence',''))
    if not evidence.is_file():return False,'MISSING_TASK_FEASIBILITY_EVIDENCE'
    return True,'SELECTED_MEASURED_CALIBRATION'


def export(manifest_path, output):
    manifest_path=Path(manifest_path).resolve();m=read(manifest_path);validate_manifest(m);base=manifest_path.parent
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];fingerprints={};checkpoint_hashes=set()
    for kind,values in m['candidates'].items():
        for value in values:
            for seed in m['calibration_seeds']:
                r=dict(shift_type=kind,requested_shift_parameter=value,calibration_seed=seed,
                       candidate_origin='exploratory_engineering_heuristic',nominal_physical_parameter=None,
                       effective_simulator_parameter=None,measured_physical_response=None,control_performance=None,
                       task_feasibility_evidence=None,severity_selection_rationale='Pending measurements/review',evidence_status='PENDING_SERVER_MEASUREMENTS')
                directory=base/'probes'/kind/f'seed-{seed}'
                path=directory/f'{kind}-{value:g}.json';nompath=directory/f'{kind}-{NOMINAL[kind]:g}.json'
                if path.exists() and nompath.exists():
                    meta=read(directory/'metadata.json')
                    if meta['seed']!=seed or meta['shift']!=kind or meta.get('tape_clock')!='native_control_step' or value not in meta['values'] or not meta.get('no_learning'):
                        raise ValueError('Probe provenance mismatch')
                    fingerprint=json.dumps([meta['git_head'],meta['source_sha256'],meta['versions']],sort_keys=True)
                    if kind in fingerprints and fingerprints[kind]!=fingerprint:raise ValueError('Mixed source/runtime within shift')
                    fingerprints[kind]=fingerprint
                    nom,shifted=read(nompath),read(path)
                    metrics,covered,rb=paired_response(nom,shifted,meta,kind,value,m['motion_detection_m'])
                    control=control_evidence(frozen_dir(base,m,kind,value),seed)
                    if control:
                        if control['code_revision']!=meta['git_head']:raise ValueError('Frozen/probe revision mismatch')
                        if control['source_sha256']!=meta.get('repository_source_sha256'):raise ValueError('Frozen/probe source byte mismatch; rerun probes with updated collector')
                        if any(control['environment_versions'].get(k)!=v for k,v in meta['versions'].items()):raise ValueError('Frozen/probe runtime mismatch')
                        checkpoint_hashes.add(control['checkpoint_sha256'])
                        if len(checkpoint_hashes)>1:raise ValueError('Mixed Frozen checkpoints')
                    r.update(nominal_physical_parameter=nom['inventory'],effective_simulator_parameter=rb or dict(verified_command_parameter=value,control_freq=shifted['inventory']['control_freq']),
                             measured_physical_response=metrics,control_performance=control,
                             evidence_status='MEASURED' if covered and control else 'PARTIAL',
                             probe_path=str(path),probe_sha256=digest(path),tape_sha256=meta['actions_sha256'])
                rows.append(r)
    chosen={};decisions=[]
    for kind,values in m['candidates'].items():
        for value in values:
            group=[r for r in rows if r['shift_type']==kind and r['requested_shift_parameter']==value]
            review=m['selection_reviews'].get(f'{kind}:{value:g}')
            if review and review.get('task_feasibility_evidence'):
                review=dict(review)
                proof=Path(review['task_feasibility_evidence'])
                review['task_feasibility_evidence']=str((base/proof).resolve() if not proof.is_absolute() else proof)
            ok,status=eligible(group,review,m['minimum_calibration_seeds']) if value!=NOMINAL[kind] else (False,'NOMINAL_REFERENCE')
            if ok:
                proof=Path(review['task_feasibility_evidence']);proof=proof.resolve()
                chosen.setdefault(kind,[]).append(dict(value=value,origin='measured_physical_response_calibration',criteria=review,
                                                     feasibility_sha256=digest(proof),calibration_seeds=m['calibration_seeds']))
            for r in group:
                r['selection_status']=status
                if review:
                    r['severity_selection_rationale']=review.get('rationale','')
                    r['task_feasibility_evidence']=review.get('task_feasibility_evidence')
            decisions.append(f'| {kind} | {value:g} | {status} |')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (output/'physical_severity_calibration.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,sort_keys=True,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    config=dict(protocol=m['protocol'],status='CALIBRATED' if len(chosen)==4 else 'PARTIAL_CALIBRATION' if chosen else 'PENDING_MEASUREMENTS_OR_REVIEW',
                ready_for_benchmark=len(chosen)==4,calibration_seeds=m['calibration_seeds'],final_benchmark_seeds=m['final_benchmark_seeds'],
                selected_severities=chosen,manifest_sha256=digest(manifest_path))
    # JSON is valid YAML 1.2; no additional YAML runtime dependency required.
    dump(output/'calibrated_severity.yaml',config)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,13))
    for axs,(kind,metrics) in zip(axes,METRICS.items()):
        for ax,metric in zip(axs,metrics):
            count=0
            for seed in m['calibration_seeds']:
                points=[(r['requested_shift_parameter'],r['measured_physical_response'].get(metric)) for r in rows if r['shift_type']==kind and r['calibration_seed']==seed and r['measured_physical_response']]
                points=sorted((x,y) for x,y in points if y is not None)
                if points:ax.plot(*zip(*points),'o-',label=f'seed {seed}');count+=len(points)
            ax.set(title=kind,ylabel=metric,xlabel='Native delay steps' if kind=='action_delay' else 'Requested factor')
            if not count:ax.text(.5,.5,'NO SIMULATOR MEASUREMENTS',ha='center',transform=ax.transAxes)
            else:ax.legend(fontsize=7)
    fig.suptitle('Physical response calibration | Separate units per shift | No inferred measurements')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(output/f'physical_response_by_severity.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    text='''# Physical-response severity 선정\n\n기존 PILOT 값은 exploratory engineering heuristic이다. 이 산출물은 기존 config/결과를 변경하지 않는다. 측정 또는 선정 review가 없으면 selected_severities는 비어 있고 benchmark 실행용 확정값이 아니다.\n\n| Shift | Candidate | Selection status |\n|---|---:|---|\n'''+ '\n'.join(decisions)
    text+='\n\n## 측정과 선정 기준\n\nCalibration seeds: '+str(m['calibration_seeds'])+'; final benchmark seeds (미실행): '+str(m['final_benchmark_seeds'])+'''.\n\n- Gain: 접촉 전 후보 구간의 nominal 대비 measured EEF RMSE와 target RMSE를 별도 기록한다. Object 접촉 부재만으로 모든 robot collision 부재를 증명하지는 않는다.\n- Delay: 실제 issued/executed history가 요청 native delay와 일치하는지 검사한다. 운동 onset은 측정 변위 threshold 초과 시점이며 motor causal latency 추정값이 아니다.\n- Mass: native 경계 velocity 차분의 평균 가속도, lift 변화, 동시 양 finger 접촉 fraction을 기록한다.\n- Friction: 지속된 동일 contact pair의 상대 이동 누적량을 finger와 plane으로 분리한다. 이는 deformation/rotation을 포함할 수 있는 relative-motion proxy이며 마찰력 또는 순수 sliding distance로 단정하지 않는다. Contact-free 값은 0이 아닌 missing이며 선정 불가다.\n- Frozen success/episode length/goal-distance progress는 open-loop 물리 probe와 별도 CSV 필드에 저장한다. Reward 합을 task progress로 대체하지 않는다.\n- Curve는 common native prefix를 사용하고 양쪽 길이를 기록한다. 종료 길이가 다르면 같은 관측 window 내 비교만 유효하다.\n- 최소 3개 calibration seed의 완전한 physical+Frozen evidence, shift별 response threshold, 최소 Frozen success, 별도 feasibility 근거와 rationale가 있어야 선정한다. Threshold는 사용자/연구자가 기록하는 연구 기준이며 보편적인 물리 상수가 아니다. 성공률은 소표본 기술통계이며 통계적 유의성을 주장하지 않는다.\n- Final seed는 export/selection에 읽지 않는다. 지정한 final seed가 과거 다른 실험에서 사용됐는지 추가 확인해야 한다.\n'''
    (output/'severity_selection.md').write_text(text)
    dump(output/'calibration_manifest_snapshot.json',m)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    init=sub.add_parser('prepare');init.add_argument('--output',type=Path,required=True);init.add_argument('--checkpoint',default='checkpoints/maniskill2/pick-cube-1.pt')
    init.add_argument('--calibration-set',type=int,default=70);init.add_argument('--final-set',type=int,default=80);init.add_argument('--seeds',type=int,default=3,choices=range(1,11))
    exp=sub.add_parser('export');exp.add_argument('--manifest',type=Path,required=True);exp.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.mode=='prepare':prepare(args)
    else:export(args.manifest,args.output)


if __name__=='__main__':main()
