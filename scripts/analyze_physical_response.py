"""Plot numeric simulator probe outputs, separately from reference calculations.

Requires one or more output directories from probe_physical_response.py.
Does not launch a simulator or infer missing physical measurements.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tdmpc2-physical-response-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(run):
    rows=run['rows']; inv=run['inventory']
    if not rows: raise ValueError('Empty simulator trajectory')
    counts={k:0 for k in ['object-finger','object-plane','object-other']}
    impulses={k:0. for k in counts}
    contact_steps=[]
    for row in rows:
        if len(row['substeps']) != inv['sim_freq']//inv['control_freq']:
            raise ValueError('Missing physics substeps')
        fingers=set()
        for step in row['substeps']:
            for contact in step['contacts']:
                kind=contact['kind']; counts[kind]+=len(contact['points'])
                for point in contact['points']:
                    for key in ['position','normal','impulse']:
                        array=np.asarray(point[key])
                        if array.shape!=(3,) or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
                            raise ValueError(f'Non-numeric {key}')
                    magnitude=float(np.linalg.norm(point['impulse']))
                    impulses[kind]+=magnitude
                    if kind=='object-finger' and magnitude>0:
                        fingers.update(n for n in contact['actors'] if 'finger' in n)
        contact_steps.append(len(fingers))
    height=np.array([r['after']['object_pose'][2] for r in rows])
    eef=np.array([r['after']['eef_pose'][:3] for r in rows])
    target=np.array([r['target_pose_world'][:3] for r in rows])
    return dict(native_steps=len(rows),contact_point_samples=counts,
                sum_contact_impulse_magnitudes_Ns=impulses,
                native_steps_with_both_fingers=sum(n>=2 for n in contact_steps),
                maximum_height_increase_m=float(max(height)-run['initial']['object_pose'][2]),
                final_height_increase_m=float(height[-1]-run['initial']['object_pose'][2]),
                mean_target_tracking_error_m=float(np.linalg.norm(target-eef,axis=1).mean()),
                mass_kg=inv['mass'],inertia_kgm2=inv['inertia'],
                interpretation='Contact samples are not independent events. Impulse magnitude sums are not net force. Both fingers may contact at different substeps.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    groups={}; summaries={}; metadata={}
    for directory in args.runs:
        meta=json.loads((directory/'metadata.json').read_text())
        if meta.get('tape_clock')!='native_control_step': raise ValueError('Unverified action tape clock')
        kind=meta['shift']
        if kind in groups: raise ValueError('Use one run directory per shift to avoid pooling runs')
        metadata[kind]=meta;groups[kind]=[];summaries[kind]={}
        for value in meta['values']:
            path=directory/f'{kind}-{value:g}.json'
            run=json.loads(path.read_text())
            summaries[kind][str(value)]=summarize(run)
            groups[kind].append((value,run))
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
    for kind,runs in groups.items():
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for value,run in runs:
            rows=run['rows']; dt=1/run['inventory']['control_freq']
            t=(np.arange(len(rows))+1)*dt
            eef=np.array([r['after']['eef_pose'][:3] for r in rows])
            target=np.array([r['target_pose_world'][:3] for r in rows])
            height=np.array([r['after']['object_pose'][2] for r in rows])
            label=f'{value:g}'
            axes[0,0].plot(t,np.linalg.norm(eef-np.array(run['initial']['eef_pose'][:3]),axis=1),label=label)
            axes[0,1].plot(t,np.linalg.norm(target-eef,axis=1),label=label)
            if kind in ['action_gain','action_delay']:
                axes[1,0].step(t,[r['commanded'][0] for r in rows],where='post',linestyle='--',alpha=.5,label=f'{label} issued x')
                axes[1,0].step(t,[r['executed'][0] for r in rows],where='post',label=f'{label} executed x')
                axes[1,1].plot(t,height-run['initial']['object_pose'][2],label=label)
            else:
                axes[1,0].plot(t,height-run['initial']['object_pose'][2],label=label)
                for category,style in [('object-finger','-'),('object-plane','--')]:
                    impulse=[sum(np.linalg.norm(p['impulse']) for s in r['substeps'] for c in s['contacts'] if c['kind']==category for p in c['points']) for r in rows]
                    axes[1,1].plot(t,impulse,linestyle=style,label=f'{label} {category}')
        axes[0,0].set(ylabel='Measured EEF displacement (m)')
        axes[0,1].set(ylabel='World target - measured EEF norm (m)')
        axes[1,0].set(ylabel='Normalized x command' if kind in ['action_gain','action_delay'] else 'Measured object height change (m)')
        axes[1,1].set(ylabel='Measured object height change (m)' if kind in ['action_gain','action_delay'] else 'Sum of contact impulse magnitudes (N s)')
        for ax in axes.flat: ax.set_xlabel('Simulation time (s)');ax.legend(fontsize=7)
        fig.suptitle(f'SIMULATOR MEASUREMENTS: {kind}\nPrescribed native action tape; target is not measured motion')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(args.output/f'measured_{kind}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
        if kind=='contact_friction':
            fig,ax=plt.subplots(figsize=(9,4))
            for value,run in runs:
                rows=run['rows'];t=(np.arange(len(rows))+1)/run['inventory']['control_freq']
                for finger,initial in run['initial']['object_in_fingers'].items():
                    relative=np.array([r['after']['object_in_fingers'][finger][:3] for r in rows])
                    ax.plot(t,np.linalg.norm(relative-np.array(initial[:3]),axis=1),label=f'{value:g} {finger}')
            ax.set(xlabel='Simulation time (s)',ylabel='Relative position change from reset (m)',title='MEASURED relative motion; not proof of slip without sustained contact')
            ax.legend(fontsize=7);fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(args.output/f'measured_contact_relative_motion.{ext}',dpi=180,bbox_inches='tight')
            plt.close(fig)
    (args.output/'measurement_summary.json').write_text(json.dumps(dict(measurements=summaries,provenance=metadata),indent=2,allow_nan=False)+'\n')
    print(args.output)


if __name__=='__main__': main()
