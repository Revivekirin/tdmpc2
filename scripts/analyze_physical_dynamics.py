"""Reproduce command-level probes and plots; never launches a simulator or training.

Uses the actual transform AST, not a rewritten delay implementation. Physical
curves are explicitly reference calculations, not simulator measurements.
"""
import ast
import csv
import hashlib
import json
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tdmpc2-physical-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/physical_dynamics'
OUT.mkdir(parents=True, exist_ok=True)


def function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


ns = dict(np=np)
shift = ROOT / 'tdmpc2/envs/wrappers/dynamics_shift'
integer = function(shift / 'schedule.py', 'integer', ns)
transform = function(shift / 'action_shift.py', 'transform', ns)
records = []
for delay in range(4):
    obj = SimpleNamespace(kind='action_delay', indices=slice(None, -1),
                          low=np.full(4, -1., dtype=np.float32), high=np.ones(4, dtype=np.float32),
                          queue=deque(), max_delay=3, initial_action=np.zeros(4, dtype=np.float32))
    for t in range(8):
        a = np.array([.2 * (t // 2 + 1), 0, 0, (-1.) ** (t // 2)], dtype=np.float32)
        result = transform(obj, a, delay)
        expected = 0 if t < delay else .2 * ((t-delay)//2 + 1)
        np.testing.assert_allclose(result[0], expected, atol=1e-7)
        assert result[3] == a[3]
        records.append(dict(native_step=t, decision=t//2, delay=delay,
                            command=a.tolist(), executed=result.tolist(), queue_length=len(obj.queue)))
gains = []
for gain in [1., .9, .7, .5, .3]:
    obj.kind = 'action_gain'
    a = np.array([.5, -.2, .1, -1.], dtype=np.float32)
    b = transform(obj, a, gain)
    np.testing.assert_allclose(b[:3], a[:3] * gain)
    assert b[3] == a[3]
    gains.append(dict(gain=gain, commanded=a.tolist(), executed=b.tolist(),
                      reference_ee_delta_m=(.1*b[:3]).tolist(), measured_response=None))
result = dict(evidence='LOCAL_CODE_EXECUTION_ONLY; no simulator', delay=records, gain=gains,
              unknown=['latest_server_revision', 'live_physics_response', 'live_controller_config'])
(OUT/'command_probe.json').write_text(json.dumps(result, indent=2)+'\n')

def save(name):
    plt.savefig(OUT/(name+'.png'), dpi=180, bbox_inches='tight')
    plt.savefig(OUT/(name+'.pdf'), bbox_inches='tight')
    plt.close()

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(12, 4.5)); ax.axis('off')
labels = ['State obs\nEncoder / MPC', 'Issued action\n4 normalized values',
          'Repeat x2\nShift each native call', 'EE delta + gripper\nIK -> joint targets',
          'Drive + passive force\n25 physics substeps', 'Obs / reward\nReplan']
for i, label in enumerate(labels):
    x=.08+i*.168
    ax.text(x,.62,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#e9f0fa',ec='#5277a4'),fontsize=9)
    if i < 5: ax.annotate('',xy=(x+.10,.62),xytext=(x+.065,.62),arrowprops=dict(arrowstyle='->'))
ax.text(.5,.22,'STRUCTURAL DIAGRAM | Runtime reference: physics 500 Hz, native 20 Hz, policy 10 Hz\nMass / friction: reset-time property change; gain / delay: command transformation',ha='center')
ax.set_title('Policy to physics: local code + versioned ManiSkill2 reference')
save('action_pipeline')

fig, axes = plt.subplots(1,2,figsize=(10,4))
axes[0].plot([r['gain'] for r in gains],[r['executed'][0] for r in gains],'o-',label='Executed x command')
axes[0].set(xlabel='Arm command gain',ylabel='Normalized command',title='Actual transform function: input x=0.5')
axes[1].plot([r['gain'] for r in gains],[r['reference_ee_delta_m'][0]*1000 for r in gains],'o-',color='#dc8540')
axes[1].set(xlabel='Arm command gain',ylabel='Target delta (mm)',title='Reference target calculation; NOT measured EEF')
fig.suptitle('Command probe | Measured joint / EEF response unavailable',y=1.04)
fig.tight_layout();save('gain_command_vs_response')

fig, ax = plt.subplots(figsize=(10,3.8))
for d in range(4):
    rows=[r for r in records if r['delay']==d]
    ax.step(range(8),[r['executed'][0] for r in rows],where='post',marker='o',label=f'delay {d}')
ax.set(xlabel='Native step (zero-based)',ylabel='Executed arm x command',title='Actual transform execution: A,A,B,B,C,C,D,D; zero startup')
ax.legend(ncol=4);save('delay_native_step_timeline')

factors=np.array([1,1.5,2,3]); mass=.064000003*factors
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(factors,mass,'o-');axes[0].set(xlabel='Mass factor',ylabel='Mass (kg)',title='Historical mass readback x factor')
axes[1].plot(factors,1/factors,'o-');axes[1].set(xlabel='Mass factor',ylabel='Acceleration / nominal',title='Theory: same net force, a=F/m')
fig.suptitle('REFERENCE CALCULATION | No new acceleration / lift measurements',y=1.04)
fig.tight_layout();save('mass_physical_response')

factors=np.array([1,.7,.5,.3])
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(factors,factors,'o-',label='Object-plane (both scaled)')
ax.plot(factors,(factors+2)/2,'s-',label='Object-finger (finger=2 unchanged)')
ax.set(xlabel='Material factor',ylabel='Effective coefficient (conditional)',title='REFERENCE THEORY ONLY: average combine, object/plane=1')
ax.legend();save('friction_contact_response')

rows=list(csv.DictReader(open(ROOT/'results/stage3b-generalization/severity_sweep/combined-set0-v1/summary.csv')))
fig,axes=plt.subplots(2,2,figsize=(10,7))
for ax,kind in zip(axes.flat,['action_gain','action_delay','object_mass','contact_friction']):
    data=sorted([r for r in rows if r['shift_type']==kind],key=lambda r:float(r['shift_value']))
    ax.plot([float(r['shift_value']) for r in data],[100*float(r['success']) for r in data],'o-')
    ax.set(title=kind,xlabel='Native steps' if kind=='action_delay' else 'Factor',ylabel='Frozen success (%)',ylim=(-5,105))
    for r in data: ax.annotate(f"L={float(r['episode_length']):g}",(float(r['shift_value']),100*float(r['success'])),xytext=(0,12 if float(r['success'])<.15 else -17),textcoords='offset points',ha='center',fontsize=8)
fig.suptitle('EXISTING PILOT OBSERVATIONS | 10 episodes / condition, one seed set\nL = mean policy decisions; success is not physical response')
fig.tight_layout();save('shift_severity_physical_comparison')

paths=[shift/'action_shift.py',shift/'physics_shift.py',shift/'base.py',ROOT/'tdmpc2/envs/maniskill.py',ROOT/'tdmpc2/tdmpc2.py',ROOT/'scripts/run_stage3b.py',ROOT/'scripts/analyze_physical_dynamics.py',ROOT/'scripts/probe_physical_response.py']
paths += list((OUT/'reference_ms2_0_4_1_post0').rglob('*.py'))
paths += [ROOT/'results/stage3b-generalization/severity_sweep/combined-set0-v1/summary.csv',ROOT/'results/stage3c-report-20261108-170105/tables/stage3b_aggregated_results.csv']
(OUT/'source_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
print('PASS: 32 delay commands + 5 gain commands; six PNG/PDF pairs; no simulator measurements.')
