"""Regression checks for clock semantics, physical scope, and stage gates."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/nsgym/scripts'))
sys.path.insert(0,str(ROOT/'tdmpc2'))
from severity_protocol import parser,resolve
from print_severity_commands import commands
from run_severity import prerequisites
from envs.nsgym.severity import SeverityHalfCheetah


@pytest.mark.parametrize('schedule,expected',[
    ('nominal',[1]*12),('fixed',[1.5]*12),('abrupt',[1]*3+[1.5]*9),
    ('gradual',[1]*4+[7/6,4/3]+[1.5]*6),
    ('reversible',[1]*3+[1.5]*6+[1]*3),
    ('repeated',[1]*3+[1.5]*3+[1]*3+[1.5]*3)])
@pytest.mark.parametrize('parameter',['torso_mass','bthigh_damping'])
def test_actual_schedule_readback_and_reset(schedule,expected,parameter):
    e=SeverityHalfCheetah(parameter=parameter,severity=1.5,schedule=schedule,
                         onset=3,duration=3,restore=9,period=3,max_episode_steps=12)
    try:
        for _ in range(2):
            e.reset(seed=42);nominal=e.nominal_value;observed=[]
            for step in range(12):
                obs,r,term,trunc,info=e.step(np.full(6,.1))
                observed.append(e.readback()/nominal)
                assert obs.shape==(17,) and not term and trunc==(step==11)
            np.testing.assert_allclose(observed,expected,atol=1e-12,rtol=0)
    finally:e.close()


def test_generated_commands_use_real_cli_and_disjoint_seeds(tmp_path):
    for phase,argv in commands(tmp_path):
        c=resolve(parser().parse_args(argv))
        assert c['expected_checkpoint_sha256'].startswith('f44afd')
        assert c['optimizer_updates']==0
    with pytest.raises(ValueError,match='Seeds'):
        resolve(parser().parse_args(['frozen','--output',str(tmp_path),'--seeds','41000001']))


def test_large_frozen_gate_rejects_smoke(tmp_path):
    gate=tmp_path/'gate';gate.mkdir()
    (gate/'manifest.json').write_text(json.dumps(dict(stage='physical',smoke=True)))
    (gate/'completion.json').write_text(json.dumps(dict(status='PASS')))
    c=resolve(parser().parse_args(['frozen','--output',str(tmp_path/'out'),'--gate',str(gate)]))
    with pytest.raises(ValueError,match='non-smoke'):prerequisites(c)


def test_optional_large_gate(tmp_path):
    c=resolve(parser().parse_args(['nonstationary','--schedule','gradual','--output',str(tmp_path)]))
    with pytest.raises(ValueError,match='S3-S5'):prerequisites(c)


def test_disabled_legacy_gain_does_not_leak_into_nominal():
    from omegaconf import OmegaConf
    import torch
    from envs.nsgym import make_env
    envs=[]
    try:
        for gain in (.7,1.):
            cfg=OmegaConf.create(dict(task='nsgym-halfcheetah',seed=42,obs='state',multitask=False,
                dynamics_shift=dict(enabled=False,shift_gain=gain),nsgym=dict(mode='nominal',mass_scale=1.,max_episode_steps=12)))
            envs.append(make_env(cfg))
        torch.testing.assert_close(envs[0].reset(),envs[1].reset(),rtol=0,atol=0)
        for _ in range(12):
            a,b=[e.step(torch.full((6,),.1)) for e in envs]
            torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
            torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
    finally:
        for e in envs:e.close()


def test_latent_windows_and_analysis_with_explicit_synthetic_fixture(tmp_path,monkeypatch):
    # This deterministic CPU fixture tests indexing only, not a trained checkpoint.
    import types
    import csv
    import torch
    import severity_runtime as runtime
    from severity_analysis import summarize
    source=tmp_path/'fixture';source.mkdir();out=tmp_path/'latent';out.mkdir()
    c=resolve(parser().parse_args(['latent','--input',str(source),'--output',str(out)]))
    (source/'manifest.json').write_text(json.dumps(dict(stage='physical',expected_checkpoint_sha256=c['expected_checkpoint_sha256'])))
    entries=[]
    for k in (1.,1.5):
        obs=np.zeros((13,17),dtype=np.float32);obs[:,:2]=np.arange(13)[:,None]*.1*k
        action=np.full((12,6),.1,dtype=np.float32)
        file=f'synthetic-{k}.npz'
        np.savez(source/file,observations=obs,actions=action,rewards=action.sum(1),qvel=np.zeros((13,9)),actual_parameter=np.full(12,k))
        entries.append(dict(file=file,seed=41000001,source='SYNTHETIC_TEST_ONLY',source_step=0,
            condition=dict(family='mass',parameter='torso_mass',severity=k,schedule='fixed')))
    (source/'dataset_index.json').write_text(json.dumps(entries))
    model=types.SimpleNamespace(encode=lambda o,t:o[:,:2],next=lambda z,a,t:z+a[:,:2],
                                reward=lambda z,a,t:a.sum(-1,keepdim=True))
    fixture=types.SimpleNamespace(agent=types.SimpleNamespace(model=model,device='cpu',cfg=types.SimpleNamespace(num_bins=0)),
                                 verify=lambda out:None)
    monkeypatch.setattr(runtime,'FrozenAgent',lambda config:fixture)
    result=runtime.latent(c,out)
    assert result['prediction_windows']==2*(12+10+3)
    rows=list(csv.DictReader((out/'latent_errors.csv').open()))
    one=[float(x['latent_mse']) for x in rows if float(x['severity'])==1.5 and x['horizon']=='1']
    three=[float(x['latent_mse']) for x in rows if float(x['severity'])==1.5 and x['horizon']=='3']
    np.testing.assert_allclose(one,.0025,atol=1e-7)
    np.testing.assert_allclose(three,.0225,atol=1e-7)
    assert all(float(x['reward_squared_error'])<1e-12 for x in rows)
    (out/'manifest.json').write_text(json.dumps(dict(stage='latent')))
    summary=tmp_path/'summary';summary.mkdir()
    summarize(dict(input=str(out)),summary)
    assert (summary/'latent_error_by_severity.csv').exists()
    assert (summary/'latent_error_aligned.png').stat().st_size>0


def test_episode_logging_and_summary_with_test_policy(tmp_path,monkeypatch):
    import types
    import csv
    import torch
    import severity_runtime as runtime
    from severity_analysis import summarize
    out=tmp_path/'episodes';out.mkdir()
    c=resolve(parser().parse_args(['frozen','--smoke','--output',str(out)]))
    (out/'manifest.json').write_text(json.dumps(c))
    # Tests measurement plumbing without claiming pretrained-policy performance.
    fixture=types.SimpleNamespace(agent=types.SimpleNamespace(act=lambda *a,**kw:torch.full((6,),.1)),verify=lambda path:None)
    monkeypatch.setattr(runtime,'FrozenAgent',lambda config:fixture)
    result=runtime.collect(c,out)
    assert result['transitions']==48 and result['episodes']==4
    entries=json.loads((out/'dataset_index.json').read_text())
    for entry in entries:
        d=np.load(out/entry['file']);assert d['actual_parameter'].shape==(12,)
        steps=list(csv.DictReader((out/entry['steps']).open()))
        assert steps[0]['aligned_step']=='0' and steps[0]['shift_onset']=='0'
        assert steps[-1]['truncated']=='True'
    summary=tmp_path/'summary';summary.mkdir()
    summarize(dict(input=str(out)),summary)
    rows=list(csv.DictReader((summary/'frozen_generalization.csv').open()))
    assert len(rows)==4 and all(int(r['truncated_count'])==1 for r in rows)
    assert (summary/'shift_aligned_timeseries.png').stat().st_size>0
