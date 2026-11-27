"""Run in the legacy Gymnasium 0.29.1 environment; report optional dependencies."""
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tdmpc2'))
import gymnasium as gym
import torch
from omegaconf import OmegaConf
from envs import make_env

report = {'gymnasium': gym.__version__, 'imports': {}, 'tasks': {}}
for name in ['common.stage3a','common.decision_ttt','common.shift_protocol',
             'adapt_shift','envs.wrappers.dynamics_shift','envs.maniskill',
             'envs.dmcontrol','envs.metaworld','envs.myosuite','envs.mujoco']:
    try:
        importlib.import_module(name)
        report['imports'][name] = 'PASS'
    except Exception as exc:
        report['imports'][name] = f'UNAVAILABLE: {type(exc).__name__}: {exc}'
for task in ['mujoco-halfcheetah','mujoco-walker']:
    env = None
    try:
        cfg = OmegaConf.create(dict(task=task, obs='state', multitask=False, seed=17))
        env = make_env(cfg)
        obs = env.reset()
        result = env.step(torch.zeros(cfg.action_dim))
        assert torch.isfinite(obs).all() and torch.isfinite(result[0]).all()
        report['tasks'][task] = dict(status='PASS', shape=list(obs.shape), reward=float(result[1]))
    except Exception as exc:
        report['tasks'][task] = f'FAIL: {type(exc).__name__}: {exc}'
    finally:
        if env is not None:
            env.close()
print(json.dumps(report, indent=2))
if any(isinstance(value, str) and value.startswith('FAIL') for value in report['tasks'].values()):
    sys.exit(1)
if any(report['imports'][name] != 'PASS' for name in
       ['common.stage3a','common.decision_ttt','common.shift_protocol','adapt_shift',
        'envs.wrappers.dynamics_shift']):
    sys.exit(1)
