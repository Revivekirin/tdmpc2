"""Opt-in real CUDA/checkpoint/ManiSkill equivalence test (separate from CPU units).
Set TD_MPC2_MECHANISM_CHECKPOINTS to the directory of the three official files.
The five-/fifty-episode condition matrix lives in evaluate_mechanism_suite.py.
"""
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import torch
from test_planner_instrumentation import ROOT, oracle


@unittest.skipUnless(os.getenv('TD_MPC2_MECHANISM_CHECKPOINTS') and torch.cuda.is_available(),
                     'Requires opt-in official checkpoint directory and CUDA/ManiSkill runtime')
class CheckpointIntegrationTests(unittest.TestCase):
    def test_official_planner_equivalence(self):
        from hydra import compose, initialize_config_dir
        from common.benchmark_config import normalize, cli_aliases
        from common.parser import parse_cfg
        from common.seed import set_seed
        from common.mechanism_analysis import model_checksum
        from common.planner_trace import PlannerTrace
        from envs import make_env
        from tdmpc2 import TDMPC2
        checkpoints=Path(os.environ['TD_MPC2_MECHANISM_CHECKPOINTS'])
        for task, filename in [('pick-cube','pick-cube-1.pt'),('pick-cube','pick-cube-3.pt'),('stack-cube','stack-cube-3.pt')]:
            with self.subTest(checkpoint=filename):
                overrides=cli_aliases([f'task={task}',f'checkpoint={checkpoints/filename}', 'model_size=5',
                    'compile=false','episodic=false','analysis=frozen_mechanism',
                    'dynamics_shift=action_delay','schedule=fixed','dynamics_shift.parameter.shifted=2'])
                with initialize_config_dir(config_dir=str(ROOT/'tdmpc2'),version_base=None):
                    cfg=normalize(compose(config_name='config',overrides=overrides),overrides)
                with patch('hydra.utils.get_original_cwd', return_value=str(ROOT)):
                    cfg=parse_cfg(cfg)
                set_seed(1)
                env=make_env(cfg)
                try:
                    agent=TDMPC2(cfg); agent.load(cfg.checkpoint)
                    original=oracle.OriginalPlanner()
                    for key in ('cfg','model','device','discount'):
                        setattr(original,key,getattr(agent,key))
                    original._prev_mean=agent._prev_mean.clone()
                    original.plan=original._plan
                    obs=env.reset(seed=1)
                    digest=model_checksum(agent.model)
                    for decision in range(3):
                        set_seed(100+decision)
                        expected=original.act(obs,t0=decision==0)
                        cpu=torch.random.get_rng_state().clone()
                        cuda=[s.clone() for s in torch.cuda.get_rng_state_all()]
                        set_seed(100+decision)
                        agent._mechanism_trace=PlannerTrace()
                        actual=agent.act(obs,t0=decision==0)
                        self.assertTrue(torch.equal(expected,actual))
                        self.assertTrue(torch.equal(original._prev_mean,agent._prev_mean))
                        self.assertTrue(torch.equal(cpu,torch.random.get_rng_state()))
                        self.assertTrue(all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))
                        obs,_,done,_=env.step(actual)
                        if done: break
                    self.assertEqual(digest,model_checksum(agent.model))
                finally:
                    env.close()
