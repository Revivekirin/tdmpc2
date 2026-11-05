"""CPU continuation/integration smoke; no ManiSkill performance claims."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from hydra import compose, initialize_config_dir
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_stage3a import Model, settings, data
from common.stage3a import EpisodeAdapter, equal_state
from common.stage3b import run, resumable_repeat
from common.benchmark_config import normalize
from benchmark_imports import package
from benchmark_shift import logging
from test_shift_benchmark import config
from test_shift_state_and_logging import Controller
from adapt_shift import file_sha256
from analyze_stage3b import analyze, normalized_auc


class Native:
    def __init__(self):
        self.action_space=SimpleNamespace(shape=(2,),low=np.array([-1.,-1.]),high=np.array([1.,1.]))
        self.observation_space=SimpleNamespace(shape=(2,))
        self.agent=SimpleNamespace(controller=Controller())
        self._main_rng=np.random.RandomState(1)
        self._episode_rng=np.random.RandomState(2)
        self._elapsed_steps=0
    @property
    def unwrapped(self): return self
    def reset(self,seed=0):
        self.state=np.array([.1,.2],dtype=np.float32)
        self._elapsed_steps=0
        return self.state.copy()
    def step(self,action):
        self.state += np.array([action[0],-.2*action[0]],dtype=np.float32)
        self._elapsed_steps+=1
        return self.state.copy(),1.,self._elapsed_steps==3,{'success':self._elapsed_steps==3}
    def get_state(self): return self.state.copy()
    def set_state(self,state): self.state=state.copy()
    def close(self): pass


class ManiSkillWrapper:
    def __init__(self,shift): self.env=shift
    def reset(self,**kwargs): return self.env.reset(**kwargs)
    def step(self,action): return resumable_repeat(self,action)
    def native_checkpoint(self,record): pass
    def close(self): self.env.close()


class Tensor:
    def __init__(self,env): self.env=env
    def reset(self,**kwargs): return torch.tensor(self.env.reset(**kwargs))
    def step(self,action):
        obs,reward,done,info=self.env.step(action.numpy())
        return torch.tensor(obs),torch.tensor(reward),done,info
    def close(self): self.env.close()


def factory(cfg):
    shift_cfg=config('action_delay','fixed',0,1)
    shift_cfg['schedule']['type']=cfg.schedule['type']
    return Tensor(ManiSkillWrapper(package.ActionShiftWrapper(Native(),shift_cfg)))


class Agent:
    device=torch.device('cpu')
    def __init__(self,model):
        self.model=model
        self._prev_mean=torch.zeros(2,1)
    def act(self,obs,**kwargs):
        self._prev_mean.add_(.1)
        self.model._Qs(torch.ones(1,2))
        return torch.tensor([.2,0.])


class Stage3BTests(unittest.TestCase):
    def test_hydra_shifts(self):
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'tdmpc2'),version_base=None):
            for name,mag in [('action_delay',1),('action_gain',.7),('object_mass',1.5),('contact_friction',.5)]:
                cfg=normalize(compose(config_name='stage3b',overrides=[
                    f'configs/dynamics_shift@dynamics_shift={name}','configs/schedule@schedule=fixed',f'dynamics_shift.parameter.shifted={mag}']))
                self.assertEqual(cfg.dynamics_shift.parameter.type,name)
                self.assertEqual(cfg.dynamics_shift.parameter.shifted,mag)
                self.assertEqual(list(cfg.stage3a.native_budgets),[800,1600,3200,6400,12800])

    def test_continuous_and_resumed_mid_repeat_and_terminal(self):
        torch.set_num_threads(1)
        for mode in ('frozen','dynamics_only'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                tmp=Path(tmp)
                torch.manual_seed(6)
                model=Model(); model._dynamics[0]=torch.nn.Linear(4,2)
                initial=model.state_dict()
                s=settings(mode)
                s.update(batch_size=1,warmup=1,updates_per_episode=2,seed_set=0,
                         native_budgets=[1,3,5,9],fixed_data_seeds=[30,31],fixed_data_path=str(tmp/'fixed.pt'))
                transitions=[(o,torch.cat([a,torch.zeros(1)]),n) for o,a,n in data(2)]
                dataset=dict(zip(('observation','action','next_observation'),[torch.stack(x) for x in zip(*transitions)]))
                dataset.update(episode_ids=torch.tensor([0,1]),episodes=[{},{}],
                               collection_native_steps=4,collection_policy_decisions=2)
                fixed=tmp/'fixed.pt'
                torch.save(dataset,fixed)
                def execute(folder,resume=None):
                    model=Model(); model._dynamics[0]=torch.nn.Linear(4,2); model.load_state_dict(initial)
                    agent=Agent(model)
                    adapter=EpisodeAdapter(model,s,20,20)
                    cfg=SimpleNamespace(stage3a=s,stage3b={'resume':resume,'run_id':'SYNTHETIC'},seed=1,
                                        schedule={'type':'fixed'},horizon=2)
                    resolved={'stage3a':s,'stage3b':cfg.stage3b,'task':'fixture',
                        'schedule':cfg.schedule,'dynamics_shift':{'parameter':{'type':'action_delay','shifted':1}}}
                    env=factory(cfg)
                    with patch.dict(sys.modules,{'envs.wrappers.dynamics_shift.logging':logging}):
                        run(agent,adapter,env,cfg,factory,deepcopy(model._dynamics),'fixture',dataset,
                            file_sha256(fixed),fixed,tmp/folder,resolved)
                    return adapter,tmp/folder/'set-0'/mode/'1'
                full,full_path=execute('full')
                rows=[json.loads(x) for x in (full_path/'evaluations.jsonl').read_text().splitlines()]
                self.assertEqual([r['native_budget'] for r in rows],[0,1,3,5,9])
                # Native 3 terminates an episode, but its update is still pending.
                self.assertEqual(rows[2]['adaptation_episodes'],0)
                self.assertEqual(rows[2]['gradient_update_steps'],0)
                self.assertEqual(rows[3]['adaptation_episodes'],1)
                for budget in (1,3,5):
                    resumed,path=execute(f'resume{budget}',str(full_path/f'resume-budget-{budget}.pt'))
                    a,b=full.state_dict(),resumed.state_dict()
                    a.pop('update_compute_time'); b.pop('update_compute_time')
                    self.assertTrue(equal_state(a,b))
                    self.assertEqual(json.loads((path/'complete.json').read_text())['adaptation_native_steps'],9)
                self.assertEqual(full.gradient_steps,0 if mode=='frozen' else 4)
                self.assertEqual(full.transitions,5) # Last terminal decision remains pending.
                self.assertEqual(full.native_steps,8)
                self.assertEqual(full.episodes,2)
                self.assertTrue(full.in_episode)
                with self.assertRaises(FileExistsError): execute('full')

    def test_auc_uses_nonuniform_spacing(self):
        self.assertAlmostEqual(normalized_auc([0,1,4],[0,1,1]),.875)
        self.assertNotAlmostEqual(normalized_auc([0,1,4],[0,1,1]),2/3)


if __name__=='__main__': unittest.main()
