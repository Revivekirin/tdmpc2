import json
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_utils import ROOT
from training_support import config_from_file,validate_config,require_cuda,evaluate


def test_config_preserves_learning_defaults():
    c=config_from_file(ROOT/'experiments/nsgym/configs/train_nominal.yaml')
    assert c['rho']==.5 and c['discount_max']==.995 and c['lr']==3e-4
    assert c['nsgym']['mode']=='nominal' and c['seed_steps']==10000
    c['nsgym']['mass_scale']=1.3
    with pytest.raises(ValueError):validate_config(c)


def test_cuda_requirement_is_explicit():
    with pytest.raises(ValueError):require_cuda('cpu')
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match='BLOCKED'):require_cuda('cuda:0')


def test_random_reference_reproducible():
    c=config_from_file(ROOT/'experiments/nsgym/configs/train_nominal.yaml',smoke=True)
    a,_=evaluate(None,c,c['eval_seeds'],random_actions=True)
    b,_=evaluate(None,c,c['eval_seeds'],random_actions=True)
    assert a==b and len(a)==2 and a[0]['episode_length']==8


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real unchanged TD-MPC2 agent and replay require CUDA')
def test_real_training_smoke_and_frozen_checkpoint(tmp_path):
    from train_nominal import train
    from evaluate_frozen import run
    c=config_from_file(ROOT/'experiments/nsgym/configs/train_nominal.yaml',smoke=True)
    checkpoint=train(c,tmp_path/'train')
    completion=json.loads((tmp_path/'train/completion.json').read_text())
    assert completion['gradient_updates']>0 and completion['environment_transitions']==35
    config=dict(evaluation_seeds=[20000001,20000002],mass_scales=[.7,1.,1.3],integration_only=True,bootstrap_seed=1,bootstrap_repeats=100)
    run(checkpoint,config,tmp_path/'frozen')
    assert json.loads((tmp_path/'frozen/integrity.json').read_text())['status']=='PASS'


def test_real_world_model_cpu_forward_compatibility():
    # This is a real forward pass, not a substitute for the CUDA optimizer smoke.
    from training_support import build
    from common.world_model import WorldModel
    c=config_from_file(ROOT/'experiments/nsgym/configs/train_nominal.yaml',smoke=True)
    cfg,env=build(c)
    try:
        model=WorldModel(cfg)
        observation=env.reset().unsqueeze(0)
        with torch.no_grad():
            latent=model.encode(observation,None)
            action,_=model.pi(latent,None)
            nxt=model.next(latent,action,None)
            reward=model.reward(latent,action,None)
        assert nxt.shape==(1,cfg.latent_dim)
        assert all(torch.isfinite(x).all() for x in [latent,action,nxt,reward])
    finally:env.close()
