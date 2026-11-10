"""Real critic/device regression: no CUDA or full agent is required."""
from copy import deepcopy
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_utils import ROOT
from training_support import config_from_file,build,tree_hash
from common.layers import Ensemble
from common.world_model import WorldModel


def test_ensemble_values_gradients_and_optimizer_match_independent_members():
    torch.manual_seed(41)
    members=[torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,1)) for _ in range(3)]
    references=deepcopy(members)
    ensemble=Ensemble(members).to('cpu')
    assert not any(p.is_meta for p in ensemble.parameters())
    x=torch.randn(7,3)
    expected=torch.stack([m(x) for m in references])
    actual=ensemble(x)
    torch.testing.assert_close(actual,expected)
    actual.square().mean().backward();expected.square().mean().backward()
    for name,_ in references[0].named_parameters():
        reference_grad=torch.stack([dict(m.named_parameters())[name].grad for m in references])
        torch.testing.assert_close(ensemble.params[tuple(name.split('.'))].grad,reference_grad)
    opt=torch.optim.Adam(ensemble.parameters(),lr=1e-3)
    reference_opt=torch.optim.Adam([p for m in references for p in m.parameters()],lr=1e-3)
    opt.step();reference_opt.step()
    torch.testing.assert_close(ensemble(x),torch.stack([m(x) for m in references]))


def test_template_dropout_mode_and_serialization():
    ensemble=Ensemble([torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.Dropout(.5)) for _ in range(2)])
    ensemble.eval();assert not ensemble.module.training
    x=torch.ones(10,3)
    torch.testing.assert_close(ensemble(x),ensemble(x),rtol=0,atol=0)
    assert not any('.module.' in key for key in ensemble.state_dict())
    assert not any(v.is_meta for v in ensemble.state_dict().values() if isinstance(v,torch.Tensor))
    ensemble.train();assert ensemble.module.training


def test_world_model_transfer_targets_and_checkpoint_roundtrip(tmp_path):
    cfg,env=build(config_from_file(ROOT/'experiments/nsgym/configs/train_nominal.yaml',True))
    try:
        model=WorldModel(cfg)
        original=tree_hash(model.state_dict())
        model.to('cpu')
        assert tree_hash(model.state_dict())==original
        assert not any(p.is_meta for p in model.parameters())
        assert not any(b.is_meta for b in model.buffers())
        model.train();assert model._Qs.module.training and not model._target_Qs.module.training
        q=model._Qs.params['0','weight'];detached=model._detach_Qs_params['0','weight'];target=model._target_Qs_params['0','weight']
        assert q.data_ptr()==detached.data_ptr() and q.data_ptr()!=target.data_ptr()
        old=target.clone()
        with torch.no_grad():q.add_(.25)
        torch.testing.assert_close(target,old)
        expected=old.lerp(detached,cfg.tau)
        model.soft_update_target_Q();torch.testing.assert_close(target,expected)
        torch.save(model.state_dict(),tmp_path/'model.pt')
        fresh=WorldModel(cfg).to('cpu')
        fresh.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=False))
        assert tree_hash(fresh.state_dict())==tree_hash(model.state_dict())
        model.eval();fresh.eval()
        z=torch.randn(2,cfg.latent_dim);action=torch.zeros(2,cfg.action_dim)
        torch.testing.assert_close(model.Q(z,action,None,return_type='all'),fresh.Q(z,action,None,return_type='all'))
    finally:env.close()
