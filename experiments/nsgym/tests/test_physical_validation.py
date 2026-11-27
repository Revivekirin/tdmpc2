import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_physics import run, state, restore, model_arrays
from envs.nsgym import HalfCheetah


def test_physical_pipeline_reproduces(tmp_path):
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/physical_validation.json').read_text())
    config['horizon']=6
    a=run(config,tmp_path/'a'); b=run(config,tmp_path/'b')
    aa=np.load(a/'raw.npz');bb=np.load(b/'raw.npz')
    for key in aa.files:np.testing.assert_array_equal(aa[key],bb[key])
    assert json.loads((a/'checks.json').read_text())['status']=='PASS'
    x=aa['physical']; assert np.max(np.abs(x[:,:,0,1:]-x[:,:,1,1:]))>1e-8
    with pytest.raises(FileExistsError):run(config,a)


def test_state_restore_and_model_inventory():
    env=HalfCheetah()
    try:
        env.reset(seed=9);snapshot=state(env)
        env.step(np.ones(6)*.2);restore(env,snapshot,1e-10)
        np.testing.assert_allclose(state(env),snapshot,atol=1e-10,rtol=0)
        assert {'dof_damping','geom_friction','actuator_gear','body_inertia'}<=model_arrays(env).keys()
    finally:env.close()
