"""Fixed, experiment-only Reacher interventions using official NS-Gym accessors."""
import numpy as np
import mujoco
from ns_gym.schedulers import DiscreteScheduler
from ns_gym.update_functions import StepWiseUpdate
from ns_gym.wrappers import MujocoWrapper
from ns_gym.wrappers.mujoco_env import param_look_up
from .reacher import Reacher

PARAMETERS = {'D0': 'joint0_damping', 'D1': 'joint1_damping',
              'M0': 'body0_mass', 'M1': 'body1_mass'}
STATE = mujoco.mjtState.mjSTATE_INTEGRATION

class SeverityReacher(Reacher):
    def __init__(self, parameter='joint0_damping', severity=1., **kwargs):
        if parameter not in PARAMETERS.values() or not np.isfinite(severity) or severity <= 0:
            raise ValueError('Unsupported parameter or nonpositive multiplier')
        super().__init__(**kwargs)
        self.parameter, self.severity = parameter, float(severity)
        self.getter, self.setter = param_look_up('ReacherEnv', parameter)[0]
        self.nominal_value = float(self.getter(self.unwrapped))
        self.target_value = self.nominal_value * severity
        if parameter.endswith('damping'):
            joint = self.unwrapped.model.joint(parameter.removesuffix('_damping'))
            assert joint.id == self.unwrapped.model.jnt_dofadr[joint.id]
        self.nominal_inertia = self.unwrapped.model.body_inertia.copy()
        # Reassert the same fixed target at t=0 via the official scheduler.
        # The value is also applied eagerly, before the first reset observation.
        changes = {parameter: StepWiseUpdate(DiscreteScheduler({0}), [self.target_value])}
        self.env = MujocoWrapper(self.env.env, changes, change_notification=False,
                                 delta_change_notification=False, scalar_reward=True,
                                 persistent_params=True)
        self.setter(self.unwrapped, self.target_value)
        self.env._dependency_resolver()
        self.check()

    def readback(self):
        return float(self.getter(self.unwrapped))

    def check(self):
        np.testing.assert_allclose(self.readback(), self.target_value, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(self.unwrapped.model.body_inertia, self.nominal_inertia)

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.check()
        return result

    def step(self, action):
        result = super().step(action)
        self.check()
        if not np.isfinite(result[0]).all() or not np.isfinite(result[1]):
            raise FloatingPointError('Nonfinite Reacher transition')
        return result

    def integration_state(self):
        raw = self.unwrapped
        x = np.empty(mujoco.mj_stateSize(raw.model, STATE))
        mujoco.mj_getState(raw.model, raw.data, x, STATE)
        return x

    def restore(self, x):
        raw = self.unwrapped
        mujoco.mj_setState(raw.model, raw.data, x, STATE)
        mujoco.mj_forward(raw.model, raw.data)
        np.testing.assert_allclose(self.integration_state(), x, rtol=0, atol=1e-12)
        return raw._get_obs().copy()

    def physical(self):
        r = self.unwrapped
        tip = r.get_body_com('fingertip').copy()
        target = r.get_body_com('target').copy()
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(r.model, r.data, mujoco.mjtObj.mjOBJ_BODY,
                                r.model.body('fingertip').id, velocity, 0)
        return dict(q=r.data.qpos[:2].copy(), qvel=r.data.qvel[:2].copy(),
                    qacc=r.data.qacc[:2].copy(), tip=tip, tip_velocity=velocity[3:].copy(),
                    target=target, distance=float(np.linalg.norm(tip-target)),
                    force=r.data.actuator_force.copy(), time=float(r.data.time),
                    ctrl=r.data.ctrl.copy())
