import gymnasium as gym
import gym as legacy_gym
import numpy as np
from envs.wrappers.timeout import Timeout
from envs.wrappers.dynamics_shift import DynamicsShiftWrapper, ShiftWrapper, make_shift

import mani_skill2.envs


MANISKILL_TASKS = {
	'lift-cube': dict(
		env='LiftCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'pick-cube': dict(
		env='PickCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'stack-cube': dict(
		env='StackCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'pick-ycb': dict(
		env='PickSingleYCB-v0',
		control_mode='pd_ee_delta_pose',
	),
	'turn-faucet': dict(
		env='TurnFaucet-v0',
		control_mode='pd_ee_delta_pose',
	),
}


class ManiSkillWrapper(gym.Env):
	def __init__(self, env, cfg):
		super().__init__()
		self.env = env
		self.cfg = cfg
		self.observation_space = self.env.observation_space
		self.action_space = gym.spaces.Box(
			low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
			high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
			dtype=self.env.action_space.dtype,
		)

	def reset(self, **kwargs):
		return self.env.reset(**kwargs)
	
	def step(self, action):
		reward = 0
		native_steps = []
		benchmark = isinstance(self.env, ShiftWrapper)
		if benchmark:
			self.env.begin_decision()
		for _ in range(2):
			obs, r, done, info = self.env.step(action)
			if 'shift_record' in info:
				native_steps.append(info['shift_record'])
			if 'shift_gain' in info:
				native_steps.append({**{k: v for k, v in info.items() if k.startswith('shift_')},
					'reward': float(r), 'done': bool(done)})
			reward += r
			info['terminated'] = done
			if done:
				break
		if benchmark:
			self.env.end_decision()
		if native_steps:
			info['shift_native_steps'] = native_steps
		return obs, reward, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render(mode='cameras')

	def close(self):
		return self.env.close()


def make_env(cfg):
	"""
	Make ManiSkill2 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	task_cfg = MANISKILL_TASKS[cfg.task]
	env = legacy_gym.make(
		task_cfg['env'],
		obs_mode='state',
		control_mode=task_cfg['control_mode'],
		render_camera_cfgs=dict(width=384, height=384),
	)
	shift_config = cfg.get('dynamics_shift', {})
	if cfg.get('protocol_version', None) == 2:
		from common.benchmark_config import runtime_shift
		shift_config = runtime_shift(dict(dynamics_shift=shift_config, schedule=cfg.schedule))
	if shift_config.get('enabled', False):
		if cfg.get('shift_mode', 'nominal') != 'nominal':
			raise ValueError('Do not mix legacy shift_mode and dynamics_shift settings.')
		env = make_shift(env, shift_config, task=cfg.task)
	else:
		env = DynamicsShiftWrapper(
			env, mode=cfg.get('shift_mode', 'nominal'), gain=cfg.get('shift_gain', 0.7),
			switch_step=cfg.get('shift_switch_step', 100), period=cfg.get('shift_period', 200),
			amplitude=cfg.get('shift_amplitude', 0.3), clock=cfg.get('shift_clock', 'global'),
			preserve_gripper=cfg.get('shift_preserve_gripper', True),
		)
	env = ManiSkillWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=100)
	return env
