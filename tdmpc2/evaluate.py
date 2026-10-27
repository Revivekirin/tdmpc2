import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.shift_evaluation import EvaluationLog, shift_env
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	if cfg.get('dynamics_shift', {}).get('enabled', False):
		raise ValueError('Use evaluate_shift.py for structured dynamics_shift benchmark configuration.')
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)
	
	shift = shift_env(env)
	if cfg.shift_mode != 'nominal' and shift is None:
		raise ValueError('Dynamics shifts require a single-task ManiSkill environment.')
	log = EvaluationLog(cfg)
	if cfg.eval_seeds is not None and len(cfg.eval_seeds) != cfg.eval_episodes:
		raise ValueError('eval_seeds must contain one seed per evaluation episode.')
	print(f'Evaluation artifacts: {log.path.resolve()}')

	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(log.path, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		ep_rewards, ep_successes = [], []
		for i in range(cfg.eval_episodes):
			episode_seed = int(cfg.eval_seeds[i] if cfg.eval_seeds is not None else cfg.seed + i)
			if shift is not None:
				set_seed(episode_seed)
			start_native = shift.global_step if shift is not None else None
			obs, done, ep_reward, t = env.reset(task_idx=task_idx, **({"seed": episode_seed} if shift is not None else {})), False, 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done:
				action = agent.act(obs, t0=t==0, task=task_idx)
				obs, reward, done, info = env.step(action)
				log.step(task, i, t, action, reward, done, info)
				ep_reward += float(reward)
				t += 1
				if cfg.save_video:
					frames.append(env.render())
			log.episode(dict(task=task, checkpoint=str(log.checkpoint), seed=episode_seed if shift is not None else cfg.seed,
				episode_index=i, shift_mode=cfg.shift_mode, shift_gain=cfg.shift_gain,
				shift_switch_step=cfg.shift_switch_step, shift_period=cfg.shift_period,
				shift_amplitude=cfg.shift_amplitude, shift_clock=cfg.shift_clock,
				episode_return=ep_reward, success=float(info['success']), episode_length=t,
				native_global_start=start_native, native_global_end=shift.global_step if shift is not None else None,
				native_episode_length=shift.episode_step if shift is not None else None,
				output_directory=str(log.path.resolve())))
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if cfg.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
		print(f'  Std: return={np.std(ep_rewards):.3f}, success={np.std(ep_successes):.3f}')
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))
	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))

	log.finish()
	env.close()


if __name__ == '__main__':
	evaluate()
