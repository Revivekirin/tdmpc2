"""Separate NS-Gym factory; legacy envs.make_env and registrations are unchanged."""
import gymnasium as gym

from .halfcheetah import HalfCheetah
from .pusher import Pusher
from .reacher import Reacher


class TDMPC2Wrapper(gym.Wrapper):
    """Translate Gymnasium's five-return API to the existing TD-MPC2 contract."""

    def __init__(self, env):
        super().__init__(env)
        self.max_episode_steps = env.max_episode_steps

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)[0]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(terminated=terminated, truncated=truncated)
        return obs, reward, terminated or truncated, info


def make_env(cfg, *, recorder=None):
    """Explicit opt-in factory, called instead of envs.make_env for NS-Gym.

    Keep NS-Gym dependencies out of all existing environment imports. Uses the
    existing TensorWrapper without changing agent, model, MPC or adaptation.
    """
    environments = {'nsgym-halfcheetah': HalfCheetah, 'nsgym-pusher': Pusher,
                    'nsgym-reacher': Reacher}
    if cfg.task not in environments:
        raise ValueError(f'Unknown NS-Gym task: {cfg.task}')
    if cfg.get('multitask', False) or cfg.get('obs', 'state') != 'state':
        raise ValueError('NS-Gym initially supports single-task state observations only')
    if cfg.get('dynamics_shift', {}).get('enabled', False):
        raise ValueError('Use the nsgym config; legacy dynamics_shift is not supported here')
    from ..wrappers.tensor import TensorWrapper
    settings = dict(cfg.get('nsgym', {}))
    env = TensorWrapper(TDMPC2Wrapper(environments[cfg.task](seed=int(cfg.seed),
                                                        recorder=recorder, **settings)))
    # Explicit attributes avoid Gymnasium >=1.0's removal of wrapper forwarding.
    env.max_episode_steps = env.env.max_episode_steps
    cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    return env
