"""Uninstrumented planner oracle copied from repository HEAD before Stage 2.
Do not regenerate from the instrumented implementation.
"""
import torch
from common import math

class OriginalPlanner:

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
    		Select an action by planning in the latent space of the world model.
    
    		Args:
    			obs (torch.Tensor): Observation from the environment.
    			t0 (bool): Whether this is the first observation in the episode.
    			eval_mode (bool): Whether to use the mean of the action distribution.
    			task (int): Task index (only used for multi-task experiments).
    
    		Returns:
    			torch.Tensor: Action to take in the environment.
    		"""
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if self.cfg.mpc:
            return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
        z = self.model.encode(obs, task)
        action, info = self.model.pi(z, task)
        if eval_mode:
            action = info['mean']
        return action[0].cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = (0, 1)
        termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G = G + discount * (1 - termination) * reward
            discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update
            if self.cfg.episodic:
                termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.0)
        action, _ = self.model.pi(z, task)
        return G + discount * (1 - termination) * self.model.Q(z, action, task, return_type='avg')

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        """
    		Plan a sequence of actions using the learned world model.
    
    		Args:
    			z (torch.Tensor): Latent state from which to plan.
    			t0 (bool): Whether this is the first observation in the episode.
    			eval_mode (bool): Whether to use the mean of the action distribution.
    			task (Torch.Tensor): Task index (only used for multi-task experiments).
    
    		Returns:
    			torch.Tensor: Action to take in the environment.
    		"""
        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[t], _ = self.model.pi(_z, task)
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions
        for _ in range(self.cfg.iterations):
            r = torch.randn(self.cfg.horizon, self.cfg.num_samples - self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]
            value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = (value[elite_idxs], actions[:, elite_idxs])
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-09)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-09)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]
        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = (actions[0], std[0])
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)
