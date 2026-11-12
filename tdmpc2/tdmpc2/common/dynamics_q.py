"""One-step Dynamics+Q ablation; original TD-MPC2 targets and categorical loss."""
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
import math as scalar_math
import random
import time

import torch
import torch.nn.functional as F

from common import math
from common.stage3a import equal_state


def expected_updates(transitions, warmup=64, every=25):
    return max(0, transitions // every - (warmup - 1) // every)


class DynamicsQAdapter:
    """Two fresh Adams, one shared real-transition batch per scheduled event.

    Q sees detached *actual* latents. It never trains encoder/dynamics/reward/pi.
    Observing an explicit final next observation avoids crossing reset boundaries.
    """
    def __init__(self, agent, config, identity):
        self.agent, self.model, self.config = agent, agent.model, dict(config)
        self.identity = dict(identity)
        self.device = next(self.model.parameters()).device
        self.model.requires_grad_(False)
        self.model._dynamics.requires_grad_(True)
        self.model._Qs.requires_grad_(True)
        self.model.eval()
        self.dynamics_params = list(self.model._dynamics.parameters())
        self.q_params = list(self.model._Qs.parameters())
        assert not {id(p) for p in self.dynamics_params} & {id(p) for p in self.q_params}
        adam = dict(betas=tuple(config['betas']), eps=config['eps'], weight_decay=0.)
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics_params, lr=config['dynamics_lr'], **adam)
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=config['q_lr'], **adam)
        self.replay = deque(maxlen=config['capacity'])
        self.rng = random.Random(identity['adaptation_seed'])
        self.transitions = self.dynamics_steps = self.q_steps = self.target_steps = 0
        self.update_seconds = 0.
        self.evaluating = False
        self.initial_model = deepcopy(self.model.state_dict())
        self.initial_fixed = self.fixed_state()
        self.last_metrics = {}
        self.last_sample_steps = []
        assert agent.cfg.num_bins > 1 and agent.cfg.num_q >= 2

    def fixed_state(self):
        # _detach_Qs_params aliases online Q storage; it is not another optimizer.
        mutable = ('_dynamics.', '_Qs.', '_detach_Qs_params.', '_target_Qs_params.')
        return deepcopy({k: v for k, v in self.model.state_dict().items()
                         if not k.startswith(mutable)})

    def assert_fixed(self):
        assert equal_state(self.initial_fixed, self.fixed_state()), 'Frozen parameter/buffer changed'
        trainable = {id(p) for p in self.dynamics_params + self.q_params}
        assert all(p.grad is None for p in self.model.parameters() if id(p) not in trainable)

    def observe(self, observation, action, next_observation, reward, terminated, truncated,
                episode, episode_step, actual_parameter, nominal_parameter):
        if self.evaluating:
            raise RuntimeError('Held-out evaluation cannot write replay')
        assert self.identity['adaptation_seed'] not in self.config['evaluation_seeds']
        assert episode == self.transitions // self.config['horizon']
        assert episode_step == self.transitions % self.config['horizon']
        assert bool(truncated) == (episode_step + 1 == self.config['horizon'])
        assert not terminated, 'HalfCheetah continuing-task contract rejects true termination'
        assert scalar_math.isfinite(reward)
        assert abs(actual_parameter - nominal_parameter * self.identity['severity']) < 1e-10
        tensors = [torch.as_tensor(x, dtype=torch.float32).detach().cpu().clone()
                   for x in (observation, action, next_observation)]
        assert all(torch.isfinite(x).all() for x in tensors)
        record = dict(observation=tensors[0], action=tensors[1], next_observation=tensors[2],
                      reward=float(reward), discount=float(self.agent.discount), terminated=bool(terminated),
                      truncated=bool(truncated), episode=episode, episode_step=episode_step,
                      global_step=self.transitions + 1, adaptation_seed=self.identity['adaptation_seed'],
                      parameter=self.identity['parameter'], severity=self.identity['severity'],
                      actual_parameter=float(actual_parameter), nominal_parameter=float(nominal_parameter))
        self.replay.append(record)
        self.transitions += 1
        self.last_metrics = {}
        if (len(self.replay) >= max(self.config['warmup'], self.config['batch_size'])
                and self.transitions % self.config['update_every'] == 0):
            self.last_metrics = self.update()
        return self.last_metrics

    def losses(self, batch):
        obs, action, nxt = [torch.stack([r[k] for r in batch]).to(self.device)
                            for k in ('observation', 'action', 'next_observation')]
        reward, terminal = [torch.tensor([[r[k]] for r in batch], device=self.device, dtype=torch.float32)
                            for k in ('reward', 'terminated')]
        assert all(r['discount'] == float(self.agent.discount) for r in batch)
        with torch.no_grad():
            z = self.model.encode(obs, None)
            next_z = self.model.encode(nxt, None)
            # Original method: real reward + gamma*(1-terminal)*min(two target heads).
            target = self.agent._td_target(next_z, reward, terminal, None)
        assert target.shape == reward.shape and torch.isfinite(target).all()
        dyn = F.mse_loss(self.model.next(z, action, None), next_z)
        logits = self.model.Q(z.detach(), action, None, return_type='all')
        assert logits.shape == (self.agent.cfg.num_q, len(batch), self.agent.cfg.num_bins)
        q = torch.stack([math.soft_ce(head, target, self.agent.cfg).mean() for head in logits]).mean()
        assert torch.isfinite(dyn) and torch.isfinite(q)
        return dyn, q, logits, target

    def update(self):
        if self.evaluating:
            raise RuntimeError('Held-out evaluation cannot update')
        assert self.transitions % self.config['update_every'] == 0
        expected = expected_updates(self.transitions, max(self.config['warmup'], self.config['batch_size']),
                                    self.config['update_every'])
        assert self.dynamics_steps == self.q_steps == expected - 1, 'Repeated/unscheduled update'
        batch = self.rng.sample(list(self.replay), self.config['batch_size'])
        self.last_sample_steps = [r['global_step'] for r in batch]
        assert max(self.last_sample_steps) <= self.transitions
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self.dynamics_optimizer.zero_grad(set_to_none=True)
        self.q_optimizer.zero_grad(set_to_none=True)
        # Dynamics remains eval-mode, as in predictor-only TTT. Online Q uses
        # original training dropout; target Q stays in eval-mode.
        self.model._Qs.train(True)
        try:
            dyn, q, logits, target = self.losses(batch)
            (self.config['consistency_coef'] * dyn).backward()
            assert all(p.grad is None for p in self.q_params), 'Dynamics loss leaked into Q'
            dn = torch.nn.utils.clip_grad_norm_(self.dynamics_params, self.config['dynamics_grad_clip'], error_if_nonfinite=True)
            self.dynamics_optimizer.step()
            self.dynamics_optimizer.zero_grad(set_to_none=True)
            (self.config['value_coef'] * q).backward()
            assert all(p.grad is None for p in self.dynamics_params), 'Q loss leaked into dynamics'
            qn = torch.nn.utils.clip_grad_norm_(self.q_params, self.config['q_grad_clip'], error_if_nonfinite=True)
            self.q_optimizer.step()
            self.q_optimizer.zero_grad(set_to_none=True)
            self.dynamics_steps += 1
            self.q_steps += 1
            with torch.no_grad():
                target_before = deepcopy(self.model._target_Qs_params.state_dict())
                self.model.soft_update_target_Q()
                online = self.model._detach_Qs_params.state_dict()
                for key, value in self.model._target_Qs_params.state_dict().items():
                    if isinstance(value, torch.Tensor):
                        expected_target = target_before[key].lerp(online[key], self.agent.cfg.tau)
                        assert torch.allclose(value, expected_target), 'Target Q did not follow original EMA'
            self.target_steps += 1
        finally:
            self.model.eval()
        self.assert_fixed()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.update_seconds += time.perf_counter() - started
        with torch.no_grad():
            decoded = math.two_hot_inv(logits, self.agent.cfg)
            probabilities = logits.softmax(-1)
            entropy = -(probabilities * logits.log_softmax(-1)).sum(-1).mean()
        return dict(dynamics_loss=float(dyn.detach()), q_loss=float(q.detach()),
                    dynamics_gradient_norm=float(dn), q_gradient_norm=float(qn),
                    q_prediction=float(decoded.mean()), q_target=float(target.mean()),
                    q_entropy=float(entropy), q_disagreement=float(decoded.std(0).mean()))

    def parameter_audit(self):
        result = []
        groups = {'encoder': '_encoder.', 'dynamics': '_dynamics.', 'reward': '_reward.',
                  'q': '_Qs.', 'target_q': '_target_Qs_params.', 'actor': '_pi.'}
        current = self.model.state_dict()
        for name, prefix in groups.items():
            pairs = [(old, current[k]) for k, old in self.initial_model.items()
                     if k.startswith(prefix) and isinstance(old, torch.Tensor)]
            drift = sum(float((old - new).double().square().sum()) for old, new in pairs) ** .5
            result.append(dict(component=name, changed=any(not torch.equal(a, b) for a, b in pairs),
                               drift_l2=drift, elements=sum(a.numel() for a, _ in pairs)))
        self.assert_fixed()
        return result

    def state_dict(self):
        return deepcopy(dict(dynamics_optimizer=self.dynamics_optimizer.state_dict(),
                             q_optimizer=self.q_optimizer.state_dict(), replay=list(self.replay),
                             transitions=self.transitions, dynamics_steps=self.dynamics_steps,
                             q_steps=self.q_steps, target_steps=self.target_steps,
                             rng=self.rng.getstate(), update_seconds=self.update_seconds))

    def load_state_dict(self, state):
        self.dynamics_optimizer.load_state_dict(state['dynamics_optimizer'])
        self.q_optimizer.load_state_dict(state['q_optimizer'])
        self.replay = deque(deepcopy(state['replay']), maxlen=self.config['capacity'])
        for key in ('transitions', 'dynamics_steps', 'q_steps', 'target_steps', 'update_seconds'):
            setattr(self, key, state[key])
        self.rng.setstate(state['rng'])
        assert self.dynamics_steps == self.q_steps == self.target_steps
        self.assert_fixed()

    @property
    def gradient_steps(self):
        """Compatibility with the existing read-only rollout: synchronous events."""
        return self.dynamics_steps

    @property
    def update_compute_time(self):
        return self.update_seconds

    @contextmanager
    def held_out(self):
        before = deepcopy(self.model.state_dict()), self.state_dict()
        self.evaluating = True
        try:
            with torch.no_grad():
                yield
        finally:
            self.evaluating = False
            assert equal_state(before[0], self.model.state_dict()), 'Evaluation changed model'
            assert equal_state(before[1], self.state_dict()), 'Evaluation changed adaptation state'
