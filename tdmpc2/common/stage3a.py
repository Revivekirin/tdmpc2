"""Episode-boundary predictor adaptation; independent of simulator dependencies."""
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
import random
import torch
import torch.nn.functional as F


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def validate(settings):
    if settings['mode'] not in ('frozen', 'dynamics_only'):
        raise NotImplementedError('Stage 3A supports frozen and dynamics_only; joint_rl and dynamics_value are reserved.')
    for key, expected in [('update_granularity', 'episode'), ('consistency_steps', 1),
                          ('action_input', 'commanded'), ('replay_source', 'fresh_shifted'),
                          ('sampling', 'uniform_without_replacement')]:
        if settings[key] != expected:
            raise NotImplementedError(f'{key} must be {expected!r} in Stage 3A.')
    for key in ('capacity', 'batch_size', 'warmup', 'updates_per_episode', 'episodes', 'eval_every'):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    if settings['capacity'] < max(settings['batch_size'], settings['warmup']):
        raise ValueError('capacity must cover batch_size and warmup.')
    if not 0 < settings['lr'] < float('inf'):
        raise ValueError('lr must be finite and positive.')
    a, e = settings['adaptation_seeds'], settings['evaluation_seeds']
    if len(a) != settings['episodes'] or not e or len(set(a)) != len(a) or len(set(e)) != len(e):
        raise ValueError('Provide unique adaptation seeds for every episode and nonempty unique evaluation seeds.')
    if any(type(s) is not int or not 0 <= s < 2**32 for s in a + e) or set(a) & set(e):
        raise ValueError('Adaptation/evaluation seeds must be valid, disjoint seed sets.')


class EpisodeAdapter:
    def __init__(self, model, settings, consistency_coef, grad_clip_norm, seed=0):
        validate(settings)
        self.model, self.settings = model, settings
        self.coef, self.clip = consistency_coef, grad_clip_norm
        self.replay = deque(maxlen=settings['capacity'])
        self.rng = random.Random(seed)
        self.device = next(model.parameters()).device
        model.requires_grad_(False)
        self.params = list(model._dynamics.parameters())
        if settings['mode'] == 'dynamics_only':
            model._dynamics.requires_grad_(True)
            self.optimizer = torch.optim.Adam(self.params, lr=settings['lr'])
        else:
            self.optimizer = None
        model.eval()
        self.initial_fixed = self.fixed_state()
        self.episodes = self.transitions = self.native_steps = self.gradient_steps = 0
        self.evaluating = self.in_episode = False
        self.last_metrics = dict(consistency_loss=None, predictor_gradient_norm=None, predictor_update_norm=None)

    def fixed_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                if not k.startswith('_dynamics.')}

    def begin_episode(self):
        if self.evaluating or self.in_episode:
            raise RuntimeError('Invalid adaptation episode boundary.')
        self.in_episode = True
        self.episode_predictor = deepcopy(self.model._dynamics.state_dict())

    def observe(self, observation, commanded_action, next_observation, native_steps):
        if self.evaluating or not self.in_episode:
            raise RuntimeError('Replay addition requires an active adaptation episode.')
        self.replay.append(tuple(x.detach().cpu().clone() for x in
                                 (observation, commanded_action, next_observation)))
        self.transitions += 1
        self.native_steps += native_steps

    def loss(self, batch):
        obs, action, next_obs = [torch.stack(x).to(self.device) for x in zip(*batch)]
        with torch.no_grad():
            z = self.model.encode(obs, None)
            target = self.model.encode(next_obs, None)
        return F.mse_loss(self.model.next(z, action, None), target)

    def update(self):
        """Separate API for future timing policies; currently only legal between episodes."""
        if self.evaluating or self.in_episode:
            raise RuntimeError('Updates are prohibited inside episodes and held-out evaluation.')
        if self.optimizer is None or len(self.replay) < max(self.settings['warmup'], self.settings['batch_size']):
            return self.last_metrics
        metrics = []
        for _ in range(self.settings['updates_per_episode']):
            batch = self.rng.sample(list(self.replay), self.settings['batch_size'])
            before = [p.detach().clone() for p in self.params]
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss(batch)
            (self.coef * loss).backward()
            norm = torch.nn.utils.clip_grad_norm_(self.params, self.clip, error_if_nonfinite=True)
            self.optimizer.step()
            delta = torch.sqrt(sum((p.detach()-q).square().sum() for p, q in zip(self.params, before)))
            metrics.append([loss.item(), norm.item(), delta.item()])
            self.gradient_steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.last_metrics = dict(zip(self.last_metrics, torch.tensor(metrics).mean(0).tolist()))
        self.assert_fixed()
        return self.last_metrics

    def end_episode(self):
        if self.evaluating or not self.in_episode:
            raise RuntimeError('No adaptation episode to end.')
        if not equal_state(self.episode_predictor, self.model._dynamics.state_dict()):
            raise AssertionError('Predictor changed inside episode.')
        self.in_episode = False
        self.episodes += 1
        return self.update()

    def assert_fixed(self):
        if not equal_state(self.initial_fixed, self.fixed_state()):
            raise AssertionError('A frozen encoder/reward/Q/policy/target parameter changed.')

    @contextmanager
    def held_out(self):
        if self.evaluating or self.in_episode:
            raise RuntimeError('Evaluation requires an episode boundary.')
        model = deepcopy(self.model.state_dict())
        optimizer = deepcopy(self.optimizer.state_dict()) if self.optimizer else None
        replay = list(self.replay)
        counters = (self.episodes, self.transitions, self.native_steps, self.gradient_steps)
        rng = self.rng.getstate()
        self.evaluating = True
        try:
            with torch.no_grad():
                yield
        finally:
            self.evaluating = False
            assert equal_state(model, self.model.state_dict()), 'Evaluation mutated model'
            assert equal_state(optimizer, self.optimizer.state_dict() if self.optimizer else None), 'Evaluation mutated optimizer'
            assert equal_state(replay, list(self.replay)), 'Evaluation mutated replay'
            assert counters == (self.episodes, self.transitions, self.native_steps, self.gradient_steps)
            assert rng == self.rng.getstate()
            self.assert_fixed()

    def metrics(self):
        self.assert_fixed()
        return dict(adaptation_episodes=self.episodes, adaptation_transitions=self.transitions,
                    adaptation_native_steps=self.native_steps, adaptation_policy_decisions=self.transitions,
                    gradient_update_steps=self.gradient_steps, replay_size=len(self.replay),
                    reward_q_parameter_invariance=True, **self.last_metrics)
