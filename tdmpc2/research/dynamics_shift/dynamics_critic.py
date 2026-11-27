"""One-step Dynamics+Q ablation; original TD-MPC2 targets and categorical loss."""
from contextlib import contextmanager
from copy import deepcopy
import math as scalar_math
import random
import time

import torch

from common import math
from .audit import ParameterInvarianceAudit, equal_state
from .objectives import (DEFAULT_OBJECTIVES, DynamicsObjectiveConfig,
                         MultiStepDynamicsConsistency, ObjectiveHooks,
                         replay_encoded_critic_loss)
from .replay import AdaptationReplayBuffer
from .schedule import UpdateSchedule, expected_updates


class DynamicsCriticAdaptation:
    """Two fresh Adams, one shared real-transition batch per scheduled event.

    Q sees detached *actual* latents. It never trains encoder/dynamics/reward/pi.
    Observing an explicit final next observation avoids crossing reset boundaries.
    """
    def __init__(self, agent, config, identity, *, objectives: ObjectiveHooks | None = None):
        objective_config = DynamicsObjectiveConfig.from_mapping(config)
        multi_step = objective_config.dynamics_objective == "multi_step_imagined_dynamics_consistency"
        if multi_step and objectives is not None:
            raise ValueError("Custom objective overrides are unsupported with multi-step dynamics")
        if multi_step and objective_config.rollout_horizon > config['horizon']:
            raise ValueError("Rollout horizon cannot exceed the episode horizon")
        self.agent, self.model, self.config = agent, agent.model, dict(config)
        self.identity = dict(identity)
        self.objective_config = objective_config
        self.multi_step = multi_step
        self.objectives = (ObjectiveHooks(MultiStepDynamicsConsistency(
            objective_config.rollout_horizon, objective_config.temporal_discount,
            objective_config.normalization_mode), replay_encoded_critic_loss)
            if multi_step else DEFAULT_OBJECTIVES if objectives is None else objectives)
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
        self.replay = AdaptationReplayBuffer(maxlen=config['capacity'])
        self.rng = random.Random(identity['adaptation_seed'])
        self.schedule = UpdateSchedule(self.config, 'update_every')
        self.transitions = self.dynamics_steps = self.q_steps = self.target_steps = 0
        self.update_seconds = 0.
        self.evaluating = False
        self.initial_model = deepcopy(self.model.state_dict())
        self.audit = ParameterInvarianceAudit(
            self.model, ('_dynamics.', '_Qs.', '_detach_Qs_params.', '_target_Qs_params.'))
        self.initial_fixed = self.audit.initial_fixed
        self.last_metrics = {}
        self.last_sample_steps = []
        self.last_sequence_steps = []
        self.sequence_skips = 0
        assert agent.cfg.num_bins > 1 and agent.cfg.num_q >= 2

    def fixed_state(self):
        # _detach_Qs_params aliases online Q storage; it is not another optimizer.
        return self.audit.fixed_state()

    def assert_fixed(self):
        self.audit.assert_fixed(self.initial_fixed, 'Frozen parameter/buffer changed')
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
        if self.schedule.due(len(self.replay), self.transitions):
            if self.multi_step and len(self.replay.contiguous_sequences(
                    self.objective_config.rollout_horizon)) < self.config['batch_size']:
                self.sequence_skips += 1
            else:
                self.last_metrics = self.update()
        return self.last_metrics

    def _sequence_tensors(self, sequences):
        horizon = self.objective_config.rollout_horizon
        if not sequences or any(len(window) != horizon for window in sequences):
            raise ValueError("Expected full contiguous sequence windows")
        for window in sequences:
            for previous, current in zip(window, window[1:]):
                if previous['episode'] != current['episode'] or \
                        previous['episode_step'] + 1 != current['episode_step'] or \
                        previous['global_step'] + 1 != current['global_step'] or \
                        previous['terminated'] or previous['truncated'] or \
                        not torch.equal(previous['next_observation'], current['observation']):
                    raise ValueError("Sequence crosses an episode boundary or missing step")
        initial_obs = torch.stack([window[0]['observation'] for window in sequences]).to(self.device)
        actions = torch.stack([torch.stack([window[step]['action'] for window in sequences])
                               for step in range(horizon)]).to(self.device)
        next_obs = torch.stack([torch.stack([window[step]['next_observation'] for window in sequences])
                                for step in range(horizon)]).to(self.device)
        with torch.no_grad():
            initial_z = self.model.encode(initial_obs, None)
            targets = self.model.encode(next_obs.flatten(0, 1), None).reshape(
                horizon, len(sequences), -1)
        valid_mask = torch.ones(actions.shape[:2], device=self.device, dtype=torch.bool)
        return initial_z, actions, targets, valid_mask

    def losses(self, batch, sequences=None):
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
        if self.multi_step:
            if sequences is None:
                raise ValueError("Multi-step dynamics requires contiguous sequences")
            if self.objective_config.rollout_horizon == 1:
                if any(len(window) != 1 for window in sequences):
                    raise ValueError("H=1 requires one transition per sequence")
                sequence_rows = [window[0] for window in sequences]
                if len(sequence_rows) != len(batch) or any(a is not b for a, b in zip(sequence_rows, batch)):
                    raise ValueError("H=1 parity requires the shared transition batch")
                dyn = self.objectives.dynamics(self.model, z, action.unsqueeze(0),
                                               next_z.unsqueeze(0))
            else:
                initial_z, actions, targets, valid_mask = self._sequence_tensors(sequences)
                dyn = self.objectives.dynamics(self.model, initial_z, actions, targets, valid_mask)
        else:
            dyn = self.objectives.dynamics(self.model, z, action, next_z)
        q, logits = self.objectives.critic(self.model, z, action, target, self.agent.cfg)
        assert logits.shape == (self.agent.cfg.num_q, len(batch), self.agent.cfg.num_bins)
        assert torch.isfinite(dyn) and torch.isfinite(q)
        return dyn, q, logits, target

    def update(self):
        if self.evaluating:
            raise RuntimeError('Held-out evaluation cannot update')
        assert self.transitions % self.config['update_every'] == 0
        expected = self.schedule.expected_updates(self.transitions)
        assert self.dynamics_steps == self.q_steps == expected - 1 - self.sequence_skips, 'Repeated/unscheduled update'
        batch = self.replay.sample(self.rng, self.config['batch_size'])
        self.last_sample_steps = [r['global_step'] for r in batch]
        sequences = None
        if self.multi_step:
            horizon = self.objective_config.rollout_horizon
            sequences = ([(row,) for row in batch] if horizon == 1 else
                         self.replay.sample_sequences(self.rng, self.config['batch_size'], horizon))
            self.last_sequence_steps = [[row['global_step'] for row in window] for window in sequences]
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
            dyn, q, logits, target = self.losses(batch, sequences)
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
        state = dict(dynamics_optimizer=self.dynamics_optimizer.state_dict(),
                             q_optimizer=self.q_optimizer.state_dict(), replay=list(self.replay),
                             transitions=self.transitions, dynamics_steps=self.dynamics_steps,
                             q_steps=self.q_steps, target_steps=self.target_steps,
                             rng=self.rng.getstate(), update_seconds=self.update_seconds)
        if self.multi_step:
            state['objective_configuration'] = self.objective_config.as_dict()
            state['sequence_skips'] = self.sequence_skips
        return deepcopy(state)

    def load_state_dict(self, state):
        saved_objective = state.get('objective_configuration')
        if self.multi_step:
            if saved_objective != self.objective_config.as_dict():
                raise ValueError('Multi-step objective configuration mismatch or missing')
        elif saved_objective is not None:
            raise ValueError('Cannot load multi-step adaptation state into one-step adapter')
        self.dynamics_optimizer.load_state_dict(state['dynamics_optimizer'])
        self.q_optimizer.load_state_dict(state['q_optimizer'])
        self.replay = AdaptationReplayBuffer(deepcopy(state['replay']), maxlen=self.config['capacity'])
        for key in ('transitions', 'dynamics_steps', 'q_steps', 'target_steps', 'update_seconds'):
            setattr(self, key, state[key])
        self.rng.setstate(state['rng'])
        self.sequence_skips = state.get('sequence_skips', 0) if self.multi_step else 0
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
