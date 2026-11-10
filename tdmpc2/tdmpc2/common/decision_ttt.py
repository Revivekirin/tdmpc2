"""Policy-decision scheduling over the unchanged Stage 3B one-step optimizer."""
from copy import deepcopy
from common.stage3a import EpisodeAdapter


class DecisionAdapter(EpisodeAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.settings['mode'] != 'decision_ttt':
            raise ValueError('DecisionAdapter requires decision_ttt.')
        self.scheduler = dict(first_eligible_update_decision=None,
                              first_actual_update_decision=None, decisions_with_updates=0,
                              decisions_skipped_insufficient_replay=0, last_update_decision=0)
        self.last_decision = {}

    def observe(self, observation, commanded_action, next_observation, native_steps):
        super().observe(observation, commanded_action, next_observation, native_steps)
        before, elapsed = self.gradient_steps, self.update_compute_time
        ready = len(self.replay) >= max(self.settings['warmup'], self.settings['batch_size'])
        scheduled = self.transitions % self.settings['update_every_decisions'] == 0
        reason = None
        if not ready:
            reason = 'insufficient_replay'
            self.scheduler['decisions_skipped_insufficient_replay'] += 1
        elif not scheduled:
            reason = 'schedule_interval'
        else:
            if self.scheduler['first_eligible_update_decision'] is None:
                self.scheduler['first_eligible_update_decision'] = self.transitions
            self.update()
        triggered = self.gradient_steps > before
        if triggered:
            self.scheduler['decisions_with_updates'] += 1
            if self.scheduler['first_actual_update_decision'] is None:
                self.scheduler['first_actual_update_decision'] = self.transitions
        self.last_decision = dict(record_type='decision_update', update_granularity='decision',
            episode_index=self.episodes, policy_decision_index=self.transitions,
            native_step_index=self.native_steps, update_triggered=triggered,
            update_skip_reason=reason, gradient_updates=self.gradient_steps,
            replay_size=len(self.replay), samples_processed=self.gradient_steps*self.settings['batch_size'],
            predictor_consistency_loss=self.last_metrics['consistency_loss'] if triggered else None,
            predictor_gradient_norm=self.last_metrics['predictor_gradient_norm'] if triggered else None,
            predictor_update_norm=self.last_metrics['predictor_update_norm'] if triggered else None,
            update_wall_time_ms=1000*(self.update_compute_time-elapsed))
        return self.last_decision

    def update(self):
        if self.evaluating or not self.in_episode:
            raise RuntimeError('Decision updates require an active adaptation episode.')
        if (self.transitions <= self.scheduler['last_update_decision']
                or self.transitions % self.settings['update_every_decisions']):
            return self.last_metrics
        before = self.gradient_steps
        result = self._optimizer_steps(self.settings['updates_per_decision'])
        if self.gradient_steps > before:
            self.scheduler['last_update_decision'] = self.transitions
        return result

    def end_episode(self):
        if self.evaluating or not self.in_episode:
            raise RuntimeError('No adaptation episode to end.')
        self.assert_fixed()
        self.in_episode = False
        self.episodes += 1
        return self.last_metrics

    def state_dict(self):
        return dict(super().state_dict(), scheduler=deepcopy(self.scheduler),
                    last_decision=deepcopy(self.last_decision))

    def load_state_dict(self, state):
        if 'scheduler' not in state or state['settings']['mode'] != 'decision_ttt':
            raise ValueError('Decision resume requires a decision-wise scheduler checkpoint.')
        super().load_state_dict(state)
        self.scheduler = deepcopy(state['scheduler'])
        self.last_decision = deepcopy(state['last_decision'])

    def metrics(self):
        return dict(super().metrics(), update_granularity='decision', **self.scheduler,
                    gradient_updates=self.gradient_steps,
                    samples_processed=self.gradient_steps*self.settings['batch_size'],
                    update_wall_time_ms=1000*self.update_compute_time)
