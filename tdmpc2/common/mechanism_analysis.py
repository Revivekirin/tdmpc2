"""Frozen decision-interval diagnostics integrated with Benchmark v2.

Observed-policy returns and model-only rescoring are descriptive proxies, never
counterfactual ground truth. No optimizer or simulator-state mutation occurs.
"""
import csv
import hashlib
import io
import json
import random
import zipfile
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from common import math
from common.planner_trace import PlannerTrace, array
from common.mechanism_metrics import ranking_comparison, ranks
from common.shift_evaluation import json_value


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def model_checksum(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, torch.Size):
            digest.update(b'torch.Size:')
            digest.update(repr(tuple(value)).encode())
        elif isinstance(value, torch.device):
            digest.update(b'torch.device:')
            digest.update(str(value).encode())
        else:
            raise TypeError(f'Unsupported model state entry {name}: {type(value).__name__}')
    return digest.hexdigest()


def prediction_error(predicted, observed):
    if predicted.shape != observed.shape:
        raise ValueError(f'Latent shape mismatch: {predicted.shape} != {observed.shape}')
    residual = (predicted-observed).square()
    return dict(latent_prediction_mse=float(residual.mean()), latent_prediction_squared_l2=float(residual.sum()),
                latent_prediction_relative_error=float(residual.sum() / observed.square().sum().clamp_min(1e-12)))


def rollout(model, z, actions):
    """Open loop with recorded commands, no intermediate latent correction."""
    for action in actions:
        z = model.next(z, action, None)
    return z


def execution_semantics(action, records, compatible):
    executed = [np.asarray(r['executed_action']) for r in records]
    valid = (compatible and len(executed) == 2 and all(x.shape == action.shape for x in executed)
             and all('commanded_action' not in r or np.array_equal(np.asarray(r['commanded_action']), action) for r in records)
             and all(np.array_equal(executed[0], x) for x in executed)
             and all(np.all(np.abs(x) <= 1) for x in executed))
    difference = float(np.mean([np.linalg.norm(action-x) for x in executed])) if executed else None
    return dict(commanded_executed_action_difference=difference,
                prediction_action_semantics='policy_decision_command; executed=controller_command_not_torque',
                prediction_validity='VALID' if len(records) == 2 else 'PARTIAL_REPEAT',
                executed_prediction_validity='VALID_REPEATED_CONTROLLER_COMMAND' if valid else 'NOT_IDENTIFIABLE'), executed, valid


class MechanismAnalysis:
    def __init__(self, cfg, agent, path, action_space):
        self.cfg, self.agent, self.options = cfg, agent, cfg.analysis
        self.path = Path(path) / 'mechanism'
        self.path.mkdir()
        if agent.model.training:
            raise ValueError('Frozen diagnostics require model.eval()')
        self.digest = model_checksum(agent.model)
        self.compatible = (tuple(action_space.shape) == (cfg.action_dim,)
                           and np.all(action_space.low == -1) and np.all(action_space.high == 1))
        self.tables = {}
        self.groups = defaultdict(list)
        self.archive = zipfile.ZipFile(self.path / 'planner_candidates.npz', 'w', compression=zipfile.ZIP_DEFLATED)
        self.closed = False
        self.history_groups = defaultdict(list)
        self.episode_summaries = []
        self.last_change = None
        self.write_json('manifest.json', dict(schema_version=1, analysis=cfg.analysis,
            checkpoint=str(cfg.checkpoint), task=cfg.task, discount=float(agent.discount), episodic=cfg.episodic,
            mpc=cfg.mpc, action_mode='stochastic (evaluate_shift calls eval_mode=False)',
            action_space_compatible=self.compatible, model_transition_unit='policy decision, up to 2 native steps',
            actual_score='captured original sampled terminal policy action and random Q-head pair',
            diagnostic_score='mean terminal policy action and mean of all decoded Q heads',
            candidate_storage='first bounded IDs plus selected final candidate; rankings use all candidates',
            model_sha256_before=self.digest,
            relative_error_definition='squared L2 residual / max(squared L2 observed latent, 1e-12)',
            elapsed_shift_clock='native steps at end of decision since most recent parameter change',
            truncated_return='observed finite continuation, not complete Q target',
            previous_actions_window=5))

    def write_json(self, name, data):
        (self.path/name).write_text(json.dumps(data, default=json_value, indent=2, allow_nan=False)+'\n')

    def row(self, name, row):
        if name not in self.tables:
            f = (self.path/(name+'.csv')).open('w', newline='')
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            self.tables[name] = (f, writer)
        serial = {k: json.dumps(v, default=json_value) if isinstance(v, (list, dict, np.ndarray)) else v for k,v in row.items()}
        self.tables[name][1].writerow(serial)

    def save_array(self, key, value):
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        self.archive.writestr(key+'.npy', buffer.getvalue())

    def begin_episode(self, episode, seed):
        self.episode, self.seed, self.transitions = episode, seed, []

    def before_action(self, decision):
        self.decision = decision
        self.agent._mechanism_trace = (PlannerTrace() if self.options['planning'] and self.cfg.mpc
                                      and decision % self.options['decision_stride'] == 0 else None)

    @torch.no_grad()
    def observe(self, obs, action, next_obs, reward, done, info, events):
        with preserve_rng():
            self._observe(obs, action, next_obs, reward, done, info, events)

    def _observe(self, obs, action, next_obs, reward, done, info, events):
        model, cfg = self.agent.model, self.cfg
        z = model.encode(obs.to(self.agent.device).unsqueeze(0), None)
        zn = model.encode(next_obs.to(self.agent.device).unsqueeze(0), None)
        a = action.to(self.agent.device).unsqueeze(0)
        records = info['shift_native_steps']
        first = records[0]
        for event in events:
            previous = event['previous_parameter']
            reference = first['nominal_parameter'] if previous is None else previous
            if not np.allclose(event['applied_parameter'], reference, rtol=0, atol=1e-8):
                self.last_change = event['native_step']
        context = dict(episode_id=self.episode, seed=self.seed, policy_decision_step=self.decision,
                       global_native_step=first['global_native_step'], shift_parameter=[r['applied_parameter'] for r in records],
                       regime_id=[r['regime_id'] for r in records],
                       exposed=any(not np.allclose(r['applied_parameter'], r['nominal_parameter'], rtol=0, atol=1e-8) for r in records),
                       native_steps_since_shift=None if self.last_change is None else records[-1]['global_native_step']-self.last_change)
        semantics, executed, valid = execution_semantics(array(action), records, self.compatible)
        diagnostic = dict(context, **semantics, **prediction_error(model.next(z, a, None), zn))
        diagnostic['executed_latent_prediction_mse'] = (prediction_error(model.next(z, torch.as_tensor(executed[0], device=z.device, dtype=a.dtype).unsqueeze(0), None), zn)['latent_prediction_mse'] if valid else None)
        q = math.two_hot_inv(model.Q(z, a, None, return_type='all'), cfg).reshape(-1)
        predicted_reward = float(math.two_hot_inv(model.reward(z, a, None), cfg).item())
        _, pi_info = model.pi(zn, None)
        # Analysis-only all-head mean Bellman proxy, deliberately not training target.
        next_q = math.two_hot_inv(model.Q(zn, pi_info['mean'], None, return_type='all'), cfg).mean()
        terminated = bool(records[-1].get('analysis_native_terminated', info.get('terminated', False)))
        bootstrap_mask = 1 - int(terminated and cfg.episodic)
        proxy = float(reward) + float(self.agent.discount)*bootstrap_mask*float(next_q)
        rq = dict(context, predicted_reward=predicted_reward, actual_reward=float(reward),
                  reward_absolute_error=abs(predicted_reward-float(reward)), Q_all=array(q),
                  Q_mean=float(q.mean()), Q_min=float(q.min()), Q_std=float(q.std(unbiased=False)),
                  Q_range=float(q.max()-q.min()), bellman_proxy_target=proxy,
                  bellman_proxy_residual=float(q.mean())-proxy,
                  target_semantics='analysis_only_online_all_head_mean_next_policy_mean',
                  terminated=terminated, time_limit=bool(records[-1].get('analysis_native_truncated', False) or (done and not terminated)),
                  termination_semantics='native legacy Gym done excluding TimeLimit.truncated; outer decision timeout separately', episodic=cfg.episodic)
        self.transitions.append(dict(z=z, next_z=zn, action=a, reward=float(reward), diagnostic=diagnostic, rq=rq))
        for repeat, record in enumerate(records):
            self.row('action_history', dict(context, action_repeat_index=repeat,
                native_step=record['native_step'], requested_action=record['commanded_action'],
                native_executed_action=record['executed_action'], current_shift=record['applied_parameter'],
                delay_queue_state_if_available=record.get('analysis_delay_queue_before', 'NOT_APPLICABLE'),
                previous_actions=[array(t['action']).reshape(-1).tolist() for t in self.transitions[max(0,len(self.transitions)-6):-1]]))
        if self.options['save_latents']:
            self.save_array(f'latent_e{self.episode}_d{self.decision}', array(z))
            self.save_array(f'next_latent_e{self.episode}_d{self.decision}', array(zn))
        trace = self.agent._mechanism_trace
        if trace is not None:
            self.planning(z, context, trace)
        elif not cfg.mpc and self.decision % self.options['decision_stride'] == 0:
            self.row('planner_summary', dict(context, status='NOT_APPLICABLE', reason='actor_only'))
        self.agent._mechanism_trace = None

    def planning(self, z, context, trace):
        model, cfg = self.agent.model, self.cfg
        previous_elites = None
        for iteration, row in enumerate(trace.iterations):
            actions = torch.as_tensor(row['actions'], device=z.device)
            latent = z.repeat(actions.shape[1], 1)
            reward_sum = torch.zeros(actions.shape[1], 1, device=z.device)
            terminated = torch.zeros_like(reward_sum)
            discount = 1
            for action in actions:
                reward_sum += discount*(1-terminated)*math.two_hot_inv(model.reward(latent, action, None), cfg)
                latent = model.next(latent, action, None)
                discount *= float(self.agent.discount)
                if cfg.episodic:
                    terminated = (terminated+(model.termination(latent, None)>.5).float()).clamp(max=1)
            _, pi_info = model.pi(latent, None)
            q = math.two_hot_inv(model.Q(latent, pi_info['mean'], None, return_type='all'), cfg).mean(0)
            terminal = discount*(1-terminated)*q
            dr, dq = array(reward_sum).reshape(-1), array(terminal).reshape(-1)
            ds = dr+dq
            n = len(ds)
            ids = list(range(min(n, self.options['max_logged_candidates'])))
            selected = trace.selection['selected_candidate_id'] if iteration == len(trace.iterations)-1 else None
            if selected is not None and selected not in ids:
                ids[-1] = selected
                ids.sort()
            key = f'e{self.episode}_d{self.decision}_i{iteration}'
            self.save_array(key+'_candidate_id', np.asarray(ids))
            self.save_array(key+'_candidate_source', np.where(np.asarray(ids)<row['num_prior'], 'policy_prior', 'sampled'))
            self.save_array(key+'_elite_membership', np.isin(ids, row['elite_ids']))
            if self.options['save_candidate_actions']:
                self.save_array(key+'_candidate_action_sequence', row['actions'][:, ids])
            if self.options['save_candidate_scores']:
                for name in ('actual_planner_score', 'predicted_reward_sum', 'terminal_q_contribution', 'candidate_rank'):
                    self.save_array(key+'_'+name, row[name][ids])
                for name, values in dict(diagnostic_score=ds, diagnostic_reward_sum=dr, diagnostic_terminal_q=dq).items():
                    self.save_array(key+'_'+name, values[ids])
            elites = set(row['elite_ids'].tolist())
            summary = dict(context, status='VALID', iteration_id=iteration, candidate_count=n, logged_candidates=len(ids),
                selected_candidate_id=selected, selected_first_action=trace.selection['selected_first_action'] if selected is not None else None,
                actual_reward_vs_q=ranking_comparison(row['predicted_reward_sum'], row['terminal_q_contribution']),
                actual_total_vs_reward=ranking_comparison(row['actual_planner_score'], row['predicted_reward_sum']),
                actual_total_vs_q=ranking_comparison(row['actual_planner_score'], row['terminal_q_contribution']),
                actual_vs_diagnostic=ranking_comparison(row['actual_planner_score'], ds),
                diagnostic_reward_vs_q=ranking_comparison(dr, dq) if self.options['counterfactual'] else None,
                elite_slot_overlap=None if previous_elites is None else len(elites & previous_elites)/len(elites | previous_elites),
                elite_overlap_semantics='slot overlap only; sampled trajectories change across iterations')
            if self.options['counterfactual'] and self.options['save_counterfactual_details']:
                self.save_array(key+'_reward_only_rank', ranks(dr)[ids])
                self.save_array(key+'_q_only_rank', ranks(dq)[ids])
            self.row('planner_summary', summary)
            previous_elites = elites

    @torch.no_grad()
    def end_episode(self, success):
        with preserve_rng():
            future = 0.
            ended = self.transitions[-1]['rq']
            mc_valid = bool(ended['terminated'] and self.cfg.episodic)
            for transition in reversed(self.transitions):
                future = transition['reward'] + float(self.agent.discount)*future
                transition['rq'].update(observed_discounted_return=future,
                    mc_validity='COMPLETE_EPISODIC_RETURN' if mc_valid else 'TRUNCATED_POLICY_CONTINUATION',
                    Q_minus_observed_return=transition['rq']['Q_mean']-future,
                    episode_success=bool(success))
            for index, transition in enumerate(self.transitions):
                diagnostic = dict(transition['diagnostic'], episode_success=bool(success))
                if self.options['one_step']:
                    self.row('one_step', diagnostic)
                self.row('reward_q', transition['rq'])
                if self.options['multi_step'] and index % self.options['decision_stride'] == 0:
                    for horizon in self.options['rollout_horizons']:
                        if index+horizon > len(self.transitions):
                            continue
                        sequence = self.transitions[index:index+horizon]
                        predicted = rollout(self.agent.model, transition['z'], [t['action'] for t in sequence])
                        self.row('multistep', dict(episode_id=self.episode, seed=self.seed, policy_decision_step=index,
                            horizon=horizon, rollout_semantics='open_loop_recorded_commands_policy_decision_interval',
                            prediction_validity='PARTIAL_REPEAT' if any(t['diagnostic']['prediction_validity']!='VALID' for t in sequence) else 'VALID',
                            **prediction_error(predicted, sequence[-1]['next_z'])))
                group = (str(self.cfg.checkpoint), self.cfg.task, self.cfg.dynamics_shift['parameter']['type'],
                         self.cfg.schedule['type'], json.dumps(diagnostic['regime_id']), diagnostic['exposed'],
                         bool(success), self.cfg.mpc, diagnostic['native_steps_since_shift'])
                self.groups[group].append((self.episode, diagnostic['latent_prediction_mse'], transition['rq']['reward_absolute_error']))
                mismatch = diagnostic['commanded_executed_action_difference']
                history_key = (group[:8], index == 0, mismatch is not None and mismatch > 1e-8)
                self.history_groups[history_key].append(diagnostic['latent_prediction_mse'])
            self.episode_summaries.append(dict(episode_id=self.episode, seed=self.seed, success=bool(success),
                latent_prediction_mse=float(np.mean([t['diagnostic']['latent_prediction_mse'] for t in self.transitions])),
                reward_absolute_error=float(np.mean([t['rq']['reward_absolute_error'] for t in self.transitions]))))
            for f, _ in self.tables.values():
                f.flush()
            self.transitions = []

    def finish(self, error=None):
        if self.closed:
            return
        self.closed = True
        self.agent._mechanism_trace = None
        for f, _ in self.tables.values():
            f.close()
        self.archive.close()
        unchanged = model_checksum(self.agent.model) == self.digest
        groups = []
        keys = ('checkpoint','task','shift_type','schedule','regime','exposed','episode_success','mpc','native_steps_since_shift')
        for group, values in self.groups.items():
            n = len(set(v[0] for v in values))
            groups.append(dict(zip(keys, group), samples=len(values), episodes=n,
                status='INSUFFICIENT_SAMPLES' if n<5 else ('INSUFFICIENT_EXPOSURE' if not group[5] and self.cfg.schedule['type']!='nominal' else 'DESCRIPTIVE_ONLY'),
                latent_prediction_mse=float(np.mean([v[1] for v in values])),
                reward_absolute_error=float(np.mean([v[2] for v in values]))))
        history = [dict(checkpoint=k[0][0], task=k[0][1], shift_type=k[0][2], schedule=k[0][3],
                        regime=k[0][4], exposed=k[0][5], episode_success=k[0][6], mpc=k[0][7],
                        empty_command_history=k[1], command_execution_mismatch=k[2], samples=len(v),
                        latent_prediction_mse=float(np.mean(v)), status='DESCRIPTIVE_ONLY_NO_CAUSAL_CLAIM')
                   for k,v in self.history_groups.items()]
        self.write_json('mechanism_summary.json', dict(groups=groups, episodes=self.episode_summaries, history_conditioned=history, error=error,
            interpretation='Associations only; no candidate correctness, epistemic uncertainty, adaptation or plasticity claim'))
        self.write_json('validity_report.json', dict(model_parameters_unchanged=unchanged,
            model_sha256_after=model_checksum(self.agent.model), exact_state_counterfactual='UNVERIFIED',
            pairing='approximate_pairing; independent closed-loop rollouts, not common-action branches',
            true_candidate_returns='UNAVAILABLE', simulator_validation='UNVERIFIED_BY_THIS_ANALYSIS',
            training_target='NOT_COMPUTED; Bellman analysis proxy only', error=error))
        if not unchanged:
            raise RuntimeError('Frozen model state changed during mechanism analysis')
