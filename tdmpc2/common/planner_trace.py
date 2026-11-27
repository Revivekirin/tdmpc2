"""Observer of the actual MPPI computation. Never samples or changes tensors."""
import numpy as np
from common.mechanism_metrics import ranks, ranking_comparison


def array(tensor):
    return tensor.detach().cpu().numpy().copy()


class PlannerTrace:
    def __init__(self):
        self.iterations = []
        self.parts = None
        self.selection = None
        self.encoded_z = None

    def encoded(self, z):
        self.encoded_z = array(z)

    def value(self, reward, terminal):
        self.parts = (array(reward).reshape(-1), array(terminal).reshape(-1))

    def iteration(self, actions, score, elites, num_prior):
        reward, terminal = self.parts
        values = array(score).reshape(-1)
        self.iterations.append(dict(actions=array(actions), actual_planner_score=values,
                                    predicted_reward_sum=reward, terminal_q_contribution=terminal,
                                    elite_ids=array(elites), candidate_rank=ranks(values), num_prior=num_prior))

    def selected(self, candidate, first_action):
        self.selection = dict(selected_candidate_id=int(candidate.item()), selected_first_action=array(first_action))

    def summaries(self):
        for iteration, row in enumerate(self.iterations):
            yield dict(iteration_id=iteration,
                       actual_reward_vs_q=ranking_comparison(row['predicted_reward_sum'], row['terminal_q_contribution']),
                       actual_total_vs_reward=ranking_comparison(row['actual_planner_score'], row['predicted_reward_sum']),
                       actual_total_vs_q=ranking_comparison(row['actual_planner_score'], row['terminal_q_contribution']))
