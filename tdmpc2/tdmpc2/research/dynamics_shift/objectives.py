"""Numerical expression shared exactly by predictor-only and D+Q updates."""

import torch.nn.functional as F


def one_step_dynamics_mse(model, encoded_observation, action, encoded_next_observation):
    return F.mse_loss(model.next(encoded_observation, action, None), encoded_next_observation)
