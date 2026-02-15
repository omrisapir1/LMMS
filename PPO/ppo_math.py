from __future__ import annotations

from typing import Tuple

import torch


def clipped_policy_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ratio = torch.exp(logp_new - logp_old)
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    obj_1 = ratio * advantages
    obj_2 = clipped * advantages
    policy_loss = -torch.mean(torch.minimum(obj_1, obj_2))
    clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_range).to(torch.float32))
    return policy_loss, clipfrac


def value_mse_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return torch.mean((values - returns) ** 2)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_true_var = torch.var(y_true)
    if float(y_true_var) < 1e-12:
        return 0.0
    return float(1.0 - torch.var(y_true - y_pred) / y_true_var)
