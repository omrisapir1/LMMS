from __future__ import annotations

import torch

from PPO.ppo_math import clipped_policy_loss


def test_clipped_policy_loss_and_clipfrac() -> None:
    logp_old = torch.log(torch.tensor([0.5, 0.5]))
    logp_new = torch.log(torch.tensor([0.8, 0.2]))
    adv = torch.tensor([1.0, -1.0])

    loss, clipfrac = clipped_policy_loss(
        logp_new=logp_new,
        logp_old=logp_old,
        advantages=adv,
        clip_range=0.2,
    )

    assert torch.isfinite(loss)
    # Both ratios are outside [0.8, 1.2], so clipfrac should be 1.
    assert abs(float(clipfrac.item()) - 1.0) < 1e-8
