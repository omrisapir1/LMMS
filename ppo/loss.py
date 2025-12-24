from typing import Any, Dict

import torch


def compute_ppo_losses(
    *,
    logits_new: torch.Tensor,     # [N, V]
    values_new: torch.Tensor,     # [N]
    batch: Dict[str, torch.Tensor],
    advantages: torch.Tensor,     # [N]
    clip_epsilon: float,
) -> Dict[str, torch.Tensor]:
    """
    Compute Phase-1 PPO losses (separate): policy, value, entropy, and return ratio.

    Notes:
    - No KL penalty, no value clipping.
    - Entropy scaling is handled outside this function.
    - No reward/advantage normalization here (advantages provided).
    - Entropy is recomputed from current policy logits (not rollout batch) and applied only on latent steps.
    """
    if advantages.requires_grad:
        raise RuntimeError("advantages must not require gradients")
    required = ("actions", "logprob_old", "rewards", "phases")
    for k in required:
        if k not in batch:
            raise KeyError(f"Batch missing required key: {k}")

    actions = batch["actions"]          # [N] long
    logprob_old = batch["logprob_old"]  # [N] float
    rewards = batch["rewards"]          # [N] float
    phases = batch["phases"]            # List[str] length N

    if logprob_old.requires_grad:
        raise RuntimeError("logprob_old must not require gradients")

    # Basic shape checks
    if logits_new.ndim != 2:
        raise RuntimeError(f"logits_new must be [N, V], got shape {tuple(logits_new.shape)}")
    N = logits_new.shape[0]
    for name, t in {
        "values_new": values_new,
        "actions": actions,
        "logprob_old": logprob_old,
        "rewards": rewards,
        "advantages": advantages,
    }.items():
        if t.ndim != 1:
            raise RuntimeError(f"{name} must be 1D [N], got ndim={t.ndim}")
        if t.shape[0] != N:
            raise RuntimeError(f"{name} length {t.shape[0]} != N {N}")
    if not isinstance(phases, list) or len(phases) != N:
        raise RuntimeError("batch['phases'] must be a list of length N")

    # Required equality for core shapes
    if not (advantages.shape == rewards.shape == values_new.shape):
        raise RuntimeError("Shape mismatch among advantages, rewards, and values_new.")

    # Recompute log-probs under new policy
    log_probs_new = torch.log_softmax(logits_new, dim=-1)      # [N, V]
    logprob_new = log_probs_new.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [N]

    # Importance sampling ratio
    ratio = torch.exp(logprob_new - logprob_old)  # [N]

    # Numerical safety checks for ratio
    if not torch.isfinite(ratio).all():
        raise RuntimeError("Non-finite values in importance sampling ratio.")

    # Clipped surrogate policy loss (includes both latent and answer steps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    # Value loss: no clipping, no coefficient here
    value_loss = torch.mean((values_new - rewards) ** 2)

    # Entropy loss from current policy (new logits), applied only on latent steps
    dist_new = torch.distributions.Categorical(logits=logits_new)
    entropy_new = dist_new.entropy()  # [N]
    latent_mask = torch.tensor([p == "latent" for p in phases], device=entropy_new.device, dtype=torch.float32)
    latent_count = latent_mask.sum()
    if latent_count.item() > 0:
        entropy_loss = -((entropy_new * latent_mask).sum() / latent_count)
    else:
        entropy_loss = torch.zeros((), device=entropy_new.device)

    # Final numerical safety checks
    for name, t in {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "ratio": ratio,
    }.items():
        if not torch.isfinite(t).all():
            raise RuntimeError(f"Non-finite values detected in {name}.")

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "ratio": ratio,
    }
