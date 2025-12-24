from typing import Any, Dict

import torch


def compute_ppo_losses(
    *,
    logits_new: torch.Tensor,     # [N, V]
    values_new: torch.Tensor,     # [N]
    batch: Dict[str, torch.Tensor],
    advantages: torch.Tensor,     # [N]
    clip_epsilon: float,
    tokenizer: Any = None,
    vocab_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Compute Phase-1 PPO losses (separate): policy, value, entropy, and return ratio.

    Notes:
    - No KL penalty, no value clipping.
    - Entropy scaling is handled outside this function.
    - No reward/advantage normalization here (advantages provided).
    - Entropy is recomputed from current policy logits (not rollout batch) and applied only on latent steps.
    - CRITICAL FIX: Reapply phase-dependent action masks before computing logprob_new and entropy.
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
    N, V = logits_new.shape
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

    # Build phase-dependent mask over vocabulary [N, V]
    if tokenizer is None:
        raise RuntimeError("tokenizer must be provided to compute_ppo_losses for phase masking.")

    # Cache token id sets on first use for this tokenizer instance
    cache_key = id(tokenizer)
    if not hasattr(compute_ppo_losses, "_token_cache"):
        compute_ppo_losses._token_cache = {}
    cache = compute_ppo_losses._token_cache  # type: ignore[attr-defined]
    if cache_key not in cache:
        # z-token strings are deterministic in Phase-1
        z_tokens = [f"<z{i}>" for i in range(vocab_size)]
        z_token_ids = tokenizer.convert_tokens_to_ids(z_tokens)
        if any(i is None for i in z_token_ids):
            missing = [t for t, i in zip(z_tokens, z_token_ids) if i is None]
            raise ValueError(f"Some z-tokens missing ids in tokenizer: {missing}")
        z_token_ids = [int(i) for i in z_token_ids]

        digit_tokens = [str(i) for i in range(10)]
        digit_token_ids = tokenizer.convert_tokens_to_ids(digit_tokens)
        if any(i is None for i in digit_token_ids):
            missing = [t for t, i in zip(digit_tokens, digit_token_ids) if i is None]
            raise ValueError(f"Some digit tokens missing ids in tokenizer: {missing}")
        digit_token_ids = [int(i) for i in digit_token_ids]

        cache[cache_key] = {
            "z_ids": torch.tensor(sorted(set(z_token_ids)), dtype=torch.long),
            "digit_ids": torch.tensor(sorted(set(digit_token_ids)), dtype=torch.long),
        }
    ids = cache[cache_key]
    z_ids: torch.Tensor = ids["z_ids"].to(logits_new.device)
    digit_ids: torch.Tensor = ids["digit_ids"].to(logits_new.device)

    # Construct mask
    mask = torch.full((N, V), -1e9, dtype=logits_new.dtype, device=logits_new.device)
    for i, phase in enumerate(phases):
        if phase == "latent":
            mask[i, z_ids] = 0.0
        elif phase == "answer":
            mask[i, digit_ids] = 0.0
        else:
            raise RuntimeError(f"Unknown phase '{phase}' in batch['phases']")

    masked_logits_new = logits_new + mask

    # Recompute log-probs under new policy with masking
    log_probs_new = torch.log_softmax(masked_logits_new, dim=-1)      # [N, V]
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

    # Entropy loss from current policy (new masked logits), applied only on latent steps
    dist_new = torch.distributions.Categorical(logits=masked_logits_new)
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

    print("\n[PPO STEP DEBUG]")
    for i in range(N):
        print(f" step {i}:")
        print(f"  phase: {phases[i]}")
        print(f"  action_id: {actions[i].item()}")
        if tokenizer is not None:
            print(f"  action_tok: {tokenizer.convert_ids_to_tokens([actions[i].item()])[0]}")
        print(f"  reward: {rewards[i].item():.3f}")
        print(f"  value_new: {values_new[i].item():.3f}")
        print(f"  advantage: {advantages[i].item():.3f}")
        print(f"  logprob_old: {logprob_old[i].item():.3f}")
        print(f"  logprob_new: {logprob_new[i].item():.3f}")
        print(f"  ratio: {ratio[i].item():.3f}")
        print(f"  entropy: {entropy_new[i].item():.3f}")
        print("-----")

    print("[PPO AGGREGATES]")
    print(" policy_loss:", policy_loss.item())
    print(" value_loss:", value_loss.item())
    print(" entropy_loss:", entropy_loss.item())
    print(" ratio_mean:", ratio.mean().item())
    print(" ratio_std:", ratio.std().item())

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "ratio": ratio,
    }
