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
    answer_temperature: float = 1.0,
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
    allowed_ids_per_step = batch.get("allowed_token_ids")  # List[List[int]] length N

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
    if allowed_ids_per_step is None or not isinstance(allowed_ids_per_step, list) or len(allowed_ids_per_step) != N:
        raise RuntimeError("batch['allowed_token_ids'] must be a list of length N")

    # Required equality for core shapes
    if not (advantages.shape == rewards.shape == values_new.shape):
        raise RuntimeError("Shape mismatch among advantages, rewards, and values_new.")

    # Build per-step mask from allowed ids recorded during rollout [N, V]
    mask = torch.full((N, V), -1e9, dtype=logits_new.dtype, device=logits_new.device)
    for i in range(N):
        allowed_ids_i = allowed_ids_per_step[i]
        if not isinstance(allowed_ids_i, list):
            raise RuntimeError(f"allowed_token_ids[{i}] must be a list")
        if len(allowed_ids_i) == 0:
            # Should only occur if phase is done, which shouldn't be in batch
            raise RuntimeError(f"Empty allowed ids at step {i}")
        idx = torch.tensor(sorted(set(int(x) for x in allowed_ids_i)), dtype=torch.long, device=logits_new.device)
        mask[i].index_fill_(0, idx, 0.0)

    # For debugging: compute unscaled masked logits as well
    masked_logits_unscaled = logits_new + mask
    masked_logits_new = (logits_new + mask) / answer_temperature

    # Recompute log-probs under new policy with masking (scaled and unscaled for diagnostics)
    log_probs_new = torch.log_softmax(masked_logits_new, dim=-1)      # [N, V]
    log_probs_new_unscaled = torch.log_softmax(masked_logits_unscaled, dim=-1)
    logprob_new = log_probs_new.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [N]
    logprob_new_unscaled = log_probs_new_unscaled.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

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

    # Detailed per-step diagnostics
    print("\n[PPO STEP DEBUG]")
    try:
        print(f" answer_temperature_used_in_loss: {answer_temperature}")
    except Exception as e:
        print(f" [DEBUG] preview failed: {e}")

    for i in range(N):
        a_id = int(actions[i].item())
        phase = phases[i]
        mask_val = float(mask[i, a_id].item())
        try:
            a_tok = tokenizer.convert_ids_to_tokens([a_id])[0] if tokenizer is not None else str(a_id)
        except Exception:
            a_tok = str(a_id)

        # LSE diagnostics
        lse_scaled = float(torch.logsumexp(masked_logits_new[i], dim=-1).item())
        lse_unscaled = float(torch.logsumexp(masked_logits_unscaled[i], dim=-1).item())
        masked_logit_scaled = float(masked_logits_new[i, a_id].item())
        masked_logit_unscaled = float(masked_logits_unscaled[i, a_id].item())

        # Top-3 allowed tokens by prob (scaled)
        probs_i = torch.softmax(masked_logits_new[i], dim=-1)
        # Only consider allowed indices for top-k
        allowed_i = (mask[i] == 0.0)
        probs_allowed = probs_i.masked_fill(~allowed_i, -1.0)
        topk_vals, topk_idx = torch.topk(probs_allowed, k=min(3, V))
        topk_idx = topk_idx.tolist()
        topk_vals = topk_vals.tolist()
        try:
            topk_toks = tokenizer.convert_ids_to_tokens(topk_idx) if tokenizer is not None else [str(x) for x in topk_idx]
        except Exception:
            topk_toks = [str(x) for x in topk_idx]

        # Allowed set preview from batch
        allowed_ids_preview = allowed_ids_per_step[i][:10]
        try:
            allowed_toks_preview = tokenizer.convert_ids_to_tokens(allowed_ids_preview) if tokenizer is not None else [str(x) for x in allowed_ids_preview]
        except Exception:
            allowed_toks_preview = [str(x) for x in allowed_ids_preview]

        print(f" step {i}:")
        print(f"  phase: {phase}")
        print(f"  action_id: {a_id}  action_tok: {a_tok}  mask_value_for_action: {mask_val}")
        print(f"  reward: {rewards[i].item():.3f}  value_new: {values_new[i].item():.6f}  advantage: {advantages[i].item():.6f}")
        print(f"  logprob_old: {logprob_old[i].item():.9f}")
        print(f"  logprob_new_scaled: {logprob_new[i].item():.9f}  logprob_new_unscaled: {logprob_new_unscaled[i].item():.9f}")
        print(f"  delta(new_scaled - old): {(logprob_new[i] - logprob_old[i]).item():.9f}")
        print(f"  LSE_scaled: {lse_scaled:.6f}  LSE_unscaled: {lse_unscaled:.6f}")
        print(f"  masked_logit_scaled[a]: {masked_logit_scaled:.6f}  masked_logit_unscaled[a]: {masked_logit_unscaled:.6f}")
        print(f"  allowed_count: {len(allowed_ids_per_step[i])}  allowed_ids[:10]: {allowed_ids_preview}  allowed_toks[:10]: {allowed_toks_preview}")
        print(f"  top_allowed_scaled: {[f'({idx},{tok},{val:.6f})' for idx, tok, val in zip(topk_idx, topk_toks, topk_vals)]}")
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
