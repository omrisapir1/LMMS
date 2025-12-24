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
    # Show short summaries of allowed id sets
    try:
        z_ids_list = z_ids[:10].tolist()
        digit_ids_list = digit_ids[:10].tolist()
        z_toks = tokenizer.convert_ids_to_tokens(z_ids_list) if tokenizer is not None else [str(i) for i in z_ids_list]
        digit_toks = tokenizer.convert_ids_to_tokens(digit_ids_list) if tokenizer is not None else [str(i) for i in digit_ids_list]
        print(f" z_ids[:10]: {z_ids_list}  z_toks[:10]: {z_toks}")
        print(f" digit_ids[:10]: {digit_ids_list}  digit_toks[:10]: {digit_toks}")
        print(f" answer_temperature_used_in_loss: {answer_temperature}")
    except Exception as e:
        print(f" [DEBUG] token id preview failed: {e}")

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

        print(f" step {i}:")
        print(f"  phase: {phase}")
        print(f"  action_id: {a_id}  action_tok: {a_tok}  mask_value_for_action: {mask_val}")
        print(f"  reward: {rewards[i].item():.3f}  value_new: {values_new[i].item():.6f}  advantage: {advantages[i].item():.6f}")
        print(f"  logprob_old: {logprob_old[i].item():.9f}")
        print(f"  logprob_new_scaled: {logprob_new[i].item():.9f}  logprob_new_unscaled: {logprob_new_unscaled[i].item():.9f}")
        print(f"  delta(new_scaled - old): {(logprob_new[i] - logprob_old[i]).item():.9f}")
        print(f"  LSE_scaled: {lse_scaled:.6f}  LSE_unscaled: {lse_unscaled:.6f}")
        print(f"  masked_logit_scaled[a]: {masked_logit_scaled:.6f}  masked_logit_unscaled[a]: {masked_logit_unscaled:.6f}")
        # Allowed set size preview
        allowed_count_i = int(allowed_i.sum().item())
        allowed_ids_preview = torch.nonzero(allowed_i, as_tuple=False).view(-1)[:10].tolist()
        try:
            allowed_toks_preview = tokenizer.convert_ids_to_tokens(allowed_ids_preview) if tokenizer is not None else [str(x) for x in allowed_ids_preview]
        except Exception:
            allowed_toks_preview = [str(x) for x in allowed_ids_preview]
        print(f"  allowed_count: {allowed_count_i}  allowed_ids[:10]: {allowed_ids_preview}  allowed_toks[:10]: {allowed_toks_preview}")
        # Top-k within allowed
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
