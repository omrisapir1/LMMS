from typing import Any, Dict, List

import torch


def compute_ppo_losses(
    *,
    logits_token_new: torch.Tensor,     # [N, V]
    logits_answer_new: torch.Tensor,    # [N, 10]
    values_new: torch.Tensor,           # [N]
    batch: Dict[str, Any],
    advantages: torch.Tensor,           # [N]
    clip_epsilon: float,
    vocab_size: int = 64,               # kept for error text compatibility
) -> Dict[str, torch.Tensor]:
    """
    Compute PPO losses (policy, value, entropy, ratio) for mixed action spaces.

    - Token actions use logits_token_new with vocab masking.
    - Answer actions use logits_answer_new with class masking.
    - Entropy is computed ONLY for token actions (answer steps contribute zero entropy).
    - Temperature is NOT applied here (rollout-only).
    """
    if advantages.requires_grad:
        raise RuntimeError("advantages must not require gradients")

    required = ("actions", "logprob_old", "rewards", "action_kinds")
    for k in required:
        if k not in batch:
            raise KeyError(f"Batch missing required key: {k}")

    actions: torch.Tensor = batch["actions"]          # [N]
    logprob_old: torch.Tensor = batch["logprob_old"]  # [N]
    rewards: torch.Tensor = batch["rewards"]          # [N]
    action_kinds: List[str] = batch["action_kinds"]   # list length N
    allowed_ids_per_step = batch.get("allowed_action_ids")  # List[List[int]] length N

    if logprob_old.requires_grad:
        raise RuntimeError("logprob_old must not require gradients")

    # Basic shape checks
    if logits_token_new.ndim != 2:
        raise RuntimeError(f"logits_token_new must be [N, V], got shape {tuple(logits_token_new.shape)}")
    if logits_answer_new.ndim != 2:
        raise RuntimeError(f"logits_answer_new must be [N, 10], got ndim={logits_answer_new.ndim}")
    if logits_answer_new.shape[1] != 10:
        raise RuntimeError(f"logits_answer_new second dim must be 10, got {logits_answer_new.shape[1]}")

    N = values_new.shape[0]
    if logits_token_new.shape[0] != N:
        raise RuntimeError(f"Batch-size mismatch: logits_token_new.shape[0]={logits_token_new.shape[0]} != N={N}")
    if logits_answer_new.shape[0] != N:
        raise RuntimeError(f"Batch-size mismatch: logits_answer_new.shape[0]={logits_answer_new.shape[0]} != N={N}")

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

    if not isinstance(action_kinds, list) or len(action_kinds) != N:
        raise RuntimeError("batch['action_kinds'] must be a list of length N")
    if allowed_ids_per_step is None or not isinstance(allowed_ids_per_step, list) or len(allowed_ids_per_step) != N:
        raise RuntimeError("batch['allowed_action_ids'] must be a list of length N")

    if not (advantages.shape == rewards.shape == values_new.shape == actions.shape == logprob_old.shape):
        raise RuntimeError("Shape mismatch among core 1D tensors.")

    # Per-step outputs (must end up length N)
    logprob_new_list: List[torch.Tensor] = []
    entropy_list: List[torch.Tensor] = []

    for i in range(N):
        kind = action_kinds[i]
        allowed_ids_i = allowed_ids_per_step[i]
        if not isinstance(allowed_ids_i, list):
            raise RuntimeError(f"allowed_action_ids[{i}] must be a list")
        if len(allowed_ids_i) == 0:
            raise RuntimeError(f"Empty allowed ids at step {i}")

        if kind == "token":
            v = logits_token_new.shape[1]
            logits_i = logits_token_new[i]
            if logits_i.shape[0] != v:
                raise RuntimeError(f"Token logits length mismatch at step {i}: {logits_i.shape[0]} vs {v}")

            for id_ in allowed_ids_i:
                if not (0 <= int(id_) < v):
                    raise RuntimeError(
                        f"Out-of-range allowed id at step {i} (token): id={id_}, valid range [0, {v})"
                    )

            idx = torch.tensor(sorted(set(int(x) for x in allowed_ids_i)), dtype=torch.long, device=logits_i.device)

            neg_large = torch.finfo(logits_i.dtype).min
            mask_i = torch.full((v,), neg_large, dtype=logits_i.dtype, device=logits_i.device)
            mask_i.index_fill_(0, idx, 0.0)
            masked_logits_i = logits_i + mask_i

            log_probs_i = torch.log_softmax(masked_logits_i, dim=-1)
            a_i = int(actions[i].item()) if torch.is_tensor(actions) else int(actions[i])

            # Validate chosen action correctness
            if a_i not in allowed_ids_i:
                raise RuntimeError(
                    f"Chosen action not in allowed ids at step {i} (token): action={a_i}, allowed={allowed_ids_i}"
                )

            # Append logprob_new for this step
            logprob_new_list.append(log_probs_i[a_i])

            # Entropy for token steps
            dist_i = torch.distributions.Categorical(logits=masked_logits_i)
            entropy_list.append(dist_i.entropy())

            # Debug first 3 steps
            if True:
                print({
                    "step": i,
                    "kind": "token",
                    "action": a_i,
                    "logprob_old": float(logprob_old[i].item()),
                    "logprob_new": float(log_probs_i[a_i].item()),
                    "advantage": float(advantages[i].item()),
                })

        elif kind == "answer":
            c = logits_answer_new.shape[1]
            logits_i = logits_answer_new[i]
            if c != 10:
                raise RuntimeError(f"Answer logits second dim must be 10, got {c}")
            if logits_i.shape[0] != c:
                raise RuntimeError(f"Answer logits length mismatch at step {i}: {logits_i.shape[0]} vs {c}")

            for id_ in allowed_ids_i:
                if not (0 <= int(id_) < 10):
                    raise RuntimeError(
                        f"Out-of-range allowed id at step {i} (answer): id={id_}, valid range [0, 10)"
                    )

            idx = torch.tensor(sorted(set(int(x) for x in allowed_ids_i)), dtype=torch.long, device=logits_i.device)

            neg_large = torch.finfo(logits_i.dtype).min
            mask_i = torch.full((c,), neg_large, dtype=logits_i.dtype, device=logits_i.device)
            mask_i.index_fill_(0, idx, 0.0)
            masked_logits_i = logits_i + mask_i

            log_probs_i = torch.log_softmax(masked_logits_i, dim=-1)
            a_i = int(actions[i].item()) if actions.dtype == torch.long else int(actions[i])

            # Validate chosen action correctness
            if not (0 <= a_i < 10):
                raise RuntimeError(f"Chosen action out of range at step {i} (answer): action={a_i}, valid [0,10)")
            if a_i not in allowed_ids_i:
                raise RuntimeError(
                    f"Chosen action not in allowed ids at step {i} (answer): action={a_i}, allowed={allowed_ids_i}"
                )

            # Append logprob_new for this step (FIXED)
            logprob_new_list.append(log_probs_i[a_i])

            # Entropy is defined as 0 for answer steps (FIXED to keep length N)
            entropy_list.append(torch.zeros((), device=logits_i.device, dtype=logits_i.dtype))

            # Debug first 3 steps
            if i True:
                print({
                    "step": i,
                    "kind": "answer",
                    "action": a_i,
                    "logprob_old": float(logprob_old[i].item()),
                    "logprob_new": float(log_probs_i[a_i].item()),
                    "advantage": float(advantages[i].item()),
                })

        else:
            raise RuntimeError(f"Unknown action kind at step {i}: {kind}")

    # Stack results (now guaranteed length N)
    if len(logprob_new_list) != N:
        raise RuntimeError(f"logprob_new_list length {len(logprob_new_list)} != N {N}")
    if len(entropy_list) != N:
        raise RuntimeError(f"entropy_list length {len(entropy_list)} != N {N}")

    logprob_new = torch.stack(logprob_new_list)  # [N]
    entropy = torch.stack(entropy_list)          # [N]

    ratio = torch.exp(logprob_new - logprob_old)  # [N]
    if not torch.isfinite(ratio).all():
        raise RuntimeError("Non-finite values in importance sampling ratio.")

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    value_loss = torch.mean((values_new - rewards) ** 2)

    # Entropy loss: mean over token steps only
    token_mask = torch.tensor([k == "token" for k in action_kinds], device=entropy.device, dtype=torch.float32)
    token_count = token_mask.sum()
    if token_count.item() > 0:
        entropy_loss = -((entropy * token_mask).sum() / token_count)
    else:
        entropy_loss = torch.zeros((), device=entropy.device)

    for name, t in {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "ratio": ratio,
    }.items():
        if not torch.isfinite(t).all():
            raise RuntimeError(f"Non-finite values detected in {name}.")

    # Debug: sanity assertions
    assert not logprob_old.requires_grad
    assert logprob_new.requires_grad
    assert ratio.mean().item() < 5.0
    assert ("temperature" not in compute_ppo_losses.__code__.co_names), "Temperature found in PPO loss code"

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "ratio": ratio,
    }
