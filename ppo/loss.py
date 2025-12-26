from typing import Any, Dict

import torch


def compute_ppo_losses(
    *,
    logits_token_new: torch.Tensor,     # [N, V]
    logits_answer_new: torch.Tensor,   # [N, 10]
    values_new: torch.Tensor,     # [N]
    batch: Dict[str, Any],
    advantages: torch.Tensor,     # [N]
    clip_epsilon: float,
    vocab_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Compute PPO losses (policy, value, entropy, ratio) for mixed action spaces.

    - Token actions use logits_token_new with vocab masking.
    - Answer actions use logits_answer_new with class masking.
    - Entropy is computed ONLY for token actions.
    - Temperature is NOT applied here (rollout-only).
    """
    if advantages.requires_grad:
        raise RuntimeError("advantages must not require gradients")
    required = ("actions", "logprob_old", "rewards", "action_kinds")
    for k in required:
        if k not in batch:
            raise KeyError(f"Batch missing required key: {k}")

    actions = batch["actions"]          # [N] long
    logprob_old = batch["logprob_old"]  # [N] float
    rewards = batch["rewards"]          # [N] float
    action_kinds = batch["action_kinds"] # List[str], length N
    allowed_ids_per_step = batch.get("allowed_action_ids")  # List[List[int]] length N
    step_uid = batch.get("step_uid", None)  # List[int]
    global_step = batch.get("global_step", None)
    episode_index = batch.get("episode_index", None)

    if logprob_old.requires_grad:
        raise RuntimeError("logprob_old must not require gradients")

    # Basic shape checks (conditional for mixed spaces)
    if logits_token_new.ndim != 2:
        raise RuntimeError(f"logits_token_new must be [N, V], got shape {tuple(logits_token_new.shape)}")
    if logits_answer_new.ndim != 2:
        raise RuntimeError(f"logits_answer_new must be [N, 10], got ndim={logits_answer_new.ndim}")
    if logits_answer_new.shape[1] != 10:
        raise RuntimeError(f"logits_answer_new second dim must be 10, got {logits_answer_new.shape[1]}")

    N = values_new.shape[0]
    # Enforce batch-size consistency across outputs
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
    if step_uid is None or not isinstance(step_uid, list) or len(step_uid) != N:
        raise RuntimeError("batch['step_uid'] must be a list of length N")
    if episode_index is None or not torch.is_tensor(episode_index) or episode_index.ndim != 1 or episode_index.shape[0] != N:
        raise RuntimeError("batch['episode_index'] must be a 1D tensor of length N")

    # Required equality for core shapes
    if not (advantages.shape == rewards.shape == values_new.shape == actions.shape == logprob_old.shape):
        raise RuntimeError("Shape mismatch among core 1D tensors.")

    # Prepare outputs per step
    logprob_new_list = []
    entropy_list = []

    # Iterate per step; build masked logits and compute logprob and entropy as specified
    for i in range(N):
        kind = action_kinds[i]
        allowed_ids_i = allowed_ids_per_step[i]
        if not isinstance(allowed_ids_i, list):
            raise RuntimeError(f"allowed_action_ids[{i}] must be a list")
        if len(allowed_ids_i) == 0:
            raise RuntimeError(f"Empty allowed ids at step {i}")

        if kind == "token":
            # Use vocab logits; apply vocab mask
            v = logits_token_new.shape[1]
            logits_i = logits_token_new[i]
            if logits_i.shape[0] != v:
                raise RuntimeError(f"Token logits length mismatch at step {i}: {logits_i.shape[0]} vs {v}")
            # Validate allowed ids range for vocab
            for id_ in allowed_ids_i:
                if not (0 <= int(id_) < v):
                    raise RuntimeError(
                        f"Out-of-range allowed id at step {i} (token): id={id_}, valid range [0, {vocab_size})"
                    )
            # Device-safe idx on logits_i.device
            idx = torch.tensor(sorted(set(int(x) for x in allowed_ids_i)), dtype=torch.long, device=logits_i.device)
            # Build mask using smallest finite value to strongly exclude
            neg_large = torch.finfo(logits_i.dtype).min
            mask_i = torch.full((v,), neg_large, dtype=logits_i.dtype, device=logits_i.device)
            mask_i.index_fill_(0, idx, 0.0)
            masked_logits_i = logits_i + mask_i  # no temperature
            log_probs_i = torch.log_softmax(masked_logits_i, dim=-1)
            # Validate chosen action correctness
            a_i = int(actions[i])
            if a_i not in allowed_ids_i:
                print("[PPO CHECK] action not in allowed ids (token)", {
                    "i": i, "a": a_i, "allowed": allowed_ids_i, "step_uid": step_uid[i]
                })
                raise RuntimeError(
                    f"Chosen action not in allowed ids at step {i} (token): action={a_i}, allowed={allowed_ids_i}"
                )
            logprob_new_list.append(log_probs_i[a_i])
            # Entropy only for token steps
            dist_i = torch.distributions.Categorical(logits=masked_logits_i)
            entropy_list.append(dist_i.entropy())
            # Debug instrumentation: print first few steps and assert invariant on epoch 0
            if i < 10:
                delta = float(log_probs_i[a_i].item() - logprob_old[i].item())
                ratio_dbg = float(torch.exp(log_probs_i[a_i] - logprob_old[i]).item())
                print(
                    "[PPO CHECK] step_uid=" + str(step_uid[i]) +
                    ", epi=" + str(int(episode_index[i].item())) +
                    ", kind=token" +
                    ", action=" + str(a_i) +
                    ", allowed_ids=" + str(allowed_ids_i) +
                    ", logprob_old=" + str(float(logprob_old[i].item())) +
                    ", logprob_new=" + str(float(log_probs_i[a_i].item())) +
                    ", delta=" + str(delta) +
                    ", ratio=" + str(ratio_dbg)
                )
                if abs(delta) > 1e-3:
                    print("[PPO CHECK] MISMATCH CONTEXT (token)", {
                        "i": i,
                        "epi": int(episode_index[i].item()),
                        "step_uid": step_uid[i],
                        "masked_logits": masked_logits_i.detach().cpu().tolist(),
                        "mask_nonzero_count": int((mask_i != 0).sum().item()),
                        "idx_used": sorted(set(int(x) for x in allowed_ids_i)),
                    })
                if isinstance(global_step, int) and global_step == 0:
                    assert abs(delta) < 1e-3, "logprob_old/new mismatch at identical policy (token)"
        elif kind == "answer":
            # Use answer logits; apply class mask
            c = logits_answer_new.shape[1]
            logits_i = logits_answer_new[i]
            if c != 10:
                raise RuntimeError(f"Answer logits second dim must be 10, got {c}")
            if logits_i.shape[0] != c:
                raise RuntimeError(f"Answer logits length mismatch at step {i}: {logits_i.shape[0]} vs {c}")
            # Validate allowed ids range for classes
            for id_ in allowed_ids_i:
                if not (0 <= int(id_) < 10):
                    raise RuntimeError(
                        f"Out-of-range allowed id at step {i} (answer): id={id_}, valid range [0, 10)"
                    )
            # Device-safe idx on logits_i.device
            idx = torch.tensor(sorted(set(int(x) for x in allowed_ids_i)), dtype=torch.long, device=logits_i.device)
            neg_large = torch.finfo(logits_i.dtype).min
            mask_i = torch.full((c,), neg_large, dtype=logits_i.dtype, device=logits_i.device)
            mask_i.index_fill_(0, idx, 0.0)
            masked_logits_i = logits_i + mask_i  # no temperature
            log_probs_i = torch.log_softmax(masked_logits_i, dim=-1)
            # Validate chosen action correctness
            a_i = int(actions[i].item()) if actions.dtype != torch.long else int(actions[i])
            if not (0 <= a_i < 10):
                raise RuntimeError(
                    f"Chosen action out of range at step {i} (answer): action={a_i}, valid range [0, 10)"
                )
            if a_i not in allowed_ids_i:
                print("[PPO CHECK] action not in allowed ids (answer)", {
                    "i": i, "a": a_i, "allowed": allowed_ids_i, "step_uid": step_uid[i]
                })
                raise RuntimeError(
                    f"Chosen action not in allowed ids at step {i} (answer): action={a_i}, allowed={allowed_ids_i}"
                )
            logprob_new_list.append(log_probs_i[a_i])
            # Entropy is zero for answer steps
            entropy_list.append(torch.zeros((), dtype=logits_i.dtype, device=logits_i.device))
            # Debug instrumentation
            if i < 10:
                delta = float(log_probs_i[a_i].item() - logprob_old[i].item())
                ratio_dbg = float(torch.exp(log_probs_i[a_i] - logprob_old[i]).item())
                print(
                    "[PPO CHECK] step_uid=" + str(step_uid[i]) +
                    ", epi=" + str(int(episode_index[i].item())) +
                    ", kind=answer" +
                    ", action=" + str(a_i) +
                    ", allowed_ids=" + str(allowed_ids_i) +
                    ", logprob_old=" + str(float(logprob_old[i].item())) +
                    ", logprob_new=" + str(float(log_probs_i[a_i].item())) +
                    ", delta=" + str(delta) +
                    ", ratio=" + str(ratio_dbg)
                )
                if abs(delta) > 1e-3:
                    print("[PPO CHECK] MISMATCH CONTEXT (answer)", {
                        "i": i,
                        "epi": int(episode_index[i].item()),
                        "step_uid": step_uid[i],
                        "masked_logits": masked_logits_i.detach().cpu().tolist(),
                        "mask_nonzero_count": int((mask_i != 0).sum().item()),
                        "idx_used": sorted(set(int(x) for x in allowed_ids_i)),
                    })
                if isinstance(global_step, int) and global_step == 0:
                    assert abs(delta) < 1e-3, "logprob_old/new mismatch at identical policy (answer)"
        else:
            raise RuntimeError(f"Unknown action kind at step {i}: {kind}")

    # Stack results
    logprob_new = torch.stack(logprob_new_list)  # [N]
    entropy = torch.stack(entropy_list)          # [N]

    # Importance sampling ratio
    ratio = torch.exp(logprob_new - logprob_old)  # [N]

    # Numerical safety checks for ratio
    if not torch.isfinite(ratio).all():
        raise RuntimeError("Non-finite values in importance sampling ratio.")

    # Clipped surrogate policy loss (includes both token and answer steps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    # Value loss: no clipping, no coefficient here
    value_loss = torch.mean((values_new - rewards) ** 2)

    # Entropy loss: mean over token steps only (entropy for answer steps is zero)
    token_mask = torch.tensor([k == "token" for k in action_kinds], device=entropy.device, dtype=torch.float32)
    token_count = token_mask.sum()
    if token_count.item() > 0:
        entropy_loss = -((entropy * token_mask).sum() / token_count)
    else:
        entropy_loss = torch.zeros((), device=entropy.device)

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
