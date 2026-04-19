from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from PPO.reward import compute_reward


@dataclass
class GRPOTrajectory:
    prompt_id: int
    sample_id: str
    question: str
    prompt_ids: List[int]
    prompt_attention_mask: List[int]
    actions: List[int]
    action_types: List[str]
    old_logp: List[float]
    advantages: List[float]
    returns: List[float]
    reward_info: Dict[str, object]


def _extract_z_phase_from_vllm_row_with_budget(
    *,
    row: Dict[str, object],
    answer_token_id: int,
    budget: int,
) -> Tuple[List[int], bool]:
    if int(budget) <= 0:
        return [], False
    token_ids_full = [int(x) for x in list(row.get("token_ids", []))]
    token_ids = token_ids_full[: int(budget)]
    was_truncated_by_budget = len(token_ids_full) > int(budget)
    if int(answer_token_id) in token_ids:
        pos = token_ids.index(int(answer_token_id))
        return token_ids[:pos], True

    has_answer = False
    if (not was_truncated_by_budget) and row.get("stop_reason") is not None:
        try:
            has_answer = int(row.get("stop_reason")) == int(answer_token_id)
        except Exception:
            has_answer = False
    return token_ids, bool(has_answer)


def _group_mean_center(values: Sequence[float]) -> List[float]:
    if len(values) == 0:
        return []
    mean_val = float(sum(float(x) for x in values) / float(len(values)))
    return [float(x) - mean_val for x in values]


def _action_types_for_z(z_len: int) -> List[str]:
    if int(z_len) < 0:
        raise ValueError("z_len must be >= 0")
    return ["z"] * int(z_len) + ["answer"]


def _action_types_for_digits(num_digits: int = 5) -> List[str]:
    return ["digit"] * int(num_digits)


def _iter_batches(seq: Sequence[int], batch_size: int) -> List[List[int]]:
    bsz = max(int(batch_size), 1)
    out: List[List[int]] = []
    i = 0
    n = len(seq)
    while i < n:
        out.append([int(x) for x in seq[i : i + bsz]])
        i += bsz
    return out


def collect_grpo_batch(
    *,
    prepared: Sequence[Dict[str, object]],
    tokenizer,
    vllm_engine: Any,
    cfg,
    answer_token_id: int,
    digit_token_ids: Sequence[int],
    reward_rng: torch.Generator,
    answer_logit_bias: float = 0.0,
) -> Tuple[List[GRPOTrajectory], Dict[str, float]]:
    if len(prepared) == 0:
        return [], {}

    supports_token_prompts = bool(vllm_engine.supports_prompt_token_ids())
    if not supports_token_prompts:
        raise RuntimeError("GRPO rollout requires vLLM prompt_token_ids support")

    n_z = int(cfg.rollout.n_z_traces)
    n_digit = int(cfg.rollout.n_digit_traces)
    vllm_batch_size = max(int(getattr(cfg.rollout, "vllm_batch_size", 1)), 1)
    if n_z <= 0 or n_digit <= 0:
        raise ValueError("rollout.n_z_traces and rollout.n_digit_traces must be > 0")

    digit_allowed_set = set(int(x) for x in digit_token_ids)
    digit_to_value = {int(tok): idx for idx, tok in enumerate(digit_token_ids)}

    trajectories: List[GRPOTrajectory] = []
    total_digit_rollouts = 0
    total_exact = 0

    prepared_idx = list(range(len(prepared)))
    z_infos_by_prompt: Dict[int, List[Dict[str, object]]] = {}

    # Z phase batched by rollout.vllm_batch_size.
    for idx_batch in _iter_batches(prepared_idx, vllm_batch_size):
        batch_prompt_token_ids = [
            [int(x) for x in list(prepared[i]["prompt_ids"])]
            for i in idx_batch
        ]
        z_rows = vllm_engine.generate_z(
            prompt_token_ids=batch_prompt_token_ids,
            num_samples_per_prompt=n_z,
            max_new_tokens=int(cfg.rollout.max_z_new_tokens),
            temperature=float(cfg.rollout.z_temperature),
            top_p=float(cfg.rollout.z_top_p),
            min_p=float(cfg.rollout.z_min_p),
            repetition_penalty=float(cfg.rollout.z_repetition_penalty),
            greedy=False,
            logit_bias={int(answer_token_id): float(answer_logit_bias)} if float(answer_logit_bias) != 0.0 else None,
        )
        expected_z = len(idx_batch) * n_z
        if len(z_rows) != expected_z:
            raise RuntimeError(f"Z rollout count mismatch: got={len(z_rows)} expected={expected_z}")

        row_pos = 0
        for i in idx_batch:
            item = prepared[i]
            prompt_ids = [int(x) for x in list(item["prompt_ids"])]
            prompt_attn = [int(x) for x in list(item["prompt_attention_mask"])]
            prompt_id = int(item["prompt_id"])

            z_infos: List[Dict[str, object]] = []
            for zi in range(n_z):
                row = dict(z_rows[row_pos])
                row_pos += 1
                z_prefix, has_answer = _extract_z_phase_from_vllm_row_with_budget(
                    row=row,
                    answer_token_id=int(answer_token_id),
                    budget=int(cfg.rollout.max_z_new_tokens),
                )
                if not has_answer:
                    raise RuntimeError(
                        "GRPO Z phase must end with <ANSWER>; increase rollout.max_z_new_tokens or adjust sampling"
                    )
                z_actions = [int(x) for x in z_prefix] + [int(answer_token_id)]
                z_types = _action_types_for_z(len(z_prefix))
                z_infos.append(
                    {
                        "z_index": int(zi),
                        "z_actions": z_actions,
                        "z_action_types": z_types,
                        "prefix_ids": prompt_ids + z_actions,
                        "prefix_attn": prompt_attn + [1] * len(z_actions),
                        "digit_rewards": [],
                    }
                )
            z_infos_by_prompt[prompt_id] = z_infos

    # Digit phase batched by rollout.vllm_batch_size (over parent Z prompts).
    digit_jobs: List[Tuple[Dict[str, object], Dict[str, object]]] = []
    for item in prepared:
        prompt_id = int(item["prompt_id"])
        if prompt_id not in z_infos_by_prompt:
            raise RuntimeError(f"Missing z infos for prompt_id={prompt_id}")
        for z in z_infos_by_prompt[prompt_id]:
            digit_jobs.append((item, z))

    for job_batch_idx in _iter_batches(list(range(len(digit_jobs))), vllm_batch_size):
        batch_jobs = [digit_jobs[j] for j in job_batch_idx]
        digit_prompt_token_ids = [[int(x) for x in list(z["prefix_ids"])] for (_item, z) in batch_jobs]
        digit_rows = vllm_engine.generate_digits(
            prompt_token_ids=digit_prompt_token_ids,
            num_samples_per_prompt=n_digit,
            num_digits=5,
            temperature=float(cfg.rollout.digit_temperature),
            top_p=float(cfg.rollout.digit_top_p),
            greedy=bool(cfg.rollout.digit_greedy),
            min_p=0.0,
            repetition_penalty=1.0,
        )
        expected_digits = len(batch_jobs) * n_digit
        if len(digit_rows) != expected_digits:
            raise RuntimeError(
                f"Digit rollout count mismatch: got={len(digit_rows)} expected={expected_digits}"
            )

        row_pos = 0
        for item, z in batch_jobs:
            prompt_id = int(item["prompt_id"])
            question = str(item["question"])
            true_digits = [int(x) for x in list(item["true_digits"])]
            sample_base = str(item["sample_id_base"])
            z_idx = int(z["z_index"])

            for di in range(n_digit):
                digit_ids = [int(x) for x in list(digit_rows[row_pos])]
                row_pos += 1
                if len(digit_ids) != 5:
                    raise RuntimeError(f"Digit phase must return exactly 5 tokens, got {len(digit_ids)}")
                bad = [tid for tid in digit_ids if tid not in digit_allowed_set]
                if bad:
                    raise RuntimeError(f"Digit phase emitted non-digit token ids: {bad}")

                pred_digits = [int(digit_to_value[int(tid)]) for tid in digit_ids]
                reward_info = compute_reward(
                    pred_digits=pred_digits,
                    true_digits=true_digits,
                    terminated_reason="answer_with_5_digits",
                    partial_scale=float(cfg.reward.partial_scale),
                    keep_prob=cfg.reward.keep_prob,
                    length_penalty=float(cfg.reward.length_penalty),
                    correct_length_discount=float(cfg.reward.correct_length_discount),
                    reward_if_max_len=float(cfg.reward.reward_if_max_len),
                    num_generated_tokens=len(z["z_actions"]) + 5,
                    generator=reward_rng,
                )
                reward_scalar = float(reward_info["reward_final"])
                total_digit_rollouts += 1
                total_exact += int(bool(reward_info.get("exact_match", False)))
                z["digit_rewards"].append(reward_scalar)

                child_mean = float(
                    sum(float(x) for x in z["digit_rewards"]) / float(len(z["digit_rewards"]))
                )
                trajectories.append(
                    GRPOTrajectory(
                        prompt_id=prompt_id,
                        sample_id=f"{sample_base}_z{z_idx}_d{di}",
                        question=question,
                        prompt_ids=list(z["prefix_ids"]),
                        prompt_attention_mask=list(z["prefix_attn"]),
                        actions=list(digit_ids),
                        action_types=_action_types_for_digits(5),
                        old_logp=[],
                        advantages=[0.0] * 5,
                        returns=[float(reward_scalar)] * 5,
                        reward_info={
                            "phase": "digit",
                            "prompt_id": prompt_id,
                            "z_index": z_idx,
                            "digit_index": di,
                            "reward_final": float(reward_scalar),
                            "exact_match": bool(reward_info.get("exact_match", False)),
                            "pred_digits": pred_digits,
                            "true_digits": list(true_digits),
                            "partial": float(reward_info.get("reward_partial", 0.0)),
                            "parent_reward_running_mean": float(child_mean),
                        },
                    )
                )

    # Advantages and Z trajectories per prompt.
    for item in prepared:
        prompt_id = int(item["prompt_id"])
        question = str(item["question"])
        true_digits = [int(x) for x in list(item["true_digits"])]
        prompt_ids = [int(x) for x in list(item["prompt_ids"])]
        prompt_attn = [int(x) for x in list(item["prompt_attention_mask"])]
        sample_base = str(item["sample_id_base"])
        z_infos = z_infos_by_prompt.get(prompt_id, [])
        if len(z_infos) != n_z:
            raise RuntimeError(f"Missing Z infos for prompt_id={prompt_id}: got={len(z_infos)} expected={n_z}")

        for z in z_infos:
            z_idx = int(z["z_index"])
            rewards = [float(x) for x in list(z["digit_rewards"])]
            if len(rewards) != n_digit:
                raise RuntimeError(
                    f"Digit reward count mismatch for prompt_id={prompt_id}, z={z_idx}: "
                    f"got={len(rewards)} expected={n_digit}"
                )
            centered = _group_mean_center(rewards)
            match = [
                t
                for t in trajectories
                if int(t.prompt_id) == prompt_id
                and str(t.reward_info.get("phase")) == "digit"
                and int(t.reward_info.get("z_index", -1)) == z_idx
            ]
            if len(match) != len(centered):
                raise RuntimeError("Digit grouping mismatch while assigning centered advantages")
            for t, adv in zip(match, centered):
                t.advantages = [float(adv)] * len(t.actions)
                t.reward_info["adv_centered"] = float(adv)

        z_rewards = [
            float(sum(float(x) for x in list(z["digit_rewards"])) / float(n_digit))
            for z in z_infos
        ]
        z_centered = _group_mean_center(z_rewards)
        for z, z_r, z_adv in zip(z_infos, z_rewards, z_centered):
            z_idx = int(z["z_index"])
            trajectories.append(
                GRPOTrajectory(
                    prompt_id=prompt_id,
                    sample_id=f"{sample_base}_z{z_idx}",
                    question=question,
                    prompt_ids=list(prompt_ids),
                    prompt_attention_mask=list(prompt_attn),
                    actions=[int(x) for x in list(z["z_actions"])],
                    action_types=list(z["z_action_types"]),
                    old_logp=[],
                    advantages=[float(z_adv)] * len(list(z["z_actions"])),
                    returns=[float(z_r)] * len(list(z["z_actions"])),
                    reward_info={
                        "phase": "z",
                        "prompt_id": prompt_id,
                        "z_index": z_idx,
                        "z_reward_mean_children": float(z_r),
                        "adv_centered": float(z_adv),
                        "child_rewards": [float(x) for x in list(z["digit_rewards"])],
                        "true_digits": list(true_digits),
                    },
                )
            )

    stats = {
        "num_trajectories": float(len(trajectories)),
        "num_digit_rollouts": float(total_digit_rollouts),
        "digit_exact_rate": 0.0 if total_digit_rollouts == 0 else float(total_exact / total_digit_rollouts),
    }
    return trajectories, stats
