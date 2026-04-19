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


def collect_grpo_batch(
    *,
    prepared: Sequence[Dict[str, object]],
    tokenizer,
    vllm_engine: Any,
    cfg,
    answer_token_id: int,
    digit_token_ids: Sequence[int],
    reward_rng: torch.Generator,
) -> Tuple[List[GRPOTrajectory], Dict[str, float]]:
    if len(prepared) == 0:
        return [], {}

    supports_token_prompts = bool(vllm_engine.supports_prompt_token_ids())
    if not supports_token_prompts:
        raise RuntimeError("GRPO rollout requires vLLM prompt_token_ids support")

    n_z = int(cfg.rollout.n_z_traces)
    n_digit = int(cfg.rollout.n_digit_traces)
    if n_z <= 0 or n_digit <= 0:
        raise ValueError("rollout.n_z_traces and rollout.n_digit_traces must be > 0")

    digit_allowed_set = set(int(x) for x in digit_token_ids)
    digit_to_value = {int(tok): idx for idx, tok in enumerate(digit_token_ids)}

    trajectories: List[GRPOTrajectory] = []
    total_digit_rollouts = 0
    total_exact = 0

    for item in prepared:
        prompt_id = int(item["prompt_id"])
        question = str(item["question"])
        true_digits = [int(x) for x in list(item["true_digits"])]
        prompt_ids = [int(x) for x in list(item["prompt_ids"])]
        prompt_attn = [int(x) for x in list(item["prompt_attention_mask"])]
        sample_base = str(item["sample_id_base"])

        z_rows = vllm_engine.generate_z(
            prompt_token_ids=[prompt_ids],
            num_samples_per_prompt=n_z,
            max_new_tokens=int(cfg.rollout.max_z_new_tokens),
            temperature=float(cfg.rollout.z_temperature),
            top_p=float(cfg.rollout.z_top_p),
            min_p=float(cfg.rollout.z_min_p),
            repetition_penalty=float(cfg.rollout.z_repetition_penalty),
            greedy=False,
        )
        if len(z_rows) != n_z:
            raise RuntimeError(f"Z rollout count mismatch: got={len(z_rows)} expected={n_z}")

        z_infos: List[Dict[str, object]] = []
        for zi, row in enumerate(z_rows):
            z_prefix, has_answer = _extract_z_phase_from_vllm_row_with_budget(
                row=dict(row),
                answer_token_id=int(answer_token_id),
                budget=int(cfg.rollout.max_z_new_tokens),
            )
            if not has_answer:
                raise RuntimeError(
                    "GRPO Z phase must end with <ANSWER>; increase rollout.max_z_new_tokens or adjust sampling"
                )
            z_actions = [int(x) for x in z_prefix] + [int(answer_token_id)]
            z_types = _action_types_for_z(len(z_prefix))
            z_prefix_full = prompt_ids + z_actions
            z_attn_full = prompt_attn + [1] * len(z_actions)
            z_infos.append(
                {
                    "z_index": int(zi),
                    "z_actions": z_actions,
                    "z_action_types": z_types,
                    "prefix_ids": z_prefix_full,
                    "prefix_attn": z_attn_full,
                    "digit_rewards": [],
                }
            )

        # Per-Z second group: sample n_digit_traces 5-digit completions.
        digit_prompt_token_ids = [list(z["prefix_ids"]) for z in z_infos]
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
        if len(digit_rows) != n_z * n_digit:
            raise RuntimeError(
                f"Digit rollout count mismatch: got={len(digit_rows)} expected={n_z * n_digit}"
            )

        row_idx = 0
        for z in z_infos:
            z_idx = int(z["z_index"])
            z_children: List[Tuple[List[int], float, Dict[str, object]]] = []
            for di in range(n_digit):
                digit_ids = [int(x) for x in list(digit_rows[row_idx])]
                row_idx += 1
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

                z_children.append((digit_ids, reward_scalar, reward_info))
                z["digit_rewards"].append(reward_scalar)

                child_mean = float(
                    sum(float(x) for x in z["digit_rewards"]) / float(len(z["digit_rewards"]))
                )

                # We assign centered advantages after all children are collected.
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

        # Set digit advantages by parent-Z group mean-centering.
        for z in z_infos:
            z_idx = int(z["z_index"])
            rewards = [float(x) for x in list(z["digit_rewards"])]
            centered = _group_mean_center(rewards)
            # Update trajectories that belong to this prompt/z pair.
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

        # Z-group reward: average of n_digit_traces under each Z.
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
