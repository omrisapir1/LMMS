from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def parse_final_answer_to_digits(raw: object) -> Optional[List[int]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None

    if value < 0 or value > 99999:
        return None
    return [int(ch) for ch in f"{value:05d}"]


def sample_keep_mask(
    true_digits: Sequence[int],
    keep_prob: Sequence[float],
    generator: Optional[torch.Generator],
) -> List[int]:
    if len(true_digits) != 5:
        raise ValueError("true_digits must have length 5")
    if len(keep_prob) != 5:
        raise ValueError("keep_prob must have length 5")

    mask: List[int] = []
    for idx, digit in enumerate(true_digits):
        if int(digit) != 0:
            mask.append(1)
            continue
        p = float(keep_prob[idx])
        draw = torch.rand((), generator=generator).item()
        mask.append(1 if draw < p else 0)
    return mask


def compute_reward(
    *,
    pred_digits: Sequence[int],
    true_digits: Sequence[int],
    terminated_by_answer: bool,
    partial_scale: float,
    keep_prob: Sequence[float],
    length_penalty: float,
    num_generated_tokens: int,
    generator: Optional[torch.Generator],
) -> Dict[str, object]:
    if len(pred_digits) != 5 or len(true_digits) != 5:
        raise ValueError("pred_digits and true_digits must have length 5")

    exact_match = all(int(a) == int(b) for a, b in zip(pred_digits, true_digits))

    if not terminated_by_answer:
        return {
            "reward_full": 1 if exact_match else 0,
            "partial_scale": float(partial_scale),
            "keep_prob": [float(x) for x in keep_prob],
            "applied_mask": [0, 0, 0, 0, 0],
            "applied_count": 0,
            "correct_count": 0,
            "reward_partial": 0.0,
            "length_penalty": float(length_penalty),
            "reward": 0.0,
            "reward_final": 0.0,
            "exact_match": exact_match,
        }

    if exact_match:
        reward = 1.0
        partial = 1.0
        applied_mask = [1, 1, 1, 1, 1]
        applied_count = 5
        correct_count = 5
    else:
        applied_mask = sample_keep_mask(true_digits=true_digits, keep_prob=keep_prob, generator=generator)
        applied_count = int(sum(applied_mask))
        correct_count = int(sum(m * int(int(p) == int(t)) for m, p, t in zip(applied_mask, pred_digits, true_digits)))
        if applied_count == 0:
            partial = 0.0
        else:
            partial = float(partial_scale) * (float(correct_count) / float(applied_count))
        partial = max(0.0, min(1.0, partial))
        reward = partial

    reward_final = max(0.0, float(reward) - float(length_penalty) * float(num_generated_tokens))

    return {
        "reward_full": 1 if exact_match else 0,
        "partial_scale": float(partial_scale),
        "keep_prob": [float(x) for x in keep_prob],
        "applied_mask": [int(x) for x in applied_mask],
        "applied_count": int(applied_count),
        "correct_count": int(correct_count),
        "reward_partial": float(partial),
        "length_penalty": float(length_penalty),
        "reward": float(reward),
        "reward_final": float(reward_final),
        "exact_match": exact_match,
    }
