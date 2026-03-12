from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def parse_final_answer_to_digits(raw: object) -> Optional[List[int]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("+"):
        text = text[1:]
    if (not text) or (not text.isdigit()) or len(text) > 5:
        return None
    return [int(ch) for ch in text.zfill(5)]


def parse_answer_digits(raw: object) -> Optional[List[int]]:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    if len(raw) != 5:
        return None
    out: List[int] = []
    for x in raw:
        try:
            v = int(x)
        except (TypeError, ValueError):
            return None
        if v < 0 or v > 9:
            return None
        out.append(v)
    return out


def sample_keep_mask(
    true_digits: Sequence[int],
    pred_digits: Sequence[int],
    keep_prob: Sequence[float],
    generator: Optional[torch.Generator],
) -> List[int]:
    if len(true_digits) != 5:
        raise ValueError("true_digits must have length 5")
    if len(keep_prob) != 5:
        raise ValueError("keep_prob must have length 5")

    mask: List[int] = []
    for idx, (true_digit, pred_digit) in enumerate(zip(true_digits, pred_digits)):
        if int(true_digit) != 0 or int(pred_digit) != 0:
            mask.append(1)
            continue
        p = float(keep_prob[idx])
        draw = torch.rand((), generator=generator).item()
        mask.append(1 if draw < p else 0)
    return mask


def compute_reward(
    *,
    pred_digits: Optional[Sequence[int]],
    true_digits: Sequence[int],
    terminated_reason: str,
    partial_scale: float,
    keep_prob: Sequence[float],
    length_penalty: float,
    reward_if_max_len: float,
    num_generated_tokens: int,
    generator: Optional[torch.Generator],
) -> Dict[str, object]:
    if len(true_digits) != 5:
        raise ValueError("true_digits must have length 5")

    is_complete_answer = terminated_reason == "answer_with_5_digits"
    if not is_complete_answer:
        return {
            "reward_full": 0,
            "partial_scale": float(partial_scale),
            "keep_prob": [float(x) for x in keep_prob],
            "applied_mask": [0, 0, 0, 0, 0],
            "applied_count": 0,
            "correct_count": 0,
            "reward_partial": 0.0,
            "length_penalty": float(length_penalty),
            "reward_if_max_len": float(reward_if_max_len),
            "reward": float(reward_if_max_len),
            "reward_final": float(reward_if_max_len),
            "exact_match": False,
        }
    if pred_digits is None or len(pred_digits) != 5:
        raise ValueError("pred_digits must have length 5 when terminated_reason=answer_with_5_digits")

    exact_match = all(int(a) == int(b) for a, b in zip(pred_digits, true_digits))

    if exact_match:
        reward = 1.0
        partial = 1.0
        applied_mask = [1, 1, 1, 1, 1]
        applied_count = 5
        correct_count = 5
        length_reward = 0.0
    else:
        applied_mask = sample_keep_mask(true_digits=true_digits, pred_digits=pred_digits, keep_prob=keep_prob, generator=generator)
        applied_count = int(sum(applied_mask))
        correct_count = int(sum(m * int(int(p) == int(t)) for m, p, t in zip(applied_mask, pred_digits, true_digits)))
        if applied_count == 0:
            partial = 0.0
        else:
            partial = float(partial_scale) * (float(correct_count) / float(applied_count))
        partial = max(0.0, min(1.0, partial))
        reward = partial
        length_reward = - float(length_penalty) * float(num_generated_tokens)

    # reward_final = max(0.0, float(reward) - float(length_penalty) * float(num_generated_tokens))
    # reward_final = float(reward) - float(length_penalty) * float(num_generated_tokens)
    reward_final = float(reward) - length_reward

    return {
        "reward_full": 1 if exact_match else 0,
        "partial_scale": float(partial_scale),
        "keep_prob": [float(x) for x in keep_prob],
        "applied_mask": [int(x) for x in applied_mask],
        "applied_count": int(applied_count),
        "correct_count": int(correct_count),
        "reward_partial": float(partial),
        "length_penalty": float(length_penalty),
        "reward_if_max_len": float(reward_if_max_len),
        "reward": float(reward),
        "reward_final": float(reward_final),
        "exact_match": exact_match,
    }
