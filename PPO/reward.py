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


def _sample_episode_zero_zero_mask(
    *,
    true_digits: Sequence[int],
    keep_prob: Sequence[float],
    generator: Optional[torch.Generator],
) -> List[int]:
    if len(true_digits) != 5:
        raise ValueError("true_digits must have length 5")
    if len(keep_prob) != 5:
        raise ValueError("keep_prob must have length 5")
    out: List[int] = []
    for idx, true_digit in enumerate(true_digits):
        if int(true_digit) != 0:
            out.append(1)
            continue
        p = float(keep_prob[idx])
        draw = torch.rand((), generator=generator).item()
        out.append(1 if draw < p else 0)
    return out


def _applied_mask_from_episode_mask(
    *,
    true_digits: Sequence[int],
    pred_digits: Sequence[int],
    episode_zero_zero_keep_mask: Sequence[int],
) -> List[int]:
    if len(true_digits) != 5 or len(pred_digits) != 5:
        raise ValueError("true_digits/pred_digits must both have length 5")
    if len(episode_zero_zero_keep_mask) != 5:
        raise ValueError("episode_zero_zero_keep_mask must have length 5")

    out: List[int] = []
    for idx, (true_digit, pred_digit) in enumerate(zip(true_digits, pred_digits)):
        if int(true_digit) != 0:
            out.append(1)
        elif int(pred_digit) != 0:
            out.append(1)
        else:
            out.append(int(episode_zero_zero_keep_mask[idx]))
    return out


def _round_answer_reward_from_mask(
    *,
    true_digits: Sequence[int],
    pred_digits: Sequence[int],
    partial_scale: float,
    episode_zero_zero_keep_mask: Sequence[int],
) -> Dict[str, object]:
    if len(true_digits) != 5 or len(pred_digits) != 5:
        raise ValueError("true_digits/pred_digits must both have length 5")

    exact_match = all(int(a) == int(b) for a, b in zip(pred_digits, true_digits))
    if exact_match:
        reward = 1.0
        partial = 1.0
        applied_mask = [1, 1, 1, 1, 1]
        applied_count = 5
        correct_count = 5
    else:
        applied_mask = _applied_mask_from_episode_mask(
            true_digits=true_digits,
            pred_digits=pred_digits,
            episode_zero_zero_keep_mask=episode_zero_zero_keep_mask,
        )
        applied_count = int(sum(applied_mask))
        correct_count = int(sum(m * int(int(p) == int(t)) for m, p, t in zip(applied_mask, pred_digits, true_digits)))
        if applied_count == 0:
            partial = 0.0
        else:
            partial = float(partial_scale) * (float(correct_count) / float(applied_count))
        partial = max(0.0, min(1.0, partial))
        reward = float(partial)

    return {
        "answer_reward": float(reward),
        "reward_partial": float(partial),
        "exact_match": bool(exact_match),
        "applied_mask": [int(x) for x in applied_mask],
        "applied_count": int(applied_count),
        "correct_count": int(correct_count),
        "reward_full": 1 if exact_match else 0,
    }


def compute_multi_round_reward(
    *,
    round_pred_digits: Sequence[Optional[Sequence[int]]],
    true_digits: Sequence[int],
    terminated_reason: str,
    partial_scale: float,
    keep_prob: Sequence[float],
    length_penalty: float,
    correct_length_discount: float,
    early_success: float,
    reward_if_max_len: float,
    rounds_penalty_coef: float,
    num_generated_tokens: int,
    round_count: int,
    generator: Optional[torch.Generator],
) -> Dict[str, object]:
    if len(true_digits) != 5:
        raise ValueError("true_digits must have length 5")
    if int(round_count) < 0:
        raise ValueError("round_count must be >= 0")
    if float(rounds_penalty_coef) < 0.0:
        raise ValueError("rounds_penalty_coef must be >= 0")
    if float(early_success) < 0.0 or float(early_success) > 1.0:
        raise ValueError("early_success must be in [0.0, 1.0]")

    episode_mask = _sample_episode_zero_zero_mask(
        true_digits=true_digits,
        keep_prob=keep_prob,
        generator=generator,
    )

    round_answer_rewards: List[float] = []
    round_exact_flags: List[bool] = []
    round_applied_masks: List[List[int]] = []
    round_applied_counts: List[int] = []
    round_correct_counts: List[int] = []

    has_complete_round = False
    for pred in round_pred_digits:
        if pred is None:
            round_answer_rewards.append(0.0)
            round_exact_flags.append(False)
            round_applied_masks.append([0, 0, 0, 0, 0])
            round_applied_counts.append(0)
            round_correct_counts.append(0)
            continue
        pred_digits = [int(x) for x in pred]
        if len(pred_digits) != 5:
            raise ValueError("Each non-None round_pred_digits item must have length 5")
        has_complete_round = True
        rr = _round_answer_reward_from_mask(
            true_digits=true_digits,
            pred_digits=pred_digits,
            partial_scale=partial_scale,
            episode_zero_zero_keep_mask=episode_mask,
        )
        round_answer_rewards.append(float(rr["answer_reward"]))
        round_exact_flags.append(bool(rr["exact_match"]))
        round_applied_masks.append([int(x) for x in rr["applied_mask"]])
        round_applied_counts.append(int(rr["applied_count"]))
        round_correct_counts.append(int(rr["correct_count"]))

    is_finalize_terminated = str(terminated_reason) == "finalize"
    finalize_round_index = int(len(round_pred_digits) - 1) if is_finalize_terminated and len(round_pred_digits) > 0 else -1
    finalize_round_has_complete_answer = bool(
        finalize_round_index >= 0
        and finalize_round_index < len(round_pred_digits)
        and round_pred_digits[finalize_round_index] is not None
    )
    finalize_round_exact_match = bool(
        finalize_round_has_complete_answer and bool(round_exact_flags[finalize_round_index])
    )
    any_exact_indices = [idx for idx, is_exact in enumerate(round_exact_flags) if bool(is_exact)]
    any_exact_match = bool(any_exact_indices)
    first_exact_round_index = int(any_exact_indices[0]) if any_exact_match else -1

    selected_round_index = -1
    selected_round_answer_reward = 0.0
    reward_mode = "no_complete_round"
    if finalize_round_exact_match:
        selected_round_index = int(finalize_round_index)
        selected_round_answer_reward = 1.0
        reward_mode = "finalize_exact"
    elif is_finalize_terminated and any_exact_match:
        selected_round_index = int(first_exact_round_index)
        selected_round_answer_reward = float(early_success)
        reward_mode = "early_success_discounted"
    elif is_finalize_terminated and finalize_round_has_complete_answer:
        selected_round_index = int(finalize_round_index)
        selected_round_answer_reward = float(round_answer_rewards[finalize_round_index])
        reward_mode = "finalize_partial_or_zero"
    elif has_complete_round:
        reward_mode = "non_finalize_no_credit"

    best_round_index = int(selected_round_index)
    best_round_answer_reward = float(selected_round_answer_reward)
    best_exact_match = bool(finalize_round_exact_match)

    token_penalty = float(length_penalty) * float(num_generated_tokens)
    if best_exact_match:
        token_penalty *= float(correct_length_discount)

    rounds_penalty = float(rounds_penalty_coef) * float(round_count)
    max_len_term = float(reward_if_max_len) if str(terminated_reason) == "max_new_tokens" else 0.0
    reward_final = float(best_round_answer_reward) - float(token_penalty) - float(rounds_penalty) + float(max_len_term)

    if selected_round_index >= 0:
        best_applied_mask = [int(x) for x in round_applied_masks[selected_round_index]]
        best_applied_count = int(round_applied_counts[selected_round_index])
        best_correct_count = int(round_correct_counts[selected_round_index])
    else:
        best_applied_mask = [0, 0, 0, 0, 0]
        best_applied_count = 0
        best_correct_count = 0

    return {
        "reward_full": 1 if best_exact_match else 0,
        "partial_scale": float(partial_scale),
        "keep_prob": [float(x) for x in keep_prob],
        "applied_mask": best_applied_mask,
        "applied_count": int(best_applied_count),
        "correct_count": int(best_correct_count),
        "reward_partial": float(best_round_answer_reward),
        "length_penalty": float(length_penalty),
        "correct_length_discount": float(correct_length_discount),
        "early_success": float(early_success),
        "rounds_penalty_coef": float(rounds_penalty_coef),
        "reward_if_max_len": float(max_len_term),
        "reward": float(best_round_answer_reward),
        "reward_final": float(reward_final),
        "exact_match": bool(best_exact_match),
        "round_count": int(round_count),
        "round_answer_rewards": [float(x) for x in round_answer_rewards],
        "best_round_answer_reward": float(best_round_answer_reward),
        "best_round_index": int(best_round_index),
        "reward_selection_mode": str(reward_mode),
        "finalize_round_index": int(finalize_round_index),
        "finalize_round_exact_match": bool(finalize_round_exact_match),
        "any_exact_match": bool(any_exact_match),
        "first_exact_round_index": int(first_exact_round_index),
        "token_penalty": float(token_penalty),
        "rounds_penalty": float(rounds_penalty),
        "reached_max_length": bool(str(terminated_reason) == "max_new_tokens"),
    }


def compute_reward(
    *,
    pred_digits: Optional[Sequence[int]],
    true_digits: Sequence[int],
    terminated_reason: str,
    partial_scale: float,
    keep_prob: Sequence[float],
    length_penalty: float,
    correct_length_discount: float,
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
            "correct_length_discount": float(correct_length_discount),
            "reward_if_max_len": float(reward_if_max_len),
            "reward": float(reward_if_max_len),
            "reward_final": float(reward_if_max_len),
            "exact_match": False,
        }
    if pred_digits is None or len(pred_digits) != 5:
        raise ValueError("pred_digits must have length 5 when terminated_reason=answer_with_5_digits")
    exact_match = all(int(a) == int(b) for a, b in zip(pred_digits, true_digits))
    length_reward = float(length_penalty) * float(num_generated_tokens)
    if exact_match:
        reward = 1.0
        partial = 1.0
        applied_mask = [1, 1, 1, 1, 1]
        applied_count = 5
        correct_count = 5
        length_reward *= correct_length_discount
    else:
        applied_mask = sample_keep_mask(
            true_digits=true_digits,
            pred_digits=pred_digits,
            keep_prob=keep_prob,
            generator=generator,
        )
        applied_count = int(sum(applied_mask))
        correct_count = int(sum(m * int(int(p) == int(t)) for m, p, t in zip(applied_mask, pred_digits, true_digits)))
        if applied_count == 0:
            partial = 0.0
        else:
            partial = float(partial_scale) * (float(correct_count) / float(applied_count))
        partial = max(0.0, min(1.0, partial))
        reward = float(partial)

    reward_final = float(reward) - float(length_reward)

    return {
        "reward_full": 1 if exact_match else 0,
        "partial_scale": float(partial_scale),
        "keep_prob": [float(x) for x in keep_prob],
        "applied_mask": [int(x) for x in applied_mask],
        "applied_count": int(applied_count),
        "correct_count": int(correct_count),
        "reward_partial": float(partial),
        "length_penalty": float(length_penalty),
        "correct_length_discount": float(correct_length_discount),
        "reward_if_max_len": float(reward_if_max_len),
        "reward": float(reward),
        "reward_final": float(reward_final),
        "exact_match": exact_match,
    }
