from __future__ import annotations

import torch

from PPO.reward import compute_reward, parse_final_answer_to_digits


def test_parse_final_answer_digits() -> None:
    assert parse_final_answer_to_digits("42") == [0, 0, 0, 4, 2]
    assert parse_final_answer_to_digits("-3") is None
    assert parse_final_answer_to_digits("abc") is None


def test_reward_for_max_len_termination_is_zero() -> None:
    out = compute_reward(
        pred_digits=[1, 2, 3, 4, 5],
        true_digits=[1, 2, 3, 4, 5],
        terminated_by_answer=False,
        partial_scale=0.5,
        keep_prob=(0.02, 0.05, 0.1, 0.5, 1.0),
        length_penalty=0.01,
        num_generated_tokens=20,
        generator=torch.Generator().manual_seed(0),
    )
    assert out["reward_final"] == 0.0


def test_partial_reward_mask_sampled_once() -> None:
    gen = torch.Generator().manual_seed(123)
    out = compute_reward(
        pred_digits=[0, 1, 2, 3, 4],
        true_digits=[0, 1, 9, 0, 4],
        terminated_by_answer=True,
        partial_scale=0.5,
        keep_prob=(1.0, 1.0, 1.0, 1.0, 1.0),
        length_penalty=0.0,
        num_generated_tokens=3,
        generator=gen,
    )

    assert out["applied_count"] == 5
    assert out["correct_count"] == 3
    assert abs(out["reward_partial"] - 0.3) < 1e-8
    assert out["reward_final"] == out["reward"]
