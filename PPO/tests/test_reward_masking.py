from __future__ import annotations

import torch

from PPO.reward import compute_multi_round_reward, compute_reward, parse_answer_digits, parse_final_answer_to_digits


def test_parse_final_answer_digits() -> None:
    assert parse_final_answer_to_digits("42") == [0, 0, 0, 4, 2]
    assert parse_final_answer_to_digits("+42") == [0, 0, 0, 4, 2]
    assert parse_final_answer_to_digits("00123") == [0, 0, 1, 2, 3]
    assert parse_final_answer_to_digits("-3") is None
    assert parse_final_answer_to_digits("abc") is None
    assert parse_final_answer_to_digits("123456") is None


def test_parse_answer_digits_supports_ints_and_strings() -> None:
    assert parse_answer_digits([0, 1, 2, 3, 4]) == [0, 1, 2, 3, 4]
    assert parse_answer_digits(["0", "1", "2", "3", "4"]) == [0, 1, 2, 3, 4]
    assert parse_answer_digits([1, 2, 3]) is None
    assert parse_answer_digits([0, 1, 2, 3, 12]) is None


def test_reward_for_max_len_termination_uses_config() -> None:
    out = compute_reward(
        pred_digits=None,
        true_digits=[1, 2, 3, 4, 5],
        terminated_reason="max_new_tokens",
        partial_scale=0.5,
        keep_prob=(0.02, 0.05, 0.1, 0.5, 1.0),
        length_penalty=0.01,
        correct_length_discount=0.1,
        reward_if_max_len=0.17,
        num_generated_tokens=20,
        generator=torch.Generator().manual_seed(0),
    )
    assert out["reward_final"] == 0.17


def test_partial_reward_mask_sampled_once() -> None:
    gen = torch.Generator().manual_seed(123)
    out = compute_reward(
        pred_digits=[0, 1, 2, 3, 4],
        true_digits=[0, 1, 9, 0, 4],
        terminated_reason="answer_with_5_digits",
        partial_scale=0.5,
        keep_prob=(1.0, 1.0, 1.0, 1.0, 1.0),
        length_penalty=0.0,
        correct_length_discount=0.1,
        reward_if_max_len=0.0,
        num_generated_tokens=3,
        generator=gen,
    )

    assert out["applied_count"] == 5
    assert out["correct_count"] == 3
    assert abs(out["reward_partial"] - 0.3) < 1e-8
    assert out["reward_final"] == out["reward"]


def test_multi_round_finalize_exact_gets_full_reward() -> None:
    out = compute_multi_round_reward(
        round_pred_digits=[
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ],
        true_digits=[1, 2, 3, 4, 5],
        terminated_reason="finalize",
        partial_scale=0.5,
        keep_prob=(0.02, 0.05, 0.1, 0.5, 1.0),
        length_penalty=0.01,
        correct_length_discount=0.1,
        early_success=0.7,
        reward_if_max_len=-0.1,
        rounds_penalty_coef=0.02,
        num_generated_tokens=10,
        round_count=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert out["best_round_index"] == 1
    assert out["best_round_answer_reward"] == 1.0
    assert out["reward_selection_mode"] == "finalize_exact"
    # token_penalty = 10*0.01*0.1 = 0.01; rounds_penalty = 2*0.02 = 0.04
    assert abs(float(out["token_penalty"]) - 0.01) < 1e-8
    assert abs(float(out["rounds_penalty"]) - 0.04) < 1e-8
    assert abs(float(out["reward_final"]) - 0.95) < 1e-8


def test_multi_round_early_success_discount_applies_when_finalize_not_exact() -> None:
    out = compute_multi_round_reward(
        round_pred_digits=[
            [1, 2, 3, 4, 5],  # early exact
            [1, 2, 3, 4, 0],  # final round not exact
        ],
        true_digits=[1, 2, 3, 4, 5],
        terminated_reason="finalize",
        partial_scale=0.5,
        keep_prob=(0.02, 0.05, 0.1, 0.5, 1.0),
        length_penalty=0.0,
        correct_length_discount=0.1,
        early_success=0.7,
        reward_if_max_len=-0.1,
        rounds_penalty_coef=0.0,
        num_generated_tokens=10,
        round_count=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert out["best_round_index"] == 0
    assert abs(float(out["best_round_answer_reward"]) - 0.7) < 1e-8
    assert out["reward_selection_mode"] == "early_success_discounted"
    assert out["exact_match"] is False
    assert out["reward_full"] == 0
    assert abs(float(out["reward_final"]) - 0.7) < 1e-8


def test_multi_round_no_finalize_no_credit_and_max_len_term() -> None:
    out = compute_multi_round_reward(
        round_pred_digits=[[1, 2, 3, 4, 5], None],
        true_digits=[1, 2, 3, 4, 5],
        terminated_reason="max_new_tokens",
        partial_scale=0.5,
        keep_prob=(0.02, 0.05, 0.1, 0.5, 1.0),
        length_penalty=0.01,
        correct_length_discount=0.1,
        early_success=0.7,
        reward_if_max_len=-0.1,
        rounds_penalty_coef=0.0,
        num_generated_tokens=20,
        round_count=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert out["best_round_answer_reward"] == 0.0
    assert out["best_round_index"] == -1
    assert out["reward_selection_mode"] == "non_finalize_no_credit"
    # 0 - 0.2 + (-0.1)
    assert abs(float(out["reward_final"]) + 0.3) < 1e-8
