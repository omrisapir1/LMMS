from __future__ import annotations

import torch
import torch.nn.functional as F

from RSFT.logic import (
    TARGET_ANSWER,
    TARGET_DIGIT,
    TARGET_IGNORE,
    TARGET_VERIFY,
    TARGET_Z,
    RoundTrace,
    build_training_example,
    compute_rsft_losses,
    exact_digit_match,
    extract_z_before_answer_from_row,
)


def test_exact_digit_filtering() -> None:
    assert exact_digit_match([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert not exact_digit_match([1, 2, 3, 4, 0], [1, 2, 3, 4, 5])


def test_overlength_examples_are_dropped() -> None:
    rounds = [
        RoundTrace(
            z_token_ids=[4, 5, 6],
            digit_token_ids=[8, 9, 10, 11, 12],
            pred_digits=[0, 0, 0, 0, 1],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=13,
            verify_target_token_id=13,
            is_correct=True,
        )
    ]
    ex = build_training_example(
        prompt_ids=[1, 2, 3],
        rounds=rounds,
        answer_token_id=7,
        finalize_token_id=13,
        retry_token_id=14,
        max_length=5,
    )
    assert ex is None


def test_masking_contract_failed_then_success() -> None:
    rounds = [
        RoundTrace(
            z_token_ids=[10],
            digit_token_ids=[21, 21, 21, 21, 21],
            pred_digits=[0, 0, 0, 0, 0],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=14,
            verify_target_token_id=14,
            is_correct=False,
        ),
        RoundTrace(
            z_token_ids=[11],
            digit_token_ids=[21, 21, 21, 21, 22],
            pred_digits=[0, 0, 0, 0, 1],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=13,
            verify_target_token_id=13,
            is_correct=True,
        ),
    ]
    ex = build_training_example(
        prompt_ids=[1, 2],
        rounds=rounds,
        answer_token_id=20,
        finalize_token_id=13,
        retry_token_id=14,
        max_length=64,
    )
    assert ex is not None
    tcls = ex["target_class"]
    assert TARGET_VERIFY in tcls
    assert tcls.count(TARGET_DIGIT) == 5
    assert tcls.count(TARGET_VERIFY) == 2


def test_verify_always_and_z_digit_only_final_round() -> None:
    rounds = [
        RoundTrace(
            z_token_ids=[30, 31],
            digit_token_ids=[41, 41, 41, 41, 41],
            pred_digits=[0, 0, 0, 0, 0],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=51,
            verify_target_token_id=51,
            is_correct=False,
        ),
        RoundTrace(
            z_token_ids=[32],
            digit_token_ids=[41, 41, 41, 41, 42],
            pred_digits=[0, 0, 0, 0, 1],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=50,
            verify_target_token_id=50,
            is_correct=True,
        ),
    ]
    ex = build_training_example(
        prompt_ids=[1, 2],
        rounds=rounds,
        answer_token_id=40,
        finalize_token_id=50,
        retry_token_id=51,
        max_length=128,
    )
    assert ex is not None
    # With two rounds, we must supervise exactly:
    # - verify: 2 tokens (every round)
    # - digits: 5 tokens (final round only)
    # - z/answer: len(final_z)+1 tokens (final round only)
    tcls = ex["target_class"]
    assert tcls.count(TARGET_VERIFY) == 2
    assert tcls.count(TARGET_DIGIT) == 5
    assert tcls.count(TARGET_Z) == 1
    assert tcls.count(TARGET_ANSWER) == 1


def test_zero_success_sequence_is_verify_only() -> None:
    rounds = [
        RoundTrace(
            z_token_ids=[60],
            digit_token_ids=[71, 71, 71, 71, 71],
            pred_digits=[0, 0, 0, 0, 0],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=81,
            verify_target_token_id=81,
            is_correct=False,
        ),
        RoundTrace(
            z_token_ids=[61, 62],
            digit_token_ids=[71, 71, 71, 71, 71],
            pred_digits=[0, 0, 0, 0, 0],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=81,
            verify_target_token_id=81,
            is_correct=False,
        ),
    ]
    ex = build_training_example(
        prompt_ids=[1, 2],
        rounds=rounds,
        answer_token_id=70,
        finalize_token_id=80,
        retry_token_id=81,
        max_length=128,
    )
    assert ex is not None
    tcls = ex["target_class"]
    assert tcls.count(TARGET_VERIFY) == 2
    assert tcls.count(TARGET_Z) == 0
    assert tcls.count(TARGET_ANSWER) == 0
    assert tcls.count(TARGET_DIGIT) == 0


def test_zero_success_with_correct_but_retry_round_is_allowed_verify_only() -> None:
    rounds = [
        RoundTrace(
            z_token_ids=[90],
            digit_token_ids=[71, 71, 71, 71, 72],
            pred_digits=[0, 0, 0, 0, 1],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=81,  # executed retry even though correct
            verify_target_token_id=80,    # target finalize
            is_correct=True,
        ),
        RoundTrace(
            z_token_ids=[91],
            digit_token_ids=[71, 71, 71, 71, 71],
            pred_digits=[0, 0, 0, 0, 0],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=81,
            verify_target_token_id=81,
            is_correct=False,
        ),
    ]
    ex = build_training_example(
        prompt_ids=[1, 2],
        rounds=rounds,
        answer_token_id=70,
        finalize_token_id=80,
        retry_token_id=81,
        max_length=128,
    )
    assert ex is not None
    tcls = ex["target_class"]
    assert tcls.count(TARGET_VERIFY) == 2
    assert tcls.count(TARGET_Z) == 0
    assert tcls.count(TARGET_ANSWER) == 0
    assert tcls.count(TARGET_DIGIT) == 0


def test_masking_contract_for_three_losses() -> None:
    labels = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    target_class = torch.tensor([[TARGET_Z, TARGET_ANSWER, TARGET_DIGIT, TARGET_VERIFY]], dtype=torch.long)

    logits = torch.tensor(
        [
            [
                [0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 1.0],
                [0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0, 5.0],
            ]
        ],
        dtype=torch.float32,
    )

    losses = compute_rsft_losses(
        logits=logits,
        labels=labels,
        target_class=target_class,
        z_token_ids=[5],
        answer_token_id=6,
        digit_token_ids=[7, 8],
        verify_token_ids=[3, 8],
        finalize_token_id=3,
        retry_token_id=8,
        w_z=1.0,
        w_answer=1.0,
        w_digits=1.0,
        w_verify=1.0,
    )

    expected_z = F.cross_entropy(
        torch.tensor([[4.0, 0.0]], dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
        reduction="mean",
    )
    expected_answer = F.cross_entropy(
        torch.tensor([[0.0, 4.0]], dtype=torch.float32),
        torch.tensor([1], dtype=torch.long),
        reduction="mean",
    )
    expected_digits = F.cross_entropy(
        torch.tensor([[5.0, 1.0]], dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
        reduction="mean",
    )
    expected_verify = F.cross_entropy(
        torch.tensor([[3.0, 5.0]], dtype=torch.float32),
        torch.tensor([1], dtype=torch.long),
        reduction="mean",
    )

    assert torch.allclose(losses["l_z"], expected_z, atol=1e-6)
    assert torch.allclose(losses["l_answer"], expected_answer, atol=1e-6)
    assert torch.allclose(losses["l_digits"], expected_digits, atol=1e-6)
    assert torch.allclose(losses["l_verify"], expected_verify, atol=1e-6)
    assert torch.allclose(losses["loss"], expected_z + expected_answer + expected_digits + expected_verify, atol=1e-6)


def test_extract_z_before_answer_from_row_handles_implicit_answer_stop() -> None:
    row = {
        "token_ids": [101, 102, 103],
        "stop_reason": 999,
        "finish_reason": "stop",
    }
    z = extract_z_before_answer_from_row(row, answer_token_id=999)
    assert z == [101, 102, 103]


def test_correct_but_executed_retry_uses_finalize_verify_target_label() -> None:
    rounds = [
        RoundTrace(
            z_token_ids=[10],
            digit_token_ids=[21, 21, 21, 21, 22],
            pred_digits=[0, 0, 0, 0, 1],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=14,  # executed <RETRY> to continue
            verify_target_token_id=13,    # supervise <FINALIZE>
            is_correct=True,
        ),
        RoundTrace(
            z_token_ids=[11],
            digit_token_ids=[21, 21, 21, 21, 22],
            pred_digits=[0, 0, 0, 0, 1],
            true_digits=[0, 0, 0, 0, 1],
            executed_verify_token_id=13,
            verify_target_token_id=13,
            is_correct=True,
        ),
    ]
    ex = build_training_example(
        prompt_ids=[1, 2],
        rounds=rounds,
        answer_token_id=20,
        finalize_token_id=13,
        retry_token_id=14,
        max_length=128,
    )
    assert ex is not None
    labels = ex["labels"]
    target_class = ex["target_class"]
    verify_labels = [labels[i] for i in range(len(labels)) if target_class[i] == TARGET_VERIFY]
    assert verify_labels == [13, 13]


def test_verify_weighting_alpha_zero_matches_unweighted_mean_per_example() -> None:
    # verify labels: [retry, retry, finalize]
    labels = torch.tensor([[-100, -100, 20, 20, 10]], dtype=torch.long)
    target_class = torch.tensor([[TARGET_IGNORE, TARGET_IGNORE, TARGET_VERIFY, TARGET_VERIFY, TARGET_VERIFY]], dtype=torch.long)

    logits = torch.zeros((1, 5, 32), dtype=torch.float32)
    # retry token id=20, finalize token id=10
    logits[0, 2, 10] = 1.0
    logits[0, 2, 20] = 3.0
    logits[0, 3, 10] = 1.0
    logits[0, 3, 20] = 3.0
    logits[0, 4, 10] = 1.5
    logits[0, 4, 20] = 2.5

    losses = compute_rsft_losses(
        logits=logits,
        labels=labels,
        target_class=target_class,
        z_token_ids=[1],
        answer_token_id=2,
        digit_token_ids=[3, 4],
        verify_token_ids=[10, 20],
        finalize_token_id=10,
        retry_token_id=20,
        w_z=0.0,
        w_answer=0.0,
        w_digits=0.0,
        w_verify=1.0,
        verify_finalize_alpha=0.0,
    )

    restricted = torch.tensor(
        [
            [3.0, 1.0],
            [3.0, 1.0],
            [2.5, 1.5],
        ],
        dtype=torch.float32,
    )
    expected = F.cross_entropy(
        restricted,
        torch.tensor([0, 0, 1], dtype=torch.long),
        reduction="mean",
    )
    assert torch.allclose(losses["l_verify"], expected, atol=1e-6)


def test_verify_weighting_boosts_finalize_for_retry_heavy_success() -> None:
    # success-like verify labels with many retries then one finalize: [R, R, R, F]
    labels = torch.tensor([[20, 20, 20, 10]], dtype=torch.long)
    target_class = torch.tensor([[TARGET_VERIFY, TARGET_VERIFY, TARGET_VERIFY, TARGET_VERIFY]], dtype=torch.long)

    logits = torch.zeros((1, 4, 32), dtype=torch.float32)
    # retries are easy, finalize is hard
    logits[0, 0, 20] = 5.0
    logits[0, 0, 10] = 1.0
    logits[0, 1, 20] = 5.0
    logits[0, 1, 10] = 1.0
    logits[0, 2, 20] = 5.0
    logits[0, 2, 10] = 1.0
    logits[0, 3, 10] = 1.1
    logits[0, 3, 20] = 4.9

    losses_alpha0 = compute_rsft_losses(
        logits=logits,
        labels=labels,
        target_class=target_class,
        z_token_ids=[1],
        answer_token_id=2,
        digit_token_ids=[3, 4],
        verify_token_ids=[10, 20],
        finalize_token_id=10,
        retry_token_id=20,
        w_z=0.0,
        w_answer=0.0,
        w_digits=0.0,
        w_verify=1.0,
        verify_finalize_alpha=0.0,
    )
    losses_alpha1 = compute_rsft_losses(
        logits=logits,
        labels=labels,
        target_class=target_class,
        z_token_ids=[1],
        answer_token_id=2,
        digit_token_ids=[3, 4],
        verify_token_ids=[10, 20],
        finalize_token_id=10,
        retry_token_id=20,
        w_z=0.0,
        w_answer=0.0,
        w_digits=0.0,
        w_verify=1.0,
        verify_finalize_alpha=1.0,
    )

    assert float(losses_alpha1["l_verify"].item()) > float(losses_alpha0["l_verify"].item())


def test_verify_weighting_failure_only_is_normalized() -> None:
    # failure-only labels: [R, R, R], no finalize present.
    labels = torch.tensor([[20, 20, 20]], dtype=torch.long)
    target_class = torch.tensor([[TARGET_VERIFY, TARGET_VERIFY, TARGET_VERIFY]], dtype=torch.long)

    logits = torch.zeros((1, 3, 32), dtype=torch.float32)
    logits[0, 0, 20] = 3.0
    logits[0, 0, 10] = 1.0
    logits[0, 1, 20] = 2.0
    logits[0, 1, 10] = 1.2
    logits[0, 2, 20] = 1.5
    logits[0, 2, 10] = 1.0

    losses_alpha0 = compute_rsft_losses(
        logits=logits,
        labels=labels,
        target_class=target_class,
        z_token_ids=[1],
        answer_token_id=2,
        digit_token_ids=[3, 4],
        verify_token_ids=[10, 20],
        finalize_token_id=10,
        retry_token_id=20,
        w_z=0.0,
        w_answer=0.0,
        w_digits=0.0,
        w_verify=1.0,
        verify_finalize_alpha=0.0,
    )
    losses_alpha2 = compute_rsft_losses(
        logits=logits,
        labels=labels,
        target_class=target_class,
        z_token_ids=[1],
        answer_token_id=2,
        digit_token_ids=[3, 4],
        verify_token_ids=[10, 20],
        finalize_token_id=10,
        retry_token_id=20,
        w_z=0.0,
        w_answer=0.0,
        w_digits=0.0,
        w_verify=1.0,
        verify_finalize_alpha=2.0,
    )

    assert torch.allclose(losses_alpha0["l_verify"], losses_alpha2["l_verify"], atol=1e-6)
