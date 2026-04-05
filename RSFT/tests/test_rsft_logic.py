from __future__ import annotations

import torch
import torch.nn.functional as F

from RSFT.logic import (
    RolloutCandidate,
    TARGET_ANSWER,
    TARGET_DIGIT,
    TARGET_Z,
    build_training_example,
    compute_rsft_losses,
    exact_digit_match,
    extract_z_before_answer_from_row,
    select_shortest_valid,
)


def test_exact_digit_filtering() -> None:
    assert exact_digit_match([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert not exact_digit_match([1, 2, 3, 4, 0], [1, 2, 3, 4, 5])


def test_selection_picks_any_correct_candidate() -> None:
    candidates = [
        RolloutCandidate(
            prompt_idx=0,
            rollout_idx=0,
            z_token_ids=[10, 11],
            z_avg_logprob=-1.2,
            digit_token_ids=[101, 102, 103, 104, 105],
            pred_digits=[1, 2, 3, 4, 5],
            true_digits=[1, 2, 3, 4, 5],
        ),
        RolloutCandidate(
            prompt_idx=0,
            rollout_idx=1,
            z_token_ids=[12],
            z_avg_logprob=-0.8,
            digit_token_ids=[101, 102, 103, 104, 105],
            pred_digits=[1, 2, 3, 4, 5],
            true_digits=[1, 2, 3, 4, 5],
        ),
        RolloutCandidate(
            prompt_idx=0,
            rollout_idx=2,
            z_token_ids=[13],
            z_avg_logprob=-2.1,
            digit_token_ids=[101, 102, 103, 104, 105],
            pred_digits=[1, 2, 3, 4, 5],
            true_digits=[1, 2, 3, 4, 5],
        ),
    ]
    chosen = select_shortest_valid(candidates)
    assert chosen is not None
    assert chosen.z_token_ids in ([10, 11], [12], [13])


def test_acceptance_filtering_requires_exact_match() -> None:
    candidates = [
        RolloutCandidate(
            prompt_idx=0,
            rollout_idx=0,
            z_token_ids=[10],
            z_avg_logprob=-1.0,
            digit_token_ids=[101, 102, 103, 104, 105],
            pred_digits=[1, 2, 3, 4, 0],
            true_digits=[1, 2, 3, 4, 5],
        )
    ]
    chosen = select_shortest_valid(candidates)
    assert chosen is None


def test_overlength_examples_are_dropped() -> None:
    ex = build_training_example(
        prompt_ids=[1, 2, 3],
        z_token_ids=[4, 5, 6],
        answer_token_id=7,
        digit_token_ids=[8, 9, 10, 11, 12],
        max_length=5,
    )
    assert ex is None


def test_masking_contract_for_two_losses() -> None:
    # 3-token sequence with explicit targets: z, answer, digit.
    labels = torch.tensor([[5, 6, 7]], dtype=torch.long)
    target_class = torch.tensor([[TARGET_Z, TARGET_ANSWER, TARGET_DIGIT]], dtype=torch.long)

    logits = torch.tensor(
        [
            [
                [0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
        digit_token_ids=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        w_z_ans=1.0,
        w_digits=1.0,
    )

    expected_z_ans = F.cross_entropy(
        torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.long),
        reduction="mean",
    )
    expected_digits = F.cross_entropy(
        torch.tensor([[5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
        reduction="mean",
    )

    assert torch.allclose(losses["l_z_ans"], expected_z_ans, atol=1e-6)
    assert torch.allclose(losses["l_digits"], expected_digits, atol=1e-6)
    assert torch.allclose(losses["loss"], expected_z_ans + expected_digits, atol=1e-6)


def test_extract_z_before_answer_from_row_handles_implicit_answer_stop() -> None:
    row = {
        "token_ids": [101, 102, 103],
        "stop_reason": 999,
        "finish_reason": "stop",
    }
    z = extract_z_before_answer_from_row(row, answer_token_id=999)
    assert z == [101, 102, 103]
