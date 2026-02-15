from __future__ import annotations

import torch

from PPO.masking import build_allowed_token_ids, introspect_z_token_ids, masked_log_probs_and_entropy


class DummyTokenizer:
    def __init__(self):
        self.vocab = {
            "<Z_1>": 11,
            "<Z_0>": 10,
            "<ANSWER>": 42,
            "foo": 3,
        }

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, token: str):
        return self.vocab.get(token, -1)


def test_introspect_z_ids_sorted_by_index() -> None:
    tok = DummyTokenizer()
    assert introspect_z_token_ids(tok) == [10, 11]


def test_build_allowed_ids() -> None:
    tok = DummyTokenizer()
    assert build_allowed_token_ids(tok) == [10, 11, 42]


def test_masked_log_probs_and_entropy_only_use_allowed() -> None:
    logits = torch.tensor([0.1, 0.2, 0.3, 2.0, -0.7, 1.2])
    allowed = [1, 3, 5]

    log_probs, entropy = masked_log_probs_and_entropy(logits=logits, allowed_token_ids=allowed)
    probs = torch.exp(log_probs)

    # Forbidden indices must have effectively zero probability.
    assert probs[0].item() < 1e-8
    assert probs[2].item() < 1e-8
    assert probs[4].item() < 1e-8

    # Allowed indices should sum to ~1.
    assert abs(probs[allowed].sum().item() - 1.0) < 1e-6
    assert entropy.item() > 0
