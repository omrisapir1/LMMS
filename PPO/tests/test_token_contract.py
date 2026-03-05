from __future__ import annotations

import pytest

from PPO.token_contract import resolve_digit_token_ids, validate_answer_token_single


class DummyTokenizer:
    def __init__(self):
        self.single = {str(i): [i] for i in range(10)}
        self.special = {"<ANSWER>": [99]}

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        if text in self.single:
            return list(self.single[text])
        if text in self.special:
            return list(self.special[text])
        return []


def test_answer_token_single_ok() -> None:
    tok = DummyTokenizer()
    validate_answer_token_single(tok, "<ANSWER>", 99)


def test_answer_token_multi_errors() -> None:
    tok = DummyTokenizer()
    tok.special["<ANSWER>"] = [99, 100]
    with pytest.raises(RuntimeError, match="Answer tokenization contract violated"):
        validate_answer_token_single(tok, "<ANSWER>", 99)


def test_digit_single_tokens_ok() -> None:
    tok = DummyTokenizer()
    ids = resolve_digit_token_ids(tok)
    assert ids == list(range(10))


def test_digit_multi_token_errors() -> None:
    tok = DummyTokenizer()
    tok.single["7"] = [7, 70]
    with pytest.raises(RuntimeError, match="Digit tokenization contract violated"):
        resolve_digit_token_ids(tok)
