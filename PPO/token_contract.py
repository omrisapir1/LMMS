from __future__ import annotations

from typing import List


def resolve_digit_token_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    for d in "0123456789":
        tok = tokenizer.encode(d, add_special_tokens=False)
        if len(tok) != 1:
            raise RuntimeError(f"Digit tokenization contract violated for '{d}': got {tok}")
        ids.append(int(tok[0]))
    return ids


def validate_single_token(tokenizer, token_text: str, token_id: int, *, label: str) -> None:
    enc = tokenizer.encode(token_text, add_special_tokens=False)
    if enc != [int(token_id)]:
        raise RuntimeError(
            f"{label} tokenization contract violated for {token_text}: got {enc}, expected [{token_id}]"
        )


def validate_answer_token_single(tokenizer, answer_token: str, answer_token_id: int) -> None:
    validate_single_token(tokenizer, answer_token, answer_token_id, label="Answer")
