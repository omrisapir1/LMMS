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


def validate_answer_token_single(tokenizer, answer_token: str, answer_token_id: int) -> None:
    enc = tokenizer.encode(answer_token, add_special_tokens=False)
    if enc != [int(answer_token_id)]:
        raise RuntimeError(
            f"Answer tokenization contract violated for {answer_token}: got {enc}, expected [{answer_token_id}]"
        )
