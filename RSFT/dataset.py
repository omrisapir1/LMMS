from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset

from PPO.reward import parse_answer_digits, parse_final_answer_to_digits
from phase1.dataset import SYSTEM_PROMPT


@dataclass
class PromptExample:
    question: str
    prompt_ids: List[int]
    true_digits: List[int]


def build_prompt_text(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(question)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_true_digits(row: Dict[str, object], *, answer_digits_field: str, answer_field: str) -> Optional[List[int]]:
    if answer_digits_field in row:
        parsed = parse_answer_digits(row.get(answer_digits_field))
        if parsed is not None:
            return parsed
    return parse_final_answer_to_digits(row.get(answer_field))


def load_hf_records(dataset_name: str, split: str) -> List[Dict[str, object]]:
    ds = load_dataset(dataset_name, split=split)
    return [dict(x) for x in ds]


def prepare_prompt_examples(
    *,
    records: Sequence[Dict[str, object]],
    tokenizer,
    question_field: str,
    answer_digits_field: str,
    answer_field: str,
) -> List[PromptExample]:
    out: List[PromptExample] = []
    for row in records:
        q = row.get(question_field)
        if q is None:
            continue
        true_digits = parse_true_digits(row, answer_digits_field=answer_digits_field, answer_field=answer_field)
        if true_digits is None:
            continue

        prompt_text = build_prompt_text(tokenizer, str(q))
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_attention_mask=False).get("input_ids", [])
        if not isinstance(prompt_ids, list) or len(prompt_ids) == 0:
            continue

        out.append(PromptExample(question=str(q), prompt_ids=[int(x) for x in prompt_ids], true_digits=[int(x) for x in true_digits]))
    return out


def sample_unique_prompt_batch(
    *,
    examples: Sequence[PromptExample],
    ordered_indices: Sequence[int],
    cursor: int,
    batch_size: int,
    seen_questions: set[str],
) -> tuple[List[PromptExample], int]:
    selected: List[PromptExample] = []
    idx_cursor = int(cursor)
    n = len(ordered_indices)
    while idx_cursor < n and len(selected) < int(batch_size):
        ex = examples[int(ordered_indices[idx_cursor])]
        idx_cursor += 1
        if ex.question in seen_questions:
            continue
        seen_questions.add(ex.question)
        selected.append(ex)
    return selected, idx_cursor


def make_digit_id_to_value_map(digit_token_ids: Sequence[int]) -> Dict[int, int]:
    if len(digit_token_ids) != 10:
        raise ValueError("digit_token_ids must have length 10")
    return {int(tok_id): idx for idx, tok_id in enumerate(digit_token_ids)}
