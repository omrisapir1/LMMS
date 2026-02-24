"""
Phase 1 dataset.
Consumes Hugging Face records with fields:
- question
- answer
- generated_answer

Derives:
- thoughts via split_thoughts(generated_answer)
- K = number of thoughts
- digit labels from answer

Dataset is stage-agnostic; stage logic is injected via num_latent_fn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .split_logic import split_thoughts


SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

LATENT_TOKEN = "<|latent|>"
# ANSWER_TOKEN is documented-only; actual token is provided via dataset constructor
ANSWER_TOKEN = "<ANSWER>"


def build_prompt(question: str, answer: str, tokenizer) -> Dict[str, List[int]]:
    """
    Build chat prompt text and return tokenized fields (`input_ids`, `attention_mask`).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        # {"role": "assistant", "content": answer},
    ]



    with_chat = tokenizer.apply_chat_template(messages,
                                              tokenize=False,
                                              add_generation_prompt=True)
    with_chat += answer
    return tokenizer(with_chat,  add_special_tokens=False, padding=False, return_attention_mask=True)


def format_answer(thoughts: List[str], K: int, num_latent: int, answer_token: str) -> str:
    """
    New semantics:
    - num_latent == 0:
        t0 t1 ... t(K-1)
    - num_latent == 1:
        t0 t1 ... t(K-2) <LATENT> <ANSWER>
    - num_latent >= 2:
        <LATENT> x (num_latent - 1)
        t(num_latent - 1) ... t(K - 2)
        <LATENT>
        <ANSWER>
    """
    num_latent = max(0, min(int(num_latent), int(K)))
    lines: List[str] = []

    if num_latent == 0:
        # No latent tokens at all
        for t in thoughts:
            lines.append(t)
        return "\n".join(lines)

    if num_latent == 1:
        # Replace last thought only
        for t in thoughts[:-1]:
            lines.append(t)
        lines.append(LATENT_TOKEN +'\n' + answer_token)
        return "\n".join(lines)

    # num_latent >= 2
    left_latents = num_latent - 1
    assert left_latents <= K - 1

    # Left latents
    lines.append('\n'.join([LATENT_TOKEN for _ in range(left_latents)]))

    # Middle thoughts
    for t in thoughts[left_latents:-1]:
        lines.append(t)

    # Final latent + answer

    return "\n".join(lines) + '\n' + LATENT_TOKEN + '\n' + answer_token


@dataclass
class Phase1Sample:
    question: str
    answer_int: int
    thoughts: List[str]
    K: int


class Phase1Dataset(Dataset):
    """
    Stage-agnostic dataset.
    - derives thoughts from generated_answer
    - clamps K by truncating thoughts to max_thoughts
    - applies external num_latent_fn(K)
    - appends exact 5 digit tokens after <ANSWER>
    """

    def __init__(
        self,
        *,
        records: Iterable[Dict],
        tokenizer,
        num_latent_fn: Callable[[int], int],
        k_filter: Optional[Callable[[int], bool]] = None,
        max_thoughts: int,
        answer_token: str = ANSWER_TOKEN,
        min_chars: int = 100,
        max_chars: int = 300,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_latent_fn = num_latent_fn
        self.k_filter = k_filter
        self.max_thoughts = int(max_thoughts)
        self.answer_token = str(answer_token)
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.samples: List[Phase1Sample] = []
        self.stats = {
            "dropped_missing_fields": 0,
            "dropped_bad_answer": 0,
            "dropped_empty_thoughts": 0,
        }
        self.digit_token_ids = self._resolve_digit_token_ids()
        self.answer_token_id = self.tokenizer.convert_tokens_to_ids(self.answer_token)
        if self.answer_token_id is None or int(self.answer_token_id) < 0:
            raise RuntimeError(f"Failed to resolve answer token id for {self.answer_token}")
        self.latent_token_id = self.tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
        if self.latent_token_id is None or int(self.latent_token_id) < 0:
            raise RuntimeError(f"Failed to resolve latent token id for {LATENT_TOKEN}")

        for rec in records:
            question = rec.get("question")
            answer = rec.get("answer")
            generated_answer = rec.get("generated_answer")
            if question is None or answer is None or generated_answer is None:
                self.stats["dropped_missing_fields"] += 1
                continue
            try:
                answer_int = int(answer)
            except (TypeError, ValueError):
                self.stats["dropped_bad_answer"] += 1
                continue
            if not (0 <= answer_int <= 99999):
                self.stats["dropped_bad_answer"] += 1
                continue

            thoughts = split_thoughts(
                str(generated_answer),
                min_chars=self.min_chars,
                max_chars=self.max_chars,
            )
            if not thoughts:
                self.stats["dropped_empty_thoughts"] += 1
                continue
            if len(thoughts) > self.max_thoughts:
                thoughts = thoughts[: self.max_thoughts]
            K = len(thoughts)
            if self.k_filter is not None and not bool(self.k_filter(K)):
                continue
            self.samples.append(
                Phase1Sample(
                    question=str(question),
                    answer_int=answer_int,
                    thoughts=thoughts,
                    K=K,
                )
            )

    def _resolve_digit_token_ids(self) -> List[int]:
        out: List[int] = []
        for d in "0123456789":
            ids = self.tokenizer.encode(d, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"Digit tokenization check failed for '{d}': "
                    f"expected 1 token, got {len(ids)} tokens -> {ids}."
                )
            out.append(int(ids[0]))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _answer_digit_ids(self, answer_int: int) -> tuple[str, List[int], List[int]]:
        digit_str = f"{int(answer_int):05d}"
        digit_ids: List[int] = []
        digit_values: List[int] = []
        for ch in digit_str:
            ids = self.tokenizer.encode(ch, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"Answer digit '{ch}' did not tokenize to exactly 1 token: {ids}"
                )
            digit_ids.append(int(ids[0]))
            digit_values.append(int(ch))
        if len(digit_ids) != 5:
            raise RuntimeError(
                f"Internal error: expected 5 digit tokens, got {len(digit_ids)}."
            )
        return digit_str, digit_ids, digit_values

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = self.samples[idx]
        num_latent = int(self.num_latent_fn(item.K))
        num_latent = max(0, min(num_latent, item.K))
        answer_text = format_answer(
            thoughts=item.thoughts,
            K=item.K,
            num_latent=num_latent,
            answer_token=self.answer_token,
        )
        prompt = build_prompt(item.question, answer_text, self.tokenizer)
        input_ids = list(prompt["input_ids"])
        attention_mask = list(prompt["attention_mask"])
        digit_str, digit_ids, digit_values = self._answer_digit_ids(item.answer_int)

        input_ids.extend(digit_ids)
        attention_mask.extend([1] * 5)

        answer_count = sum(1 for x in input_ids if int(x) == int(self.answer_token_id))
        if answer_count != 1:
            raise RuntimeError(
                f"Expected exactly one <ANSWER> token in sample, found {answer_count}."
            )

        labels = [-100] * len(input_ids)
        if len(input_ids) >= 2:
            labels[:-1] = input_ids[1:]
        digit_mask = [0] * len(input_ids)
        answer_pos = input_ids.index(int(self.answer_token_id))
        for i in range(5):
            label_pos = answer_pos + i
            source_pos = answer_pos + 1 + i
            if source_pos >= len(input_ids):
                raise RuntimeError("Digit supervision positions are out of range.")
            labels[label_pos] = input_ids[source_pos]
            digit_mask[label_pos] = 1

        latent_count = sum(1 for x in input_ids if int(x) == int(self.latent_token_id))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "digit_mask": digit_mask,
            "digit_values": digit_values,
            "digit_token_ids": digit_ids,
            "answer_token_id": int(self.answer_token_id),
            "latent_count": int(latent_count),
            "K": int(item.K),
            "digit_str": digit_str,
            "sample_idx": int(idx),
        }


class Phase1Collator:
    def __init__(
        self,
        *,
        tokenizer,
        max_length: int,
        answer_token: str = ANSWER_TOKEN,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.answer_token = answer_token
        self.answer_token_id = self.tokenizer.convert_tokens_to_ids(answer_token)
        if self.answer_token_id is None or int(self.answer_token_id) < 0:
            raise RuntimeError(f"Failed to resolve answer token id for {answer_token}.")
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer must have pad_token_id or eos_token_id.")
            self.pad_token_id = int(self.tokenizer.eos_token_id)
        self.stats = {
            "dropped_answer_missing_or_multi": 0,
            "dropped_truncated_digits": 0,
        }

    def _build_supervision(
        self,
        input_ids: Sequence[int],
        attention_mask: Sequence[int],
        latent_count: int,
        K: int,
        digit_values: Sequence[int],
        sample_idx: int,
    ) -> Optional[Dict[str, List[int]]]:
        answer_positions = [i for i, tid in enumerate(input_ids) if int(tid) == int(self.answer_token_id)]
        if len(answer_positions) != 1:
            self.stats["dropped_answer_missing_or_multi"] += 1
            return None
        answer_pos = int(answer_positions[0])
        digit_positions = [answer_pos + 1 + i for i in range(5)]
        if any(p >= len(input_ids) for p in digit_positions):
            self.stats["dropped_truncated_digits"] += 1
            return None

        labels = [-100] * len(input_ids)
        if len(input_ids) >= 2:
            labels[:-1] = input_ids[1:]
        digit_mask = [0] * len(input_ids)
        label_digit_positions: List[int] = []
        for i in range(5):
            lp = answer_pos + i
            digit_mask[lp] = 1
            label_digit_positions.append(lp)
        digit_target_ids = [int(input_ids[p]) for p in digit_positions]
        if len(digit_values) != 5:
            raise RuntimeError("digit_values must contain exactly 5 entries.")
        return {
            "input_ids": list(int(x) for x in input_ids),
            "attention_mask": list(int(x) for x in attention_mask),
            "labels": labels,
            "digit_mask": digit_mask,
            "digit_position_indices": label_digit_positions,
            "digit_target_token_ids": digit_target_ids,
            "digit_values": [int(x) for x in digit_values],
            "latent_count": int(latent_count),
            "K": int(K),
            "sample_idx": int(sample_idx),
        }

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        processed: List[Dict[str, List[int]]] = []
        for sample in batch:
            input_ids = list(sample["input_ids"])[: self.max_length]
            attention_mask = list(sample["attention_mask"])[: self.max_length]
            built = self._build_supervision(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_count=int(sample["latent_count"]),
                K=int(sample["K"]),
                digit_values=sample["digit_values"],
                sample_idx=int(sample["sample_idx"]),
            )
            if built is not None:
                processed.append(built)

        if not processed:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "labels": torch.empty(0, 0, dtype=torch.long),
                "digit_mask": torch.empty(0, 0, dtype=torch.bool),
                "digit_position_indices": torch.empty(0, 5, dtype=torch.long),
                "digit_target_token_ids": torch.empty(0, 5, dtype=torch.long),
                "digit_values": torch.empty(0, 5, dtype=torch.long),
                "latent_count": torch.empty(0, dtype=torch.long),
                "K": torch.empty(0, dtype=torch.long),
                "sample_idx": torch.empty(0, dtype=torch.long),
            }

        max_len = max(len(x["input_ids"]) for x in processed)
        input_ids_t = torch.full((len(processed), max_len), int(self.pad_token_id), dtype=torch.long)
        attention_mask_t = torch.zeros((len(processed), max_len), dtype=torch.long)
        labels_t = torch.full((len(processed), max_len), -100, dtype=torch.long)
        digit_mask_t = torch.zeros((len(processed), max_len), dtype=torch.bool)
        digit_pos_t = torch.full((len(processed), 5), -1, dtype=torch.long)
        digit_target_t = torch.full((len(processed), 5), -100, dtype=torch.long)
        digit_values_t = torch.full((len(processed), 5), -1, dtype=torch.long)
        latent_count_t = torch.zeros((len(processed),), dtype=torch.long)
        k_t = torch.zeros((len(processed),), dtype=torch.long)
        sample_idx_t = torch.zeros((len(processed),), dtype=torch.long)

        for i, row in enumerate(processed):
            n = len(row["input_ids"])
            input_ids_t[i, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
            attention_mask_t[i, :n] = torch.tensor(row["attention_mask"], dtype=torch.long)
            labels_t[i, :n] = torch.tensor(row["labels"], dtype=torch.long)
            digit_mask_t[i, :n] = torch.tensor(row["digit_mask"], dtype=torch.bool)
            digit_pos_t[i] = torch.tensor(row["digit_position_indices"], dtype=torch.long)
            digit_target_t[i] = torch.tensor(row["digit_target_token_ids"], dtype=torch.long)
            digit_values_t[i] = torch.tensor(row["digit_values"], dtype=torch.long)
            latent_count_t[i] = int(row["latent_count"])
            k_t[i] = int(row["K"])
            sample_idx_t[i] = int(row["sample_idx"])

        if bool((digit_pos_t < 0).any().item()):
            raise RuntimeError(
                "Collator produced invalid digit_position_indices (<0) in non-empty batch."
            )

        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "labels": labels_t,
            "digit_mask": digit_mask_t,
            "digit_position_indices": digit_pos_t,
            "digit_target_token_ids": digit_target_t,
            "digit_values": digit_values_t,
            "latent_count": latent_count_t,
            "K": k_t,
            "sample_idx": sample_idx_t,
        }
