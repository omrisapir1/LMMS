from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset

ANSWER_TOKEN = "<ANSWER>"
THOUGHT_EMBEDDING_USER_PROMPT_TEMPLATE = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only answer) inside \\boxed{{}}.\n\n{problem}"
)

# 0=ignore(prompt), 1=z, 2=answer, 3=digit
TARGET_IGNORE = 0
TARGET_Z = 1
TARGET_ANSWER = 2
TARGET_DIGIT = 3


@dataclass
class SFTSample:
    question: str
    z_ids: List[int]
    answer_digits: List[int]
    source: str


def _digits_from_answer_int(answer_int: object) -> Optional[List[int]]:
    if answer_int is None:
        return None
    try:
        value = int(answer_int)
    except Exception:
        return None
    if value < 0 or value > 99999:
        return None
    return [int(ch) for ch in f"{value:05d}"]


def _validate_digits(digits: List[int]) -> List[int]:
    if len(digits) != 5:
        raise ValueError(f"answer_digits must have length 5, got {len(digits)}")
    out: List[int] = []
    for d in digits:
        v = int(d)
        if not (0 <= v <= 9):
            raise ValueError(f"answer digit out of range [0,9]: {v}")
        out.append(v)
    return out


class SFTDataset(Dataset):
    def __init__(
        self,
        *,
        records: Iterable[Dict],
        tokenizer,
        vocab_size: int,
        answer_token: str = ANSWER_TOKEN,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = int(vocab_size)
        self.answer_token = str(answer_token)

        self.answer_token_id = int(self.tokenizer.convert_tokens_to_ids(self.answer_token))
        if self.answer_token_id < 0:
            raise RuntimeError(f"Failed to resolve token id for {self.answer_token}")

        self.z_token_texts = [f"<z_{i}>" for i in range(self.vocab_size)]
        self.z_token_ids = [int(self.tokenizer.convert_tokens_to_ids(t)) for t in self.z_token_texts]
        if any(tid < 0 for tid in self.z_token_ids):
            raise RuntimeError("Some Z tokens were not added to tokenizer.")

        self.digit_token_ids = self._resolve_digit_token_ids()
        answer_enc = self.tokenizer.encode(self.answer_token, add_special_tokens=False)
        if answer_enc != [self.answer_token_id]:
            raise RuntimeError(
                f"Tokenization contract violated for {self.answer_token}: got {answer_enc}, "
                f"expected [{self.answer_token_id}]"
            )
        for i in range(self.vocab_size):
            z_tok = f"<z_{i}>"
            z_enc = self.tokenizer.encode(z_tok, add_special_tokens=False)
            if z_enc != [self.z_token_ids[i]]:
                raise RuntimeError(
                    f"Tokenization contract violated for {z_tok}: got {z_enc}, "
                    f"expected [{self.z_token_ids[i]}]"
                )
        self.samples: List[SFTSample] = []

        dropped = 0
        for row in records:
            q = row.get("question")
            z = row.get("z_ids")
            d = row.get("answer_digits")
            digits: Optional[List[int]] = None
            if d is not None:
                try:
                    digits = _validate_digits([int(x) for x in d])
                except Exception:
                    digits = None
            if digits is None:
                from_int = _digits_from_answer_int(row.get("answer_int"))
                if from_int is not None:
                    digits = _validate_digits(from_int)

            if q is None or z is None or digits is None:
                dropped += 1
                continue
            z_ids = [int(x) for x in z]
            if any(x < 0 or x >= self.vocab_size for x in z_ids):
                dropped += 1
                continue
            self.samples.append(
                SFTSample(
                    question=str(q),
                    z_ids=z_ids,
                    answer_digits=digits,
                    source=str(row.get("source", "")),
                )
            )

        self.stats = {"dropped": dropped, "kept": len(self.samples)}

    def _resolve_digit_token_ids(self) -> List[int]:
        ids: List[int] = []
        for d in "0123456789":
            tok = self.tokenizer.encode(d, add_special_tokens=False)
            if len(tok) != 1:
                raise RuntimeError(
                    f"Digit tokenization check failed for '{d}': expected 1 token, got {tok}"
                )
            ids.append(int(tok[0]))
        return ids

    def __len__(self) -> int:
        return len(self.samples)

    def _build_prompt_text(self, question: str) -> str:
        user_prompt = THOUGHT_EMBEDDING_USER_PROMPT_TEMPLATE.format(problem=question)
        messages = [
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        prompt_text = self._build_prompt_text(sample.question)
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]

        z_token_ids = [self.z_token_ids[z] for z in sample.z_ids]
        digit_ids = [self.digit_token_ids[d] for d in sample.answer_digits]
        expected_suffix = z_token_ids + [self.answer_token_id] + digit_ids
        input_ids = list(prompt_ids) + list(expected_suffix)
        attention_mask = [1] * len(input_ids)

        token_class = [TARGET_IGNORE] * len(prompt_ids)
        token_class += [TARGET_Z] * len(z_token_ids)
        token_class += [TARGET_ANSWER]
        token_class += [TARGET_DIGIT] * 5

        target_class = [TARGET_IGNORE] * len(input_ids)
        for pos in range(len(input_ids) - 1):
            target_class[pos] = token_class[pos + 1]

        labels = [-100] * len(input_ids)
        for pos in range(len(input_ids) - 1):
            tcls = target_class[pos]
            if tcls in (TARGET_Z, TARGET_ANSWER, TARGET_DIGIT):
                labels[pos] = int(input_ids[pos + 1])

        z_target_positions = [i for i, t in enumerate(target_class) if t == TARGET_Z]
        digit_target_positions = [i for i, t in enumerate(target_class) if t == TARGET_DIGIT]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_class": target_class,
            "z_len": len(z_token_ids),
            "z_target_positions": z_target_positions,
            "digit_target_positions": digit_target_positions,
        }


class SFTCollator:
    def __init__(self, *, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer requires pad_token_id or eos_token_id")
            self.pad_token_id = int(self.tokenizer.eos_token_id)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        clipped: List[Dict] = []
        for ex in batch:
            keep = min(len(ex["input_ids"]), self.max_length)
            ex2 = {
                "input_ids": ex["input_ids"][:keep],
                "attention_mask": ex["attention_mask"][:keep],
                "labels": ex["labels"][:keep],
                "target_class": ex["target_class"][:keep],
                "z_len": ex["z_len"],
                "z_target_positions": [p for p in ex["z_target_positions"] if p < keep],
                "digit_target_positions": [p for p in ex["digit_target_positions"] if p < keep],
            }
            clipped.append(ex2)

        max_len = max(len(x["input_ids"]) for x in clipped)

        def pad(vals: List[int], pad_value: int) -> List[int]:
            if len(vals) < max_len:
                return vals + [pad_value] * (max_len - len(vals))
            return vals

        input_ids = torch.tensor([pad(x["input_ids"], self.pad_token_id) for x in clipped], dtype=torch.long)
        attention_mask = torch.tensor([pad(x["attention_mask"], 0) for x in clipped], dtype=torch.long)
        labels = torch.tensor([pad(x["labels"], -100) for x in clipped], dtype=torch.long)
        target_class = torch.tensor([pad(x["target_class"], TARGET_IGNORE) for x in clipped], dtype=torch.long)
        z_lens = torch.tensor([int(x["z_len"]) for x in clipped], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_class": target_class,
            "z_lens": z_lens,
        }
