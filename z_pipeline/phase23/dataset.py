from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, WeightedRandomSampler

from .utils import k_to_bucket, suffix_pad


TARGET_DIST = {
    "K1": 0.075,
    "K2": 0.10,
    "K3": 0.125,
    "K4_7": 0.300,
    "K8_12": 0.20,
    "K13_20": 0.20,
}


@dataclass
class UnifiedItem:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    digit_labels: torch.Tensor
    k_value: int
    k_bucket: str


def _validate_answer_digits(answer_digits: object) -> torch.Tensor:
    if not isinstance(answer_digits, (list, tuple)):
        raise ValueError("answer_digits must be a list/tuple of length 5")
    if len(answer_digits) != 5:
        raise ValueError("answer_digits must have length 5")

    vals: List[int] = []
    for d in answer_digits:
        v = int(d)
        if not (0 <= v <= 9):
            raise ValueError("answer_digits values must be in [0,9]")
        vals.append(v)
    return torch.tensor(vals, dtype=torch.long)


def _digits_to_int(digits: torch.Tensor) -> int:
    out = 0
    for d in digits.tolist():
        out = out * 10 + int(d)
    return out


class UnifiedDataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer,
        latent_token_id: int,
        answer_token_id: int,
        k_max: int,
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
        data_path: Optional[str] = None,
        hf_dataset=None,
        max_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.latent_token_id = int(latent_token_id)
        self.answer_token_id = int(answer_token_id)
        self.k_max = int(k_max)
        self.max_length = max_length

        if hf_dataset is not None:
            self.ds = hf_dataset
        elif dataset_name is not None:
            self.ds = load_dataset(dataset_name, split=split or "train")
        elif data_path is not None:
            self.ds = self._load_from_data_path(data_path=data_path, split=split or "train")
        else:
            raise ValueError("Provide one of: hf_dataset, local data_path, or dataset_name")

        self._buckets: List[str] = []
        for i in range(len(self.ds)):
            ex = self.ds[i]
            if "num_latents" not in ex:
                raise KeyError("Dataset samples must contain 'num_latents'")
            k_val = int(ex["num_latents"])
            if not (1 <= k_val <= self.k_max):
                raise ValueError(f"K={k_val} out of range [1,{self.k_max}]")
            # Buckets are used ONLY for sampling balance, not for conditioning or loss computation.
            self._buckets.append(k_to_bucket(k_val))

    @staticmethod
    def _load_from_data_path(*, data_path: str, split: str):
        # If the path doesn't exist locally, treat it as an HF dataset repo id.
        if not os.path.exists(data_path):
            return load_dataset(data_path, split=split)

        # Existing local file: pick loader by extension.
        if os.path.isfile(data_path):
            ext = os.path.splitext(data_path)[1].lower()
            if ext in {".json", ".jsonl", ".ndjson"}:
                return load_dataset("json", data_files=data_path, split=split)
            if ext == ".parquet":
                return load_dataset("parquet", data_files=data_path, split=split)
            if ext == ".csv":
                return load_dataset("csv", data_files=data_path, split=split)
            raise ValueError(
                f"Unsupported local dataset file extension: {ext}. "
                "Use .json/.jsonl/.ndjson/.parquet/.csv or pass an HF dataset repo id."
            )

        # Existing local directory: let datasets resolve it (script/builder format).
        return load_dataset(data_path, split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]

        question = ex["question"]
        k_val = int(ex["num_latents"])
        if "digit_labels" in ex:
            raw_digits = ex["digit_labels"]
        elif "answer_digits" in ex:
            raw_digits = ex["answer_digits"]
        else:
            raise KeyError(
                "Dataset samples must contain either 'digit_labels' or 'answer_digits' "
                "(list/tuple of 5 digits)"
            )

        digit_labels = _validate_answer_digits(raw_digits)
        answer_digits = _digits_to_int(digit_labels)

        if not (1 <= k_val <= self.k_max):
            raise ValueError(f"K={k_val} out of range [1,{self.k_max}]")

        q_ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        if self.max_length is not None:
            max_q = self.max_length - k_val - 1
            if max_q < 1:
                raise RuntimeError("max_length too small for [question + latents + <ANSWER>]")
            if len(q_ids) > max_q:
                q_ids = q_ids[:max_q]

        input_ids = q_ids + [self.latent_token_id] * k_val + [self.answer_token_id]
        if sum(1 for tok in input_ids if tok == self.latent_token_id) != k_val:
            raise RuntimeError("Constructed sequence latent count does not match num_latents")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "digit_labels": digit_labels,
            "K": torch.tensor(k_val, dtype=torch.long),
            "K_bucket": self._buckets[idx],
            "question": question,
            "answer_digits": torch.tensor(answer_digits, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    input_ids = [b["input_ids"] for b in batch]
    digit_labels = torch.stack([b["digit_labels"] for b in batch], dim=0)
    k_vals = torch.stack([b["K"] for b in batch], dim=0)
    k_buckets = [b["K_bucket"] for b in batch]
    questions = [str(b["question"]) for b in batch]
    answer_digits = torch.stack([b["answer_digits"] for b in batch], dim=0)

    input_padded, attention_mask = suffix_pad(input_ids, pad_value=pad_token_id)

    return {
        "input_ids": input_padded,
        "attention_mask": attention_mask,
        "digit_labels": digit_labels,
        "K": k_vals,
        "K_bucket": k_buckets,
        "question": questions,
        "answer_digits": answer_digits,
    }


def build_rebalanced_sampler(
    dataset: UnifiedDataset,
    target_dist: Dict[str, float],
) -> WeightedRandomSampler:
    counts: Dict[str, int] = {k: 0 for k in target_dist}
    for bucket in dataset._buckets:
        if bucket not in counts:
            raise RuntimeError(f"Unexpected bucket '{bucket}'")
        counts[bucket] += 1

    for bucket, c in counts.items():
        if c == 0:
            raise RuntimeError(f"Bucket '{bucket}' has zero samples")

    weights = [target_dist[b] / counts[b] for b in dataset._buckets]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
