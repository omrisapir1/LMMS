from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from datasets import load_dataset

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


def compute_digit_labels(final_answer: int) -> torch.Tensor:
    if final_answer < 0 or final_answer > 99999:
        raise ValueError("final_answer out of range 0..99999")
    digits = [
        (final_answer // 10000) % 10,
        (final_answer // 1000) % 10,
        (final_answer // 100) % 10,
        (final_answer // 10) % 10,
        final_answer % 10,
    ]
    return torch.tensor(digits, dtype=torch.long)


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
        self.max_length = max_length
        self.k_max = int(k_max)

        if hf_dataset is not None:
            self.ds = hf_dataset
        elif data_path is not None:
            self.ds = load_dataset("json", data_files=data_path, split=split or "train")
        else:
            if dataset_name is None:
                raise ValueError("dataset_name or data_path must be provided")
            self.ds = load_dataset(dataset_name, split=split or "train")

        self._buckets: List[str] = []
        for i in range(len(self.ds)):
            k_val = int(self.ds[i]["K"])
            if not (1 <= k_val <= self.k_max):
                raise ValueError(f"K={k_val} out of range")
            # Buckets are used ONLY for sampling balance, not for conditioning or loss computation.
            self._buckets.append(k_to_bucket(k_val))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.ds[idx]
        question = ex["question"]
        k_val = int(ex["K"])
        final_answer = int(ex["final_answer"])
        if not (1 <= k_val <= self.k_max):
            raise ValueError(f"K={k_val} out of range")

        # Tokenize question
        q_ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        if self.max_length is not None:
            max_q = self.max_length - k_val - 1
            if max_q < 1:
                raise RuntimeError("max_length too small to fit latent tokens and <ANSWER>")
            if len(q_ids) > max_q:
                q_ids = q_ids[:max_q]

        input_ids = q_ids + [self.latent_token_id] * k_val + [self.answer_token_id]
        # Source of truth is the built sequence: enforce exact latent-slot count.
        if sum(1 for t in input_ids if t == self.latent_token_id) != k_val:
            raise RuntimeError("Built sequence latent-token count does not match K")
        attention_mask = [1] * len(input_ids)
        digit_labels = compute_digit_labels(final_answer)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "digit_labels": digit_labels,
            "K": torch.tensor(k_val, dtype=torch.long),
            "K_bucket": self._buckets[idx],
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    input_ids = [b["input_ids"] for b in batch]

    digit_labels = torch.stack([b["digit_labels"] for b in batch], dim=0)
    k_vals = torch.stack([b["K"] for b in batch], dim=0)
    k_buckets = [b["K_bucket"] for b in batch]

    input_padded, attention_mask = suffix_pad(input_ids, pad_value=pad_token_id)

    return {
        "input_ids": input_padded,
        "attention_mask": attention_mask,
        "digit_labels": digit_labels,
        "K": k_vals,
        "K_bucket": k_buckets,
    }


def build_rebalanced_sampler(
    dataset: UnifiedDataset,
    target_dist: Dict[str, float],
) -> WeightedRandomSampler:
    counts: Dict[str, int] = {k: 0 for k in target_dist}
    for b in dataset._buckets:
        if b not in counts:
            raise RuntimeError(f"Unexpected bucket '{b}'")
        counts[b] += 1

    for b, c in counts.items():
        if c == 0:
            raise RuntimeError(f"Bucket '{b}' has 0 samples")

    weights = []
    for b in dataset._buckets:
        weights.append(target_dist[b] / counts[b])

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
