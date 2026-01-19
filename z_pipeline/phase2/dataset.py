# phase2/dataset.py
#
# Phase-2 Dataset (adjusted)
#
# Key changes:
# - input_ids contain EXACTLY K <|latent|> tokens (variable length)
# - padding is SUFFIX ONLY (handled in collate_fn)
# - latent_states and z_mask are padded to Kmax (model-side convenience)
#
# HF dataset columns expected:
#   question: str
#   latent_states: list[list[float]]   # [K, H]  (already states[:-1])
#   answer_digits: list[int]            # [5]
#   num_latents: int                    # K
#   K_bucket: str
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

TARGET_DIST = {
    "K1": 0.225,
    "K2": 0.175,
    "K3": 0.100,
    "K4_7": 0.200,
    "K8_12": 0.150,
    "K13_20": 0.150,
}


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@dataclass
class Phase2Item:
    input_ids: torch.Tensor        # [T]
    attention_mask: torch.Tensor   # [T]
    latent_states: torch.Tensor    # [Kmax, H]
    z_mask: torch.Tensor           # [Kmax]
    digit_labels: torch.Tensor     # [5]


class Phase2Dataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer,
        dataset_name: str,
        split: str,                        # "train" | "eval"
        k_max: int,
        latent_token_id: int,
        answer_token_id: int,
        seed: int = 42,
        rebalance_train: bool = True,
        target_dist: Optional[Dict[str, float]] = None,
        max_length: Optional[int] = None,
    ):
        if split not in ("train", "eval"):
            raise ValueError("split must be 'train' or 'eval'")

        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.k_max = int(k_max)
        self.latent_token_id = int(latent_token_id)
        self.answer_token_id = int(answer_token_id)
        self.seed = int(seed)

        self.pad_id = tokenizer.pad_token_id or 0

        if max_length is None:
            ml = getattr(tokenizer, "model_max_length", None)
            if ml is None or ml > 10**8:
                raise ValueError("tokenizer.model_max_length not set; pass max_length explicitly")
            self.max_length = int(ml)
        else:
            self.max_length = int(max_length)

        self.ds = load_dataset(dataset_name, split=split)

        self.rebalance_train = bool(rebalance_train)
        self.target_dist = dict(target_dist) if target_dist is not None else dict(TARGET_DIST)

        if self.split == "train" and self.rebalance_train:
            self.indices = self._build_rebalanced_indices()
        else:
            self.indices = list(range(len(self.ds)))

    def __len__(self) -> int:
        return len(self.indices)

    # ─────────────────────────────────────────────
    # Rebalancing
    # ─────────────────────────────────────────────

    def _build_rebalanced_indices(self) -> List[int]:
        s = sum(self.target_dist.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"target_dist must sum to 1.0, got {s}")

        buckets: Dict[str, List[int]] = {k: [] for k in self.target_dist}
        for i in range(len(self.ds)):
            b = self.ds[i].get("K_bucket", None)
            if b not in buckets:
                raise RuntimeError(f"Unexpected K_bucket='{b}' at row {i}")
            buckets[b].append(i)

        for b, idxs in buckets.items():
            if not idxs:
                raise RuntimeError(f"Bucket '{b}' has 0 samples")

        N = len(self.ds)
        target_counts = {b: int(round(self.target_dist[b] * N)) for b in buckets}
        diff = N - sum(target_counts.values())
        if diff != 0:
            biggest = max(self.target_dist, key=self.target_dist.get)
            target_counts[biggest] += diff

        rng = np.random.default_rng(self.seed)
        out: List[int] = []
        for b, n in target_counts.items():
            pool = buckets[b]
            out.extend(rng.choice(pool, size=n, replace=True).tolist())

        rng.shuffle(out)
        return out

    # ─────────────────────────────────────────────
    # Item
    # ─────────────────────────────────────────────

    def _tokenize_prompt_ids(self, question: str) -> List[int]:
        prompt = build_prompt(self.tokenizer, question)
        enc = self.tokenizer(prompt, add_special_tokens=False, truncation=False)
        return list(enc["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.ds[self.indices[idx]]

        question = ex["question"]
        latent_states_list = ex["latent_states"]   # [K, H]
        answer_digits = ex["answer_digits"]
        K = int(ex["num_latents"])

        if K <= 0 or K > self.k_max:
            raise RuntimeError(f"num_latents={K} out of range")

        prompt_ids = self._tokenize_prompt_ids(question)

        # Build input_ids: prompt + K LATENT + ANSWER
        input_ids = (
            prompt_ids
            + [self.latent_token_id] * K
            + [self.answer_token_id]
        )

        if len(input_ids) > self.max_length:
            raise RuntimeError(
                f"Sequence too long: len={len(input_ids)} > max_length={self.max_length}"
            )

        attention_mask = [1] * len(input_ids)

        # latent_states padding
        H = len(latent_states_list[0])
        latent_states = torch.zeros((self.k_max, H), dtype=torch.float32)
        latent_states[:K] = torch.tensor(latent_states_list, dtype=torch.float32)

        z_mask = torch.zeros(self.k_max, dtype=torch.bool)
        z_mask[:K] = True

        digit_labels = torch.tensor(answer_digits, dtype=torch.long)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "latent_states": latent_states,
            "z_mask": z_mask,
            "digit_labels": digit_labels,
        }


# ─────────────────────────────────────────────
# Collate function (IMPORTANT)
# ─────────────────────────────────────────────
def phase2_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pads input_ids / attention_mask to batch max length (SUFFIX padding only).
    latent_states and z_mask are already fixed-size.
    """
    max_len = max(x["input_ids"].size(0) for x in batch)

    # Use pad token ID from dataset (already resolved correctly)
    pad_id = getattr(batch[0]["input_ids"], "new_tensor")(0).item()
    # NOTE: pad_id will be overwritten below from attention_mask logic

    def pad_1d(x: torch.Tensor, pad_value: int) -> torch.Tensor:
        if x.size(0) == max_len:
            return x
        pad_len = max_len - x.size(0)
        pad = torch.full(
            (pad_len,),
            pad_value,
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, pad], dim=0)

    # ------------------------------------------------------------------
    # input_ids + attention_mask (suffix padding)
    # ------------------------------------------------------------------
    input_ids = torch.stack([
        pad_1d(x["input_ids"], pad_value=batch[0]["input_ids"].new_tensor(0).item())
        for x in batch
    ])

    attention_mask = torch.stack([
        pad_1d(x["attention_mask"], pad_value=0)
        for x in batch
    ])

    # ------------------------------------------------------------------
    # Fixed-size tensors (already padded)
    # ------------------------------------------------------------------
    latent_states = torch.stack([x["latent_states"] for x in batch])
    z_mask = torch.stack([x["z_mask"] for x in batch])
    digit_labels = torch.stack([x["digit_labels"] for x in batch])

    # ------------------------------------------------------------------
    # Safety: ensure suffix padding only
    # ------------------------------------------------------------------
    for i in range(attention_mask.size(0)):
        am = attention_mask[i]
        if (am == 0).any():
            first_zero = (am == 0).nonzero(as_tuple=False)[0].item()
            if not torch.all(am[first_zero:] == 0):
                raise RuntimeError("Non-suffix padding detected in attention_mask")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "latent_states": latent_states,
        "z_mask": z_mask,
        "digit_labels": digit_labels,
    }

def compute_keep_prob_from_dataset(dataset, alpha=0.3, min_k=0.05):
    zero_counts = [0] * 5
    total = 0

    # If a Phase2Dataset was passed, iterate underlying HF dataset to avoid heavy __getitem__
    source_iter = getattr(dataset, "ds", None)
    if source_iter is None:
        source_iter = dataset

    for ex in source_iter:
        # HF dataset rows have 'answer_digits'; batched items have 'digit_labels'
        digits = ex.get("answer_digits")

        for i in range(5):
            val = int(digits[i])

            if val == 0:
                zero_counts[i] += 1
        total += 1

    keep_prob = []
    for i in range(5):
        zr = zero_counts[i] / max(total, 1)
        if zr == 0:
            kp = 1.0
        else:
            kp = min(1.0, max(min_k, alpha / zr))
        keep_prob.append(kp)

    return keep_prob



__all__ = [
    "Phase2Dataset",
    "Phase2Item",
    "phase2_collate_fn",
    "TARGET_DIST",
    "SYSTEM_PROMPT",
    "build_prompt",
]
