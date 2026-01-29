# phase3/dataset.py
#
# Phase-3 Dataset
#
# Contract (per item):
# {
#   "input_ids":      [T]   prompt + Zs + <ANSWER>
#   "attention_mask": [T]   suffix-padding only
#   "digit_labels":   [5]
#   "num_latents":    int   K
#   "K_bucket":       str   one of K1, K2, K3, K4_7, K8_12, K13_20
# }
#
# Key properties:
# - deterministic
# - no model calls
# - suffix padding only
# - exactly one <ANSWER> token (last non-pad token)
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


# ---------------------------------------------------------------------
# Target K distribution (train-time rebalancing)
# ---------------------------------------------------------------------

TARGET_DIST = {
    "K1": 0.075,
    "K2": 0.10,
    "K3": 0.125,
    "K4_7": 0.300,
    "K8_12": 0.20,
    "K13_20": 0.20,
}


# ---------------------------------------------------------------------
# Dataset item (for type clarity)
# ---------------------------------------------------------------------

@dataclass
class Phase3Item:
    input_ids: torch.Tensor        # [T]
    attention_mask: torch.Tensor   # [T]
    digit_labels: torch.Tensor     # [5]


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class Phase3Dataset(Dataset):
    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
        hf_dataset=None,                      # 👈 NEW
        z_token_ids: List[int],
        answer_token_id: int,
        seed: int = 42,
        rebalance_train: bool = True,
        target_dist: Optional[Dict[str, float]] = None,
        max_length: Optional[int] = None,
    ):
        self.z_token_ids = set(int(x) for x in z_token_ids)
        self.answer_token_id = int(answer_token_id)
        self.seed = int(seed)

        self.rebalance_train = bool(rebalance_train)
        self.target_dist = dict(target_dist) if target_dist is not None else dict(TARGET_DIST)
        self.max_length = max_length

        # -------------------------
        # Dataset source
        # -------------------------
        if hf_dataset is not None:
            self.ds = hf_dataset
            self.split = "train" if rebalance_train else "eval"
        else:
            if dataset_name is None or split is None:
                raise ValueError(
                    "Phase3Dataset: either hf_dataset OR (dataset_name + split) must be provided"
                )
            self.ds = load_dataset(dataset_name, split=split)
            self.split = split

        # -------------------------
        # Indices / rebalancing
        # -------------------------
        if self.split == "train" and self.rebalance_train:
            self.indices = self._build_rebalanced_indices()
        else:
            self.indices = list(range(len(self.ds)))


    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------
    # Rebalancing (identical logic to Phase-2)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Item
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.ds[self.indices[idx]]

        input_ids = ex["input_ids"]
        attention_mask = ex["attention_mask"]
        digit_labels = ex["digit_labels"]
        K = int(ex["num_latents"])

        # ----------------------------
        # Basic checks
        # ----------------------------
        if not isinstance(input_ids, list) or not isinstance(attention_mask, list):
            raise RuntimeError("input_ids and attention_mask must be lists")

        if len(input_ids) != len(attention_mask):
            raise RuntimeError("input_ids and attention_mask length mismatch")

        if self.max_length is not None and len(input_ids) > self.max_length:
            raise RuntimeError(
                f"Sequence too long: len={len(input_ids)} > max_length={self.max_length}"
            )

        # exactly one <ANSWER> token
        answer_positions = [i for i, t in enumerate(input_ids) if t == self.answer_token_id]
        if len(answer_positions) != 1:
            raise RuntimeError(f"Expected exactly one <ANSWER> token, got {len(answer_positions)}")

        answer_pos = answer_positions[0]

        # must be last non-pad token
        if attention_mask[answer_pos] != 1 or any(attention_mask[answer_pos + 1 :]):
            raise RuntimeError("<ANSWER> must be the last non-pad token")

        # count Z tokens before <ANSWER>
        z_count = sum(
            1 for t in input_ids[:answer_pos] if t in self.z_token_ids
        )
        if z_count != K:
            raise RuntimeError(
                f"num_latents mismatch: declared K={K}, found {z_count} Z tokens"
            )

        # digit labels
        if not isinstance(digit_labels, list) or len(digit_labels) != 5:
            raise RuntimeError("digit_labels must be a list of length 5")

        # ----------------------------
        # Tensors
        # ----------------------------
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "digit_labels": torch.tensor(digit_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------

def phase3_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pads input_ids / attention_mask to batch max length (SUFFIX padding only).
    Stacks digit_labels.
    """

    max_len = max(x["input_ids"].size(0) for x in batch)

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

    input_ids = torch.stack([
        pad_1d(x["input_ids"], pad_value=0)
        for x in batch
    ])

    attention_mask = torch.stack([
        pad_1d(x["attention_mask"], pad_value=0)
        for x in batch
    ])

    digit_labels = torch.stack([
        x["digit_labels"] for x in batch
    ])

    # Safety: suffix padding only
    for i in range(attention_mask.size(0)):
        am = attention_mask[i]
        if (am == 0).any():
            first_zero = (am == 0).nonzero(as_tuple=False)[0].item()
            if not torch.all(am[first_zero:] == 0):
                raise RuntimeError("Non-suffix padding detected in attention_mask")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "digit_labels": digit_labels,
    }


__all__ = [
    "Phase3Dataset",
    "Phase3Item",
    "phase3_collate_fn",
    "TARGET_DIST",
]
