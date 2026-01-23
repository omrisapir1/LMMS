# phase2/eval.py
#
# Phase-2 evaluation:
# 1) Digit exact-match accuracy (all 5 digits correct)
# 2) Overall Zi distribution (argmax) + Zi distribution for K==1 rows
# 3) For K=2..20: average dominant-Z ratio per row
#
from __future__ import annotations

from typing import Dict, List, Optional
from collections import Counter, defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader

from .dataset import Phase2Dataset, phase2_collate_fn


@torch.no_grad()
def evaluate_phase2(
    *,
    model,
    tokenizer,
    dataset_name: str,
    batch_size: int,
    latent_token_id: int,
    answer_token_id: int,
    k_max: int = 20,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Returns:
    {
      "digit_em": float,
      "z_distribution": np.ndarray [V],
      "z_distribution_k1": np.ndarray [V],
      "dominant_z_ratio_by_k": { K: float }
    }
    """

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # ------------------------------------------------------------------
    # Dataset (eval split, no rebalance, deterministic)
    # ------------------------------------------------------------------
    ds = Phase2Dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split="eval",
        k_max=k_max,
        latent_token_id=latent_token_id,
        answer_token_id=answer_token_id,
        rebalance_train=False,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=phase2_collate_fn,
    )


    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    total_rows = 0
    correct_rows = 0

    z_counts_global: Counter = Counter()
    z_counts_k1: Counter = Counter()

    dominant_ratios_by_k: Dict[int, List[float]] = defaultdict(list)
    unique_ratios_by_k: Dict[int, List[float]] = defaultdict(list)
    adjacent_repeat_by_k: Dict[int, List[float]] = defaultdict(list)

    total_rows_by_k: Dict[int, int] = defaultdict(int)
    correct_rows_by_k: Dict[int, int] = defaultdict(int)

    V: Optional[int] = None


    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        latent_states = batch["latent_states"].to(device)
        z_mask = batch["z_mask"].to(device)
        digit_labels = batch["digit_labels"].to(device)

        B, Kmax = z_mask.shape

        # --------------------------------------------------------------
        # Forward pass
        # --------------------------------------------------------------
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_states=latent_states,
            z_mask=z_mask,
            temperature=None,              # ignored in eval
            return_z_probs=True,
        )

        if "z_probs" not in out:
            raise RuntimeError("Phase2Model must return z_probs during eval")

        if "digit_logits" not in out:
            raise RuntimeError("Phase2Model must return digit_logits during eval")

        z_probs = out["z_probs"]          # [B, Kmax, V]
        digit_logits = out["digit_logits"]  # [B, 5, 10]

        if z_probs.ndim != 3:
            raise RuntimeError("z_probs must be [B, K, V]")

        if digit_logits.shape != (B, 5, 10):
            raise RuntimeError("digit_logits must be [B, 5, 10]")

        _, _, V_batch = z_probs.shape
        if V is None:
            V = V_batch
        elif V != V_batch:
            raise RuntimeError("Z vocab size mismatch across batches")

        # --------------------------------------------------------------
        # Argmax Z selection
        # --------------------------------------------------------------
        z_ids = torch.argmax(z_probs, dim=-1)  # [B, Kmax]

        # --------------------------------------------------------------
        # Digit exact match
        # --------------------------------------------------------------
        pred_digits = torch.argmax(digit_logits, dim=-1)  # [B, 5]
        em = (pred_digits == digit_labels).all(dim=1)     # [B]
        correct_rows += int(em.sum().item())
        total_rows += B

        # --------------------------------------------------------------
        # Z statistics
        # --------------------------------------------------------------
        for b in range(B):
            K = int(z_mask[b].sum().item())
            if K <= 0:
                continue

            total_rows_by_k[K] += 1
            if em[b].item():
                correct_rows_by_k[K] += 1

            row_z = z_ids[b, :K].tolist()

            # Global distribution
            z_counts_global.update(row_z)

            # K == 1 distribution
            if K == 1:
                z_counts_k1[row_z[0]] += 1

            # -------------------------------
            # Row-level Z diagnostics
            # -------------------------------

            # Dominant-Z ratio
            if K >= 2:
                freq = Counter(row_z)
                dominant_ratios_by_k[K].append(max(freq.values()) / K)

            # Unique ratio
            unique_ratios_by_k[K].append(len(set(row_z)) / K)

            # Adjacent repeat rate
            if K >= 2:
                repeats = sum(
                    1 for i in range(K - 1) if row_z[i] == row_z[i + 1]
                )
                adjacent_repeat_by_k[K].append(repeats / (K - 1))


    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    if total_rows == 0:
        raise RuntimeError("Evaluation dataset is empty")

    digit_em = correct_rows / total_rows

    # Z distributions
    z_distribution = np.zeros(V, dtype=np.float64)
    z_distribution_k1 = np.zeros(V, dtype=np.float64)

    total_z = sum(z_counts_global.values())
    if total_z > 0:
        for z, c in z_counts_global.items():
            z_distribution[z] = c / total_z

    total_z_k1 = sum(z_counts_k1.values())
    if total_z_k1 > 0:
        for z, c in z_counts_k1.items():
            z_distribution_k1[z] = c / total_z_k1

    # ------------------------------------------------------------------
    # Global effective vocab size
    # ------------------------------------------------------------------
    eps = 1e-12
    entropy = -np.sum(
        z_distribution * np.log(z_distribution + eps)
    )
    effective_vocab_size = float(np.exp(entropy))


    digit_em_by_k: Dict[int, float] = {}
    for K, total in total_rows_by_k.items():
        if total == 0:
            continue
        digit_em_by_k[K] = correct_rows_by_k[K] / total

    # Row metrics by K
    unique_ratio_by_k = {
        K: float(np.mean(v))
        for K, v in unique_ratios_by_k.items()
        if v
    }

    adjacent_repeat_rate_by_k = {
        K: float(np.mean(v))
        for K, v in adjacent_repeat_by_k.items()
        if v
    }

    return {
        "digit_em": float(digit_em),
        "digit_em_by_k": digit_em_by_k,

        "effective_vocab_size": effective_vocab_size,

        "unique_ratio_by_k": unique_ratio_by_k,
        "adjacent_repeat_rate_by_k": adjacent_repeat_rate_by_k,
    }
