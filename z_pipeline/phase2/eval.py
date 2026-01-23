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

import torch


@torch.no_grad()
def row_exclusive_assign_from_logits(
    z_logits: torch.Tensor,   # [B, Kmax, V]  (logits OR probs)
    z_mask: torch.Tensor,     # [B, Kmax]     (bool or 0/1)
) -> torch.Tensor:
    """
    Row-exclusive Z assignment.

    For each row:
      - Each latent picks a Z
      - No Z can be used more than once within the row
      - Greedy, confidence-ordered, deterministic

    Returns:
        z_ids: LongTensor[B, Kmax]
    """
    assert z_logits.ndim == 3, "z_logits must be [B, K, V]"
    assert z_mask.ndim == 2, "z_mask must be [B, K]"

    B, Kmax, V = z_logits.shape
    device = z_logits.device

    z_mask = z_mask.bool()
    z_ids = torch.zeros((B, Kmax), device=device, dtype=torch.long)

    for b in range(B):
        K = int(z_mask[b].sum().item())
        if K == 0:
            continue
        if K > V:
            raise RuntimeError(
                f"Row {b}: K={K} > V={V}, row-exclusive assignment impossible"
            )

        row_logits = z_logits[b, :K]   # [K, V]

        # ---- confidence = gap between best and second-best ----
        top2 = torch.topk(row_logits, k=2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]   # larger = more confident

        # process confident latents first
        order = torch.argsort(margins, descending=True)

        used = set()

        for idx in order.tolist():
            scores = row_logits[idx]
            candidates = torch.argsort(scores, descending=True)

            assigned = False
            for c in candidates.tolist():
                if c not in used:
                    z_ids[b, idx] = c
                    used.add(c)
                    assigned = True
                    break

            if not assigned:
                raise RuntimeError(
                    f"Row {b}: failed to assign latent {idx} uniquely"
                )

        # sanity
        if len(used) != K:
            raise RuntimeError(
                f"Row {b}: assigned {len(used)} Zs for K={K}"
            )

    return z_ids


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

    def filter_indices_by_min_k(ds: Phase2Dataset, min_k: int):
        return [
            i for i, ex in enumerate(ds.ds)
            if int(ex["num_latents"]) >= min_k
        ]

    # ds.indices = filter_indices_by_min_k(ds, min_k=17)

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
    total_rows_by_k: Dict[int, int] = defaultdict(int)
    correct_rows_by_k: Dict[int, int] = defaultdict(int)

    V: Optional[int] = None  # Z vocab size (inferred once)

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
        # z_ids = torch.argmax(z_probs, dim=-1)  # [B, Kmax]
        z_ids = row_exclusive_assign_from_logits(z_probs, z_mask)
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

            # Dominant-Z ratio for K >= 2
            if K >= 2:
                freq = Counter(row_z)
                m = max(freq.values())
                dominant_ratios_by_k[K].append(m / K)

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

    # Dominant-Z ratios by K
    dominant_z_ratio_by_k: Dict[int, float] = {}
    for K in range(2, k_max + 1):
        vals = dominant_ratios_by_k.get(K, [])
        if not vals:
            continue
        dominant_z_ratio_by_k[K] = float(np.mean(vals))

    digit_em_by_k: Dict[int, float] = {}
    for K, total in total_rows_by_k.items():
        if total == 0:
            continue
        digit_em_by_k[K] = correct_rows_by_k[K] / total

    # ------------------------------------------------------------------
    return {
        "digit_em": float(digit_em),
        "digit_em_by_k": digit_em_by_k,
        "z_distribution": z_distribution,
        "z_distribution_k1": z_distribution_k1,
        "dominant_z_ratio_by_k": dominant_z_ratio_by_k,
    }
