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

def sample_z_ids(
    z_probs: torch.Tensor,   # [B, K, V]
    z_mask: torch.Tensor,    # [B, K]
    temperature: float,
) -> torch.Tensor:
    """
    Returns z_ids [B, K] using temperature sampling.
    temperature = 0 → argmax
    """
    if temperature == 0.0:
        return torch.argmax(z_probs, dim=-1)

    B, K, V = z_probs.shape
    z_ids = torch.zeros(B, K, device=z_probs.device, dtype=torch.long)

    logits = torch.log(z_probs + 1e-12) / temperature

    for b in range(B):
        k = int(z_mask[b].sum().item())
        if k == 0:
            continue
        probs = torch.softmax(logits[b, :k], dim=-1)  # [k, V]
        z_ids[b, :k] = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return z_ids


def _accumulate_metrics_for_z_ids(
    *,
    z_ids: torch.Tensor,        # [B, Kmax]
    z_mask: torch.Tensor,       # [B, Kmax]
    digit_logits: torch.Tensor, # [B, 5, 10]
    digit_labels: torch.Tensor,
    accum,
):
    B, Kmax = z_mask.shape

    pred_digits = torch.argmax(digit_logits, dim=-1)
    em = (pred_digits == digit_labels).all(dim=1)

    accum["total_rows"] += B
    accum["correct_rows"] += int(em.sum().item())

    for b in range(B):
        K = int(z_mask[b].sum().item())
        if K <= 0:
            continue

        row_z = z_ids[b, :K].tolist()

        accum["total_rows_by_k"][K] += 1
        if em[b].item():
            accum["correct_rows_by_k"][K] += 1

        accum["z_counts"].update(row_z)

        # Unique ratio
        accum["unique_ratios_by_k"][K].append(len(set(row_z)) / K)

        # Adjacent repeat rate
        if K >= 2:
            repeats = sum(row_z[i] == row_z[i + 1] for i in range(K - 1))
            accum["adjacent_repeat_by_k"][K].append(repeats / (K - 1))


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

    if device is None:
        device = next(model.parameters()).device

    model.eval()

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

    temps = {
        "argmax": 0.0,
        "t0.5": 0.5,
        "t1.0": 1.0,
        "t2.0": 2.0,
    }

    modes = {}

    for mode_name, temp in temps.items():
        accum = dict(
            total_rows=0,
            correct_rows=0,
            z_counts=Counter(),
            total_rows_by_k=defaultdict(int),
            correct_rows_by_k=defaultdict(int),
            unique_ratios_by_k=defaultdict(list),
            adjacent_repeat_by_k=defaultdict(list),
        )

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            latent_states = batch["latent_states"].to(device)
            z_mask = batch["z_mask"].to(device)
            digit_labels = batch["digit_labels"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_states=latent_states,
                z_mask=z_mask,
                temperature=None,
                return_z_probs=True,
            )

            z_probs = out["z_probs"]
            digit_logits = out["digit_logits"]

            z_ids = sample_z_ids(z_probs, z_mask, temp)

            _accumulate_metrics_for_z_ids(
                z_ids=z_ids,
                z_mask=z_mask,
                digit_logits=digit_logits,
                digit_labels=digit_labels,
                accum=accum,
            )

        # ---- finalize metrics ----
        V = model.z_vocab_size
        z_distribution = np.zeros(V, dtype=np.float64)
        total_z = sum(accum["z_counts"].values())
        for z, c in accum["z_counts"].items():
            z_distribution[z] = c / max(total_z, 1)

        entropy = -np.sum(z_distribution * np.log(z_distribution + 1e-12))
        effective_vocab_size = float(np.exp(entropy))

        digit_em = accum["correct_rows"] / max(accum["total_rows"], 1)

        digit_em_by_k = {
            K: accum["correct_rows_by_k"][K] / accum["total_rows_by_k"][K]
            for K in accum["total_rows_by_k"]
            if accum["total_rows_by_k"][K] > 0
        }

        unique_ratio_by_k = {
            K: float(np.mean(v))
            for K, v in accum["unique_ratios_by_k"].items()
            if v
        }

        adjacent_repeat_rate_by_k = {
            K: float(np.mean(v))
            for K, v in accum["adjacent_repeat_by_k"].items()
            if v
        }

        modes[mode_name] = dict(
            digit_em=digit_em,
            digit_em_by_k=digit_em_by_k,
            effective_vocab_size=effective_vocab_size,
            unique_ratio_by_k=unique_ratio_by_k,
            adjacent_repeat_rate_by_k=adjacent_repeat_rate_by_k,
        )

    return {"modes": modes}

