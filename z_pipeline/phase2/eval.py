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

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # ---------------------------------------------------------
    # Evaluation modes
    # ---------------------------------------------------------
    eval_modes = {
        "argmax": None,
        "temp_0.3": 0.3,
        "temp_0.7": 0.7,
        "temp_1.0": 1.0,
    }

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
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

    all_results = {}

    # =========================================================
    # Loop over decoding modes
    # =========================================================
    for mode, temperature in eval_modes.items():

        total_rows = 0
        correct_rows = 0

        z_counts_global = Counter()
        z_counts_k1 = Counter()

        unique_ratios_by_k = defaultdict(list)
        adjacent_repeat_by_k = defaultdict(list)

        total_rows_by_k = defaultdict(int)
        correct_rows_by_k = defaultdict(int)

        V = None

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            latent_states = batch["latent_states"].to(device)
            z_mask = batch["z_mask"].to(device)
            digit_labels = batch["digit_labels"].to(device)

            B, Kmax = z_mask.shape

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_states=latent_states,
                z_mask=z_mask,
                temperature=None,   # model already outputs probs
                return_z_probs=True,
            )

            z_probs = out["z_probs"]          # [B, K, V]
            z_logits = out["z_logits"]       
            digit_logits = out["digit_logits"]

            _, _, V_batch = z_probs.shape
            if V is None:
                V = V_batch

            # --------------------------------------------------
            # Z selection per mode
            # --------------------------------------------------
            if temperature is None:
                z_ids = torch.argmax(z_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=z_logits)
                z_ids = dist.sample()

            # --------------------------------------------------
            # Digit EM
            # --------------------------------------------------
            pred_digits = torch.argmax(digit_logits, dim=-1)
            em = (pred_digits == digit_labels).all(dim=1)

            correct_rows += int(em.sum().item())
            total_rows += B

            # --------------------------------------------------
            # Row-level stats
            # --------------------------------------------------
            for b in range(B):
                K = int(z_mask[b].sum().item())
                if K <= 0:
                    continue

                total_rows_by_k[K] += 1
                if em[b].item():
                    correct_rows_by_k[K] += 1

                row_z = z_ids[b, :K].tolist()

                z_counts_global.update(row_z)
                if K == 1:
                    z_counts_k1[row_z[0]] += 1

                unique_ratios_by_k[K].append(len(set(row_z)) / K)

                if K >= 2:
                    repeats = sum(
                        1 for i in range(K - 1)
                        if row_z[i] == row_z[i + 1]
                    )
                    adjacent_repeat_by_k[K].append(repeats / (K - 1))

        # --------------------------------------------------
        # Aggregate metrics
        # --------------------------------------------------
        digit_em = correct_rows / max(1, total_rows)

        z_distribution = np.zeros(V)
        total_z = sum(z_counts_global.values())
        for z, c in z_counts_global.items():
            z_distribution[z] = c / max(1, total_z)

        z_distribution_k1 = np.zeros(V)
        total_k1 = sum(z_counts_k1.values())
        for z, c in z_counts_k1.items():
            z_distribution_k1[z] = c / max(1, total_k1)

        eps = 1e-12
        entropy = -np.sum(z_distribution * np.log(z_distribution + eps))
        effective_vocab_size = float(np.exp(entropy))

        digit_em_by_k = {
            K: correct_rows_by_k[K] / total
            for K, total in total_rows_by_k.items()
            if total > 0
        }

        unique_ratio_by_k = {
            K: float(np.mean(v)) for K, v in unique_ratios_by_k.items()
        }

        adjacent_repeat_rate_by_k = {
            K: float(np.mean(v)) for K, v in adjacent_repeat_by_k.items()
        }

        all_results[mode] = {
            "digit_em": digit_em,
            "digit_em_by_k": digit_em_by_k,
            "z_distribution": z_distribution,
            "z_distribution_k1": z_distribution_k1,
            "effective_vocab_size": effective_vocab_size,
            "unique_ratio_by_k": unique_ratio_by_k,
            "adjacent_repeat_rate_by_k": adjacent_repeat_rate_by_k,
        }

    return all_results

