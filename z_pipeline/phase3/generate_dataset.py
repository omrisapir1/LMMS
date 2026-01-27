# phase3/generate_dataset.py
#
# Phase-3 dataset generation:
# - Uses trained Phase-2 model to convert latent states -> Z tokens
# - Materializes explicit Z-token sequences
# - Saves dataset compatible with phase3/dataset.py
#
from __future__ import annotations

from typing import Dict, List, Iterable

import torch
import numpy as np
from datasets import Dataset, DatasetDict

from z_pipeline.phase2.dataset import Phase2Dataset, phase2_collate_fn
from z_pipeline.phase2.model import Phase2ZModel


# ------------------------------------------------------------
# K bucket logic (EXACTLY aligned with Phase-2 / Phase-3)
# ------------------------------------------------------------

def k_to_bucket(K: int) -> str:
    if K == 1:
        return "K1"
    if K == 2:
        return "K2"
    if K == 3:
        return "K3"
    if 4 <= K <= 7:
        return "K4_7"
    if 8 <= K <= 12:
        return "K8_12"
    if 13 <= K <= 20:
        return "K13_20"
    raise ValueError(f"Unsupported num_latents K={K}")


# ------------------------------------------------------------
# Core generation
# ------------------------------------------------------------

@torch.no_grad()
def generate_phase3_dataset(
    *,
    phase2_ckpt: Dict,
    dataset_name: str,
    split: str,
    batch_size: int,
    z_mode: str = "hard_argmax",      # "hard_argmax" | "hard_sample"
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> Dataset:
    """
    Generates a Phase-3 dataset split.

    Returns a HuggingFace Dataset with fields:
      input_ids, attention_mask, digit_labels, num_latents, K_bucket
    """

    model: Phase2ZModel = phase2_ckpt["model"]
    tokenizer = phase2_ckpt["tokenizer"]

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # --------------------------------------------------
    # Build Phase-2 dataset (source of latent_states)
    # --------------------------------------------------
    latent_token_id = model.latent_token_id
    answer_token_id = model.answer_token_id

    ds = Phase2Dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        k_max=max(len(model.z_token_ids), 20),  # safe upper bound
        latent_token_id=latent_token_id,
        answer_token_id=answer_token_id,
        rebalance_train=False,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=phase2_collate_fn,
    )

    rows: List[Dict] = []

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        latent_states = batch["latent_states"].to(device)
        z_mask = batch["z_mask"].to(device)
        digit_labels = batch["digit_labels"]

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_states=latent_states,
            z_mask=z_mask,
            z_mode=z_mode,
            temperature=temperature,
            return_z_probs=False,
        )

        z_ids = out["z_ids"]  # [B, Kmax]
        B, Kmax = z_ids.shape

        for b in range(B):
            K = int(z_mask[b].sum().item())
            if K <= 0:
                continue

            # --------------------------------------------------
            # Rebuild input_ids: prompt + Zs + <ANSWER>
            # --------------------------------------------------
            original_ids = input_ids[b].tolist()
            answer_pos = original_ids.index(answer_token_id)

            prefix = original_ids[:answer_pos]
            z_seq = z_ids[b, :K].tolist()
            new_ids = prefix + [model.z_token_ids[z] for z in z_seq] + [answer_token_id]

            attn = [1] * len(new_ids)

            rows.append({
                "input_ids": new_ids,
                "attention_mask": attn,
                "digit_labels": digit_labels[b].tolist(),
                "num_latents": K,
                "K_bucket": k_to_bucket(K),
            })

    return Dataset.from_list(rows)


# ------------------------------------------------------------
# Convenience wrapper (train + eval)
# ------------------------------------------------------------

def generate_phase3_dataset_dict(
    *,
    phase2_ckpt: Dict,
    dataset_name: str,
    batch_size: int,
    z_mode: str = "hard_argmax",
    temperature: float = 1.0,
) -> DatasetDict:

    train_ds = generate_phase3_dataset(
        phase2_ckpt=phase2_ckpt,
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        z_mode=z_mode,
        temperature=temperature,
    )

    eval_ds = generate_phase3_dataset(
        phase2_ckpt=phase2_ckpt,
        dataset_name=dataset_name,
        split="eval",
        batch_size=batch_size,
        z_mode=z_mode,
        temperature=temperature,
    )

    return DatasetDict({
        "train": train_ds,
        "eval": eval_ds,
    })


__all__ = [
    "generate_phase3_dataset",
    "generate_phase3_dataset_dict",
]
