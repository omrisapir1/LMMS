# phase3/generate_dataset.py
#
# Phase-3 dataset generation:
#   Convert Phase-2 latent placeholders into discrete Z tokens
#
# Input:
#   Phase-2 checkpoint (in-memory)
#   Phase-2 dataset
#
# Output:
#   HF dataset with:
#     - input_ids: Question + Zs + <ANSWER>
#     - attention_mask
#     - num_latents
#     - answer_digits
#
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from z_pipeline.phase2.dataset import Phase2Dataset, phase2_collate_fn


@torch.no_grad()
def generate_phase3_dataset(
    *,
    phase2_ckpt: Dict,
    dataset_name: str,
    split: str,                     # "train" | "eval"
    batch_size: int,
    latent_token_id: int,
    answer_token_id: int,
    k_max: int,
    z_mode: str = "hard_argmax",     # or "hard_sample"
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dataset:
    """
    Generate Phase-3 dataset by replacing <LATENT> placeholders with Z tokens.

    Returns:
        HuggingFace Dataset
    """

    model = phase2_ckpt["model"]
    tokenizer = phase2_ckpt["tokenizer"]
    z_token_ids: List[int] = phase2_ckpt["z_token_ids"]

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # --------------------------------------------------
    # Phase-2 dataset (source)
    # --------------------------------------------------
    ds = Phase2Dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
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

    # --------------------------------------------------
    # Output buffers
    # --------------------------------------------------
    out_input_ids: List[List[int]] = []
    out_attention_mask: List[List[int]] = []
    out_num_latents: List[int] = []
    out_answer_digits: List[List[int]] = []

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        latent_states = batch["latent_states"].to(device)
        z_mask = batch["z_mask"].to(device)
        digit_labels = batch["digit_labels"]

        B, T = input_ids.shape

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_states=latent_states,
            z_mask=z_mask,
            z_mode=z_mode,
            temperature=float(temperature),
            return_z_probs=False,
        )

        z_ids = out["z_ids"]  # [B, Kmax]

        for b in range(B):
            K = int(z_mask[b].sum().item())
            assert K > 0

            # --------------------------------------------------
            # Build new input_ids
            # --------------------------------------------------
            row_ids = input_ids[b].tolist()

            # Find latent placeholder positions
            latent_positions = [
                i for i, t in enumerate(row_ids) if t == latent_token_id
            ]
            assert len(latent_positions) == K

            # Replace <LATENT> with Z token ids
            for k in range(K):
                z_idx = int(z_ids[b, k].item())
                row_ids[latent_positions[k]] = z_token_ids[z_idx]

            # Trim suffix padding (attention_mask==0)
            attn = attention_mask[b].tolist()
            if 0 in attn:
                first_pad = attn.index(0)
                row_ids = row_ids[:first_pad]
                attn = attn[:first_pad]

            out_input_ids.append(row_ids)
            out_attention_mask.append(attn)
            out_num_latents.append(K)
            out_answer_digits.append(digit_labels[b].tolist())

    # --------------------------------------------------
    # Build HF dataset
    # --------------------------------------------------
    return Dataset.from_dict(
        {
            "input_ids": out_input_ids,
            "attention_mask": out_attention_mask,
            "num_latents": out_num_latents,
            "answer_digits": out_answer_digits,
        }
    )
