# phase3/loss_eval.py
#
# Phase-3 loss-only evaluation on a Dataset / DataLoader
#
from __future__ import annotations

from typing import Dict
import torch
from torch.utils.data import DataLoader

from .loss import Phase3Loss
from .train import (
    _apply_pad_id_safely,
    _mask_sft_to_start_at_first_z,
)


@torch.no_grad()
def evaluate_phase3_losses(
    *,
    model,
    data_loader: DataLoader,
    loss_fn: Phase3Loss,
    z_token_ids: list[int],
    pad_token_id: int | None,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    """
    Evaluate Phase-3 losses (SFT / Answer / KL) on a dataset.

    Returns averaged losses.
    """

    model.eval()

    total_loss = 0.0
    total_sft = 0.0
    total_answer = 0.0
    total_kl = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        digit_labels = batch["digit_labels"].to(device)

        input_ids = _apply_pad_id_safely(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
        )

        sft_attention = _mask_sft_to_start_at_first_z(
            input_ids=input_ids,
            attention_mask=attention_mask,
            z_token_ids=z_token_ids,
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            return_full_logits=False,
        )

        losses = loss_fn.compute(
            model=model,
            logits=out.logits,
            input_ids=input_ids,
            attention_mask=sft_attention,
            digit_logits=out.digit_logits,
            digit_labels=digit_labels,
        )

        B = input_ids.size(0)

        total_loss += losses["loss_total"].item() * B
        total_sft += losses["loss_sft"].item() * B
        total_answer += losses["loss_answer"].item() * B
        total_kl += losses["loss_kl"].item() * B
        total_samples += B

    model.train()

    return {
        "loss_total": total_loss / total_samples,
        "loss_sft": total_sft / total_samples,
        "loss_answer": total_answer / total_samples,
        "loss_kl": total_kl / total_samples,
        "num_samples": total_samples,
    }
