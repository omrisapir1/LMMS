# phase3/utils.py
#
# Shared utilities for Phase-3 training & evaluation
#

from __future__ import annotations
import torch
from typing import Optional


def apply_pad_id_safely(
    *,
    input_ids: torch.Tensor,          # [B,T]
    attention_mask: torch.Tensor,     # [B,T]
    pad_token_id: Optional[int],
) -> torch.Tensor:
    """
    Replace pad regions (attention_mask == 0) with tokenizer.pad_token_id if available.
    """
    if pad_token_id is None:
        return input_ids
    return input_ids.masked_fill(attention_mask == 0, int(pad_token_id))


def mask_sft_to_start_at_first_z(
    *,
    input_ids: torch.Tensor,          # [B,T]
    attention_mask: torch.Tensor,     # [B,T]
    z_token_ids: list[int],
) -> torch.Tensor:
    """
    Returns a new attention_mask' where tokens BEFORE the first Z token are masked out.
    """
    z_set = set(int(x) for x in z_token_ids)
    B, T = input_ids.shape
    out = attention_mask.clone()

    for b in range(B):
        first_z = None
        for t in range(T):
            if out[b, t].item() == 0:
                break
            if int(input_ids[b, t].item()) in z_set:
                first_z = t
                break

        if first_z is None:
            raise RuntimeError(
                "SFT mask: no Z token found in a sample (dataset contract violation)"
            )

        if first_z > 0:
            out[b, :first_z] = 0

    return out
