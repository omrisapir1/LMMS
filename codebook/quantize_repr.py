from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch

QuantizeMode = Literal["raw", "delta"]


def normalize_quantize_mode(mode: str) -> QuantizeMode:
    normalized = str(mode).strip().lower()
    if normalized not in ("raw", "delta"):
        raise ValueError(f"quantize_mode must be 'raw' or 'delta', got {mode!r}")
    return normalized  # type: ignore[return-value]


def transform_sequence_np(latents: np.ndarray, *, mode: QuantizeMode) -> np.ndarray:
    if latents.ndim != 2:
        raise ValueError(f"Expected rank-2 latent array, got shape={tuple(latents.shape)}")
    x = np.ascontiguousarray(latents, dtype=np.float32)
    if mode == "raw" or x.shape[0] == 0:
        return x

    out = np.empty_like(x)
    out[0] = x[0]
    if x.shape[0] > 1:
        out[1:] = x[1:] - x[:-1]
    return out


def transform_flat_torch(
    latents: torch.Tensor,
    *,
    sequence_lengths: Sequence[int],
    mode: QuantizeMode,
) -> torch.Tensor:
    if latents.ndim != 2:
        raise ValueError(f"Expected rank-2 latent tensor, got shape={tuple(latents.shape)}")
    if mode == "raw" or latents.shape[0] == 0:
        return latents

    out = torch.empty_like(latents, dtype=torch.float32)
    start = 0
    for k_raw in sequence_lengths:
        k = int(k_raw)
        if k < 0:
            raise ValueError(f"sequence length must be >= 0, got {k}")
        if k == 0:
            continue
        end = start + k
        out[start] = latents[start]
        if k > 1:
            out[start + 1 : end] = latents[start + 1 : end] - latents[start : end - 1]
        start = end
    if start != int(latents.shape[0]):
        raise ValueError(
            f"Sequence boundary mismatch: consumed={start} vs latents={int(latents.shape[0])}"
        )
    return out
