from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .dataset import iter_valid_latent_rows
    from .model import Codebook
    from .quantize_repr import normalize_quantize_mode, transform_sequence_np
except ImportError:
    from dataset import iter_valid_latent_rows  # type: ignore
    from model import Codebook  # type: ignore
    from quantize_repr import normalize_quantize_mode, transform_sequence_np  # type: ignore


def random_init_codebook(model: Codebook, *, seed: int = 42) -> Dict[str, int]:
    device = model.embeddings.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    init = torch.randn(model.vocab_size, model.dim, generator=gen, device=device, dtype=torch.float32)
    init = F.normalize(init, p=2, dim=-1, eps=1e-12)
    model.initialize_embeddings(init)
    return {"sampled_vectors": 0}


def _subsample_sequence_vectors(
    vecs: np.ndarray, max_vectors: int, rng: np.random.Generator
) -> np.ndarray:
    if max_vectors <= 0:
        raise ValueError("max_vectors must be > 0")
    if vecs.shape[0] <= max_vectors:
        return np.ascontiguousarray(vecs, dtype=np.float32)
    indices = rng.choice(vecs.shape[0], size=max_vectors, replace=False)
    sampled = vecs[indices]
    return np.ascontiguousarray(sampled, dtype=np.float32)


def init_codebook_with_kmeans(
    *,
    model: Codebook,
    input_dir: str,
    dim: int,
    sample_size: int = 500_000,
    read_batch_size: int = 256,
    fit_batch_size: int = 8_192,
    seed: int = 42,
    kmeans_max_vectors_per_sequence: int = 32,
    quantize_mode: str = "delta",
) -> Dict[str, int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if kmeans_max_vectors_per_sequence <= 0:
        raise ValueError("kmeans_max_vectors_per_sequence must be > 0")
    mode = normalize_quantize_mode(quantize_mode)

    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "scikit-learn is required for k-means initialization. "
            "Install scikit-learn or pass --no_kmeans."
        ) from exc

    kmeans = MiniBatchKMeans(
        n_clusters=model.vocab_size,
        batch_size=max(model.vocab_size * 2, int(fit_batch_size)),
        random_state=seed,
        n_init="auto",
        reassignment_ratio=0.01,
    )

    pending: list[np.ndarray] = []
    pending_count = 0
    seen_vectors = 0
    fitted_once = False
    rng = np.random.default_rng(seed)

    def flush_pending(force: bool = False) -> None:
        nonlocal pending_count, fitted_once
        if pending_count == 0:
            return
        if not force and pending_count < fit_batch_size:
            return
        if not fitted_once and pending_count < model.vocab_size:
            return
        batch = np.concatenate(pending, axis=0).astype(np.float32, copy=False)
        kmeans.partial_fit(batch)
        fitted_once = True
        pending.clear()
        pending_count = 0

    for row in iter_valid_latent_rows(input_dir, dim=dim, read_batch_size=read_batch_size):
        if seen_vectors >= sample_size:
            break
        row_vecs = transform_sequence_np(row.latent_vectors, mode=mode)
        vecs = _subsample_sequence_vectors(
            row_vecs, kmeans_max_vectors_per_sequence, rng
        )
        remaining = sample_size - seen_vectors
        if vecs.shape[0] > remaining:
            vecs = vecs[:remaining]
        if vecs.shape[0] == 0:
            continue
        # Align k-means geometry with cosine assignment by clustering on unit-norm latents.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-12)
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        pending.append(vecs)
        pending_count += int(vecs.shape[0])
        seen_vectors += int(vecs.shape[0])
        flush_pending(force=False)

    flush_pending(force=True)

    if not fitted_once:
        raise RuntimeError(
            f"Not enough valid vectors for k-means init: saw {seen_vectors}, need >= {model.vocab_size}"
        )

    centers = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32))
    centers = F.normalize(centers, p=2, dim=-1, eps=1e-12).to(model.embeddings.device)
    model.initialize_embeddings(centers)
    return {"sampled_vectors": int(seen_vectors)}
