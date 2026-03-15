from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodebookConfig:
    input_dir: str = "omrisap/nvidia-math-vectorized"
    output_dir: str = "runs/codebook"
    vocab_size: int = 2048
    dim: int = 1024
    batch_size: int = 128  # max sequences per step
    max_vectors_per_batch: int = 8_192
    epochs: int = 3
    ema_decay: float = 0.99
    beta: float = 0.25
    lambda_kl: float = 0.000
    adjacent_overlap_tau: float = 0.1
    lambda_adjacent_overlap: float = 0.3
    kmeans_sample_size: int = 500_000
    kmeans_max_vectors_per_sequence: int = 128
    no_kmeans: bool = False
    seed: int = 42

    # Streaming / runtime knobs
    read_batch_size: int = 256
    kmeans_fit_batch_size: int = 8_192
    export_quantize_chunk_size: int = 16_384
    log_interval: int = 5

    # Fixed constants requested by the spec
    usage_laplace_alpha: float = 1.0
    usage_ema_eta: float = 0.99
    eps: float = 1e-5
