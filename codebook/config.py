from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodebookConfig:
    input_dir: str
    output_dir: str
    vocab_size: int = 256
    dim: int = 1536
    batch_size: int = 2048  # sequences per step
    epochs: int = 3
    ema_decay: float = 0.995
    beta: float = 0.25
    lambda_kl: float = 0.001
    kmeans_sample_size: int = 500_000
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
