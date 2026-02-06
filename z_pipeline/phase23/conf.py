from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List


# ─────────────────────────────────────────────
# Model configuration
# ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    # Path to Phase-1 checkpoint directory
    phase1_dir: str = "omrisap/LMMS_phase1"

    # Z vocabulary size
    v_z: int = 512

    # Temperatures
    # tau_student: controls exploration in student Z mixtures used for embedding injection
    tau_student: float = 1.0
    # tau_teacher: controls self-distillation target sharpness
    tau_teacher: float = 2.0

    # Whether to compute self-teacher distributions for KL
    use_self_teacher: bool = True

    # Token strings (do NOT hardcode elsewhere)
    z_prefix: str = "Z_"
    latent_token: str = "<|latent|>"
    answer_token: str = "<ANSWER>"


# ─────────────────────────────────────────────
# Data configuration
# ─────────────────────────────────────────────

@dataclass
class DataConfig:
    # HuggingFace dataset name (optional)
    dataset_name: Optional[str] = "omrisap/phaseZ"
    train_split: str = "train"
    eval_split: str = "eval"

    # Optional local dataset path (jsonl / parquet / HF-compatible)
    data_path: Optional[str] = None

    # Max sequence length (None = no truncation)
    max_length: Optional[int] = None

    batch_size: int = 8

    # Whether to rebalance training samples by K buckets
    rebalance_train: bool = True

    # Maximum number of latent steps
    k_max: int = 20

    # Target K distribution for rebalancing
    target_k_dist: Dict[str, float] = field(default_factory=lambda: {
        "K1": 0.075,
        "K2": 0.10,
        "K3": 0.125,
        "K4_7": 0.300,
        "K8_12": 0.20,
        "K13_20": 0.20,
    })


# ─────────────────────────────────────────────
# Loss configuration
# ─────────────────────────────────────────────

@dataclass
class LossConfig:
    # Main loss weights
    lambda_ans: float = 1.0
    # Self-distillation on Z distributions (not external supervision)
    lambda_softz: float = 1.0
    lambda_cf: float = 1.0

    # Optional usage shaping (default OFF)
    lambda_usage: float = 0.0

    # Temperature for digit probability distributions
    digit_temperature: float = 1.0
    # Per-digit keep probability for zero labels (MSB -> LSB). Matches Phase-2/3 logic.
    keep_prob: Optional[List[float]] = field(default_factory=lambda: [0.02, 0.05, 0.1, 0.5, 1.0])

    # Future hook (e.g., PPO): freeze self-distill after N steps
    freeze_softz_after_steps: Optional[int] = None

    # Counterfactual perturbation schedule (by K)
    counterfactual_schedule: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0,
        2: 0.10, 3: 0.15, 4: 0.20, 5: 0.30,
        6: 0.40, 7: 0.50, 8: 0.60, 9: 0.65,
        10: 0.70, 11: 0.75, 12: 0.80,
        13: 0.85, 14: 0.85,
        15: 0.90, 16: 0.90, 17: 0.90,
        18: 0.90, 19: 0.90, 20: 0.90,
    })


# ─────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────

@dataclass
class TrainConfig:
    lr: float = 3e-5
    weight_decay: float = 0.00

    # Total training steps
    steps: int = 1000

    # Gradient accumulation
    grad_accum: int = 1

    # Logging / checkpoint cadence
    print_every: int = 5
    eval_every: int = 0
    save_every: int = 500

    seed: int = 42
    output_dir: str = "./runs/phase23"


# ─────────────────────────────────────────────
# Root config
# ─────────────────────────────────────────────

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
