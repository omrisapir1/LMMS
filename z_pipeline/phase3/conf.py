# phase3/conf.py
#
# Configuration for Phase-3 training
#
# Phase-3 goals:
# - Jointly optimize digit accuracy and Z-language behavior
# - Teacher-forced training only (no generation during training)
# - SFT + Answer loss + KL diversity loss
#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

@dataclass
class Phase3DataConfig:
    batch_size: int = 8
    max_length: int = 2048

    # Rebalancing (train only)
    rebalance_train: bool = True

    # Target K distribution (mirrors Phase-2 buckets)
    target_dist: Dict[str, float] = field(default_factory=lambda: {
        "K1": 0.075,
        "K2": 0.10,
        "K3": 0.125,
        "K4_7": 0.300,
        "K8_12": 0.20,
        "K13_20": 0.20,
    })


# ------------------------------------------------------------
# Losses
# ------------------------------------------------------------

@dataclass
class Phase3LossConfig:
    # Main weights
    print_every: int = 20
    lambda_answer: float = 1.0
    lambda_sft: float = 0.1
    lambda_kl: float = 0.1

    # KL on digit distributions temperature
    digit_temperature: float = 1.0
    keep_prob: Optional[Dict[int, float]] = (0.02, 0.05, 0.1, 0.5, 1)

    # Probability of choosing "reverse" vs "random"
    # based on number of Z tokens in the sequence
    #
    # reverse_prob[K] = P(reverse), 1 - P(reverse) = P(random)
    reverse_prob_by_k: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0,
        2: 0.10,
        3: 0.15,
        4: 0.20,
        5: 0.30,
        6: 0.40,
        7: 0.50,
        8: 0.60,
        9: 0.65,
        10: 0.70,
        11: 0.75,
        12: 0.80,
        13: 0.85,
        14: 0.85,
        15: 0.9,
        16: 0.9,
        17: 0.9,
        18: 0.9,
        19: 0.9,
        20: 0.9,
    })


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

@dataclass
class Phase3OptimConfig:
    lr: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    # Gradient clipping
    max_grad_norm: Optional[float] = 1.0


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

@dataclass
class Phase3TrainConfig:
    num_epochs: int = 30
    eval_every_steps: int = 5000
    batch_size: int = 8
    loss_batch_size: int = 2
    gradient_accumulation_steps: int = 1



# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

@dataclass
class Phase3EvalConfig:
    batch_size: int = 8

    # Generation control
    max_generation_tokens: int = 32
    sampling_temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


# ------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------

@dataclass
class Phase3CheckpointConfig:
    save_dir: str = "./phase3_ckpts"
    save_every_steps: int = 5000
    save_best: bool = True
    save_at_start: bool = True


# ------------------------------------------------------------
# Top-level config
# ------------------------------------------------------------

@dataclass
class Phase3Config:
    seed: int = 42
    phase2_repo_id: str = "omrisap/phaseZ"
    phase3_dataset_repo_id: str = "omrisap/phase3_train_dataset"

    data: Phase3DataConfig = field(default_factory=Phase3DataConfig)
    loss: Phase3LossConfig = field(default_factory=Phase3LossConfig)
    optim: Phase3OptimConfig = field(default_factory=Phase3OptimConfig)
    train: Phase3TrainConfig = field(default_factory=Phase3TrainConfig)
    eval: Phase3EvalConfig = field(default_factory=Phase3EvalConfig)
    ckpt: Phase3CheckpointConfig = field(default_factory=Phase3CheckpointConfig)

    def validate(self) -> "Phase3Config":
        # -----------------------------
        # Data
        # -----------------------------
        s = sum(self.data.target_dist.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"data.target_dist must sum to 1.0, got {s}")

        # -----------------------------
        # Loss
        # -----------------------------
        if self.loss.lambda_answer < 0 or self.loss.lambda_sft < 0 or self.loss.lambda_kl < 0:
            raise ValueError("Loss lambdas must be non-negative")

        if self.loss.digit_temperature <= 0:
            raise ValueError("digit_temperature must be > 0")

        # -----------------------------
        # Optim
        # -----------------------------
        if self.optim.lr <= 0:
            raise ValueError("optim.lr must be > 0")

        # -----------------------------
        # Training
        # -----------------------------
        if self.train.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")

        # -----------------------------
        # Eval
        # -----------------------------
        if self.eval.max_generation_tokens <= 0:
            raise ValueError("max_generation_tokens must be > 0")

        return self


__all__ = [
    "Phase3Config",
    "Phase3DataConfig",
    "Phase3LossConfig",
    "Phase3OptimConfig",
    "Phase3TrainConfig",
    "Phase3EvalConfig",
    "Phase3CheckpointConfig",
]
