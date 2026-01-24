# phase2/conf.py
#
# Configuration for Phase-2:
#   Learn Z-token embeddings + Z-selector from digit loss only.
#
# This config is consumed by:
#   - phase2/train.py
#   - phase2/eval.py
#   - pipeline/run_experiment.py
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict


# ─────────────────────────────────────────────────────────────
# Temperature schedule
# ─────────────────────────────────────────────────────────────

@dataclass
class TemperatureSchedule:
    """
    Temperature applied to Z-selector logits before softmax.

    temp(step) is monotonic decreasing.

    type:
      - "linear"
      - "cosine"
      - "exponential"
    """
    type: Literal["linear", "cosine", "exponential"] = "cosine"

    temp_start: float = 2
    temp_end: float = 1

    # Number of optimizer steps over which annealing happens
    anneal_steps: int = 3000

    # After anneal_steps, temperature is held at temp_end
    # for additional stabilization before early stopping
    cooldown_steps: int = 500

    def total_steps(self) -> int:
        return self.anneal_steps + self.cooldown_steps


# ─────────────────────────────────────────────────────────────
# Loss weights
# ─────────────────────────────────────────────────────────────

@dataclass
class Phase2LossConfig:
    """
    Loss = lambda_answer * AnswerLoss
         + lambda_kl     * ZUsageKLLoss
    """
    lambda_answer: float = 1
    lambda_kl: float = 0.05
    lambda_row: float = 0.05
    keep_prob: Optional[Dict[int, float]] = (0.02, 0.05, 0.1, 0.5, 1)



# ─────────────────────────────────────────────────────────────
# Dataset / batching
# ─────────────────────────────────────────────────────────────

@dataclass
class Phase2DataConfig:
    dataset_name: str = "omrisap/phaseZ"

    train_split: str = "train"
    eval_split: str = "eval"

    k_max: int = 20

    batch_size: int = 24
    eval_batch_size: int = 64

    num_workers: int = 4
    pin_memory: bool = True

    # Rebalancing by K_bucket (TRAIN only)
    rebalance_train: bool = True

    # Hard sequence length guard
    max_length: Optional[int] = None  # if None, uses tokenizer.model_max_length


# ─────────────────────────────────────────────────────────────
# Optimization
# ─────────────────────────────────────────────────────────────
@dataclass
class Phase2PretrainConfig:
    enable: bool = True
    steps: int = 2500
    lr: float = 5e-2
    temperature: float = 1.5


@dataclass
class Phase2OptimConfig:
    """
    Optimizer applies ONLY to:
      - Z-selector parameters
      - Z-token embedding rows (masked inside model)
    """
    lr: float = 1e-3
    weight_decay: float = 0.0

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Gradient clipping
    max_grad_norm: Optional[float] = 1.0


# ─────────────────────────────────────────────────────────────
# Evaluation / early stopping
# ─────────────────────────────────────────────────────────────

@dataclass
class Phase2EvalConfig:
    """
    Eval uses ARGMAX Z selection (temperature → 0 behavior).

    Early stopping is allowed ONLY after:
      - temperature annealing finished
      - cooldown_steps completed
    """
    eval_every_steps: int = 500

    # Early stopping criteria (checked on eval metric)
    patience: int = 5
    min_delta: float = 1e-4

    # Safety: do not stop before this many steps
    min_steps: Optional[int] = None  # if None, derived from temp schedule

    # Optional selector health checks
    max_entropy: Optional[float] = None
    min_mean_max_prob: Optional[float] = None

@dataclass
class Phase2ClusterConfig:

    n_iter: int = 25


# ─────────────────────────────────────────────────────────────
# Top-level Phase-2 config
# ─────────────────────────────────────────────────────────────


class Phase2Config:
    """
    Full Phase-2 configuration.
    """
    # Identity / logging
    run_name: str = "phase2_z_learning"
    seed: int = 42
    print_every: int = 20

    # Z vocabulary
    z_vocab_size: int = 1024

    # Token ids (must be provided by caller after tokenizer expansion)
    latent_token_id: int | None = None
    answer_token_id: int | None = None

    # Sub-configs
    data: Phase2DataConfig | None = None
    optim: Phase2OptimConfig = Phase2OptimConfig()
    loss: Phase2LossConfig = Phase2LossConfig()
    temp: TemperatureSchedule = TemperatureSchedule()
    eval: Phase2EvalConfig = Phase2EvalConfig()
    cluster: Phase2ClusterConfig = Phase2ClusterConfig()
    pretrain: Phase2PretrainConfig = Phase2PretrainConfig()

    # Misc
    force_base_eval: bool = True

    def finalize(self):
        """
        Call once after tokenizer/model are loaded.
        Performs derived-field initialization and validation.
        """
        if self.latent_token_id is None:
            raise ValueError("latent_token_id must be set")
        if self.answer_token_id is None:
            raise ValueError("answer_token_id must be set")
        if self.data is None:
            raise ValueError("data config must be provided")

        # Default min_steps = finish anneal + cooldown
        if self.eval.min_steps is None:
            self.eval.min_steps = self.temp.total_steps()

        if self.z_vocab_size <= 0:
            raise ValueError("z_vocab_size must be > 0")
        if self.data.k_max <= 0:
            raise ValueError("k_max must be > 0")

        if self.temp.temp_start <= self.temp.temp_end:
            raise ValueError("temp_start must be > temp_end")

        return self
