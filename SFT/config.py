from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class PhaseConfig:
    z_ratio: float
    batch_size: int
    gradient_accumulation_steps: int
    max_tokens: int
    epochs: float
    cf_loss: bool
    min_z_tokens: int = 1


@dataclass
class SFTConfig:
    # Model / tokenizer inputs
    base_model_or_checkpoint: str = "Qwen/Qwen3.5-9B-Base"
    train_dataset_name: str = "omrisap/nvidia_math_512_47K"
    train_dataset_split: str = "train"

    # Discrete Z vocab
    vocab_size: int = 512

    # Training
    seed: int = 42
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    optimizer_name: str = "adamw_8bit"  # one of: adamw_8bit, adamw, adamw_fused
    model_dtype: str = "bf16"
    attn_implementation: str = "flash_attention_4"
    max_length: int = 16000
    torch_device: str = "cuda:0"
    max_steps: Optional[int] = None

    # Objective weights
    z_label_smoothing: float = 0.005
    alpha_z: float = 1.0
    alpha_answer: float = 0.5
    alpha_digits: float = 1.0
    keep_prob: tuple[float, ...] = (0.2, 0.3, 0.45, 0.75, 1.0)
    debug_loss_checks: bool = False

    # Counterfactual dependence regularizer
    cf_enabled: bool = True
    cf_every_n_steps: int = 2
    cf_prob_tuple: tuple[float, float, float] = (0.5, 0.25, 0.25)  # (truncate, reverse, random)
    cf_lambda: float = 1.0
    cf_kl_margin: float = 0.5
    cf_eps: float = 1e-8
    cf_min_z_len: int = 2
    cf_trunc_range: tuple[float, float] = (0.5, 1.0)

    # Curriculum phases
    phases: Sequence[PhaseConfig] = (
    PhaseConfig(
        z_ratio=0.0,
        min_z_tokens=1,
        batch_size=8,
        gradient_accumulation_steps=8,
        max_tokens=3000,
        epochs=0.25,
        cf_loss=False,
    ),
    PhaseConfig(
        z_ratio=0.1,
        min_z_tokens=1,
        batch_size=8,
        gradient_accumulation_steps=8,
        max_tokens=3000,
        epochs=0.5,
        cf_loss=False,
    ),
    PhaseConfig(
        z_ratio=0.25,
        min_z_tokens=1,
        batch_size=8,
        gradient_accumulation_steps=8,
        max_tokens=4000,
        epochs=1.5,
        cf_loss=True,
    ),
    PhaseConfig(
        z_ratio=0.5,
        min_z_tokens=1,
        batch_size=8,
        gradient_accumulation_steps=8,
        max_tokens=5000,
        epochs=0.5,
        cf_loss=True,
    ),
    PhaseConfig(
        z_ratio=0.75,
        min_z_tokens=1,
        batch_size=8,
        gradient_accumulation_steps=8,
        max_tokens=7000,
        epochs=1,
        cf_loss=True,
    ),
    PhaseConfig(
        z_ratio=0.9,
        min_z_tokens=1,
        batch_size=8,
        gradient_accumulation_steps=8,
        max_tokens=11000,
        epochs=1,
        cf_loss=True,
    ),
    PhaseConfig(
        z_ratio=1.0,
        min_z_tokens=1,
        batch_size=32,
        gradient_accumulation_steps=2,
        max_tokens=9999999999,
        epochs=2.0,
        cf_loss=True,
    ),
)

    # Logging / checkpointing / resume
    run_root: str = "runs/sft_curriculum"
    run_name: Optional[str] = None
    log_interval_steps: int = 1
    save_every_epoch: bool = True
    save_phase_end: bool = True
    resume_from: Optional[str] = "runs/sft_curriculum/20260323_123730__V512__lr2e-05/last"

    # DataLoader
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
