from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFTConfig:
    # Model / tokenizer inputs
    base_model_or_checkpoint: str = "omrisap/phase1_improved"
    train_dataset_name: str = "omrisap/SFT_512_40K"
    train_dataset_split: str = "train"
    eval_dataset_name: str = "omrisap/SFT_eval"
    eval_dataset_split: str = "eval"

    # Discrete Z vocab
    vocab_size: int = 512

    # Training
    seed: int = 42
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    max_steps: int = 15_000
    warmup_steps: int = 500
    max_length: int = 2048

    # Objective weights
    z_label_smoothing: float = 0.00
    w_z: float = 0.1
    w_answer: float = 0.05
    w_digits: float = 1.0

    # Evaluation / generation
    eval_interval_steps: int = 500
    pass_at_n: int = 16
    k_max: int = 128
    temperature: float = 1.0
    top_p: float = 0.95

    # Logging / checkpointing
    run_root: str = "runs/sft_z"
    log_interval_steps: int = 20
    save_interval_steps: int = 200
    save_every_steps: int = 2000
    keep_last_k: int = 3
    save_best: bool = True
    save_ppo_init: bool = False

    # DataLoader
    dataloader_num_workers: int = 2
    eval_dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
