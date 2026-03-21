from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFTConfig:
    # Model / tokenizer inputs
    base_model_or_checkpoint: str = "Qwen/Qwen3.5-9B-Base"
    train_dataset_name: str = "omrisap/nvidia_math_64_47K"
    train_dataset_split: str = "train"
    eval_dataset_name: str = "omrisap/SFT_eval"
    eval_dataset_split: str = "eval"

    # Discrete Z vocab
    vocab_size: int = 64

    # Training
    seed: int = 42
    batch_size: int = 32
    eval_batch_size: int = 1024
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    optimizer_name: str = "adamw_8bit"
    trainable_layer_spec: str = "all"
    gradient_accumulation_steps: int = 8
    max_steps: int = 60_000
    warmup_steps: int = 0
    max_length: int = 2048
    torch_device: str = "cuda:0"
    debug_prefix_repeat: int = 5000
    debug_prefix_text: str = " DEBUG_PREFIX"

    # Objective weights
    z_label_smoothing: float = 0.05
    w_z: float = 1
    w_start_answer: float = 0.025
    w_start_digits: float = 0.05
    w_end_answer: float = 0.25
    w_end_digits: float = 0.5
    start_weights_steps: int = 1000
    goes_up_weights_steps: int = 1000
    keep_prob: tuple[float, ...] = (0.2, 0.3, 0.45, 0.75, 1.0)

    # Counterfactual dependence regularizer
    cf_enabled: bool = True
    cf_every_n_steps: int = 2
    cf_prob_tuple: tuple[float, float, float] = (0.5, 0.25, 0.25)  # (truncate, reverse, random)
    cf_lambda: float = 1.0
    cf_kl_margin: float = 0.5
    cf_eps: float = 1e-8
    cf_min_z_len: int = 2
    cf_trunc_range: tuple[float, float] = (0.5, 1.0)

    # Evaluation / generation
    eval_interval_steps: int = 500
    pass_at_n: int = 16
    k_max: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    vllm_cuda_visible_devices: str = "1"

    # Logging / checkpointing
    run_root: str = "runs/sft_z"
    log_interval_steps: int = 1
    save_interval_steps: int = 2000
    save_every_steps: int = 2000
    keep_last_k: int = 3
    save_best: bool = True
    save_ppo_init: bool = False

    # DataLoader
    dataloader_num_workers: int = 2
    eval_dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
