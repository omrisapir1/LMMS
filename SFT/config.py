from __future__ import annotations

from dataclasses import dataclass
import math


BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 2
DATASET_SIZE = 40_000
START_WEIGHTS_STEPS_FRACTION = 0.15
GOES_UP_WEIGHTS_STEPS_FRACTION = 0.5
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
UPDATES_PER_EPOCH = math.ceil(DATASET_SIZE / EFFECTIVE_BATCH_SIZE)
START_WEIGHTS_STEPS = max(1, int(UPDATES_PER_EPOCH * START_WEIGHTS_STEPS_FRACTION))
GOES_UP_WEIGHTS_STEPS = max(1, int(UPDATES_PER_EPOCH * GOES_UP_WEIGHTS_STEPS_FRACTION))

@dataclass
class SFTConfig:
    # Model / tokenizer inputs
    base_model_or_checkpoint: str = "unsloth/gpt-oss-20b"
    train_dataset_name: str = "omrisap/nvidia_math_64_47K"
    train_dataset_split: str = "train"
    eval_dataset_name: str = "omrisap/SFT_eval"
    eval_dataset_split: str = "eval"

    # Discrete Z vocab
    vocab_size: int = 64

    # Training
    seed: int = 42
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    max_steps: int = 60_000
    max_length: int = 512
    torch_device: str = "cuda:0"

    # GPT-OSS loading
    attn_implementation: str = "eager"
    dequantize_mxfp4: bool = True
    force_bfloat16: bool = True

    # PEFT / LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    lora_target_modules: str = "all-linear"
    lora_enable_moe_target_parameters: bool = True
    lora_moe_param_substrings: tuple[str, ...] = (
        "gate_up_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
        "w1",
        "w2",
        "w3",
    )

    # Objective weights
    z_label_smoothing: float = 0.05
    w_z: float = 1
    w_start_answer: float = 0.025
    w_start_digits: float = 0.05
    w_end_answer: float = 0.25
    w_end_digits: float = 0.5
    start_weights_steps: int = START_WEIGHTS_STEPS
    goes_up_weights_steps: int = GOES_UP_WEIGHTS_STEPS
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
    eval_interval_steps: int = 500000
    pass_at_n: int = 16
    k_max: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    vllm_cuda_visible_devices: str = "1"

    # Logging / checkpointing
    run_root: str = "runs/sft_z"
    log_interval_steps: int = 1
    save_interval_steps: int = 2000
    save_every_steps: int = 100
    keep_last_k: int = 3
    save_best: bool = True
    save_ppo_init: bool = False

    # DataLoader
    dataloader_num_workers: int = 2
    eval_dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
