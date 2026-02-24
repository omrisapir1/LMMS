from dataclasses import dataclass

@dataclass
class Phase1Config:
    # Training
    seed: int = 42
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 2
    answer_loss_weight: float = 0.5
    permutation_loss_weight: float = 1.0

    # Curriculum / stages
    max_thoughts: int = 8
    max_length: int = 1024
    eval_interval_batches: int = 100

    min_delta: float = 0.01  # 1% improvement threshold
    stage_patience: tuple = (1, 2, 1, 2, 1, 2, 2, 6)
    max_steps_first_stage: int = 40
    permutation_loss_interval_batches: int = 8


    keep_prob: tuple[float, ...] = (0.05, 0.1, 0.15, 0.75, 1.0)

    # Dataset (Hugging Face)
    dataset_name: str = "omrisap/GSM8k-Aug_qwen_62K_CoTsplitted"
    dataset_train_split: str = "train"
    dataset_eval_split: str = "eval"

    # Model
    base_model: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    torch_dtype: str = "bfloat16"

    # Logging
    log_dir: str = "runs/phase1"
    logg_loss_interval_batches: int = 10

    # DataLoader
    dataloader_num_workers: int = 4
    eval_dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
