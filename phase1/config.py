from dataclasses import dataclass

@dataclass
class Phase1Config:
    # Training
    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 4

    # Curriculum / stages
    max_thoughts: int = 8
    max_length: int = 2048
    eval_interval_batches: int = 1000

    min_delta: float = 0.01  # 1% improvement threshold
    stage_patience: tuple = (3, 3, 3, 3, 3, 3, 3, 5)
    max_steps_first_stage: int = 2
    permutation_loss_interval_batches: int = 8

    # Downsampling
    target_p0: float = 0.25
    keep_prob_path: str = "probs.json"

    # Dataset (Hugging Face)
    dataset_name: str = "omrisap/GSM8k-Aug_qwen_62K_CoTsplitted"
    dataset_train_split: str = "train"
    dataset_eval_split: str = "eval"

    # Model
    phase0_repo: str = "omrisap/LMMS_phase0"
    torch_dtype: str = "bfloat16"

    # Logging
    log_dir: str = "runs/phase1"
    logg_loss_interval_batches: int = 10
