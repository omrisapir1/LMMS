from dataclasses import dataclass

@dataclass
class Phase1Config:
    # Training
    seed: int = 42
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # Curriculum / stages
    max_thoughts: int = 8
    max_length: int = 2048
    eval_interval_batches: int = 100
    stage_exit_thresholds: tuple = (
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80
    )

    # Downsampling
    target_p0: float = 0.25
    keep_prob_path: str = "data/keep_prob.json"

    # Dataset (Hugging Face)
    dataset_name: str = "omrisap/GSM8k-Aug_qwen_62K_CoTsplitted"
    dataset_train_split: str = "train"
    dataset_eval_split: str = "eval"

    # Model
    phase0_repo: str = "omrisap/LMMS_phase0"
    torch_dtype: str = "bfloat16"

    # Logging
    log_dir: str = "runs/phase1"
