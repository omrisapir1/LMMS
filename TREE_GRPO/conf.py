from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

MAX_TOKENS = 512
BATCH_SIZE = 16


@dataclass
class ModelConfig:
    init_ckpt: str = "omrisap/LMMS_RSFT"
    answer_token: str = "<ANSWER>"
    finalize_token: str = "<FINALIZE>"
    retry_token: str = "<RETRY>"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "omrisap/numina_openmath"
    train_split: str = "train"
    question_field: str = "problem"
    answer_digits_field: str = "answer_digits"
    answer_field: str = "final_answer"
    rsft_trained_questions_path: str = "PPO/rsft_trained_questions.json"


@dataclass
class TreeConfig:
    # v1 is intentionally shallow:
    #   root siblings (fixed to 4 by collector)
    #   one retry expansion layer (children of root-retry nodes only)
    root_siblings: int = 4
    max_retry_parents_from_root: int = 2
    retry_children_per_parent: int = 2
    max_retry_depth: int = 1
    c_retry: float = 0.05
    gamma: float = 0.95
    c_branch: float = 0.0
    advantage_clip: float = 3.0


@dataclass
class RolloutConfig:
    backend: str = "vllm"  # "vllm" | "hf"
    max_new_tokens: int = MAX_TOKENS
    temperature: float = 1.2
    top_p: float = 0.95
    verify_temperature: float = 1.2
    verify_p: float = 0.95
    min_p: float = 0.03
    repetition_penalty: float = 1.1
    digit_temperature: float = 1.0
    digit_top_p: float = 0.90
    digit_greedy: bool = False
    vllm_enabled: bool = True
    vllm_sync_every: int = 2
    vllm_batch_size: int = BATCH_SIZE
    vllm_tp_size: int = 1
    gpu_memory_utilization: float = 0.95
    vllm_seed: Optional[int] = None
    vllm_tmp_ckpt_dir: str = ""
    vllm_engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    torch_device: str = "cuda:0"
    ref_model_device: str = "cuda:0"
    vllm_cuda_visible_devices: str = "1"
    tree_prompts_per_update: int = 1


@dataclass
class PPOConfig:
    clip_range: float = 0.2
    c_v: float = 0.25
    c_ent: float = 0.002
    kl_coef: float = 0.01
    ppo_epochs: int = 1
    minibatch_size: int = 16
    max_grad_norm: float = 1.0
    update_ref_model_each_steps: int = 100
    value_warmup_steps: int = 0
    value_warmup_lr: float = 1e-4


@dataclass
class TrainConfig:
    lr: float = 5e-6
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    updates: int = 10000
    grad_accum_steps: int = 16
    seed: int = 42
    output_dir: str = "./runs/tree_grpo_v1"
    save_every: int = 25


@dataclass
class RuntimeConfig:
    use_bf16: bool = True
    use_length_bucketing: bool = True
    length_bucket_width: int = 64
    compile_update_stats: bool = False


@dataclass
class LoggingConfig:
    log_action_tokens: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_SET_ALLOWED_PREFIXES: Sequence[str] = (
    "model.",
    "data.",
    "rollout.",
    "tree.",
    "ppo.",
    "train.",
    "runtime.",
    "logging.",
)
