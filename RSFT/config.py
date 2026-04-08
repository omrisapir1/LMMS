from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass
class ModelConfig:
    init_ckpt: str = "omrisap/nemotron-7B-12K"
    answer_token: str = "<ANSWER>"
    finalize_token: str = "<FINALIZE>"
    retry_token: str = "<RETRY>"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "omrisap/numina_openmath"
    train_split: str = "train"
    eval_split: str = "eval"
    question_field: str = "question"
    answer_digits_field: str = "answer_digits"
    answer_field: str = "final_answer"


@dataclass
class RolloutConfig:
    backend: str = "vllm"  # "vllm" | "hf"
    vllm_batch_size: int = 64
    rollouts_per_prompt: int = 8
    max_rounds: int = 30
    max_new_tokens: int = 512
    temperature: float = 1.2
    top_p: float = 0.95
    min_p: float = 0.03
    repetition_penalty: float = 1.05
    digit_greedy: bool = False
    vllm_tp_size: int = 1
    gpu_memory_utilization: float = 0.95
    vllm_seed: Optional[int] = None
    vllm_engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    torch_device: str = "cuda:0"
    vllm_cuda_visible_devices: str = "1"
    sync_every_n_steps: int = 2

    def __post_init__(self) -> None:
        if int(self.max_rounds) < 1:
            raise ValueError("rollout.max_rounds must be >= 1")


@dataclass
class TrainConfig:
    train_batch_size: int = 2
    lr: float = 3e-5
    warmup_lr: Optional[float] = 1e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_steps: int = 10000
    max_grad_norm: float = 1.0
    max_length: int = 1024
    seed: int = 42
    use_bf16: bool = True
    optimizer_8bit: bool = True
    warmup_steps: int = 30

    def __post_init__(self) -> None:
        if int(self.warmup_steps) < 0:
            raise ValueError("train.warmup_steps must be >= 0")
        if self.warmup_lr is not None and float(self.warmup_lr) <= 0.0:
            raise ValueError("train.warmup_lr must be > 0 when provided")


@dataclass
class LossConfig:
    w_z_ans: float = 1.0
    w_digits: float = 1.0
    w_verify: float = 2.0

    def __post_init__(self) -> None:
        if float(self.w_z_ans) < 0.0:
            raise ValueError("loss.w_z_ans must be >= 0")
        if float(self.w_digits) < 0.0:
            raise ValueError("loss.w_digits must be >= 0")
        if float(self.w_verify) < 0.0:
            raise ValueError("loss.w_verify must be >= 0")


@dataclass
class EvalConfig:
    eval_every_steps: int = 500
    eval_at_start: bool = False
    vllm_batch_size: int = 256
    pass_at_n: int = 16
    k_max: int = 512
    max_eval_questions: int = 1024


@dataclass
class LoggingConfig:
    output_dir: str = "./runs/rsft"
    log_every: int = 1
    save_every: int = 5000
    keep_last: int = 3


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_SET_ALLOWED_PREFIXES: Sequence[str] = (
    "model.",
    "data.",
    "rollout.",
    "train.",
    "loss.",
    "eval.",
    "logging.",
)
