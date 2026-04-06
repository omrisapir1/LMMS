from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass
class ModelConfig:
    init_ckpt: str = "omrisap/nemotron-7B-12K"
    answer_token: str = "<ANSWER>"
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


@dataclass
class TrainConfig:
    train_batch_size: int = 2
    lr: float = 4e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_steps: int = 10000
    max_grad_norm: float = 1.0
    max_length: int = 1024
    seed: int = 42
    use_bf16: bool = True
    optimizer_8bit: bool = True


@dataclass
class LossConfig:
    w_z_ans: float = 1.0
    w_digits: float = 1.0
    use_prompt_weighting: bool = True


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
