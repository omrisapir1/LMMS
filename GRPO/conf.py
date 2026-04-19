from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

MAX_TOKENS = 512
BATCH_SIZE = 16


@dataclass
class ModelConfig:
    init_ckpt: str = "omrisap/nemotron-7B-12K"
    answer_token: str = "<ANSWER>"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "omrisap/numina_openmath"
    train_split: str = "train"
    question_field: str = "problem"
    answer_digits_field: str = "answer_digits"
    answer_field: str = "final_answer"


@dataclass
class RolloutConfig:
    backend: str = "vllm"
    n_z_traces: int = 24
    n_digit_traces: int = 6
    max_z_new_tokens: int = MAX_TOKENS
    z_temperature: float = 1.2
    z_top_p: float = 0.95
    z_min_p: float = 0.03
    z_repetition_penalty: float = 1.1
    answer_start_logit_bias: float = 0
    steps_for_linear_schaduler_logit_bias: int = 0
    digit_temperature: float = 1.0
    digit_top_p: float = 0.9
    digit_greedy: bool = False
    vllm_enabled: bool = True
    vllm_sync_every: int = 2
    vllm_batch_size: int = 4096
    vllm_tp_size: int = 1
    gpu_memory_utilization: float = 0.95
    vllm_seed: Optional[int] = None
    vllm_tmp_ckpt_dir: str = ""
    vllm_engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    torch_device: str = "cuda:0"
    vllm_cuda_visible_devices: str = "1"
    prompts_per_update: int = 8


@dataclass
class RewardConfig:
    partial_scale: float = 0.25
    keep_prob: Tuple[float, float, float, float, float] = (0.02, 0.05, 0.1, 0.5, 1.0)
    length_penalty: float = 0.0
    correct_length_discount: float = 1.0
    reward_if_max_len: float = 0.0


@dataclass
class GRPOConfig:
    clip_range: float = 0.2
    c_ent: float = 0.001
    kl_coef: float = 0.05
    ppo_epochs: int = 1
    minibatch_size: int = 128
    max_tokens_per_mini_batch: int = 4096*2
    max_grad_norm: float = 1.0
    update_ref_model_each_steps: int = 100


@dataclass
class TrainConfig:
    lr: float = 2e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    updates: int = 10000
    grad_accum_steps: int = 16
    seed: int = 42
    output_dir: str = "./runs/grpo"
    save_every: int = 25


@dataclass
class RuntimeConfig:
    use_bf16: bool = True
    use_length_bucketing: bool = True
    length_bucket_width: int = 64
    old_logp_eval_batch_size: int = 128

@dataclass
class LoggingConfig:
    log_action_tokens: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_SET_ALLOWED_PREFIXES: Sequence[str] = (
    "model.",
    "data.",
    "rollout.",
    "reward.",
    "grpo.",
    "train.",
    "runtime.",
    "logging.",
)
