from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


MAX_TOKENS = 256
BACH_SIZE = 64

@dataclass
class ModelConfig:
    init_ckpt: str = "omrisap/SFT_Z_model"
    answer_token: str = "<ANSWER>"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "omrisap/LMMS_PPO_200K"
    train_split: str = "train"
    question_field: str = "problem"
    answer_digits_field: str = "answer_digits"
    answer_field: str = "final_answer"


@dataclass
class RolloutConfig:
    backend: str = "vllm"  # "vllm" | "hf"
    max_new_tokens: int = MAX_TOKENS
    temperature: float = 1.0
    top_p: float = 0.95
    digit_greedy: bool = False
    action_scope: str = "ppo_full"  # "ppo_only_z_tokens" | "ppo_full"
    vllm_enabled: bool = True
    vllm_sync_every: int = 2
    vllm_batch_size: int = BACH_SIZE
    vllm_tp_size: int = 1
    vllm_seed: Optional[int] = None
    vllm_tmp_ckpt_dir: str = ""
    vllm_engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    torch_device: str = "cuda:0"
    vllm_cuda_visible_devices: str = "1"
    episodes_per_batch: int = 64
    max_tokens_per_batch: int = MAX_TOKENS * BACH_SIZE


@dataclass
class RewardConfig:
    partial_scale: float = 0.5
    keep_prob: Tuple[float, float, float, float, float] = (0.02, 0.05, 0.1, 0.5, 1.0)
    length_penalty: float = 0.001
    reward_if_max_len: float = 0.0


@dataclass
class PPOConfig:
    clip_range: float = 0.2
    c_v: float = 0.5
    c_ent: float = 0.001
    ppo_epochs: int = 2
    minibatch_size: int = 32
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True


@dataclass
class TrainConfig:
    lr: float = 2e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    updates: int = 10000
    grad_accum_steps: int = 4
    seed: int = 42
    output_dir: str = "./runs/ppo"
    save_every: int = 1000
    keep_last: int = 3


@dataclass
class RuntimeConfig:
    use_bf16: bool = True
    debug_restricted_logits_check: bool = False


@dataclass
class LoggingConfig:
    log_action_tokens: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
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
    "reward.",
    "ppo.",
    "train.",
    "runtime.",
    "logging.",
)
