from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Sequence, Tuple


@dataclass
class ModelConfig:
    init_ckpt: str = "omrisap/LMMS_phase23_step_2500_128"
    answer_token: str = "<ANSWER>"
    z_prefix: str = "Z_"
    v_z_fallback: int = 128


@dataclass
class DataConfig:
    dataset_name: str = "omrisap/LMMS_numina_250K"
    train_split: str = "train"
    question_field: str = "question"
    answer_field: str = "final_answer"


@dataclass
class RolloutConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    episodes_per_batch: int = 128
    max_tokens_per_batch: int = 4096


@dataclass
class RewardConfig:
    partial_scale: float = 0.5
    keep_prob: Tuple[float, float, float, float, float] = (0.02, 0.05, 0.1, 0.5, 1.0)
    length_penalty: float = 0.0
    reward_if_max_len: float = 0.0


@dataclass
class PPOConfig:
    clip_range: float = 0.2
    c_v: float = 0.5
    c_ent: float = 0.01
    ppo_epochs: int = 4
    minibatch_size: int = 32
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True


@dataclass
class TrainConfig:
    lr: float = 3e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    updates: int = 1000
    grad_accum_steps: int = 1
    seed: int = 42
    output_dir: str = "./runs/ppo"
    save_every: int = 200
    keep_last: int = 3


@dataclass
class RuntimeConfig:
    use_bf16: bool = True


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
