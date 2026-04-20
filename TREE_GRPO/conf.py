from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    # Root remains fixed.
    root_siblings: int = 4
    # Depth-dependent branching policy for retry nodes, k in {4,2,1}.
    tree_p4_by_depth: List[float] = field(default_factory=lambda: [0.5, 0.25, 0.0, 0., 0., 0., 0., 0.0])
    tree_p2_by_depth: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0., 0.0])
    tree_p1_by_depth: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0])

    # tree_p4_by_depth: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.35, 0.2, 0.1, 0.1, 0.0])
    # tree_p2_by_depth: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.25, 0.35, 0.3, 0.2, 0.2, 0.0])
    # tree_p1_by_depth: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.25, 0.30, 0.5, 0.7, 0.7, 1.0])
    # Per-prompt safety budgets.
    max_total_nodes_per_prompt: int = 320
    max_leaves_per_prompt: int = 200
    max_active_nodes_per_wave: int = 256
    max_expanded_retry_nodes_per_level: int = 64
    max_retry_depth: int = 50
    c_retry: float = 0.05
    gamma: float = 0.95
    c_branch: float = 0.0
    advantage_clip: float = 3.0
    max_probes_per_prompt: int = 2


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
    vllm_batch_size: int = BATCH_SIZE * 100
    vllm_tp_size: int = 1
    gpu_memory_utilization: float = 0.95
    vllm_seed: Optional[int] = None
    vllm_tmp_ckpt_dir: str = ""
    vllm_engine_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_num_seqs": 256*10,
            "max_num_batched_tokens": 16384,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "swap_space": 8,
            "disable_log_stats": True,
        }
    )
    torch_device: str = "cuda:0"
    ref_model_device: str = "cuda:0"
    vllm_cuda_visible_devices: str = "1"
    tree_prompts_per_update: int = 8
    prefetch_next_rollout: bool = True


@dataclass
class PPOConfig:
    clip_range: float = 0.1
    c_v: float = 0.25
    c_ent: float = 0.002
    kl_coef: float = 0.005
    ppo_epochs: int = 1
    minibatch_size: int = 16
    max_grad_norm: float = 1.0
    update_ref_model_each_steps: int = 100
    value_warmup_steps: int = 0
    value_warmup_lr: float = 1e-4


@dataclass
class TrainConfig:
    lr: float = 2e-5
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
