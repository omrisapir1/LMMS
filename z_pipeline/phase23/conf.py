from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union


@dataclass
class ModelConfig:
    # Phase-1 repo/directory used as base for Phase23 construction.
    phase1_dir: str = "omrisap/LMMS_phase1"

    # Z vocabulary size (<Z_0> ... <Z_{v_z-1}>).
    v_z: int = 512

    # Gumbel-Softmax temperature schedule (controls exploration hardness for GS-ST).
    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.3
    gumbel_anneal_steps: int = 3000

    # Token strings.
    z_prefix: str = "Z_"
    latent_token: str = "<|latent|>"
    answer_token: str = "<ANSWER>"


@dataclass
class DataConfig:
    dataset_name: Optional[str] = "omrisap/phaseZ"
    train_split: str = "train"
    eval_split: str = "eval"

    # Optional local path (json/jsonl/parquet) OR HF dataset repo id.
    data_path: Optional[str] = None

    max_length: Optional[int] = None
    batch_size: int = 64
    rebalance_train: bool = True
    k_max: int = 20

    target_k_dist: Dict[str, float] = field(default_factory=lambda: {
        "K1": 0.075,
        "K2": 0.10,
        "K3": 0.125,
        "K4_7": 0.300,
        "K8_12": 0.20,
        "K13_20": 0.20,
    })


@dataclass
class LossConfig:
    lambda_ans: float = 0.1
    lambda_ans_start: float = 0.05
    lambda_ans_end: float = 0.5
    lambda_ans_anneal_steps: int = 1000
    lambda_sft: float = 0.05
    lambda_cf: float = 1.0
    lambda_batch: float = 0.01
    lambda_consistency: float = 0.0
    # Auxiliary SFT term: penalize p(<ANSWER>) at latent slots.
    lambda_no_answer_on_latent: float = 0.95

    digit_temperature: float = 0.1
    # Keep-prob from Phase3 style; accepts tuple/list or dict keys 0..4/1..5.
    keep_prob: Optional[Union[Mapping[int, float], Sequence[float]]] = (0.02, 0.05, 0.1, 0.5, 1)

    counterfactual_schedule: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0,
        2: 0.10, 3: 0.15, 4: 0.20, 5: 0.30,
        6: 0.40, 7: 0.50, 8: 0.60, 9: 0.65,
        10: 0.70, 11: 0.75, 12: 0.80,
        13: 0.85, 14: 0.85,
        15: 0.90, 16: 0.90, 17: 0.90,
        18: 0.90, 19: 0.90, 20: 0.90,
    })


@dataclass
class TrainConfig:
    lr: float = 3e-5
    weight_decay: float = 0.0
    steps: int = 3000
    grad_accum: int = 1

    print_every: int = 5
    eval_every: int = 50
    eval_generate_every_mult: int = 2
    eval_generate_max_new_tokens: int = 64
    eval_generate_temperature: float = 1.0
    eval_generate_top_p: float = 0.95
    save_every: int = 500
    cf_debug_every: int = 0

    # Stage A: frozen backbone warmup (LM head + Z embedding rows only).
    cf_warmup_steps: int = 100
    # Stage B: full-model unfreeze with CF attention-bias anneal to zero.
    cf_bias_anneal_steps: int = 300
    # Additive attention logit bias strength for <ANSWER> query to latent(Z) keys.
    cf_attention_bias_strength: float = 2.0
    cf_attention_bias_enabled: bool = True
    cf_bias_apply_cf_path_only: bool = True

    seed: int = 42
    output_dir: str = "./runs/phase23_gs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
