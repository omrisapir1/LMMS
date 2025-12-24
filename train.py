import os
import sys
import uuid
import time
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from policy.policy_model import PolicyModel
from policy.tokenizer_ext import extend_tokenizer_and_resize
from ppo.trainer import PPOTrainer


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
PHASE1_CONFIG_PATH = os.path.join(CONFIG_DIR, "phase1.yaml")
DEFAULTS_CONFIG_PATH = os.path.join(CONFIG_DIR, "defaults.yaml")


def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _load_yaml(path: str) -> Dict[str, Any]:
    """Minimal YAML loader using Python if PyYAML isn't available.
    Falls back to parsing via json if file is empty or simple. Prefer PyYAML if installed.
    """
    # Try PyYAML if available
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Some configs include leading file path comment lines; yaml loader will ignore comments
        return yaml.safe_load(content) or {}
    except Exception:
        # Fallback: attempt a very naive parser for empty defaults
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                return {}
            # As a last resort, try to interpret as JSON (unlikely here)
            return json.loads(text)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file '{path}': {e}")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without modifying inputs."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


@dataclass
class Phase1Config:
    phase: int
    dataset_name: str
    dataset_split: str
    filter_is_correct_by_qwen_small: bool
    filter_length_ans_max: int
    z_vocab_size: int
    latent_steps_min: int
    latent_steps_max: int
    entropy_coefficient: float
    answer_total_width: int
    seed: Optional[int] = 42

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Phase1Config":
        try:
            return Phase1Config(
                phase=int(d["phase"]),
                dataset_name=str(d["dataset"]["name"]),
                dataset_split=str(d["dataset"]["split"]),
                filter_is_correct_by_qwen_small=bool(d["dataset"]["filters"]["is_correct_by_qwen_small"]),
                filter_length_ans_max=int(d["dataset"]["filters"]["length_ans_max"]),
                z_vocab_size=int(d["model"]["z_tokens"]["vocab_size"]),
                latent_steps_min=int(d["environment"]["latent_steps"]["min"]),
                latent_steps_max=int(d["environment"]["latent_steps"]["max"]),
                entropy_coefficient=float(d["ppo"]["entropy_coefficient"]),
                answer_total_width=int(d["environment"]["answer"]["total_width"]),
                seed=int(d["seed"]),
            )
        except KeyError as e:
            raise KeyError(f"Missing required config key: {e}")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)



def log_kv(**kwargs: Any) -> None:
    msg = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    print(msg)
    sys.stdout.flush()


def load_and_filter_dataset(cfg: Phase1Config):
    if load_dataset is None:
        raise ImportError("datasets library is not installed. Please `pip install datasets`. ")

    # Load split; do not shuffle.
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    # Verify required fields exist, fail fast with helpful message.
    required_fields = ["length_ans", "is_correct_by_qwen_small"]
    for field in required_fields:
        if field not in ds.column_names:
            raise KeyError(
                f"Dataset missing required field '{field}'. Columns available: {ds.column_names}"
            )

    # Apply filters: is_correct_by_qwen_small == True and length_ans <= max
    ds_f = ds.filter(
        lambda ex: ex["is_correct_by_qwen_small"]
        and ex["length_ans"] <= cfg.filter_length_ans_max
        and ex['split'] == cfg.dataset_split
        and ex['final_ans'] == '1'
        and ex["length_ans"] == 1,
        desc="Phase-1 filters",
    )

    # Must-have checks
    size = len(ds_f)
    if size <= 0:
        raise ValueError("Filtered dataset is empty. Adjust filters or verify source dataset.")

    return ds_f


def main():
    # Load configs and merge defaults -> phase1 (phase1 overrides defaults)
    defaults = _load_yaml(DEFAULTS_CONFIG_PATH)
    phase1 = _load_yaml(PHASE1_CONFIG_PATH)
    merged = _deep_merge(defaults or {}, phase1 or {})

    cfg = Phase1Config.from_dict(merged)

    # Device & precision setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_cfg = merged.get("training", {})
    precision = str(training_cfg.get("precision", "fp32")).lower()
    # Remove global default dtype change; we'll cast the model explicitly later

    # Create run id and initialize simple logging
    run_id = f"{_now_ts()}-{uuid.uuid4().hex[:8]}"
    print("LMMS Phase-1 Entry")
    log_kv(run_id=run_id, config_path=PHASE1_CONFIG_PATH)
    log_kv(z_vocab_size=cfg.z_vocab_size, K_min=cfg.latent_steps_min, K_max=cfg.latent_steps_max, entropy_coef=cfg.entropy_coefficient)

    # Set seeds
    set_seeds(cfg.seed or 42)

    # Load and filter dataset
    ds = load_and_filter_dataset(cfg)

    # Report dataset stats
    total = len(ds)
    log_kv(total_samples=total)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        merged["model"]["base_model"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Critical: use LEFT padding so last index (-1) is the last real token
    tokenizer.padding_side = "left"

    # Initialize policy model (keep on CPU initially)
    policy = PolicyModel(merged)

    # Extend tokenizer and resize model embeddings (must be before moving to device)
    extend_tokenizer_and_resize(
        model=policy.lm,
        tokenizer=tokenizer,
        z_vocab_size=merged["model"]["z_tokens"]["vocab_size"],
        answer_token=merged["model"]["special_tokens"]["answer_token"],
    )

    # Move model to device and set dtype explicitly if needed
    if precision == "bf16" and torch.cuda.is_available():
        policy = policy.to(device=device, dtype=torch.bfloat16)
    else:
        policy = policy.to(device=device)

    if training_cfg.get("gradient_checkpointing", False):
        if hasattr(policy, "lm") and hasattr(policy.lm, "gradient_checkpointing_enable"):
            policy.lm.gradient_checkpointing_enable()

    # Initialize PPO trainer
    trainer = PPOTrainer(
        cfg=merged,
        policy=policy,
        tokenizer=tokenizer,
        device=device,
        run_id=run_id,
    )

    # Training loop
    num_epochs = int(training_cfg.get("num_epochs", 1))
    print("Starting Phase-1 training...")
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        metrics = trainer.train_epoch(ds, vocab_size=cfg.z_vocab_size)
        log_kv(
            epoch=epoch + 1,
            mean_reward=metrics.get("mean_reward"),
            policy_loss=metrics.get("policy_loss"),
            value_loss=metrics.get("value_loss"),
            entropy_loss=metrics.get("entropy_loss"),
            ratio_mean=metrics.get("ratio_mean"),
            total_steps=metrics.get("total_steps"),
        )


if __name__ == "__main__":
    main()
