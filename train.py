import os
import sys
import uuid
import time
import json
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from policy.policy_model import PolicyModel
from policy.tokenizer_ext import extend_tokenizer_and_resize
from ppo.trainer import PPOTrainer


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
PHASE1_CONFIG_PATH = os.path.join(CONFIG_DIR, "phase1.yaml")
DEFAULTS_CONFIG_PATH = os.path.join(CONFIG_DIR, "defaults.yaml")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PHASE0_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "phase0")
PHASE0_MODEL_DIR = os.path.join(PHASE0_CKPT_DIR, "model")
PHASE0_TOKENIZER_DIR = os.path.join(PHASE0_CKPT_DIR, "tokenizer")

# Engineering invariant: dataset schema required by PPOTrainer
REQUIRED_DATASET_FIELDS = ("question", "final_ans", "length_ans")


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
class TrainingPhaseConfig:
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
    def from_dict(d: Dict[str, Any]) -> "TrainingPhaseConfig":
        try:
            return TrainingPhaseConfig(
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
                seed=int(d.get("seed", 42)),
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


def _load_phase1_raw(cfg: TrainingPhaseConfig) -> Dataset:
    if load_dataset is None:
        raise ImportError("datasets library is not installed. Please `pip install datasets`. ")
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    return ds


def _adapt_phase1_to_required_schema(ds: Dataset) -> Dataset:
    # Ensure required fields exist or can be mapped; raise if missing
    cols = set(ds.column_names)
    # We expect 'question' or 'question_text' and 'final_ans' and 'length_ans'
    if "question" not in cols and "question_text" not in cols:
        raise RuntimeError(f"Phase-1 dataset missing 'question' or 'question_text'. Columns: {ds.column_names}")
    if "final_ans" not in cols:
        raise RuntimeError(f"Phase-1 dataset missing 'final_ans'. Columns: {ds.column_names}")
    if "length_ans" not in cols:
        raise RuntimeError(f"Phase-1 dataset missing 'length_ans'. Columns: {ds.column_names}")
    # Map question_text -> question if needed
    if "question" not in cols and "question_text" in cols:
        ds = ds.map(lambda ex: {"question": ex["question_text"]}, desc="Map question_text -> question")
    return ds


def _load_phase0_dataset(local_jsonl_path: str) -> List[Dict[str, Any]]:
    """Load local JSONL for protocol pretraining. Ignore extra fields gracefully.
    Returns a simple list of dicts with keys: question, final_ans, length_ans.
    """
    if not os.path.isfile(local_jsonl_path):
        raise FileNotFoundError(f"Phase 0 dataset not found at '{local_jsonl_path}'.")
    rows: List[Dict[str, Any]] = []
    with open(local_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = obj.get("question")
            a = obj.get("answer")
            if q is None or a is None:
                # Skip malformed rows silently
                continue
            try:
                a_int = int(a)
            except Exception:
                continue
            # Single-digit constraint (1..9) per Phase-0 semantics
            if a_int < 0 or a_int > 9:
                continue
            rows.append({
                "question": str(q),
                "final_ans": a_int,
                "length_ans": 1,
            })
    if len(rows) == 0:
        raise ValueError("Phase 0 dataset is empty after parsing. Check JSONL contents.")
    return rows


def validate_dataset(ds: Dataset) -> None:
    # Must be HF Dataset
    if not isinstance(ds, Dataset):
        raise RuntimeError("Dataset must be a HuggingFace Dataset, not a Python list or other type.")
    # Must expose required fields
    cols = set(ds.column_names)
    missing = [f for f in REQUIRED_DATASET_FIELDS if f not in cols]
    if missing:
        raise RuntimeError(f"Dataset missing required fields: {missing}. Ensure schema matches {REQUIRED_DATASET_FIELDS}.")


def validate_tokenizer_and_model(cfg: Dict[str, Any], tokenizer, model) -> None:
    # 1) tokenizer.vocab_size == model input embedding size
    emb = model.get_input_embeddings().weight
    vocab_model = int(emb.shape[0])
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        raise RuntimeError(
            f"Tokenizer/model vocab mismatch: tokenizer.vocab_size={len(tokenizer)} vs model_embeddings={vocab_model}. Phase-0 checkpoint incompatible or corrupted."
        )
    # 2) z-token count and prefix
    z_cfg = cfg["model"]["z_tokens"]
    z_prefix = str(z_cfg["prefix"])  # e.g., "<z>"
    z_expected = int(z_cfg["vocab_size"])  # e.g., 64
    vocab_dict = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if not vocab_dict:
        raise RuntimeError("Tokenizer does not expose vocabulary for z-token validation. Phase-0 checkpoint may be incompatible.")
    z_token_ids = [tid for tok, tid in vocab_dict.items() if isinstance(tok, str) and tok.startswith(z_prefix)]
    if len(z_token_ids) != z_expected:
        raise RuntimeError(
            f"Z-token mismatch: tokenizer extension incompatible or corrupted Phase-0 checkpoint (expected={z_expected} tokens with prefix '{z_prefix}', found={len(z_token_ids)})."
        )
    # 4) answer token existence
    ans_tok = str(cfg["model"]["special_tokens"]["answer_token"])  # "</answer>"
    ans_id = tokenizer.convert_tokens_to_ids(ans_tok)
    if ans_id is None:
        raise RuntimeError("Answer token '</answer>' missing in tokenizer. Phase-0 tokenizer extension incompatible or corrupted.")


def load_dataset_for_phase(merged_cfg: Dict[str, Any]) -> Dataset:
    phase = int(merged_cfg.get("phase", -1))
    if phase == 0:
        ds_local_path = merged_cfg.get("dataset", {}).get("local_jsonl_path")
        if not ds_local_path:
            raise KeyError("Phase 0 requires dataset.local_jsonl_path in config.")
        ds_list = _load_phase0_dataset(ds_local_path)
        ds = Dataset.from_list(ds_list)
        return ds
    elif phase == 1:
        cfg1 = TrainingPhaseConfig.from_dict(merged_cfg)
        raw = _load_phase1_raw(cfg1)
        # Respect YAML literally: apply filters exactly as configured
        # - split already loaded
        # - filter by is_correct_by_qwen_small equals configured value
        # - filter by length_ans <= length_ans_max
        raw_cols = set(raw.column_names)
        # Basic presence checks for filtering keys
        if "is_correct_by_qwen_small" not in raw_cols:
            raise RuntimeError("Phase-1 dataset missing 'is_correct_by_qwen_small' for filtering.")
        if "length_ans" not in raw_cols:
            raise RuntimeError("Phase-1 dataset missing 'length_ans' for filtering.")
        ds_f = raw.filter(
            lambda ex: ex["is_correct_by_qwen_small"] == cfg1.filter_is_correct_by_qwen_small
            and ex["length_ans"] <= cfg1.filter_length_ans_max,
            desc="Phase-1 filters (literal)",
        )
        ds = _adapt_phase1_to_required_schema(ds_f)
        return ds
    else:
        raise ValueError("Config must specify 'phase' as 0 or 1.")


def main():
    # CLI args: single entrypoint with explicit config
    parser = argparse.ArgumentParser(description="LMMS training entrypoint")
    parser.add_argument("--config", type=str, default=PHASE1_CONFIG_PATH, help="Path to YAML config (phase0.yaml or phase1.yaml)")
    args = parser.parse_args()

    # Load configs and merge defaults -> selected config (selected config overrides defaults)
    defaults = _load_yaml(DEFAULTS_CONFIG_PATH)
    selected = _load_yaml(args.config)
    merged = _deep_merge(defaults or {}, selected or {})

    phase = int(merged.get("phase", -1))
    if phase not in (0, 1):
        raise ValueError("Config must specify 'phase' as 0 or 1.")

    # Create run id and initialize simple logging
    run_id = f"{_now_ts()}-{uuid.uuid4().hex[:8]}"
    print("LMMS Entry")
    log_kv(run_id=run_id, config_path=args.config, phase=phase)

    # Device & precision setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_cfg = merged.get("training", {})
    precision = str(training_cfg.get("precision", "fp32")).lower()

    # Set seeds
    seed = int(merged.get("seed", 42))
    set_seeds(seed)

    # Load dataset via dispatcher and validate schema
    ds = load_dataset_for_phase(merged)
    validate_dataset(ds)

    # Report dataset stats
    total = len(ds)
    log_kv(total_samples=total)

    # Tokenizer and policy init
    if phase == 0:
        tokenizer = AutoTokenizer.from_pretrained(
            merged["model"]["base_model"],
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # engineering invariant
        policy = PolicyModel(merged)
        extend_tokenizer_and_resize(
            model=policy.lm,
            tokenizer=tokenizer,
            z_vocab_size=merged["model"]["z_tokens"]["vocab_size"],
            answer_token=merged["model"]["special_tokens"]["answer_token"],
        )
        # Validate tokenizer/model invariants after extension
        validate_tokenizer_and_model(merged, tokenizer, policy.lm)
    else:
        if not (os.path.isdir(PHASE0_MODEL_DIR) and os.path.isdir(PHASE0_TOKENIZER_DIR)):
            raise FileNotFoundError("Phase-0 checkpoint missing. Train Phase 0 first. Expected directories: 'checkpoints/phase0/model' and 'checkpoints/phase0/tokenizer'")
        tokenizer = AutoTokenizer.from_pretrained(
            PHASE0_TOKENIZER_DIR,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        policy = PolicyModel(merged)
        lm_ckpt = AutoModelForCausalLM.from_pretrained(
            PHASE0_MODEL_DIR,
            output_hidden_states=True,
            return_dict=True,
            trust_remote_code=True,
        )
        # Safe loading: keep same LM instance
        policy.lm.load_state_dict(lm_ckpt.state_dict())
        # Validate tokenizer/model invariants after warm start
        validate_tokenizer_and_model(merged, tokenizer, policy.lm)

    # Move model
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
    print("Starting training..." if phase == 0 else "Starting Phase-1 training...")
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        metrics = trainer.train_epoch(ds, vocab_size=int(merged["model"]["z_tokens"]["vocab_size"]))
        log_kv(
            epoch=epoch + 1,
            mean_reward=metrics.get("mean_reward"),
            policy_loss=metrics.get("policy_loss"),
            value_loss=metrics.get("value_loss"),
            entropy_loss=metrics.get("entropy_loss"),
            ratio_mean=metrics.get("ratio_mean"),
            total_steps=metrics.get("total_steps"),
        )

    # Checkpointing after Phase-0
    if phase == 0:
        os.makedirs(PHASE0_MODEL_DIR, exist_ok=True)
        os.makedirs(PHASE0_TOKENIZER_DIR, exist_ok=True)
        # Save only model and tokenizer
        policy.lm.save_pretrained(PHASE0_MODEL_DIR)
        tokenizer.save_pretrained(PHASE0_TOKENIZER_DIR)
        print(f"Saved Phase-0 checkpoint to {PHASE0_CKPT_DIR}")


if __name__ == "__main__":
    main()
