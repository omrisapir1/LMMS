# phase2/train.py
#
# Phase-2 training:
#   Learn Z-token embeddings + Z-selector from digit loss ONLY.
#
# Direct handoff contract (NO DISK):
#   def run_experiment(cfg):
#       phase2_ckpt = run_phase2(cfg.phase2)
#       phase3_metrics = run_phase3(cfg.phase3, phase2_ckpt)
#       return phase3_metrics
#
# phase2_ckpt = {
#   "model": Phase2ZModel,
#   "tokenizer": tokenizer,
#   "z_token_ids": List[int],      # ids for <Z_0>.. <Z_{V-1}>
#   "z_vocab_size": int,
# }
#
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from .conf import Phase2Config
from .dataset import Phase2Dataset, phase2_collate_fn, compute_keep_prob_from_dataset
from .loss import AnswerLoss, ZUsageKLLoss
from .eval import evaluate_phase2
from .model import Phase2ZModel


# -----------------------------
# Utils
# -----------------------------

def z_token_str(i: int) -> str:
    return f"<Z_{i}>"


def set_seed_best_effort(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_from_model(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def compute_temperature(step: int, cfg: Phase2Config) -> float:
    sched = cfg.temp
    t0 = float(sched.temp_start)
    t1 = float(sched.temp_end)
    n = int(sched.anneal_steps)

    if step <= 0:
        return t0

    if step >= n:
        return t1

    # progress in [0,1]
    p = step / max(1, n)

    if sched.type == "linear":
        return (1.0 - p) * t0 + p * t1

    if sched.type == "cosine":
        # cosine from t0 -> t1
        c = 0.5 * (1.0 + math.cos(math.pi * p))  # 1 -> 0
        return t1 + (t0 - t1) * c

    if sched.type == "exponential":
        # t = t0 * (t1/t0)^p
        ratio = t1 / max(t0, 1e-12)
        return t0 * (ratio ** p)

    raise ValueError(f"Unknown temperature schedule type: {sched.type}")


class _EvalTemperatureProxy:
    """
    eval.py calls: model(..., temperature=None, return_z_probs=True)

    Your Phase2ZModel forward expects a positive float temperature during training.
    For eval, we want argmax-ish behavior; easiest is to replace None with tiny temp.
    """
    def __init__(self, model: torch.nn.Module, eval_temp: float = 1e-6):
        self._m = model
        self._eval_temp = float(eval_temp)

    def __getattr__(self, name):
        return getattr(self._m, name)

    def __call__(self, *args, **kwargs):
        if kwargs.get("temperature", None) is None:
            kwargs["temperature"] = self._eval_temp
        return self._m(*args, **kwargs)


# -----------------------------
# Main entry point
# -----------------------------

def run_phase2(cfg: Phase2Config) -> Dict:
    """
    Returns phase2_ckpt dict (in-memory handoff).
    """
    cfg = cfg.finalize()

    set_seed_best_effort(cfg.seed)

    # -----------------------------
    # Load Phase-1 model (for base LM + digit heads)
    # -----------------------------
    # Expected to exist in your repo; signature matches your earlier usage.
    from load_model_phase1 import load_phase1  # type: ignore

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, phase1_model, _meta = load_phase1(device=device_str)

    # Phase-2 bypasses Coconut execution; we only need phase0 components.
    phase0 = phase1_model.phase0
    base_lm = phase0.model
    digit_heads = phase0.digit_heads

    # Ensure pad token exists (DataLoader padding relies on it being stable)
    if tokenizer.pad_token_id is None:
        # best effort: reuse EOS as PAD
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id and no eos_token_id; set pad token explicitly.")
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve token ids
    latent_token_id = int(cfg.latent_token_id)
    answer_token_id = int(cfg.answer_token_id)

    # -----------------------------
    # Expand tokenizer with Z tokens
    # -----------------------------
    z_tokens = [z_token_str(i) for i in range(cfg.z_vocab_size)]

    # Add only those not present (idempotent)
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in z_tokens if t not in existing]

    if to_add:
        tokenizer.add_tokens(to_add, special_tokens=False)

    # Resize base model embeddings to new vocab size
    try:
        base_lm.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        raise RuntimeError(f"Failed to resize_token_embeddings to {len(tokenizer)}: {e}")

    # Map <Z_i> -> token ids in order 0..V-1
    z_token_ids: List[int] = []
    for i in range(cfg.z_vocab_size):
        tid = tokenizer.convert_tokens_to_ids(z_token_str(i))
        if tid is None or tid < 0:
            raise RuntimeError(f"Failed to convert token to id: {z_token_str(i)}")
        z_token_ids.append(int(tid))

    # -----------------------------
    # Build Phase-2 model wrapper
    # -----------------------------
    model = Phase2ZModel(
        base_lm=base_lm,
        digit_heads=digit_heads,
        answer_token_id=answer_token_id,
        latent_token_id=latent_token_id,
        z_token_ids=z_token_ids,
        freeze_base=True,
        freeze_digit_heads=True,
        force_base_eval=cfg.force_base_eval,
    )

    device = get_device_from_model(model)

    # -----------------------------
    # Initialize Z embeddings (best-effort)
    # -----------------------------
    # You didn't specify a special init. We'll:
    # - initialize each Z row with N(0, std) matching existing embedding stats.
    emb = model.base.get_input_embeddings()
    with torch.no_grad():
        w = emb.weight
        # embedding stats
        std = float(w.std().item()) if w.numel() > 0 else 0.02
        for tid in z_token_ids:
            w[tid].normal_(mean=0.0, std=max(std, 1e-4))

    # Initialize selector weights
    torch.nn.init.xavier_uniform_(model.z_selector.weight)
    if model.z_selector.bias is not None:
        torch.nn.init.zeros_(model.z_selector.bias)

    # -----------------------------
    # Data
    # -----------------------------
    train_ds = Phase2Dataset(
        tokenizer=tokenizer,
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        k_max=cfg.data.k_max,
        latent_token_id=latent_token_id,
        answer_token_id=answer_token_id,
        seed=cfg.seed,
        rebalance_train=cfg.data.rebalance_train,
        max_length=cfg.data.max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        collate_fn=phase2_collate_fn,
    )

    # -----------------------------
    # Losses
    # -----------------------------
    keep_prob = compute_keep_prob_from_dataset(train_ds, alpha=0.3, min_k=0.05)
    answer_loss_fn = AnswerLoss(keep_prob=keep_prob)
    z_kl_loss_fn = ZUsageKLLoss(vocab_size=cfg.z_vocab_size)
    print(f'AnswerLoss keep_prob: {keep_prob}')

    # -----------------------------
    # Optimizer (ONLY selector + embedding weight)
    # -----------------------------
    # Note: embedding weight is a single Parameter; gradients are masked inside the model.
    optim_params = [
        {"params": list(model.z_selector.parameters()), "weight_decay": cfg.optim.weight_decay},
        {"params": [model.base.get_input_embeddings().weight], "weight_decay": cfg.optim.weight_decay},
    ]
    optimizer = AdamW(
        optim_params,
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )

    # -----------------------------
    # Training loop control
    # -----------------------------
    eval_every = int(cfg.eval.eval_every_steps)
    min_steps = int(cfg.eval.min_steps or 0)
    patience = int(cfg.eval.patience)
    min_delta = float(cfg.eval.min_delta)

    # Hard max steps so runs terminate even without early stop:
    # finish temp schedule + patience eval windows
    max_steps = min_steps + patience * eval_every

    best_em = -1.0
    no_improve = 0

    global_step = 0
    model.train()

    # -----------------------------
    # Training loop
    # -----------------------------
    loader_iter = iter(train_loader)

    while global_step < max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        global_step += 1

        # Temperature schedule (selector logits only)
        temp = compute_temperature(global_step, cfg)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        latent_states = batch["latent_states"].to(device, non_blocking=True)
        z_mask = batch["z_mask"].to(device, non_blocking=True)
        digit_labels = batch["digit_labels"].to(device, non_blocking=True)

        # Forward (need z_probs for KL loss)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_states=latent_states,
            z_mask=z_mask,
            temperature=float(temp),
            return_z_probs=True,
        )

        digit_logits = out["digit_logits"]
        z_probs = out["z_probs"]

        loss_answer = answer_loss_fn.compute(digit_logits, digit_labels)
        loss_kl = z_kl_loss_fn.compute(z_probs, z_mask)

        loss = cfg.loss.lambda_answer * loss_answer + cfg.loss.lambda_kl * loss_kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.optim.max_grad_norm is not None and cfg.optim.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.max_grad_norm))

        optimizer.step()

        # -------------------------
        # Periodic eval
        # -------------------------
        if global_step % eval_every == 0:
            eval_model = _EvalTemperatureProxy(model, eval_temp=1e-6)

            metrics = evaluate_phase2(
                model=eval_model,
                tokenizer=tokenizer,
                dataset_name=cfg.data.dataset_name,
                batch_size=cfg.data.eval_batch_size,
                latent_token_id=latent_token_id,
                answer_token_id=answer_token_id,
                k_max=cfg.data.k_max,
                device=device,
            )

            digit_em = float(metrics["digit_em"])

            # Early stopping gate
            if global_step >= min_steps:
                if digit_em > best_em + min_delta:
                    best_em = digit_em
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    # As requested: do NOT restore best; continue with current weights to Phase-3.
                    break

            # Back to train mode (base may remain eval if cfg.force_base_eval)
            model.train()

    # -----------------------------
    # Return in-memory handoff
    # -----------------------------
    phase2_ckpt = {
        "model": model,
        "tokenizer": tokenizer,
        "z_token_ids": z_token_ids,
        "z_vocab_size": int(cfg.z_vocab_size),
        "phase2_steps": int(global_step),
        "phase2_cfg": asdict(cfg),
    }
    return phase2_ckpt


__all__ = ["run_phase2"]
