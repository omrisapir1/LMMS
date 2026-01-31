# phase3/train.py
#
# Phase-3 training:
# - Teacher-forced training only (NO generate in training loop)
# - Loss = AnswerLoss + SFT loss + KL diversity loss
# - SFT loss is masked to start ONLY from the first Z token onward
# - Periodic evaluation uses Phase3Evaluator (generate_with_digits) in greedy + sample modes
# - Supports:
#     (A) in-memory DatasetDict passed from pipeline
#     (B) resume from disk: load model checkpoint + dataset saved_to_disk
#
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_from_disk

from .conf import Phase3Config
from .dataset import Phase3Dataset, phase3_collate_fn, TARGET_DIST
from .eval import Phase3Evaluator
from .loss import AnswerLoss, SFTLoss, DigitKLDiversityLoss, Phase3Loss
from .model import Phase3ZModel


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def set_seed_best_effort(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _get_device_from_model(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _mask_sft_to_start_at_first_z(
    *,
    input_ids: torch.Tensor,          # [B,T]
    attention_mask: torch.Tensor,     # [B,T]
    z_token_ids: list[int],
) -> torch.Tensor:
    """
    Returns a new attention_mask' where tokens BEFORE the first Z token are masked out (set to 0),
    keeping suffix padding semantics (still 0 after pad).
    """
    z_set = set(int(x) for x in z_token_ids)
    B, T = input_ids.shape
    out = attention_mask.clone()

    # per-row scan (B is small; T is moderate)
    for b in range(B):
        # find first Z position among non-pad tokens
        first_z = None
        for t in range(T):
            if out[b, t].item() == 0:
                break
            if int(input_ids[b, t].item()) in z_set:
                first_z = t
                break

        if first_z is None:
            # dataset contract violation: must contain Z tokens
            raise RuntimeError("SFT mask: no Z token found in a sample (dataset contract violation)")

        # mask out everything before first_z
        if first_z > 0:
            out[b, :first_z] = 0

    return out


def _apply_pad_id_safely(
    *,
    input_ids: torch.Tensor,          # [B,T]
    attention_mask: torch.Tensor,     # [B,T]
    pad_token_id: Optional[int],
) -> torch.Tensor:
    """
    phase3_collate_fn pads input_ids with 0. Replace pad regions with tokenizer.pad_token_id if available.
    """
    if pad_token_id is None:
        return input_ids
    return input_ids.masked_fill(attention_mask == 0, int(pad_token_id))


# ------------------------------------------------------------
# Checkpoint I/O (minimal)
# ------------------------------------------------------------

def _save_ckpt(
    *,
    save_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Phase3Config,
) -> str:
    _ensure_dir(save_dir)
    path = os.path.join(save_dir, f"ckpt_step_{step}.pt")
    payload = {
        "step": int(step),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    torch.save(payload, path)
    return path


def _load_ckpt(
    *,
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> int:
    payload = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None and "optim_state" in payload:
        optimizer.load_state_dict(payload["optim_state"])
    return int(payload.get("step", 0))


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

def run_phase3(
    cfg: Phase3Config,
    *,
    phase2_ckpt: Optional[Dict[str, Any]] = None,
    ds_dict: Optional[DatasetDict] = None,
    ds_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    save_dataset_to_disk: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Phase-3 training.

    Inputs (choose one path):
      A) Pipeline mode:
         - phase2_ckpt provided (contains model, tokenizer, z_token_ids)
         - ds_dict provided (DatasetDict with "train" and "eval") from generate_dataset.py
      B) Resume mode:
         - ckpt_path provided (Phase3ZModel weights)
         - ds_path provided (DatasetDict saved via datasets.save_to_disk)

    Returns:
      dict with:
        - "model": trained model
        - "final_step": int
        - "last_ckpt_path": Optional[str]
    """
    cfg = cfg.validate()
    set_seed_best_effort(cfg.seed)

    # -------------------------
    # Resolve tokenizer + z ids + answer id
    # -------------------------
    if phase2_ckpt is None and ckpt_path is None:
        raise ValueError("Provide either phase2_ckpt (pipeline mode) or ckpt_path (resume mode).")

    tokenizer = None
    z_token_ids = None
    answer_token_id = None

    if phase2_ckpt is not None:
        tokenizer = phase2_ckpt["tokenizer"]
        z_token_ids = list(phase2_ckpt["z_token_ids"])
        answer_token_id = int(phase2_ckpt["model"].answer_token_id)

    # If resume mode without phase2_ckpt: user must still pass these via checkpoint init on disk,
    # but we still need z_token_ids + answer_token_id for dataset checks and losses.
    # Minimal assumption: they are stored in the model attributes after loading ckpt.
    # We'll load model first then read them.
    # -------------------------
    # Resolve dataset dict
    # -------------------------
    if ds_dict is None:
        if ds_path is None:
            raise ValueError("Provide ds_dict (pipeline) or ds_path (resume).")
        ds_dict = load_from_disk(ds_path)
        if not isinstance(ds_dict, DatasetDict):
            raise RuntimeError("ds_path must point to a datasets.DatasetDict saved_to_disk.")

    # Optionally persist dataset for later resume
    if save_dataset_to_disk is not None:
        _ensure_dir(save_dataset_to_disk)
        ds_dict.save_to_disk(save_dataset_to_disk)

    # -------------------------
    # Build / load model
    # -------------------------
    model: Phase3ZModel
    if ckpt_path is None:
        # Pipeline path: construct from in-memory Phase2 model
        assert phase2_ckpt is not None
        phase2_model = phase2_ckpt["model"]
        model = Phase3ZModel.from_phase2_ckpt(
            phase2_ckpt=phase2_ckpt
        )
    else:
        # Resume path: need a Phase3ZModel skeleton first
        # If phase2_ckpt provided, build skeleton from phase2; else assume it was saved after init
        if phase2_ckpt is not None:
            phase2_model = phase2_ckpt["model"]
            model = Phase3ZModel.from_phase2_ckpt(
                phase2_ckpt=phase2_ckpt
            )
        else:
            raise ValueError(
                "Resume mode without phase2_ckpt is not supported in this minimal implementation. "
                "Pass phase2_ckpt as well (in-memory) so we can rebuild the model skeleton, then load weights."
            )

    device = _get_device_from_model(model)
    device = 'cuda'
    model.to(device)

    # Now that model exists, ensure we have z_token_ids + answer_token_id
    if z_token_ids is None:
        z_token_ids = list(model.z_token_ids)
    if answer_token_id is None:
        answer_token_id = int(model.answer_token_id)

    pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None

    # -------------------------
    # DataLoaders (train/eval splits are for teacher-forcing, not generation eval)
    # -------------------------
    # We wrap HF Dataset splits with Phase3Dataset-style contract checks by using dataset_name path.
    # BUT here ds_dict is already materialized with needed columns; easiest is to use it directly.
    # We'll use DataLoader over ds_dict["train"] and ds_dict["eval"].


    train_ds = Phase3Dataset(
        hf_dataset=ds_dict["train"],  # ✅ correct
        z_token_ids=list(z_token_ids),
        answer_token_id=answer_token_id,
        max_length=cfg.data.max_length,
        rebalance_train=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=phase3_collate_fn,
    )
    # -------------------------
    # Losses
    # -------------------------
    answer_loss = AnswerLoss(keep_prob=cfg.loss.keep_prob)
    sft_loss = SFTLoss(ignore_index=-100)
    kl_loss = DigitKLDiversityLoss(
        z_token_ids=list(z_token_ids),
        answer_token_id=answer_token_id,
        length_to_reverse_prob=cfg.loss.reverse_prob_by_k,
        digit_temperature=cfg.loss.digit_temperature,
        random_seed=cfg.seed,
    )
    loss_fn = Phase3Loss(
        answer_loss=answer_loss,
        sft_loss=sft_loss,
        kl_loss=kl_loss,
        lambda_answer=cfg.loss.lambda_answer,
        lambda_sft=cfg.loss.lambda_sft,
        lambda_kl=cfg.loss.lambda_kl,
    )

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )

    # -------------------------
    # Resume checkpoint
    # -------------------------
    global_step = 0
    last_ckpt_path: Optional[str] = None
    if ckpt_path is not None:
        global_step = _load_ckpt(
            ckpt_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            map_location="cpu",
        )
        model.to(device)
        print(f"[phase3/train] Resumed from {ckpt_path} at step={global_step}")

    # -------------------------
    # Evaluator (generation-based) — loads dataset once
    # -------------------------
    evaluator = Phase3Evaluator(
        tokenizer=tokenizer,
        batch_size=cfg.eval.batch_size,
        answer_token_id=answer_token_id,
        device=device,
    )

    # -------------------------
    # Training loop
    # -------------------------
    model.train()
    for epoch in range(cfg.train.num_epochs):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            digit_labels = batch["digit_labels"].to(device)

            # Safety: pad ids
            input_ids = _apply_pad_id_safely(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
            )

            # Hard error on max_length
            if input_ids.size(1) > cfg.data.max_length:
                raise RuntimeError(
                    f"Batch sequence too long: T={input_ids.size(1)} > max_length={cfg.data.max_length}"
                )

            # SFT mask: start from first Z onward
            sft_attention = _mask_sft_to_start_at_first_z(
                input_ids=input_ids,
                attention_mask=attention_mask,
                z_token_ids=list(z_token_ids),
            )

            # Forward
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = out.logits
            digit_logits = out.digit_logits

            # Losses
            losses = loss_fn.compute(
                model=model,
                logits=logits,
                input_ids=input_ids,
                attention_mask=sft_attention,  # masked attention for SFT only
                digit_logits=digit_logits,
                digit_labels=digit_labels,
            )
            loss_total = losses["loss_total"]

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()

            if cfg.optim.max_grad_norm is not None and cfg.optim.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.max_grad_norm))

            optimizer.step()
            global_step += 1

            # Print occasionally (minimal)
            if global_step % 10 == 0:
                la = float(losses["loss_answer"].item())
                ls = float(losses["loss_sft"].item())
                lk = float(losses["loss_kl"].item())
                lt = float(loss_total.item())
                print(
                    f"[phase3/train] step={global_step} "
                    f"loss_total={lt:.4f} answer={la:.4f} sft={ls:.4f} kl={lk:.4f}"
                )

            # Periodic eval (generation-based)
            if global_step % cfg.train.eval_every_steps == 0:
                model.eval()
                metrics = evaluator.evaluate(
                    model=model,
                    max_generation_tokens=cfg.eval.max_generation_tokens,
                    sampling_temperature=cfg.eval.sampling_temperature,
                    top_p=cfg.eval.top_p,
                    top_k=cfg.eval.top_k,
                )
                print("\n================ Phase-3 Eval (generation) ================")
                for mode_name, m in metrics.items():
                    print(f"\n--- {mode_name} ---")
                    print(f"Digit EM: {m['digit_em'] * 100:.2f}%")
                    print(f"Z length mean/median: {m['z_length']['mean']:.2f} / {m['z_length']['median']:.2f}")
                    print(f"Z usage entropy: {m['z_usage']['entropy']:.3f}")
                    print(f"Effective vocab size: {m['z_usage']['effective_vocab_size']:.2f}")
                print("===========================================================\n")
                model.train()

            # Periodic checkpoint
            if global_step % cfg.ckpt.save_every_steps == 0:
                last_ckpt_path = _save_ckpt(
                    save_dir=cfg.ckpt.save_dir,
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                )
                print(f"[phase3/train] Saved checkpoint: {last_ckpt_path}")

    # Final checkpoint at end (optional but useful)
    last_ckpt_path = _save_ckpt(
        save_dir=cfg.ckpt.save_dir,
        step=global_step,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
    )
    print(f"[phase3/train] Saved final checkpoint: {last_ckpt_path}")

    return {
        "model": model,
        "final_step": int(global_step),
        "last_ckpt_path": last_ckpt_path,
    }


__all__ = ["run_phase3"]
