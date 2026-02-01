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
    phase2_repo_id: str,
    ds_dict: Optional[DatasetDict] = None,
    ds_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    save_dataset_to_disk: Optional[str] = None,
) -> Dict[str, Any]:

    cfg = cfg.validate()
    set_seed_best_effort(cfg.seed)

    # --------------------------------------------------
    # Load tokenizer + Phase-2 metadata from HF
    # --------------------------------------------------
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    import json

    tokenizer = AutoTokenizer.from_pretrained(
        phase2_repo_id,
        subfolder="tokenizer",
    )

    z_meta_path = hf_hub_download(phase2_repo_id, "z_meta.json")
    with open(z_meta_path, "r") as f:
        z_meta = json.load(f)

    z_token_ids = list(map(int, z_meta["z_token_ids"]))
    answer_token_id = int(z_meta["answer_token_id"])
    pad_token_id = tokenizer.pad_token_id

    # --------------------------------------------------
    # Resolve dataset
    # --------------------------------------------------
    if ds_dict is None:
        if ds_path is None:
            raise ValueError("Provide ds_dict or ds_path")
        ds_dict = load_from_disk(ds_path)
        if not isinstance(ds_dict, DatasetDict):
            raise RuntimeError("ds_path must contain a DatasetDict")

    if save_dataset_to_disk is not None:
        _ensure_dir(save_dataset_to_disk)
        ds_dict.save_to_disk(save_dataset_to_disk)

    # --------------------------------------------------
    # Build Phase-3 model FROM HF Phase-2 repo
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Phase3ZModel.from_phase2_repo(
        repo_id=phase2_repo_id,
        fill_value=-1e4,
        answer_init_std=0.02,
        device=device,
    )

    # --------------------------------------------------
    # Resume Phase-3 checkpoint (optional)
    # --------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )

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

    # --------------------------------------------------
    # Dataset + Loader
    # --------------------------------------------------
    train_ds = Phase3Dataset(
        hf_dataset=ds_dict["train"],
        z_token_ids=z_token_ids,
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

    # --------------------------------------------------
    # Losses
    # --------------------------------------------------
    loss_fn = Phase3Loss(
        answer_loss=AnswerLoss(keep_prob=cfg.loss.keep_prob),
        sft_loss=SFTLoss(ignore_index=-100),
        kl_loss=DigitKLDiversityLoss(
            z_token_ids=z_token_ids,
            answer_token_id=answer_token_id,
            length_to_reverse_prob=cfg.loss.reverse_prob_by_k,
            digit_temperature=cfg.loss.digit_temperature,
            random_seed=cfg.seed,
        ),
        lambda_answer=cfg.loss.lambda_answer,
        lambda_sft=cfg.loss.lambda_sft,
        lambda_kl=cfg.loss.lambda_kl,
    )

    # --------------------------------------------------
    # Evaluator
    # --------------------------------------------------
    evaluator = Phase3Evaluator(
        tokenizer=tokenizer,
        batch_size=cfg.eval.batch_size,
        answer_token_id=answer_token_id,
        device=device,
    )

    # --------------------------------------------------
    # Training loop (UNCHANGED SEMANTICS)
    # --------------------------------------------------
    model.train()

    for epoch in range(cfg.train.num_epochs):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            digit_labels = batch["digit_labels"].to(device)

            input_ids = _apply_pad_id_safely(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
            )

            if input_ids.size(1) > cfg.data.max_length:
                raise RuntimeError(
                    f"Sequence too long: {input_ids.size(1)} > {cfg.data.max_length}"
                )

            sft_attention = _mask_sft_to_start_at_first_z(
                input_ids=input_ids,
                attention_mask=attention_mask,
                z_token_ids=z_token_ids,
            )

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            losses = loss_fn.compute(
                model=model,
                logits=out.logits,
                input_ids=input_ids,
                attention_mask=sft_attention,
                digit_logits=out.digit_logits,
                digit_labels=digit_labels,
            )

            loss_total = losses["loss_total"]

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()

            if cfg.optim.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.optim.max_grad_norm,
                )

            optimizer.step()
            global_step += 1

            if global_step % 10 == 0:
                print(
                    f"[phase3/train] step={global_step} "
                    f"loss={loss_total.item():.4f}"
                )

            if global_step % cfg.train.eval_every_steps == 0:
                model.eval()
                metrics = evaluator.evaluate(
                    model=model,
                    max_generation_tokens=cfg.eval.max_generation_tokens,
                    sampling_temperature=cfg.eval.sampling_temperature,
                    top_p=cfg.eval.top_p,
                    top_k=cfg.eval.top_k,
                )
                print("\n=== Phase-3 Eval ===")
                for k, m in metrics.items():
                    print(f"{k}: EM={m['digit_em']*100:.2f}%")
                model.train()

            if global_step % cfg.ckpt.save_every_steps == 0:
                last_ckpt_path = _save_ckpt(
                    save_dir=cfg.ckpt.save_dir,
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                )
                print(f"[phase3/train] Saved {last_ckpt_path}")

    last_ckpt_path = _save_ckpt(
        save_dir=cfg.ckpt.save_dir,
        step=global_step,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
    )

    return {
        "model": model,
        "final_step": global_step,
        "last_ckpt_path": last_ckpt_path,
    }


__all__ = ["run_phase3"]
