from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .conf import Config
from .dataset import UnifiedDataset, build_rebalanced_sampler, collate_fn
from .eval import evaluate
from .loss import (
    AnswerDigitLoss,
    AnswerTokenSFTLoss,
    CounterfactualAnswerLoss,
)
from .model import UnifiedZSoftModel
from .utils import effective_vocab_size, gumbel_tau_at_step, set_seed


def _save_checkpoint(
    *,
    output_dir: str,
    step: int,
    model: UnifiedZSoftModel,
    tokenizer,
    config: Config,
) -> None:
    ckpt_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "phase23_state.pt"))
    tokenizer.save_pretrained(ckpt_dir)
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def train(cfg: Config) -> None:
    set_seed(cfg.train.seed)

    if cfg.loss.lambda_sft <= 0:
        raise ValueError("Phase23 requires AnswerTokenSFTLoss: set loss.lambda_sft > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = UnifiedZSoftModel.from_phase1(
        phase1_dir=cfg.model.phase1_dir,
        v_z=cfg.model.v_z,
        device=device,
        torch_dtype=torch.bfloat16,
        z_prefix=cfg.model.z_prefix,
        latent_token=cfg.model.latent_token,
        answer_token=cfg.model.answer_token,
    )
    tokenizer = bundle.tokenizer
    model = bundle.model
    model.train()

    train_ds = UnifiedDataset(
        tokenizer=tokenizer,
        latent_token_id=model.latent_token_id,
        answer_token_id=model.answer_token_id,
        k_max=cfg.data.k_max,
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        data_path=cfg.data.data_path,
        max_length=cfg.data.max_length,
    )
    assert train_ds.k_max == cfg.data.k_max, "dataset.k_max must match cfg.data.k_max"

    if cfg.data.rebalance_train:
        sampler = build_rebalanced_sampler(train_ds, cfg.data.target_k_dist)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id")
        pad_token_id = tokenizer.eos_token_id

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        drop_last=False,
    )

    eval_loader = None
    if cfg.train.eval_every > 0:
        eval_ds = UnifiedDataset(
            tokenizer=tokenizer,
            latent_token_id=model.latent_token_id,
            answer_token_id=model.answer_token_id,
            k_max=cfg.data.k_max,
            dataset_name=cfg.data.dataset_name,
            split=cfg.data.eval_split,
            data_path=cfg.data.data_path,
            max_length=cfg.data.max_length,
        )
        assert eval_ds.k_max == cfg.data.k_max, "eval dataset.k_max must match cfg.data.k_max"
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    ans_loss_fn = AnswerDigitLoss(keep_prob=cfg.loss.keep_prob)
    sft_loss_fn = AnswerTokenSFTLoss(answer_token_id=model.answer_token_id)
    cf_loss_fn = CounterfactualAnswerLoss(
        permute_prob=cfg.loss.counterfactual_schedule,
        digit_temperature=cfg.loss.digit_temperature,
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)

    log_sums: Dict[str, float] = {
        "loss": 0.0,
        "ans": 0.0,
        "sft": 0.0,
        "cf": 0.0,
        "eff_vocab": 0.0,
    }
    log_count = 0
    log_eff_count = 0

    train_iter = iter(train_loader)
    while step < cfg.train.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        digit_labels = batch["digit_labels"].to(device)
        k_vals = batch["K"].to(device)

        g_tau = gumbel_tau_at_step(
            step=step,
            tau_start=cfg.model.gumbel_tau_start,
            tau_end=cfg.model.gumbel_tau_end,
            tau_anneal_steps=cfg.model.gumbel_anneal_steps,
        )

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gumbel_tau=g_tau,
                use_gs=True,
                return_distributions=True,
            )

            digit_logits = out["digit_logits"]
            answer_next_logits = out["answer_next_logits"]
            p_student = out.get("p_student")

            loss_ans = ans_loss_fn(digit_logits, digit_labels)
            loss_sft = sft_loss_fn(answer_next_logits)

            if cfg.loss.lambda_cf > 0 and p_student is not None:
                loss_cf = cf_loss_fn(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    p_z=p_student,
                    k_vals=k_vals,
                )
            else:
                loss_cf = torch.zeros((), device=device)

            loss_batch = torch.zeros((), device=device)
            loss_consistency = torch.zeros((), device=device)

            total = (
                cfg.loss.lambda_ans * loss_ans
                + cfg.loss.lambda_sft * loss_sft
                + cfg.loss.lambda_cf * loss_cf
                + cfg.loss.lambda_batch * loss_batch
                + cfg.loss.lambda_consistency * loss_consistency
            )

        total.backward()

        log_sums["loss"] += float(total.detach().cpu())
        log_sums["ans"] += float(loss_ans.detach().cpu())
        log_sums["sft"] += float(loss_sft.detach().cpu())
        log_sums["cf"] += float(loss_cf.detach().cpu())
        log_count += 1

        if p_student is not None:
            log_sums["eff_vocab"] += float(effective_vocab_size(p_student.detach()).mean().cpu())
            log_eff_count += 1

        if (step + 1) % cfg.train.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % cfg.train.print_every == 0:
            denom = max(log_count, 1)
            msg = (
                f"step {step + 1} | "
                f"loss={log_sums['loss'] / denom:.4f} "
                f"ans={log_sums['ans'] / denom:.4f} "
                f"sft={log_sums['sft'] / denom:.4f} "
                f"cf={log_sums['cf'] / denom:.4f} "
                f"tau={g_tau:.4f}"
            )
            if log_eff_count > 0:
                msg += f" | eff_vocab={log_sums['eff_vocab'] / log_eff_count:.2f}"
            print(msg)

            log_sums = {
                "loss": 0.0,
                "ans": 0.0,
                "sft": 0.0,
                "cf": 0.0,
                "eff_vocab": 0.0,
            }
            log_count = 0
            log_eff_count = 0

        if cfg.train.save_every > 0 and (step + 1) % cfg.train.save_every == 0:
            _save_checkpoint(
                output_dir=cfg.train.output_dir,
                step=step + 1,
                model=model,
                tokenizer=tokenizer,
                config=cfg,
            )

        if step  % cfg.train.eval_every == 0 and step > 0:
            metrics = evaluate(
                model=model,
                loader=eval_loader,
                cfg=cfg,
                device=device,
                global_step=step + 1,
            )
            print(
                f"eval@step {step + 1} | "
                + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "n")
            )
            model.train()

        step += 1


if __name__ == "__main__":
    train(Config())
