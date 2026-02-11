from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .conf import Config
from .dataset import UnifiedDataset, build_rebalanced_sampler, collate_fn
from .eval import evaluate, evaluate_generate_table
from .loss import (
    AnswerDigitLoss,
    AnswerTokenSFTLoss,
    batch_usage_entropy_loss,
    CounterfactualAnswerLoss,
)
from .model import UnifiedZSoftModel
from .utils import effective_vocab_size, gumbel_tau_at_step, set_seed


def _log_line(msg: str, log_path: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"{ts} | {msg}"
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def _cf_stage_and_bias_scale(step: int, cfg: Config) -> Tuple[str, float]:
    warmup = max(0, int(cfg.train.cf_warmup_steps))
    anneal = max(0, int(cfg.train.cf_bias_anneal_steps))

    if step < warmup:
        stage = "warmup_frozen"
        scale = 1.0
    elif step < warmup + anneal and anneal > 0:
        stage = "anneal_unfrozen"
        progress = float(step - warmup) / float(max(anneal, 1))
        scale = max(0.0, 1.0 - progress)
    else:
        stage = "main"
        scale = 0.0

    if not cfg.train.cf_attention_bias_enabled:
        scale = 0.0
    return stage, 0#float(scale)


def _lambda_ans_at_step(step: int, cfg: Config) -> float:
    warmup = max(0, int(cfg.train.cf_warmup_steps))
    anneal_steps = max(0, int(cfg.loss.lambda_ans_anneal_steps))
    start = float(cfg.loss.lambda_ans_start)
    end = float(cfg.loss.lambda_ans_end)

    if step < warmup:
        return start
    if anneal_steps <= 0:
        return start

    progress = float(step - warmup) / float(anneal_steps)
    progress = max(0.0, min(1.0, progress))
    return float(start + progress * (end - start))


def _build_optimizer(
    model: UnifiedZSoftModel,
    cfg: Config,
    stage_name: str,
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for optimizer.")
    if stage_name != "warmup_frozen":
        return torch.optim.AdamW(
            params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

    emb_weight = model.base.get_input_embeddings().weight
    emb_params = []
    other_params = []
    for p in params:
        if p is emb_weight:
            emb_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "weight_decay": cfg.train.weight_decay})
    if emb_params:
        # Keep non-Z rows fully stable; row updates are controlled only by grad mask.
        param_groups.append({"params": emb_params, "weight_decay": 0.0})

    return torch.optim.AdamW(
        param_groups,
        lr=cfg.train.lr,
    )


def _configure_stage_trainability(
    *,
    model: UnifiedZSoftModel,
    stage_name: str,
    embed_grad_hook: Optional[torch.utils.hooks.RemovableHandle],
) -> Optional[torch.utils.hooks.RemovableHandle]:
    # Remove previous embedding grad mask hook before changing stage.
    if embed_grad_hook is not None:
        embed_grad_hook.remove()
        embed_grad_hook = None

    if stage_name == "warmup_frozen":
        # Freeze everything first.
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze LM head.
        lm_head = model._get_lm_head()
        for p in lm_head.parameters():
            p.requires_grad = True

        # Unfreeze embedding matrix but allow gradients only for Z rows.
        emb_weight = model.base.get_input_embeddings().weight
        emb_weight.requires_grad = True
        z_ids = torch.tensor(model.z_token_ids, device=emb_weight.device, dtype=torch.long)
        z_mask = torch.zeros(emb_weight.size(0), device=emb_weight.device, dtype=emb_weight.dtype)
        z_mask[z_ids] = 1.0

        def _mask_non_z_rows(grad: torch.Tensor) -> torch.Tensor:
            return grad * z_mask.unsqueeze(1).to(grad.dtype)

        embed_grad_hook = emb_weight.register_hook(_mask_non_z_rows)
    else:
        # Stages B/C: full model trainable.
        for p in model.parameters():
            p.requires_grad = True

    return embed_grad_hook


def train(cfg: Config) -> None:
    set_seed(cfg.train.seed)
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    if cfg.loss.lambda_sft <= 0:
        raise ValueError("Phase23 requires AnswerTokenSFTLoss: set loss.lambda_sft > 0")
    if not cfg.train.cf_bias_apply_cf_path_only:
        raise ValueError("This training plan requires cf_bias_apply_cf_path_only=True")

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
    model.base.gradient_checkpointing_enable()
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

    ans_loss_fn = AnswerDigitLoss(keep_prob=cfg.loss.keep_prob)
    sft_loss_fn = AnswerTokenSFTLoss(
        answer_token_id=model.answer_token_id,
        lambda_no_answer_on_latent=cfg.loss.lambda_no_answer_on_latent,
    )
    cf_loss_fn = CounterfactualAnswerLoss(
        permute_prob=cfg.loss.counterfactual_schedule,
        digit_temperature=cfg.loss.digit_temperature,
        debug_every=cfg.train.cf_debug_every,
    )

    step = 0
    current_stage = ""
    current_bias_scale = 0.0
    embed_grad_hook: Optional[torch.utils.hooks.RemovableHandle] = None
    optimizer: Optional[torch.optim.Optimizer] = None

    log_sums: Dict[str, float] = {
        "loss": 0.0,
        "ans": 0.0,
        "sft": 0.0,
        "sft_ce": 0.0,
        "sft_no_answer_latent": 0.0,
        "sft_latent_p_ans": 0.0,
        "cf_det": 0.0,
        "batch": 0.0,
        "eff_vocab": 0.0,
    }
    log_count = 0
    log_eff_count = 0

    train_iter = iter(train_loader)
    while step < cfg.train.steps:

        stage_name, cf_bias_scale = _cf_stage_and_bias_scale(step, cfg)
        if stage_name != current_stage:
            embed_grad_hook = _configure_stage_trainability(
                model=model,
                stage_name=stage_name,
                embed_grad_hook=embed_grad_hook,
            )
            optimizer = _build_optimizer(model, cfg, stage_name=stage_name)
            optimizer.zero_grad(set_to_none=True)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            current_stage = stage_name
        current_bias_scale = cf_bias_scale

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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        lambda_ans_current = _lambda_ans_at_step(step, cfg)
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
            latent_answer_logit = out.get("latent_answer_logit_allowed")
            latent_logsumexp = out.get("latent_logsumexp_allowed")
            latent_slot_mask = out.get("latent_slot_mask", out.get("slot_mask"))
            p_student = out.get("p_student")

            loss_ans = ans_loss_fn(digit_logits, digit_labels)
            sft_terms = sft_loss_fn(
                answer_next_logits,
                latent_answer_logits=latent_answer_logit,
                latent_logsumexp=latent_logsumexp,
                latent_slot_mask=latent_slot_mask,
                return_details=True,
            )
            loss_sft = sft_terms["loss_total"]
            loss_sft_ce = sft_terms["loss_answer_ce"]
            loss_sft_no_answer_latent = sft_terms["loss_no_answer_latent"]
            sft_latent_p_ans = sft_terms["latent_p_ans_mean"]

            if cfg.loss.lambda_cf > 0 and p_student is not None:
                p_student_det_idx = p_student.detach().argmax(dim=-1)
                apply_det_bias = (
                    cfg.train.cf_bias_apply_cf_path_only
                    and cfg.train.cf_attention_bias_enabled
                    and current_bias_scale > 0.0
                )
                cf_det = cf_loss_fn(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    p_z=None,
                    p_z_idx_det=p_student_det_idx,
                    k_vals=k_vals,
                    cf_bias_scale=current_bias_scale,
                    cf_attention_bias_strength=cfg.train.cf_attention_bias_strength,
                    apply_cf_answer_z_bias=apply_det_bias,
                    cf_mode="det",
                    global_step=step + 1,
                    stage_name=current_stage,
                    return_details=True,
                )
                loss_cf_det = cf_det["loss_cf"]
            else:
                loss_cf_det = torch.zeros((), device=device)

            if cfg.loss.lambda_batch > 0 and p_student is not None:
                loss_batch = batch_usage_entropy_loss(
                    p_student=p_student,
                    latent_slot_mask=out.get("latent_slot_mask", out.get("slot_mask")),
                )
            else:
                loss_batch = torch.zeros((), device=device)
            loss_consistency = torch.zeros((), device=device)

            total = (
                lambda_ans_current * loss_ans
                + cfg.loss.lambda_sft * loss_sft
                + cfg.loss.lambda_cf * loss_cf_det
                + cfg.loss.lambda_batch * loss_batch
                + cfg.loss.lambda_consistency * loss_consistency
            )

        total.backward()
        step_cuda_max_mem = (
            float(torch.cuda.max_memory_allocated(device)) / float(1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )

        log_sums["loss"] += float(total.detach().cpu())
        log_sums["ans"] += float(loss_ans.detach().cpu())
        log_sums["sft"] += float(loss_sft.detach().cpu())
        log_sums["sft_ce"] += float(loss_sft_ce.detach().cpu())
        log_sums["sft_no_answer_latent"] += float(loss_sft_no_answer_latent.detach().cpu())
        log_sums["sft_latent_p_ans"] += float(sft_latent_p_ans.detach().cpu())
        log_sums["cf_det"] += float(loss_cf_det.detach().cpu())
        log_sums["batch"] += float(loss_batch.detach().cpu())
        log_count += 1

        if p_student is not None:
            log_sums["eff_vocab"] += float(effective_vocab_size(p_student.detach()).mean().cpu())
            log_eff_count += 1

        if (step + 1) % cfg.train.grad_accum == 0:
            assert optimizer is not None
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % cfg.train.print_every == 0:
            denom = max(log_count, 1)
            msg = (
                f"step {step + 1} | "
                f"loss={log_sums['loss'] / denom:.4f} "
                f"ans={log_sums['ans'] / denom:.4f} "
                f"sft_total={log_sums['sft'] / denom:.4f} "
                f"sft_ce={log_sums['sft_ce'] / denom:.4f} "
                f"sft_no_answer_latent={log_sums['sft_no_answer_latent'] / denom:.4f} "
                f"sft_latent_p_ans={log_sums['sft_latent_p_ans'] / denom:.4f} "
                f"cf_det={log_sums['cf_det'] / denom:.4f} "
                f"batch={log_sums['batch'] / denom:.4f} "
                f"tau={g_tau:.4f} "
                f"stage={current_stage} "
                f"cf_bias_scale={current_bias_scale:.3f} "
                f"lambda_ans={lambda_ans_current:.4f}"
            )
            if log_eff_count > 0:
                msg += f" | eff_vocab={log_sums['eff_vocab'] / log_eff_count:.2f}"
            if device.type == "cuda":
                msg += f" | cuda_max_mem_gb={step_cuda_max_mem:.2f}"
            _log_line(msg, log_path)

            log_sums = {
                "loss": 0.0,
                "ans": 0.0,
                "sft": 0.0,
                "sft_ce": 0.0,
                "sft_no_answer_latent": 0.0,
                "sft_latent_p_ans": 0.0,
                "cf_det": 0.0,
                "batch": 0.0,
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

        should_run_loss_eval = (
            eval_loader is not None
            and cfg.train.eval_every > 0
            and (step + 1) % cfg.train.eval_every == 0
        )
        gen_eval_every = cfg.train.eval_every * cfg.train.eval_generate_every_mult
        should_run_generation_eval = (
            eval_loader is not None
            and cfg.train.eval_every > 0
            and cfg.train.eval_generate_every_mult > 0
            and gen_eval_every > 0
            and (step + 1) % gen_eval_every == 0
        )

        if should_run_loss_eval:
            metrics = evaluate(
                model=model,
                loader=eval_loader,
                cfg=cfg,
                device=device,
                global_step=step + 1,
            )
            _log_line(
                f"eval@step {step + 1} | "
                + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "n"),
                log_path,
            )

        if should_run_generation_eval:
            evaluate_generate_table(
                model=model,
                loader=eval_loader,
                tokenizer=tokenizer,
                cfg=cfg,
                device=device,
                step=step + 1,
                pad_token_id=pad_token_id,
            )

        if should_run_loss_eval or should_run_generation_eval:
            model.train()

        step += 1


if __name__ == "__main__":
    train(Config())
