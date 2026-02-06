from __future__ import annotations

import json
import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from .conf import Config
from .dataset import UnifiedDataset, collate_fn, build_rebalanced_sampler
from .loss import AnswerDigitLoss, self_distill_z_kl_loss, CounterfactualAnswerLoss, usage_shaping_loss_stub
from .model import UnifiedZSoftModel
from .utils import set_seed, effective_vocab_size
from .eval import evaluate_on_loader


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = UnifiedZSoftModel.from_phase1(
        cfg.model.phase1_dir,
        v_z=cfg.model.v_z,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = bundle.tokenizer
    model = bundle.model
    model.train()

    dataset = UnifiedDataset(
        tokenizer=tokenizer,
        latent_token_id=model.latent_token_id,
        answer_token_id=model.answer_token_id,
        k_max=cfg.data.k_max,
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        data_path=cfg.data.data_path,
        max_length=cfg.data.max_length,
    )
    assert dataset.k_max == cfg.data.k_max, "dataset.k_max must match cfg.data.k_max"

    if cfg.data.rebalance_train:
        sampler = build_rebalanced_sampler(dataset, cfg.data.target_k_dist)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        drop_last=False,
    )

    eval_loader = None
    if cfg.train.eval_every > 0:
        eval_dataset = UnifiedDataset(
            tokenizer=tokenizer,
            latent_token_id=model.latent_token_id,
            answer_token_id=model.answer_token_id,
            k_max=cfg.data.k_max,
            dataset_name=cfg.data.dataset_name,
            split=cfg.data.eval_split,
            data_path=cfg.data.data_path,
            max_length=cfg.data.max_length,
        )

        eval_loader = DataLoader(
            eval_dataset,
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

    cf_loss_fn = CounterfactualAnswerLoss(
        permute_prob=cfg.loss.counterfactual_schedule,
        digit_temperature=cfg.loss.digit_temperature,
    )
    ans_loss_fn = AnswerDigitLoss(keep_prob=cfg.loss.keep_prob)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    log_sums = {
        "loss": 0.0,
        "ans": 0.0,
        "softz": 0.0,
        "cf": 0.0,
        "eff_vocab": 0.0,
    }
    log_count = 0
    log_eff_count = 0

    loader_iter = iter(loader)
    while step < cfg.train.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        digit_labels = batch["digit_labels"].to(device)
        k_vals = batch["K"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_self_teacher=cfg.model.use_self_teacher,
                tau_teacher=cfg.model.tau_teacher,
                tau_student=cfg.model.tau_student,
                return_distributions=True,
            )

            digit_logits = out["digit_logits"]
            p_student = out.get("p_student")
            q_teacher = out.get("q_teacher")

            loss_ans = ans_loss_fn(digit_logits, digit_labels)

            mask = (torch.arange(p_student.size(1), device=device)[None, :] < k_vals[:, None]).float()
            loss_softz = self_distill_z_kl_loss(p_student, q_teacher, mask=mask)


            if cfg.loss.lambda_cf > 0 and p_student is not None:
                loss_cf = cf_loss_fn(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    digit_logits_ref=digit_logits,
                    p_z=p_student,
                    k_vals=k_vals,
                )
            else:
                loss_cf = torch.tensor(0.0, device=device)

            loss_usage = usage_shaping_loss_stub(device=device)

            total = (
                cfg.loss.lambda_ans * loss_ans
                + cfg.loss.lambda_softz * loss_softz
                + cfg.loss.lambda_cf * loss_cf
                + cfg.loss.lambda_usage * loss_usage
            )

        total.backward()

        log_sums["loss"] += float(total.detach().cpu())
        log_sums["ans"] += float(loss_ans.detach().cpu())
        log_sums["softz"] += float(loss_softz.detach().cpu())
        log_sums["cf"] += float(loss_cf.detach().cpu())
        log_count += 1
        if p_student is not None:
            log_sums["eff_vocab"] += float(effective_vocab_size(p_student.detach()).mean().item())
            log_eff_count += 1

        if (step + 1) % cfg.train.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % cfg.train.print_every == 0:
            denom = max(log_count, 1)
            parts = {
                "loss": log_sums["loss"] / denom,
                "ans": log_sums["ans"] / denom,
                "softz": log_sums["softz"] / denom,
                "cf": log_sums["cf"] / denom,
            }
            msg = (
                f"step {step + 1} | "
                f"loss={parts['loss']:.4f} ans={parts['ans']:.4f} "
                f"softz={parts['softz']:.4f} cf={parts['cf']:.4f}"
            )
            if log_eff_count > 0:
                msg += f" | eff_vocab={log_sums['eff_vocab'] / log_eff_count:.2f}"
            print(msg)
            log_sums = {
                "loss": 0.0,
                "ans": 0.0,
                "softz": 0.0,
                "cf": 0.0,
                "eff_vocab": 0.0,
            }
            log_count = 0
            log_eff_count = 0

        if (step + 1) % cfg.train.save_every == 0:
            _save_checkpoint(
                output_dir=cfg.train.output_dir,
                step=step + 1,
                model=model,
                tokenizer=tokenizer,
                config=cfg,
            )

        if eval_loader is not None and (step + 1) % cfg.train.eval_every == 0:
            metrics = evaluate_on_loader(model=model, loader=eval_loader, cfg=cfg, device=device)
            print(
                f"eval@step {step + 1} | "
                f"loss={metrics['loss']:.4f} ans={metrics['ans']:.4f} "
                f"softz={metrics['softz']:.4f} cf={metrics['cf']:.4f}"
            )
            model.train()

        step += 1


if __name__ == "__main__":
    train(Config())
