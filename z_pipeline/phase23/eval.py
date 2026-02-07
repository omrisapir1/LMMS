from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .conf import Config
from .dataset import UnifiedDataset, collate_fn
from .loss import (
    AnswerDigitLoss,
    AnswerTokenSFTLoss,
    CounterfactualAnswerLoss,
)
from .model import UnifiedZSoftModel
from .utils import effective_vocab_size


def evaluate(
    *,
    model: UnifiedZSoftModel,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    global_step: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate an existing model on an existing eval loader.
    This is the function train.py calls every eval interval.
    """
    del global_step
    if cfg.loss.lambda_sft <= 0:
        raise ValueError("Phase23 requires AnswerTokenSFTLoss: set loss.lambda_sft > 0")

    model.eval()

    ans_loss_fn = AnswerDigitLoss(keep_prob=cfg.loss.keep_prob)
    sft_loss_fn = AnswerTokenSFTLoss(answer_token_id=model.answer_token_id)
    cf_loss_fn = CounterfactualAnswerLoss(
        permute_prob=cfg.loss.counterfactual_schedule,
        digit_temperature=cfg.loss.digit_temperature,
        seed=cfg.train.seed,
    )

    totals: Dict[str, float] = {"n": 0.0, "eff_vocab": 0.0}
    if cfg.loss.lambda_ans > 0:
        totals["ans"] = 0.0
    if cfg.loss.lambda_sft > 0:
        totals["sft"] = 0.0
    if cfg.loss.lambda_cf > 0:
        totals["cf"] = 0.0
    if cfg.loss.lambda_batch > 0:
        totals["batch"] = 0.0
    if cfg.loss.lambda_consistency > 0:
        totals["consistency"] = 0.0
    totals["loss"] = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            digit_labels = batch["digit_labels"].to(device)
            k_vals = batch["K"].to(device)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gumbel_tau=cfg.model.gumbel_tau_end,
                    use_gs=False,
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

            bsz = float(input_ids.size(0))
            totals["loss"] += float(total.detach().cpu()) * bsz
            if "ans" in totals:
                totals["ans"] += float(loss_ans.detach().cpu()) * bsz
            if "sft" in totals:
                totals["sft"] += float(loss_sft.detach().cpu()) * bsz
            if "cf" in totals:
                totals["cf"] += float(loss_cf.detach().cpu()) * bsz
            if "batch" in totals:
                totals["batch"] += float(loss_batch.detach().cpu()) * bsz
            if "consistency" in totals:
                totals["consistency"] += float(loss_consistency.detach().cpu()) * bsz
            if p_student is not None:
                totals["eff_vocab"] += float(effective_vocab_size(p_student.detach()).mean().cpu()) * bsz
            totals["n"] += bsz

    n = max(totals["n"], 1.0)
    metrics: Dict[str, float] = {k: v / n for k, v in totals.items() if k != "n"}
    metrics["n"] = float(totals["n"])
    return metrics


def evaluate_from_config(cfg: Config) -> Dict[str, float]:
    """Standalone eval entrypoint that loads model + eval dataset from config."""
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

    ds = UnifiedDataset(
        tokenizer=tokenizer,
        latent_token_id=model.latent_token_id,
        answer_token_id=model.answer_token_id,
        k_max=cfg.data.k_max,
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.eval_split,
        data_path=cfg.data.data_path,
        max_length=cfg.data.max_length,
    )
    assert ds.k_max == cfg.data.k_max, "eval dataset.k_max must match cfg.data.k_max"

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id")
        pad_token_id = tokenizer.eos_token_id

    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        drop_last=False,
    )
    return evaluate(model=model, loader=loader, cfg=cfg, device=device)


if __name__ == "__main__":
    metrics = evaluate_from_config(Config())
    print(metrics)
