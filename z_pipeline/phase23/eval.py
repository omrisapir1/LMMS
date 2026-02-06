from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .conf import Config
from .dataset import UnifiedDataset, collate_fn
from .loss import AnswerDigitLoss, self_distill_z_kl_loss, CounterfactualAnswerLoss, usage_shaping_loss_stub
from .model import UnifiedZSoftModel
from .utils import effective_vocab_size


def evaluate_on_loader(
    *,
    model: UnifiedZSoftModel,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> dict:
    model.eval()
    ans_loss_fn = AnswerDigitLoss(keep_prob=cfg.loss.keep_prob)
    cf_loss_fn = CounterfactualAnswerLoss(
        permute_prob=cfg.loss.counterfactual_schedule,
        digit_temperature=cfg.loss.digit_temperature,
    )

    totals = {
        "loss": 0.0,
        "ans": 0.0,
        "softz": 0.0,
        "cf": 0.0,
        "usage": 0.0,
        "eff_vocab": 0.0,
        "n": 0,
    }

    with torch.no_grad():
        for batch in loader:
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

                if cfg.loss.lambda_softz > 0 and cfg.model.use_self_teacher and p_student is not None and q_teacher is not None:
                    mask = (torch.arange(p_student.size(1), device=device)[None, :] < k_vals[:, None]).float()
                    loss_softz = self_distill_z_kl_loss(p_student, q_teacher, mask=mask)
                else:
                    loss_softz = torch.tensor(0.0, device=device)

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

            bsz = input_ids.size(0)
            totals["loss"] += float(total.detach().cpu()) * bsz
            totals["ans"] += float(loss_ans.detach().cpu()) * bsz
            totals["softz"] += float(loss_softz.detach().cpu()) * bsz
            totals["cf"] += float(loss_cf.detach().cpu()) * bsz
            totals["usage"] += float(loss_usage.detach().cpu()) * bsz

            if p_student is not None:
                totals["eff_vocab"] += float(effective_vocab_size(p_student.detach()).mean().cpu()) * bsz
            totals["n"] += bsz

    n = max(totals["n"], 1)
    metrics = {
        "loss": totals["loss"] / n,
        "ans": totals["ans"] / n,
        "softz": totals["softz"] / n,
        "cf": totals["cf"] / n,
        "usage": totals["usage"] / n,
        "eff_vocab": totals["eff_vocab"] / n,
        "n": totals["n"],
    }

    print(
        "eval | "
        f"loss={metrics['loss']:.4f} ans={metrics['ans']:.4f} "
        f"softz={metrics['softz']:.4f} cf={metrics['cf']:.4f} "
        f"usage={metrics['usage']:.4f} eff_vocab={metrics['eff_vocab']:.2f}"
    )
    return metrics


def evaluate(cfg: Config) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = UnifiedZSoftModel.from_phase1(
        cfg.model.phase1_dir,
        v_z=cfg.model.v_z,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = bundle.tokenizer
    model = bundle.model

    dataset = UnifiedDataset(
        tokenizer=tokenizer,
        latent_token_id=model.latent_token_id,
        answer_token_id=model.answer_token_id,
        k_max=cfg.data.k_max,
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.eval_split,
        data_path=cfg.data.data_path,
        max_length=cfg.data.max_length,
    )
    assert dataset.k_max == cfg.data.k_max, "dataset.k_max must match cfg.data.k_max"

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        drop_last=False,
    )
    return evaluate_on_loader(model=model, loader=loader, cfg=cfg, device=device)


if __name__ == "__main__":
    evaluate(Config())
