import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from typing import List, Dict, Any

from transformers import AutoTokenizer

from .dataset import Phase0Dataset
from .model import Phase0Model, Phase0Config


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────


def collate_phase0(batch: List[Dict[str, Any]]):
    """
    Pads input_ids and attention_mask to max length in batch.
    """
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    digit_labels = torch.stack([b["digit_labels"] for b in batch])

    # Pad
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0,  # tokenizer.pad_token_id (safe for causal LM)
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "digit_labels": digit_labels,
    }


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_digit_accuracy(logits, labels):
    """
    logits: [B, 5, 10]
    labels: [B, 5]
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float()
    return correct.mean(dim=0)  # [5]


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_correct = torch.zeros(5, device=device)
    total_count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        digit_labels = batch["digit_labels"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        preds = out["logits"].argmax(dim=-1)  # [B, 5]
        total_correct += (preds == digit_labels).sum(dim=0)
        total_count += digit_labels.size(0)

    acc = total_correct / total_count
    model.train()
    return acc


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='phase_0/conf.yaml')
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [cfg["model"]["answer_token"]]}
    )

    answer_token_id = tokenizer.convert_tokens_to_ids(
        cfg["model"]["answer_token"]
    )
    aid = tokenizer.convert_tokens_to_ids(cfg["model"]["answer_token"])
    assert aid != tokenizer.unk_token_id, "<ANSWER> is not a real token"

    # ---- Dataset ----
    full_ds = Phase0Dataset(
        hf_name=cfg["dataset"]["hf_name"],
        split=cfg["dataset"]["split"],
        tokenizer=tokenizer,
        max_length=cfg["dataset"]["max_length"],
    )

    # ---- Train / Eval split (95% / 5%) ----
    eval_frac = 0.05
    n_total = len(full_ds)
    n_eval = int(n_total * eval_frac)
    n_train = n_total - n_eval

    train_ds, eval_ds = random_split(
        full_ds,
        [n_train, n_eval],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_phase0,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_phase0,
    )

    print(
        f"[Dataset] train={len(train_ds)} eval={len(eval_ds)} "
        f"({eval_frac:.0%} eval)"
    )

    # ---- Model config ----
    model_config = Phase0Config(
        base_model_name=cfg["model"]["name"],
        answer_token=cfg["model"]["answer_token"],
        answer_token_id=answer_token_id,
        unfrozen_layer_pct=cfg["training"]["unfrozen_layer_pct"],
    )

    # ---- Model ----
    model = Phase0Model(model_config)
    model.model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # ---- Optimizer (only trainable params) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=cfg["training"]["lr"],
    )

    # ---- Training loop ----
    global_step = 0
    model.train()

    for epoch in range(cfg["training"]["epochs"]):
        print(f"\n=== Epoch {epoch + 1}/{cfg['training']['epochs']} ===")

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            digit_labels = batch["digit_labels"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                digit_labels=digit_labels,
            )

            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()

            if "gradient_clip" in cfg["training"]:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    cfg["training"]["gradient_clip"],
                )

            optimizer.step()

            if global_step % cfg["logging"]["log_every"] == 0:
                with torch.no_grad():
                    acc = compute_digit_accuracy(
                        out["logits"],
                        digit_labels,
                    )

                acc_str = " | ".join(
                    f"d{i}:{acc[i].item():.3f}" for i in range(5)
                )

                print(
                    f"step={global_step:06d} "
                    f"loss={loss.item():.4f} "
                    f"{acc_str}"
                )
                print(i)

            global_step += 1
        if i==10:
            break
        # ---- Evaluation after epoch ----
        eval_acc = evaluate(model, eval_loader, device)

        eval_acc_str = " | ".join(
            f"d{i}:{eval_acc[i].item():.3f}" for i in range(5)
        )
        out_dir = Path(cfg.get("output_dir", f"phase0_ckpt_epoch_{epoch+1}"))
        out_dir.mkdir(parents=True, exist_ok=True)



        print(f"[EVAL] {eval_acc_str}")
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

    # ---- Save (HF-native, single checkpoint) ----
    out_dir = Path(cfg.get("output_dir", "phase0_ckpt"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"\nSaved Phase 0 checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
