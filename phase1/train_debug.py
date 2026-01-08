"""
Phase 1 debug training loop.
- Train for EXACTLY 1000 batches
- Eval
- Save
- Reload from scratch
- Eval again
- Assert identical accuracy
"""

from typing import List
import os
import json
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from .config import Phase1Config
    from .dataset import Phase1Dataset, collate_fn
    from .loss import AnswerLoss
    from .eval import Evaluator
    from .model import Phase1CoconutModel
    from .split_logic import split_thoughts
except ImportError:
    from config import Phase1Config
    from dataset import Phase1Dataset, collate_fn
    from loss import AnswerLoss
    from eval import Evaluator
    from model import Phase1CoconutModel
    from split_logic import split_thoughts

from transformers import AutoTokenizer
from phase_0.model import Phase0Model
from datasets import load_dataset

LATENT_TOKEN = "<|latent|>"
ANSWER_TOKEN = "<ANSWER>"
PAD_TOKEN = "<|pad|>"
MAX_BATCHES = 1000


# ------------------------
# Setup
# ------------------------

def setup_tokenizer_and_model(config: Phase1Config):
    tokenizer = AutoTokenizer.from_pretrained(config.phase0_repo)

    if LATENT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([LATENT_TOKEN])
    if ANSWER_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([ANSWER_TOKEN])

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    phase0 = Phase0Model.from_pretrained(config.phase0_repo, torch_dtype=config.torch_dtype)
    phase0.model.resize_token_embeddings(len(tokenizer))

    latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    model = Phase1CoconutModel(
        phase0_model=phase0,
        latent_token_id=latent_id,
    )
    return tokenizer, model


# ------------------------
# Save / Load
# ------------------------

def save_phase1_checkpoint(*, model, tokenizer, config):
    os.makedirs(config.log_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(config.log_dir, "phase1_weights.pt"),
    )

    tokenizer.save_pretrained(config.log_dir)

    meta = {
        "phase": 1,
        "architecture": "Phase1CoconutModel",
        "phase0_repo": config.phase0_repo,
        "latent_token": LATENT_TOKEN,
        "answer_token": ANSWER_TOKEN,
        "max_thoughts": config.max_thoughts,
    }

    with open(os.path.join(config.log_dir, "phase1_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_phase1_from_scratch(config):
    tokenizer = AutoTokenizer.from_pretrained(config.log_dir)
    phase0 = Phase0Model.from_pretrained(config.phase0_repo, torch_dtype=config.torch_dtype)
    phase0.model.resize_token_embeddings(len(tokenizer))

    latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    model = Phase1CoconutModel(
        phase0_model=phase0,
        latent_token_id=latent_id,
    )

    state_dict = torch.load(
        os.path.join(config.log_dir, "phase1_weights.pt"),
        map_location="cpu",
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return tokenizer, model


# ------------------------
# Training
# ------------------------

def train_phase1(config: Phase1Config):
    ds = load_dataset(config.dataset_name)
    train_items = list(ds[config.dataset_train_split])
    val_items = list(ds[config.dataset_eval_split])

    # Preprocess
    train_items = [
        {
            **rec,
            "thoughts": split_thoughts(rec["generated_answer"]),
            "K": len(split_thoughts(rec["generated_answer"])),
        }
        for rec in train_items
        if rec.get("generated_answer") is not None
    ]

    val_items = train_items[:1000]

    tokenizer, model = setup_tokenizer_and_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = AnswerLoss(keep_prob=[1.0] * 5)

    evaluator = Evaluator(
        max_length=config.max_length,
        batch_size=64,
        max_thoughts=config.max_thoughts,
    )

    def num_latent_fn(K): return K

    dataset = Phase1Dataset(
        items=train_items,
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_latent_fn=num_latent_fn,
        max_thoughts=config.max_thoughts,
        answer_token=ANSWER_TOKEN,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # ------------------------
    # Train EXACTLY 1000 batches
    # ------------------------

    for step, batch in enumerate(loader, start=1):
        if step > MAX_BATCHES:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = loss_fn.compute(out["logits"], batch["digit_labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"[Train] step={step} loss={loss.item():.4f}")

    # ------------------------
    # Eval BEFORE save
    # ------------------------

    model.eval()
    acc_before = evaluator.compute_accuracy(model, tokenizer, val_items, stage=8)
    print(f"VAL ACC (before save): {acc_before:.6f}")

    # ------------------------
    # Save
    # ------------------------

    save_phase1_checkpoint(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    # ------------------------
    # Reload + Eval
    # ------------------------

    tokenizer2, model2 = load_phase1_from_scratch(config)
    model2.to(device)

    acc_after = evaluator.compute_accuracy(model2, tokenizer2, val_items, stage=8)
    print(f"VAL ACC (after reload): {acc_after:.6f}")

    # ------------------------
    # Invariant check
    # ------------------------

    assert abs(acc_before - acc_after) < 1e-6, \
        "❌ Accuracy mismatch after reload!"

    print("✅ SAVE / LOAD INVARIANT PASSED")


if __name__ == "__main__":
    cfg = Phase1Config()
    train_phase1(cfg)
