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
    # ------------------------
    # Load + preprocess
    # ------------------------
    ds = load_dataset(config.dataset_name)
    train_items = list(ds[config.dataset_train_split])
    val_items = list(ds[config.dataset_eval_split])

    print(f"[Data] Loaded HF dataset. Train={len(train_items)}, Eval={len(val_items)}")

    # These must exist in your file (you already have them in the big train.py)
    train_items = preprocess_items(train_items, config.max_thoughts)
    val_items = preprocess_items(val_items, config.max_thoughts)
    print(f"[Data] After filtering K > {config.max_thoughts}: Train={len(train_items)}, Eval={len(val_items)}")

    train_items = clear_in_range(train_items)
    val_items = clear_in_range(val_items)
    print(f"[Data] After filtering 0 <= answer < 100000: Train={len(train_items)}, Eval={len(val_items)}")

    val_items = val_items[:1000]  # keep your speed limit

    # ------------------------
    # Setup model/tokenizer
    # ------------------------
    tokenizer, model = setup_tokenizer_and_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # ------------------------
    # Loss / opt / evaluator
    # ------------------------
    keep_prob = _load_keep_prob(getattr(config, "keep_prob_path", ""))
    loss_fn = AnswerLoss(keep_prob=keep_prob)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    evaluator = Evaluator(
        max_length=config.max_length,
        batch_size=getattr(config, "eval_batch_size", 64),
        max_thoughts=config.max_thoughts,
    )

    # ------------------------
    # Build ONE training loader (stage=8 behavior)
    # ------------------------
    # stage=8 => use K latent tokens (your evaluator uses K at stage 8)
    def num_latent_fn(K: int) -> int:
        return int(K)

    dataset = Phase1Dataset(
        items=train_items,
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_latent_fn=num_latent_fn,
        max_thoughts=config.max_thoughts,
        answer_token=ANSWER_TOKEN,
        debug=False,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
    )

    # ------------------------
    # Train EXACTLY 1000 batches
    # ------------------------
    for step, batch in enumerate(loader, start=1):
        if step > MAX_BATCHES:
            break

        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = out.get("logits")
        if logits is None:
            continue

        loss = loss_fn.compute(logits, batch["digit_labels"])
        optimizer.zero_grad(set_to_none=True)
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
    save_phase1_checkpoint(model=model, tokenizer=tokenizer, config=config)

    # ------------------------
    # Reload from scratch + Eval again
    # ------------------------
    tokenizer2, model2 = load_phase1_from_scratch(config)
    model2.to(device)
    model2.eval()

    acc_after = evaluator.compute_accuracy(model2, tokenizer2, val_items, stage=8)
    print(f"VAL ACC (after reload): {acc_after:.6f}")

    # ------------------------
    # Invariant check
    # ------------------------
    if abs(acc_before - acc_after) >= 1e-6:
        raise RuntimeError(f"❌ Accuracy mismatch after reload! before={acc_before} after={acc_after}")

    print("✅ SAVE / LOAD INVARIANT PASSED")



if __name__ == "__main__":
    cfg = Phase1Config()
    train_phase1(cfg)
