"""
Phase 1 training loop (finalized design).
- Load train/val once from Hugging Face dataset.
- Derive thoughts and K once at load time.
- Stage logic lives in the trainer only.
- Loss only on answer digits via AnswerLoss.
- Global completion at stage 8.
"""

from typing import List
import os
import json
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import Phase1Config
from .dataset import Phase1Dataset, collate_fn
from .stage_manager import StageManager
from .loss import AnswerLoss
from .eval import Evaluator
from .model import Phase1CoconutModel
from .split_logic import split_thoughts

# Tokenizer/model setup from Phase 0
from transformers import AutoTokenizer
from phase_0.model import Phase0Model
from datasets import load_dataset

LATENT_TOKEN = "<|latent|>"
PAD_TOKEN = "<|pad|>"
ANSWER_TOKEN = "<ANSWER>"


def setup_tokenizer_and_model(config: Phase1Config):
    """
    Load tokenizer from Phase 0, add LATENT_TOKEN if missing, ensure pad_token_id, resize embeddings,
    initialize latent embedding from '.', and save tokenizer for reproducibility.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.phase0_repo)
    phase0 = Phase0Model.from_pretrained(config.phase0_repo, torch_dtype=config.torch_dtype)

    vocab = tokenizer.get_vocab()
    if LATENT_TOKEN not in vocab:
        tokenizer.add_tokens([LATENT_TOKEN])
    # Ensure <ANSWER> token exists to avoid UNK behavior during training/evaluation
    if ANSWER_TOKEN not in vocab:
        tokenizer.add_tokens([ANSWER_TOKEN])

    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            if PAD_TOKEN not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
            else:
                tokenizer.pad_token = PAD_TOKEN

    phase0.model.resize_token_embeddings(len(tokenizer))

    # Initialize latent embedding from '.' if available
    latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    src_id = tokenizer.convert_tokens_to_ids(".")
    # Optionally initialize ANSWER token embedding from '.' to stabilize early training
    answer_id = tokenizer.convert_tokens_to_ids(ANSWER_TOKEN)
    emb = phase0.model.get_input_embeddings().weight.data
    if latent_id is not None and src_id is not None and latent_id >= 0 and src_id >= 0:
        emb[latent_id].copy_(emb[src_id])
    if answer_id is not None and src_id is not None and answer_id >= 0 and src_id >= 0:
        emb[answer_id].copy_(emb[src_id])

    # Wrap Phase-0 with Phase1CoconutModel
    model = Phase1CoconutModel(
        phase0_model=phase0,
        latent_token_id=latent_id,
    )

    out_dir = getattr(config, "tokenizer_out_dir", None) or config.log_dir
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception:
        pass

    return tokenizer, model


def _load_keep_prob(path: str) -> List[float]:
    # Load keep_prob list of length 5; default to [1.0]*5 if missing/invalid
    keep_prob = [1.0] * 5
    try:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("keep_prob"), list) and len(data["keep_prob"]) == 5:
                    keep_prob = [float(x) for x in data["keep_prob"]]
    except Exception:
        # Fallback silently to defaults
        keep_prob = [1.0] * 5
    return keep_prob

def clear_in_range(items: List[dict]) -> List[dict]:
    """
    Filter out records where the answer is not in the range [0, 100000).
    """
    out = []
    for rec in items:
        ans = rec.get("answer")
        if ans is None:
            continue
        try:
            answer_int = int(ans)
            if 0 <= answer_int < 100000:
                out.append(rec)
        except ValueError:
            continue
    return out

def preprocess_items(items, max_thoughts: int):
    """
    Derive thoughts and K once:
    - thoughts = split_thoughts(generated_answer)
    - K = len(thoughts)
    - filter out K > max_thoughts

    Debug safety: raise if generated_answer is missing to surface upstream issues early.
    """
    out = []
    for rec in items:
        gen = rec.get("generated_answer")
        if gen is None:
            raise KeyError("generated_answer missing during preprocessing")

        thoughts = split_thoughts(gen)
        K = len(thoughts)

        if K > max_thoughts:
            continue

        rec2 = dict(rec)
        rec2["thoughts"] = thoughts
        rec2["K"] = K
        out.append(rec2)

    return out


def train_phase1(config: Phase1Config) -> None:
    # Load HF dataset splits once and preprocess
    ds = load_dataset(config.dataset_name)
    train_items = list(ds[config.dataset_train_split])
    val_items = list(ds[config.dataset_eval_split])
    len_train_items = len(train_items)
    len_val_items = len(val_items)
    assert len_train_items > 0, "No training items loaded"
    assert len_val_items > 0, "No validation items loaded"

    print(
        f"[Data] Loaded HF dataset. "
        f"Train split size={len_train_items}, "
        f"Eval split size={len_val_items}"
    )

    # Preprocess splits once: derive thoughts and K; filter K > max_thoughts
    train_items = preprocess_items(train_items, config.max_thoughts)
    val_items = preprocess_items(val_items, config.max_thoughts)
    print(
        f"[Data] After filtering K > {config.max_thoughts}: "
        f"Train size={len(train_items)}, Eval size={len(val_items)}"
    )

    train_items = clear_in_range(train_items)
    val_items = clear_in_range(val_items)
    print(
        f"[Data] After filtering 0 < answer < 100000: "
        f"Train size={len(train_items)}, Eval size={len(val_items)}"
    )

    # Setup tokenizer and model once
    tokenizer, model = setup_tokenizer_and_model(config)
    # Ensure model is on the correct device once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Initialize loss, optimizer, evaluator once
    keep_prob = _load_keep_prob(getattr(config, "keep_prob_path", ""))
    loss_fn = AnswerLoss(keep_prob=keep_prob)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    evaluator = Evaluator(max_length=config.max_length, batch_size=config.eval_batch_size if hasattr(config, "eval_batch_size") else 64, max_thoughts=config.max_thoughts)

    # Initialize StageManager
    sm = StageManager(config.stage_exit_thresholds)

    global_step = 0
    # Stage-driven training loop
    while True:
        s = sm.current_stage

        # num_latent_fn per current stage
        def num_latent_fn(K: int):
            if s < 8:
                assert K > s, "Training item with K <= stage leaked into stage_items"
                return s
            return K

        # Build dataset and dataloader for current stage
        # Trainer-side filtering to enforce participation rule:
        # Appears in stages 1..K-1, skips stage K, reappears in stage 8
        if s < 8:
            stage_items = [it for it in train_items if int(it.get("K", 0)) > s]
        else:
            stage_items = train_items

        print(
            f"[Stage {s}] Training with latent={s}. "
            f"Using samples with K > {s}. "
            f"Train size={len(stage_items)}"
        )
        dataset = Phase1Dataset(items=stage_items, tokenizer=tokenizer, max_length=config.max_length, num_latent_fn=num_latent_fn, max_thoughts=config.max_thoughts, answer_token=ANSWER_TOKEN)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id))

        for batch in loader:
            global_step += 1
            # Move tensors to device
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            # Forward → logits
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            # Debug-only: ensure <ANSWER> is present exactly once per sample after batching
            if global_step == 1:
                answer_id = tokenizer.convert_tokens_to_ids("<ANSWER>")
                assert (batch["input_ids"] == answer_id).sum(dim=1).eq(1).all(), \
                    "<ANSWER> missing or duplicated after batching"

            logits = out.get("logits")  # [B,5,10]
            if logits is None:
                continue

            # Loss (digits only)
            loss = loss_fn.compute(logits, batch["digit_labels"])  # scalar tensor
            print(f'[Stage {s}][Step {global_step}] Loss: {loss.item():.4f}')

            # Backward + step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Periodic evaluation on validation data only
            if global_step % config.eval_interval_batches == 0:
                model.eval()
                val_acc = evaluator.compute_accuracy(model, tokenizer, val_items, sm.current_stage)
                model.train()

                eval_id = global_step // config.eval_interval_batches
                advanced, done = sm.update_on_evaluation(eval_id=eval_id, val_acc=val_acc)

                if advanced:
                    dataset.already_been_called_to_print = False
                    # Reset optimizer on stage advance
                    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                    break  # rebuild dataset for next stage

                if done:
                    # Terminal behavior: save final model and report
                    try:
                        # Prefer save_pretrained if available
                        if hasattr(model, "save_pretrained"):
                            model.save_pretrained(config.log_dir)
                        else:
                            # Torch save as fallback
                            os.makedirs(config.log_dir, exist_ok=True)
                            torch.save(model.state_dict(), os.path.join(config.log_dir, "phase1_final.pt"))
                        # Always save tokenizer for reproducibility
                        try:
                            tokenizer.save_pretrained(config.log_dir)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    print(f"Final validation accuracy: {val_acc:.4f}")
                    return
        # Continue loop; if advanced, we rebuild dataset in next iteration


if __name__ == "__main__":
    cfg = Phase1Config()
    train_phase1(cfg)
