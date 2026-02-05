# phase2/train.py
#
# Phase-2 training:
#   Learn Z-token embeddings + Z-selector from digit loss ONLY.
#
# Direct handoff contract (NO DISK):
#   def run_experiment(cfg):
#       phase2_ckpt = run_phase2(cfg.phase2)
#       phase3_metrics = run_phase3(cfg.phase3, phase2_ckpt)
#       return phase3_metrics
#
# phase2_ckpt = {
#   "model": Phase2ZModel,
#   "tokenizer": tokenizer,
#   "z_token_ids": List[int],      # ids for <Z_0>.. <Z_{V-1}>
#   "z_vocab_size": int,
# }
#
from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from .conf import Phase2Config, Phase2PretrainConfig
from .dataset import Phase2Dataset, phase2_collate_fn, compute_keep_prob_from_dataset
from .loss import AnswerLoss, ZUsageKLLoss, RowZDiversityLoss
from .eval import evaluate_phase2
from .model import Phase2ZModel
from .conf import Phase2DataConfig
from .clustering import collect_latents_for_kmeans, kmeans_pp_deterministic, collect_row_representatives, kmeans_pp_row_aware

# -----------------------------
# Utils
# -----------------------------

def z_token_str(i: int) -> str:
    return f"<Z_{i}>"


def set_seed_best_effort(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_from_model(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def compute_temperature(step: int, cfg: Phase2Config) -> float:
    sched = cfg.temp
    t0 = float(sched.temp_start)
    t1 = float(sched.temp_end)
    n = int(sched.anneal_steps)

    if step <= 0:
        return t0

    if step >= n:
        return t1

    # progress in [0,1]
    p = step / max(1, n)

    if sched.type == "linear":
        return (1.0 - p) * t0 + p * t1

    if sched.type == "cosine":
        # cosine from t0 -> t1
        c = 0.5 * (1.0 + math.cos(math.pi * p))  # 1 -> 0
        return t1 + (t0 - t1) * c

    if sched.type == "exponential":
        # t = t0 * (t1/t0)^p
        ratio = t1 / max(t0, 1e-12)
        return t0 * (ratio ** p)

    raise ValueError(f"Unknown temperature schedule type: {sched.type}")


class _EvalTemperatureProxy:
    """
    eval.py calls: model(..., temperature=None, return_z_probs=True)

    Your Phase2ZModel forward expects a positive float temperature during training.
    For eval, we want argmax-ish behavior; easiest is to replace None with tiny temp.
    """
    def __init__(self, model: torch.nn.Module, eval_temp: float = 1e-6):
        self._m = model
        self._eval_temp = float(eval_temp)

    def __getattr__(self, name):
        return getattr(self._m, name)

    def __call__(self, *args, **kwargs):
        if kwargs.get("temperature", None) is None:
            kwargs["temperature"] = self._eval_temp
        return self._m(*args, **kwargs)


# -----------------------------
# Main entry point
# -----------------------------


def print_top_z(dist, topk=10, title="Z distribution"):
    idx = np.argsort(dist)[::-1][:topk]
    print(f"\n{title} (top {topk}):")
    for i in idx:
        if dist[i] > 0:
            print(f"  Z_{i:4d}: {dist[i]:.4f}")

def straight_through_argmax(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, K, V]
    returns: LongTensor[B, K] with straight-through gradient
    """
    with torch.no_grad():
        hard = torch.argmax(logits, dim=-1)

    one_hot = torch.zeros_like(logits).scatter_(-1, hard.unsqueeze(-1), 1.0)
    probs = torch.softmax(logits, dim=-1)

    # straight-through estimator
    st = one_hot + probs - probs.detach()
    return st


def pretrain_z_autoencoder(
    *,
    model: Phase2ZModel,
    train_loader: DataLoader,
    cfg: Phase2PretrainConfig,
    device: torch.device,
) -> None:
    """
    Phase-2a: Z auto-encoder warmup.
    Learns z_selector + Z embeddings to reconstruct latent_states.
    """
    model.train()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.z_selector.parameters()},
            {"params": model.base.get_input_embeddings().weight},
        ],
        lr=cfg.lr,
        weight_decay=0.0,
    )

    loader_iter = iter(train_loader)
    for i in range(2):
        if i:
            temp = 1.0
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.z_selector.parameters()},
                    {"params": model.base.get_input_embeddings().weight},
                ],
                lr=cfg.lr * 0.001,
                weight_decay=0.0,
            )
        else:
            temp = cfg.temperature

        print(f"[Z-AE pretrain] step={i} temp={temp:.2f}")
        for step in range(cfg.steps):

            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            latent_states = batch["latent_states"].to(device)   # [B, Kmax, H]
            z_mask = batch["z_mask"].to(device)                 # [B, Kmax]

            z_probs = model.z_autoencoder_forward(
                latent_states=latent_states,
                z_mask=z_mask,
                temperature=temp,
            )
            st_assign = straight_through_argmax(z_probs)         # [B, K, V]

            # reconstruct h
            z_emb_table = model.base.get_input_embeddings().weight[
                torch.tensor(model.z_token_ids, device=device)
            ]                                                    # [V, H]

            h_recon = torch.einsum("bkv,vh->bkh", st_assign, z_emb_table)

            # masked MSE
            diff = h_recon - latent_states
            loss = (diff[z_mask] ** 2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"[Z-AE pretrain] step={step} loss={loss.item():.6f}")

        print("Z autoencoder pretraining complete.")


def push_phase2_to_hf(
    phase2_ckpt: dict,
    cfg: Phase2Config,
    repo_id: str,
    hf_token: str,
):
    """
    Push Phase-2 artifact to HuggingFace.
    """
    from huggingface_hub import HfApi
    import json
    import tempfile
    import os

    model = phase2_ckpt["model"]
    tokenizer = phase2_ckpt["tokenizer"]

    z_meta = {
        "z_token_ids": phase2_ckpt["z_token_ids"],
        "z_vocab_size": phase2_ckpt["z_vocab_size"],
        "latent_token_id": cfg.latent_token_id,
        "answer_token_id": cfg.answer_token_id,
    }

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(os.path.join(tmp, "tokenizer"))

        with open(os.path.join(tmp, "z_meta.json"), "w") as f:
            json.dump(z_meta, f, indent=2)


        api = HfApi(token=hf_token)
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            repo_type="model",
        )




def run_phase2(cfg: Phase2Config, hf_token: str):
    """
    
    Two-stage training:
      Stage 1 (anneal): temp from cfg.temp.temp_start -> cfg.temp.temp_end, NO digit loss, digit heads frozen
      Stage 2 (cooldown): temp fixed at cfg.temp.temp_end (should be 1), digit loss ON, digit heads trainable
    """
    # Do not finalize yet; we need tokenizer-derived IDs first.
    set_seed_best_effort(cfg.seed)

    # Load Phase-1 tokenizer + model
    from z_pipeline.shared.load_model_phase1 import load_phase1, LATENT_TOKEN  # type: ignore

    device_str = "cuda" if torch.cuda.is_available() else "cpu"


    tokenizer, phase1_model, meta = load_phase1(device=device_str)

    # Ensure pad token id exists
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad or eos token; set pad token explicitly.")
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve token IDs directly from tokenizer
    latent_token_str = LATENT_TOKEN
    latent_token_id = tokenizer.convert_tokens_to_ids(latent_token_str)
    if latent_token_id is None or latent_token_id < 0:
        raise RuntimeError(f"Could not resolve latent token id for '{latent_token_str}' from tokenizer")

    answer_token_str = "<ANSWER>"
    answer_token_id = tokenizer.convert_tokens_to_ids(answer_token_str)
    if answer_token_id is None or answer_token_id < 0:
        raise RuntimeError("Could not resolve answer token id '<ANSWER>' from tokenizer")

    # Write back to cfg for downstream consumers, and set default data config
    cfg.latent_token_id = int(latent_token_id)
    cfg.answer_token_id = int(answer_token_id)
    if cfg.data is None:
        cfg.data = Phase2DataConfig()

    # Finalize once now that IDs are set
    cfg = cfg.finalize()

    # Pull phase-0 components from phase-1
    phase0 = phase1_model.phase0
    base_lm = phase0.model
    digit_heads = phase0.digit_heads

    # Expand tokenizer with Z tokens
    z_tokens = [z_token_str(i) for i in range(cfg.z_vocab_size)]
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in z_tokens if t not in existing]
    if to_add:
        tokenizer.add_tokens(to_add, special_tokens=False)

    # Resize base model embeddings to tokenizer size
    base_lm.resize_token_embeddings(len(tokenizer))

    # Map Z tokens to ids
    z_token_ids: list[int] = []
    for i in range(cfg.z_vocab_size):
        tid = tokenizer.convert_tokens_to_ids(z_token_str(i))
        if tid is None or tid < 0:
            raise RuntimeError(f"Failed to convert token to id: {z_token_str(i)}")
        z_token_ids.append(int(tid))

    # Build Phase-2 model
    model = Phase2ZModel(
        base_lm=base_lm,
        digit_heads=digit_heads,
        answer_token_id=answer_token_id,
        latent_token_id=latent_token_id,
        z_token_ids=z_token_ids,
        freeze_base=True,
        freeze_digit_heads=True,  # we will explicitly enable later for stage 2
        force_base_eval=cfg.force_base_eval,
    )

    # Ensure all submodules live on the same device as Phase-1
    device = next(base_lm.parameters()).device
    model.to(device)

    # Dataset
    train_ds = Phase2Dataset(
        tokenizer=tokenizer,
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        k_max=cfg.data.k_max,
        latent_token_id=latent_token_id,
        answer_token_id=answer_token_id,
        seed=cfg.seed,
        rebalance_train=cfg.data.rebalance_train,
        max_length=cfg.data.max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        collate_fn=phase2_collate_fn,
    )

    X, row_ptr = collect_latents_for_kmeans(train_ds)
    X_rows = collect_row_representatives(train_ds)
    centroids = kmeans_pp_row_aware(X, X_rows,  cfg.z_vocab_size, n_iters=cfg.cluster.n_iter,row_ptr=row_ptr,assign_mode='balanced_assign', seed=42)
    model.initialize_from_centroids(centroids)

    # Losses
    keep_prob = cfg.loss.keep_prob or compute_keep_prob_from_dataset(train_ds)
    answer_loss_fn = AnswerLoss(keep_prob=keep_prob)
    z_kl_loss_fn = ZUsageKLLoss(vocab_size=cfg.z_vocab_size)
    row_z_loss_fn = RowZDiversityLoss()
    print(f"AnswerLoss keep_prob: {keep_prob}")

    # Optimizer (we keep digit-head params in the optimizer, but gate training via requires_grad)
    optim_params = [
        {"params": list(model.z_selector.parameters()), "weight_decay": cfg.optim.weight_decay},
        # {"params": [model.base.get_input_embeddings().weight], "weight_decay": cfg.optim.weight_decay},
        # {"params": list(model.digit_heads.parameters()), "weight_decay": cfg.optim.weight_decay},
    ]
    optimizer = AdamW(
        optim_params,
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )

    # Training control
    eval_every = int(cfg.eval.eval_every_steps)
    print_every = int(cfg.print_every)

    anneal_steps = int(cfg.temp.anneal_steps)
    cooldown_steps = int(cfg.temp.cooldown_steps)
    max_steps = anneal_steps + cooldown_steps

    print("\n================ Phase-2 Schedule =================")
    print(f"Stage 1 (anneal):   steps [0 .. {max(0, anneal_steps - 1)}], temp {cfg.temp.temp_start} -> {cfg.temp.temp_end}, digit loss OFF")
    print(f"Stage 2 (cooldown): steps [{anneal_steps} .. {max_steps - 1}], temp = {cfg.temp.temp_end}, digit loss ON")
    print("===================================================\n")

    # Stage 1: ensure digit heads are frozen (even though model was built with freeze_digit_heads=True)
    for p in model.digit_heads.parameters():
        p.requires_grad = False

    global_step = 0
    model.train()

    # Loop
    cur_answer_loss, cur_kl_loss, cur_row_loss = 0.0, 0.0, 0.0
    loader_iter = iter(train_loader)
    temp = float("nan")

    while global_step < max_steps:
        # Stage transition at boundary
        if global_step == anneal_steps:
            print("\n=== Entering Stage 2 (cooldown): enabling digit heads + digit loss ===")
            for p in model.digit_heads.parameters():
                p.requires_grad = True

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        # Temperature schedule (explicit two-stage semantics)
        if global_step < anneal_steps:
            new_temp = compute_temperature(global_step, cfg)
        else:
            new_temp = float(cfg.temp.temp_end)

        # Print when temp changes by ~0.1 to avoid spam
        if not np.isfinite(temp) or (int(abs(new_temp) * 10) % 10) != (int(abs(temp) * 10) % 10):
            print(f"Step {global_step}: temperature={new_temp:.2f}")
        temp = float(new_temp)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        latent_states = batch["latent_states"].to(device, non_blocking=True)
        z_mask = batch["z_mask"].to(device, non_blocking=True)
        digit_labels = batch["digit_labels"].to(device, non_blocking=True)

        try:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_states=latent_states,
                z_mask=z_mask,
                temperature=float(temp),
                return_z_probs=True,
            )
        except Exception as e:
            print(f"Error at step {global_step}: {e}")
            torch.cuda.empty_cache()
            global_step += 1
            continue

        digit_logits = out["digit_logits"]
        z_probs = out["z_probs"]

        # Always compute these
        loss_kl = z_kl_loss_fn.compute(z_probs, z_mask)
        loss_row = row_z_loss_fn.compute(z_probs, z_mask)

        loss_answer = answer_loss_fn.compute(digit_logits, digit_labels)
        loss = (
            cfg.loss.lambda_answer * loss_answer
            + cfg.loss.lambda_kl * loss_kl
            + cfg.loss.lambda_row * loss_row
        )
        loss_answer_val = float(loss_answer.item())

        # Accumulate for printing
        cur_answer_loss += loss_answer_val
        cur_kl_loss += float(loss_kl.item())
        cur_row_loss += float(loss_row.item())

        if global_step % print_every == 0:
            denom = max(1, print_every)
            loss_ans_to_print = cur_answer_loss / denom
            loss_kl_to_print = cur_kl_loss / denom
            loss_row_to_print = cur_row_loss / denom
            stage = "anneal" if global_step < anneal_steps else "cooldown"
            print(
                f"Step {global_step} [{stage}] : "
                f"loss_answer={loss_ans_to_print:.4f} "
                f"loss_kl={loss_kl_to_print:.4f} "
                f"loss_row={loss_row_to_print:.4f}"
            )
            cur_answer_loss = 0.0
            cur_kl_loss = 0.0
            cur_row_loss = 0.0

        optimizer.zero_grad(set_to_none=True)
        try:
            loss.backward()
        except Exception as e:
            print(f"Backward error at step {global_step}: {e}")
            torch.cuda.empty_cache()
            global_step += 1
            continue

        if cfg.optim.max_grad_norm is not None and cfg.optim.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.max_grad_norm))

        optimizer.step()

        # Eval is diagnostic only
        if global_step % eval_every == 0 or global_step == max_steps - 1:
            eval_model = _EvalTemperatureProxy(model, eval_temp=1e-6)
            num_iterates = 2 if global_step == max_steps - 1 else 1

            for i in range(num_iterates):
                dataset_mode = 'eval' if i == 0 else 'train'
                metrics = evaluate_phase2(
                    model=eval_model,
                    tokenizer=tokenizer,
                    dataset_name=cfg.data.dataset_name,
                    batch_size=cfg.data.eval_batch_size,
                    latent_token_id=latent_token_id,
                    answer_token_id=answer_token_id,
                    k_max=cfg.data.k_max,
                    device=device,
                    dataset_mode=dataset_mode
                )
                print(f"\n================ Phase-2 Evaluation (diagnostic) {dataset_mode}================")

                for mode_name, m in metrics["modes"].items():
                    print(f"\n--- Z selection mode: {mode_name} ---")

                    # Global metrics
                    print(f"Digit EM (overall): {m['digit_em'] * 100:.2f}%")
                    print(f"Effective Z vocab size: {m['effective_vocab_size']:.1f} / {cfg.z_vocab_size}")
                    print_top_z(
                        m["z_distribution_k1"],
                        topk=5,
                        title="Z distribution for K=1 rows (top 5):",
                    )

                    # Per-K diagnostics
                    print("\nPer-K diagnostics:")
                    print("-" * 60)

                    all_Ks = sorted(
                        set(m["digit_em_by_k"].keys())
                        | set(m["unique_ratio_by_k"].keys())
                        | set(m["adjacent_repeat_rate_by_k"].keys())
                    )

                    for K in all_Ks:
                        em = m["digit_em_by_k"].get(K)
                        uniq = m["unique_ratio_by_k"].get(K)
                        adj = m["adjacent_repeat_rate_by_k"].get(K)

                        print(f"K={K:2d}:")
                        if em is not None:
                            print(f"  EM              : {em * 100:6.2f}%")
                        if uniq is not None:
                            print(f"  Unique ratio    : {uniq:6.3f}")
                        if adj is not None:
                            print(f"  Adj repeat rate : {adj:6.3f}")
                        print()

            model.train()

        global_step += 1


    push_phase2_to_hf(
        phase2_ckpt={
            "model": model,
            "tokenizer": tokenizer,
            "z_token_ids": z_token_ids,
            "z_vocab_size": int(cfg.z_vocab_size),
        },
        cfg=cfg,
        repo_id=cfg.hf_repo,
        hf_token=hf_token,
    )


__all__ = ["run_phase2"]