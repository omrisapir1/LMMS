# phase3/generate_dataset_from_hf.py
#
# Phase-3 dataset generation FROM HF Phase-2 repo
#
# - Loads Phase-2 model + tokenizer from HuggingFace
# - Converts latent states -> explicit Z tokens
# - Saves DatasetDict (train / eval) to HF dataset repo
#

from __future__ import annotations

from typing import Dict, List

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from huggingface_hub import HfApi

try:
    from z_pipeline.phase2.dataset import Phase2Dataset, phase2_collate_fn
    from z_pipeline.phase2.model import Phase2ZModel
    from z_pipeline.phase2.conf import Phase2Config
except ImportError:
    from .z_pipeline.phase2.dataset import Phase2Dataset, phase2_collate_fn
    from .z_pipeline.phase2.model import Phase2ZModel
    from .z_pipeline.phase2.conf import Phase2Config


# ------------------------------------------------------------
# K bucket logic (EXACTLY aligned with Phase-2 / Phase-3)
# ------------------------------------------------------------

def k_to_bucket(K: int) -> str:
    if K == 1:
        return "K1"
    if K == 2:
        return "K2"
    if K == 3:
        return "K3"
    if 4 <= K <= 7:
        return "K4_7"
    if 8 <= K <= 12:
        return "K8_12"
    if 13 <= K <= 20:
        return "K13_20"
    raise ValueError(f"Unsupported num_latents K={K}")


# ------------------------------------------------------------
# Core generation (single split)
# ------------------------------------------------------------

@torch.no_grad()
def _generate_split(
    *,
    model: Phase2ZModel,
    tokenizer,
    dataset_name: str,
    split: str,
    batch_size: int,
    z_mode: str,
    temperature: float,
    device: torch.device,
) -> Dataset:

    latent_token_id = model.latent_token_id
    answer_token_id = model.answer_token_id

    ds = Phase2Dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        k_max=max(len(model.z_token_ids), 20),
        latent_token_id=latent_token_id,
        answer_token_id=answer_token_id,
        rebalance_train=False,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=phase2_collate_fn,
    )

    rows: List[Dict] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        latent_states = batch["latent_states"].to(device)
        z_mask = batch["z_mask"].to(device)
        digit_labels = batch["digit_labels"]

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_states=latent_states,
            z_mask=z_mask,
            z_mode=z_mode,
            temperature=temperature,
            return_z_probs=False,
        )

        z_ids = out["z_ids"]  # [B, Kmax]

        for b in range(input_ids.size(0)):
            K = int(z_mask[b].sum().item())
            if K <= 0:
                continue

            T = int(attention_mask[b].sum().item())
            original_ids = input_ids[b, :T].tolist()

            try:
                answer_pos = original_ids.index(answer_token_id)
                first_latent_pos = original_ids.index(latent_token_id)
            except ValueError:
                raise RuntimeError("Dataset contract violation: missing <LATENT> or <ANSWER>")

            if first_latent_pos >= answer_pos:
                raise RuntimeError("<LATENT> must precede <ANSWER>")

            latent_region = original_ids[first_latent_pos:answer_pos]
            if len(latent_region) != K or any(t != latent_token_id for t in latent_region):
                raise RuntimeError("Latent region malformed")

            prefix = original_ids[:first_latent_pos]

            z_seq = z_ids[b, :K].tolist()
            z_tokens = [model.z_token_ids[int(z)] for z in z_seq]

            new_ids = prefix + z_tokens + [answer_token_id]

            rows.append({
                "input_ids": new_ids,
                "attention_mask": [1] * len(new_ids),
                "digit_labels": digit_labels[b].tolist(),
                "num_latents": K,
                "K_bucket": k_to_bucket(K),
            })

    return Dataset.from_list(rows)


# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------

def generate_phase3_dataset_from_hf(
    *,
    phase2_repo_id: str,
    phase3_dataset_repo_id: str,
    dataset_name: str,
    batch_size: int,
    z_mode: str = "hard_sample",
    temperature: float = 1.0,
    device: str = "cuda",
    hf_token: str,
) -> None:
    """
    End-to-end Phase-3 dataset generation from HF Phase-2 repo.
    """

    device = torch.device(device)

    # ---- load tokenizer + model ----
    tokenizer = AutoTokenizer.from_pretrained(
        phase2_repo_id,
        subfolder="tokenizer",
    )

    model = Phase2ZModel.from_pretrained(
        phase2_repo_id,
        device=device,
    )
    model.eval()

    # ---- generate splits ----
    train_ds = _generate_split(
        model=model,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        z_mode=z_mode,
        temperature=temperature,
        device=device,
    )

    eval_ds = _generate_split(
        model=model,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split="eval",
        batch_size=batch_size,
        z_mode=z_mode,
        temperature=temperature,
        device=device,
    )

    ds_dict = DatasetDict({
        "train": train_ds,
        "eval": eval_ds,
    })

    # ---- push to HF ----
    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=phase3_dataset_repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    ds_dict.push_to_hub(
        repo_id=phase3_dataset_repo_id,
        token=hf_token,
    )

    print(f"✅ Phase-3 dataset pushed to: {phase3_dataset_repo_id}")


__all__ = ["generate_phase3_dataset_from_hf"]

if __name__ == '__main__':
    generate_phase3_dataset_from_hf(phase2_repo_id=Phase2Config.hf_repo, phase3_dataset_repo_id="omrisap/phase3_train_dataset_0.0", dataset_name=Phase2Config.data.dataset_name, batch_size=64, device="cuda", temperature=0.0)