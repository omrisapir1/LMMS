import json
import os
from typing import Union
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from phase_0.model import Phase0Model
from phase1.model import Phase1CoconutModel

LATENT_TOKEN = "<|latent|>"


def _normalize_torch_dtype(torch_dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(torch_dtype, str):
        return getattr(torch, torch_dtype)
    return torch_dtype


def load_phase1_from_hf(repo_id: str, torch_dtype: Union[str, torch.dtype] = "bfloat16", device: str = "cuda"):
    """Load tokenizer + Phase-1 model from a Hugging Face repo.

    Returns (tokenizer, model, cfg/meta_dict).
    """
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    # Prefer meta file if present, else fall back to config.json
    meta = None
    try:
        meta_path = hf_hub_download(repo_id, "phase1_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        meta = None

    td = _normalize_torch_dtype(torch_dtype)

    if meta is None:
        config_path = hf_hub_download(repo_id, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        # load Phase-0
        phase0 = Phase0Model.from_pretrained(
            cfg["phase0_repo"],
            torch_dtype=td,
        )
        phase0.model.resize_token_embeddings(len(tokenizer))

        latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)

        model = Phase1CoconutModel(
            phase0_model=phase0,
            latent_token_id=latent_id,
        )

        # load Phase-1 weights
        weights_path = hf_hub_download(repo_id, "phase1_weights.pt")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

        model.eval().to(device)
        return tokenizer, model, cfg

    # meta path branch
    # transplant chat template from the Phase-0 tokenizer if available
    try:
        p0_tokenizer = AutoTokenizer.from_pretrained(meta["phase0_repo"])
        tokenizer.chat_template = p0_tokenizer.chat_template
    except Exception:
        pass

    phase0 = Phase0Model.from_pretrained(
        meta["phase0_repo"],
        torch_dtype=td,
    )
    phase0.model.resize_token_embeddings(len(tokenizer))

    latent_id = tokenizer.convert_tokens_to_ids(meta.get("latent_token", LATENT_TOKEN))

    model = Phase1CoconutModel(
        phase0_model=phase0,
        latent_token_id=latent_id,
    )

    weights_path = hf_hub_download(repo_id, "phase1_weights.pt")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.eval().to(device)
    return tokenizer, model, meta


def load_phase1(repo_or_dir: str = "omrisap/LMMS_phase1", device: str = "cuda", torch_dtype: Union[str, torch.dtype] = torch.bfloat16):
    """Load tokenizer + Phase-1 model from a HF repo id or a local directory.

    - If repo_or_dir is a local directory, expects files similar to the HF repo layout
      (phase1_meta.json or config.json, phase1_weights.pt, and tokenizer files).
    - Returns (tokenizer, model, meta_or_cfg_dict).
    """
    if os.path.isdir(repo_or_dir):
        # Try meta first
        meta_path = os.path.join(repo_or_dir, "phase1_meta.json")
        cfg_path = os.path.join(repo_or_dir, "config.json")
        weights_path = os.path.join(repo_or_dir, "phase1_weights.pt")

        tokenizer = AutoTokenizer.from_pretrained(repo_or_dir)

        td = _normalize_torch_dtype(torch_dtype)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            # optional chat template transplant from Phase-0 tokenizer
            try:
                p0_tokenizer = AutoTokenizer.from_pretrained(meta["phase0_repo"])
                tokenizer.chat_template = p0_tokenizer.chat_template
            except Exception:
                pass

            phase0 = Phase0Model.from_pretrained(
                meta["phase0_repo"],
                torch_dtype=td,
            )
            phase0.model.resize_token_embeddings(len(tokenizer))

            latent_id = tokenizer.convert_tokens_to_ids(meta.get("latent_token", LATENT_TOKEN))

            model = Phase1CoconutModel(
                phase0_model=phase0,
                latent_token_id=latent_id,
            )

            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            model.eval().to(device)
            return tokenizer, model, meta

        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)

            phase0 = Phase0Model.from_pretrained(
                cfg["phase0_repo"],
                torch_dtype=td,
            )
            phase0.model.resize_token_embeddings(len(tokenizer))

            latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
            model = Phase1CoconutModel(
                phase0_model=phase0,
                latent_token_id=latent_id,
            )

            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            model.eval().to(device)
            return tokenizer, model, cfg

        raise FileNotFoundError(
            f"Expected phase1_meta.json or config.json in {repo_or_dir}, plus phase1_weights.pt"
        )

    # Otherwise treat as HF repo id
    return load_phase1_from_hf(repo_or_dir, torch_dtype=torch_dtype, device=device)


__all__ = [
    "load_phase1_from_hf",
    "load_phase1",
    "LATENT_TOKEN",
]
