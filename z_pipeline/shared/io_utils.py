# shared/io_utils.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

import json
import torch
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------
# Artifact container
# ---------------------------------------------------------------------

@dataclass
class Phase2Artifacts:
    """
    Canonical in-memory representation of Phase-2 outputs.
    This is the ONLY object Phase-3 should consume.
    """
    z_embeddings: torch.Tensor          # [V, H], float32
    z_selector_state: Dict[str, torch.Tensor]
    tokenizer: PreTrainedTokenizerBase
    metadata: Dict


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_phase2_artifacts(
    artifacts: Phase2Artifacts,
    out_dir: str | Path,
) -> None:
    """
    Persist Phase-2 artifacts to disk.

    Layout:
      phase2_ckpt/
        z_embeddings.pt
        z_selector.pt
        tokenizer/
        metadata.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save tensors ----
    torch.save(artifacts.z_embeddings.cpu(), out_dir / "z_embeddings.pt")
    torch.save(artifacts.z_selector_state, out_dir / "z_selector.pt")

    # ---- Save tokenizer ----
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    artifacts.tokenizer.save_pretrained(tok_dir)

    # ---- Save metadata ----
    meta = dict(artifacts.metadata)
    meta.setdefault("date", datetime.utcnow().isoformat())

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_phase2_artifacts(
    ckpt_dir: str | Path,
    *,
    device: torch.device,
) -> Phase2Artifacts:
    """
    Load Phase-2 artifacts from disk.
    No mutation, no model logic.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise RuntimeError(f"Phase-2 checkpoint not found: {ckpt_dir}")

    # ---- Load tensors ----
    z_embeddings = torch.load(
        ckpt_dir / "z_embeddings.pt",
        map_location="cpu",
    )

    z_selector_state = torch.load(
        ckpt_dir / "z_selector.pt",
        map_location="cpu",
    )

    # ---- Load tokenizer ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer", trust_remote_code=True)

    # ---- Load metadata ----
    with open(ckpt_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Explicit dtype/device policy:
    # embeddings stay float32; caller decides casting
    return Phase2Artifacts(
        z_embeddings=z_embeddings,
        z_selector_state=z_selector_state,
        tokenizer=tokenizer,
        metadata=metadata,
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_phase2_artifacts(
    artifacts: Phase2Artifacts,
    *,
    expected_vocab_size: int,
    expected_hidden_size: int,
    expected_base_model: Optional[str] = None,
) -> None:
    """
    Hard validation before Phase-3 initialization.
    """
    meta = artifacts.metadata

    # ---- Format version ----
    fmt = meta.get("format_version")
    if fmt != "z-phase2-v1":
        raise RuntimeError(f"Incompatible Phase-2 format_version: {fmt}")

    # ---- Vocab size ----
    V, H = artifacts.z_embeddings.shape
    if V != expected_vocab_size:
        raise RuntimeError(
            f"Z vocab mismatch: Phase-2 V={V}, Phase-3 expects {expected_vocab_size}"
        )

    # ---- Hidden size ----
    if H != expected_hidden_size:
        raise RuntimeError(
            f"Hidden size mismatch: Phase-2 H={H}, Phase-3 expects {expected_hidden_size}"
        )

    # ---- Tokenizer sanity ----
    for i in range(V):
        tok = f"<Z_{i}>"
        tid = artifacts.tokenizer.convert_tokens_to_ids(tok)
        if tid == artifacts.tokenizer.unk_token_id:
            raise RuntimeError(f"Missing Z token in tokenizer: {tok}")

    # ---- Base model name ----
    if expected_base_model is not None:
        bm = meta.get("base_model_name")
        if bm != expected_base_model:
            raise RuntimeError(
                f"Base model mismatch: Phase-2 used {bm}, Phase-3 expects {expected_base_model}"
            )
