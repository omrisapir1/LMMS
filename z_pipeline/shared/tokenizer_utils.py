# shared/tokenizer_utils.py

from __future__ import annotations
from typing import List, Tuple, Dict, Iterable

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel


# ---------------------------------------------------------------------
# Z token naming
# ---------------------------------------------------------------------

def z_token_str(i: int) -> str:
    return f"<Z_{i}>"


def make_z_tokens(vocab_size: int) -> List[str]:
    return [z_token_str(i) for i in range(vocab_size)]


# ---------------------------------------------------------------------
# Tokenizer expansion
# ---------------------------------------------------------------------

def add_z_tokens(
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
) -> List[int]:
    """
    Add Z tokens <Z_0> ... <Z_{V-1}> to tokenizer.

    Returns:
        z_token_ids: List[int] of length V, in index order.

    Behavior:
    - Deterministic
    - Hard error if tokens already exist
    """
    z_tokens = make_z_tokens(vocab_size)

    # Check none already exist
    existing = [t for t in z_tokens if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id]
    if existing:
        raise RuntimeError(
            f"Some Z tokens already exist in tokenizer: {existing[:5]} "
            f"(refusing to continue to avoid ID mismatch)"
        )

    # Add as REGULAR tokens
    n_added = tokenizer.add_tokens(z_tokens)
    if n_added != vocab_size:
        raise RuntimeError(
            f"Tokenizer added {n_added} tokens, expected {vocab_size}. "
            "Tokenizer may be in an inconsistent state."
        )

    # Fetch IDs deterministically
    z_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in z_tokens]

    # Sanity: ensure contiguous block (not strictly required, but good invariant)
    if sorted(z_token_ids) != list(range(min(z_token_ids), min(z_token_ids) + vocab_size)):
        raise RuntimeError(
            "Z token IDs are not contiguous. This is unexpected and unsafe."
        )

    return z_token_ids


def get_z_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
) -> List[int]:
    """
    Retrieve Z token IDs, asserting correct existence and ordering.
    """
    z_tokens = make_z_tokens(vocab_size)
    z_ids = [tokenizer.convert_tokens_to_ids(t) for t in z_tokens]

    if any(i == tokenizer.unk_token_id for i in z_ids):
        raise RuntimeError("Some Z tokens are missing from tokenizer")

    return z_ids


# ---------------------------------------------------------------------
# Model initialization helpers
# ---------------------------------------------------------------------

def initialize_z_embeddings(
    model: PreTrainedModel,
    z_token_ids: List[int],
    z_embeddings: torch.Tensor,
    *,
    copy_to_lm_head: bool = True,
):
    """
    Initialize Z token rows in:
      - input embedding matrix
      - (optionally) LM head

    Assumptions:
    - model has get_input_embeddings()
    - model is AutoModelForCausalLM OR ties weights automatically

    z_embeddings:
      Tensor of shape [V, H], ordered by <Z_0> ... <Z_{V-1}>
    """
    emb = model.get_input_embeddings().weight  # [vocab, H]

    V, H = z_embeddings.shape
    if len(z_token_ids) != V:
        raise RuntimeError(
            f"Z embedding size mismatch: "
            f"{len(z_token_ids)} token IDs vs {V} embeddings"
        )

    if emb.shape[1] != H:
        raise RuntimeError(
            f"Hidden size mismatch: model H={emb.shape[1]}, z_embeddings H={H}"
        )

    device = emb.device
    dtype = emb.dtype

    # Copy embeddings
    for i, tok_id in enumerate(z_token_ids):
        emb.data[tok_id].copy_(z_embeddings[i].to(device=device, dtype=dtype))

    # LM head handling
    if copy_to_lm_head:
        lm_head = getattr(model, "lm_head", None)

        # Case 1: tied embeddings (common in HF)
        if lm_head is None:
            # Assume tied; nothing to do
            return

        # Case 2: explicit lm_head
        lm_weight = lm_head.weight
        if lm_weight.shape != emb.shape:
            raise RuntimeError(
                "LM head weight shape does not match embedding weight shape; "
                "cannot safely initialize Z rows"
            )

        for i, tok_id in enumerate(z_token_ids):
            lm_weight.data[tok_id].copy_(z_embeddings[i].to(device=device, dtype=dtype))


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def assert_vocab_size_match(
    z_embeddings: torch.Tensor,
    vocab_size: int,
):
    if z_embeddings.shape[0] != vocab_size:
        raise RuntimeError(
            f"Phase-2 vocab size {z_embeddings.shape[0]} "
            f"!= Phase-3 vocab size {vocab_size}"
        )
