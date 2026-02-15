from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import torch


_Z_TOKEN_RE = re.compile(r"^<Z_(\d+)>$")


def introspect_z_token_ids(tokenizer) -> List[int]:
    vocab = tokenizer.get_vocab()
    indexed: List[Tuple[int, int]] = []
    for tok, tok_id in vocab.items():
        m = _Z_TOKEN_RE.match(tok)
        if m is None:
            continue
        indexed.append((int(m.group(1)), int(tok_id)))

    indexed.sort(key=lambda x: x[0])
    return [tok_id for _, tok_id in indexed]


def resolve_answer_token_id(tokenizer, answer_token: str = "<ANSWER>") -> int:
    tok_id = tokenizer.convert_tokens_to_ids(answer_token)
    if tok_id is None or int(tok_id) < 0:
        raise RuntimeError(f"Could not resolve answer token id for {answer_token}")
    return int(tok_id)


def build_allowed_token_ids(tokenizer, answer_token: str = "<ANSWER>") -> List[int]:
    z_token_ids = introspect_z_token_ids(tokenizer)
    if not z_token_ids:
        raise RuntimeError("No Z tokens found via tokenizer introspection")
    answer_token_id = resolve_answer_token_id(tokenizer, answer_token=answer_token)
    return z_token_ids + [answer_token_id]


def mask_logits(logits: torch.Tensor, allowed_token_ids: Sequence[int], fill_value: float = -1e9) -> torch.Tensor:
    masked = logits.new_full(logits.shape, float(fill_value))
    idx = torch.tensor(list(allowed_token_ids), dtype=torch.long, device=logits.device)
    if logits.dim() == 1:
        masked.index_copy_(0, idx, logits.index_select(0, idx))
    elif logits.dim() == 2:
        masked.index_copy_(1, idx, logits.index_select(1, idx))
    else:
        raise ValueError("logits must be rank-1 or rank-2")
    return masked


def masked_log_probs_and_entropy(
    logits: torch.Tensor,
    allowed_token_ids: Sequence[int],
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    masked = mask_logits(logits / temperature, allowed_token_ids)
    log_probs = torch.log_softmax(masked, dim=-1)

    allowed_idx = torch.tensor(list(allowed_token_ids), dtype=torch.long, device=logits.device)
    allowed_logits = (logits / temperature).index_select(-1, allowed_idx)
    allowed_probs = torch.softmax(allowed_logits, dim=-1)
    entropy = -(allowed_probs * torch.log(allowed_probs.clamp_min(1e-12))).sum(dim=-1)
    return log_probs, entropy
