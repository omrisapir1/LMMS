# phase3/model.py
#
# Phase-3 model:
# - HuggingFace-compatible CausalLM behavior (keeps .generate())
# - Adds digit heads (5x Linear(H -> 10))
# - Replaces lm_head with a restricted head:
#     only Z tokens + <ANSWER> are ever non-masked.
#
# IMPORTANT:
# - The tokenizer + vocab size remain FULL.
# - Restriction is done *logically* via logits masking.
# - This keeps HF generation, caching, and RL intact.
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


# ============================================================
# Restricted LM Head
# ============================================================

class RestrictedLMHead(nn.Module):
    """
    Drop-in replacement for HF lm_head.

    Given hidden_states [B,T,H], returns logits [B,T,V_full] where only
    token ids in (z_token_ids + [answer_token_id]) are active.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size_full: int,
        z_token_ids: Sequence[int],
        answer_token_id: int,
        fill_value: float = -1e4,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.vocab_size_full = int(vocab_size_full)
        self.z_token_ids = [int(x) for x in z_token_ids]
        self.answer_token_id = int(answer_token_id)
        self.fill_value = float(fill_value)

        # Restricted vocabulary = Z tokens + <ANSWER>
        self.restricted_token_ids: List[int] = self.z_token_ids + [self.answer_token_id]
        self.restricted_size = len(self.restricted_token_ids)

        # Trainable rows only for restricted tokens
        self.weight = nn.Parameter(
            torch.empty((self.restricted_size, self.hidden_size), dtype=dtype, device=device)
        )

        ids = torch.tensor(self.restricted_token_ids, dtype=torch.long, device=device)
        self.register_buffer("_token_ids", ids, persistent=False)

    @torch.no_grad()
    def init_from_phase2(
        self,
        *,
        z_selector_weight: torch.Tensor,   # [V, H]
        answer_init_std: float,
    ) -> None:
        V, H = z_selector_weight.shape
        assert V == len(self.z_token_ids)
        assert H == self.hidden_size

        # Z rows
        self.weight.data[:V].copy_(
            z_selector_weight.to(dtype=self.weight.dtype, device=self.weight.device)
        )

        # <ANSWER> row
        self.weight.data[V].normal_(mean=0.0, std=float(answer_init_std))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        assert H == self.hidden_size

        restricted_logits = torch.matmul(
            hidden_states.to(torch.float32),
            self.weight.t().to(torch.float32),
        )  # [B,T,R]

        logits_full = torch.full(
            (B, T, self.vocab_size_full),
            self.fill_value,
            device=hidden_states.device,
            dtype=restricted_logits.dtype,
        )

        logits_full.index_copy_(2, self._token_ids, restricted_logits)
        return logits_full


# ============================================================
# Phase-3 Model
# ============================================================

class Phase3ZModel(nn.Module):
    """
    Phase-3 wrapper around a HF CausalLM.

    - Base LM is reused in-memory from Phase-2
    - lm_head is replaced with RestrictedLMHead
    - digit_heads are attached and frozen by default
    """

    def __init__(
        self,
        *,
        base_lm: nn.Module,
        digit_heads: nn.ModuleList,
        answer_token_id: int,
        z_token_ids: List[int],
        fill_value: float,
        answer_init_std: float,
    ):
        super().__init__()

        self.base = base_lm
        self.digit_heads = digit_heads

        # Freeze digit heads by default (Phase-3 semantics)
        for p in self.digit_heads.parameters():
            p.requires_grad = True

        self.answer_token_id = int(answer_token_id)
        self.z_token_ids = list(map(int, z_token_ids))

        # Sizes
        hidden_size = self.base.get_input_embeddings().embedding_dim
        vocab_size_full = self.base.get_input_embeddings().weight.shape[0]

        device = next(self.base.parameters()).device
        dtype = next(self.base.parameters()).dtype

        # Replace lm_head
        restricted_head = RestrictedLMHead(
            hidden_size=hidden_size,
            vocab_size_full=vocab_size_full,
            z_token_ids=self.z_token_ids,
            answer_token_id=self.answer_token_id,
            fill_value=fill_value,
            dtype=dtype,
            device=device,
        )

        # HF CausalLMs usually expose either lm_head OR get_output_embeddings/set_output_embeddings
        if hasattr(self.base, "lm_head"):
            self.base.lm_head = restricted_head
        elif hasattr(self.base, "set_output_embeddings"):
            self.base.set_output_embeddings(restricted_head)
        else:
            raise AssertionError(
                "Base model has no lm_head and no set_output_embeddings(). "
                f"type(base)={type(self.base)}"
            )

        self.restricted_lm_head = restricted_head

        # Default init (overwritten for Z rows later)
        with torch.no_grad():
            self.restricted_lm_head.weight.normal_(0.0, answer_init_std)

    # --------------------------------------------------
    # Construction from Phase-2 checkpoint
    # --------------------------------------------------

    @classmethod
    def from_phase2_ckpt(
        cls,
        *,
        phase2_ckpt: Dict,
        fill_value: float = -1e4,
        answer_init_std: float = 0.02,
        copy_digit_heads: bool = True,
    ) -> "Phase3ZModel":

        phase2_model = phase2_ckpt["model"]
        z_token_ids = phase2_ckpt["z_token_ids"]

        base = phase2_model.base
        if base.__class__.__name__ == "Qwen2Model":
            from .model import wrap_qwen2_body_as_causallm
            base = wrap_qwen2_body_as_causallm(base)
        z_selector = phase2_model.z_selector
        answer_token_id = phase2_model.answer_token_id

        # Copy digit heads
        if copy_digit_heads:
            new_heads = nn.ModuleList(
                [nn.Linear(phase2_model.hidden_size, 10) for _ in range(5)]
            )
            new_heads.load_state_dict(phase2_model.digit_heads.state_dict())
            digit_heads = new_heads
        else:
            digit_heads = phase2_model.digit_heads

        model = cls(
            base_lm=base,
            digit_heads=digit_heads,
            answer_token_id=answer_token_id,
            z_token_ids=z_token_ids,
            fill_value=fill_value,
            answer_init_std=answer_init_std,
        )

        # Initialize LM head Z rows from selector
        model.restricted_lm_head.init_from_phase2(
            z_selector_weight=z_selector.weight.detach(),
            answer_init_std=answer_init_std,
        )

        return model

    # --------------------------------------------------
    # Digit prediction
    # --------------------------------------------------

    def _digit_logits_from_hidden(
        self,
        hidden_last: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        mask = (input_ids == self.answer_token_id)
        assert torch.all(mask.sum(dim=1) == 1)

        idx = mask.float().argmax(dim=1)
        bidx = torch.arange(hidden_last.size(0), device=hidden_last.device)
        answer_hidden = hidden_last[bidx, idx]

        return torch.stack(
            [head(answer_hidden) for head in self.digit_heads],
            dim=1,
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs,
    ):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_last = out.hidden_states[-1]
        digit_logits = self._digit_logits_from_hidden(hidden_last, input_ids)

        if return_dict:
            out.digit_logits = digit_logits
            return out

        return {
            "logits": out.logits,
            "digit_logits": digit_logits,
            "hidden_states": out.hidden_states,
        }

    # --------------------------------------------------
    # Generation + digits
    # --------------------------------------------------

    @torch.no_grad()
    def generate_with_digits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> Dict[str, torch.Tensor]:

        generate_kwargs = dict(generate_kwargs)
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_hidden_states"] = True

        gen = self.base.generate(
            input_ids=input_ids,
            eos_token_id=self.answer_token_id,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        sequences = gen.sequences
        mask = (sequences == self.answer_token_id)
        assert torch.all(mask.sum(dim=1) >= 1)

        pos = mask.float().argmax(dim=1)

        hs = gen.hidden_states
        last_hidden = hs[-1] if torch.is_tensor(hs[-1]) else hs[-1][-1]

        bidx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        answer_hidden = last_hidden[bidx, pos]

        digit_logits = torch.stack(
            [head(answer_hidden) for head in self.digit_heads],
            dim=1,
        )

        return {
            "sequences": sequences,
            "digit_logits": digit_logits,
            "digit_preds": digit_logits.argmax(dim=-1),
        }

from transformers import Qwen2ForCausalLM

def wrap_qwen2_body_as_causallm(
    body: torch.nn.Module,
) -> Qwen2ForCausalLM:
    """
    Takes a Qwen2Model and wraps it into Qwen2ForCausalLM,
    transplanting weights exactly.
    """
    assert body.__class__.__name__ == "Qwen2Model"

    config = body.config
    causal_lm = Qwen2ForCausalLM(config)

    # transplant transformer weights
    causal_lm.model.load_state_dict(
        body.state_dict(),
        strict=True,
    )

    return causal_lm
