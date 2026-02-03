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

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from transformers import Qwen2ForCausalLM, AutoModel

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


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
        mapping = torch.full(
            (vocab_size_full,),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        for i, tid in enumerate(self.restricted_token_ids):
            mapping[tid] = i

        self.register_buffer("_full_to_restricted", mapping, persistent=False)


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_full_logits: bool = True,
    ) -> torch.Tensor:
        """
        If return_full_logits=False:
            returns [B,T,R] restricted logits
        If return_full_logits=True:
            returns [B,T,V_full] (slow, memory-heavy)
        """
        B, T, H = hidden_states.shape

        restricted_logits = torch.matmul(
            hidden_states,
            self.weight.t(),  # [B,T,R]
        )

        if not return_full_logits:
            return restricted_logits

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

        # --- sanity checks: LM head must be 1024 Z + 1 ANSWER ---
        assert len(self.z_token_ids) == 1024, f"expected 1024 z tokens, got {len(self.z_token_ids)}"
        assert self.answer_token_id not in self.z_token_ids, "answer_token_id is inside z_token_ids (dup bug)"
        assert restricted_head.restricted_size == 1025, f"expected restricted_size=1025, got {restricted_head.restricted_size}"
        assert int(restricted_head._full_to_restricted[self.answer_token_id].item()) != -1, "ANSWER not mapped"

        vocab_size_full = self.base.get_input_embeddings().weight.shape[0]
        max_id = max(self.z_token_ids + [self.answer_token_id])

        if max_id >= vocab_size_full:
            raise RuntimeError(
                f"Token id out of range: max_id={max_id} >= vocab_size_full={vocab_size_full}. "
                "Your z_meta token ids don't fit the loaded base model embeddings."
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
    def from_phase2_repo(
            cls,
            *,
            repo_id: str,
            fill_value: float = -1e4,
            answer_init_std: float = 0.02,
            device: str | torch.device = "cuda",
    ) -> "Phase3ZModel":
        # ---- load z_meta.json ----
        z_meta_path = hf_hub_download(repo_id, "z_meta.json")
        with open(z_meta_path, "r") as f:
            z_meta = json.load(f)

        z_token_ids = list(map(int, z_meta["z_token_ids"]))
        answer_token_id = int(z_meta["answer_token_id"])

        # ---- load base model body (saved by Phase2ZModel.save_pretrained) ----
        base = AutoModel.from_pretrained(repo_id, torch_dtype=torch.bfloat16)

        # Need a CausalLM for generate()
        if base.__class__.__name__ == "Qwen2Model":
            base = wrap_qwen2_body_as_causallm(base)

        # ---- load phase2_state.pt ----
        state_path = hf_hub_download(repo_id, "phase2_state.pt")
        state = torch.load(state_path, map_location="cpu")

        # ---- reconstruct digit heads and load weights ----
        hidden_size = base.get_input_embeddings().embedding_dim
        digit_heads = nn.ModuleList([nn.Linear(hidden_size, 10) for _ in range(5)])
        digit_heads.load_state_dict(state["digit_heads"])

        # ---- build Phase3 model ----
        model = cls(
            base_lm=base,
            digit_heads=digit_heads,
            answer_token_id=answer_token_id,
            z_token_ids=z_token_ids,
            fill_value=fill_value,
            answer_init_std=answer_init_std,
        )

        # init restricted head rows from z_selector weight
        z_selector_weight = state["z_selector"]["weight"]  # [V,H]
        # model.restricted_lm_head.init_from_phase2(
        #     z_selector_weight=z_selector_weight,
        #     answer_init_std=answer_init_std,
        # )

        model.to(device)
        return model

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
        # model.restricted_lm_head.init_from_phase2(
        #     z_selector_weight=z_selector.weight.detach(),
        #     answer_init_std=answer_init_std,
        # )

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
            output_hidden_states: bool = False,
            return_dict: bool = True,
            return_full_logits: bool = True,
            **kwargs,
    ):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # we still need last layer
            return_dict=True,
            **kwargs,
        )

        hidden_last = out.hidden_states[-1]

        # 🔥 THIS IS THE KEY CHANGE
        logits = self.restricted_lm_head(
            hidden_last,
            return_full_logits=return_full_logits,
        )

        digit_logits = self._digit_logits_from_hidden(hidden_last, input_ids)

        if return_dict:
            out.logits = logits
            out.digit_logits = digit_logits
            return out

        return {
            "logits": logits,
            "digit_logits": digit_logits,
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
        # print(input_ids, attention_mask, sep="\n\n")
        gen = self.base.generate(
            input_ids=input_ids,
            eos_token_id=self.answer_token_id,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        sequences = gen.sequences  # [B, T_total]

        # Make an attention mask for the full sequences
        # (If you have pad_token_id=None, set it in tokenizer or use eos as pad)
        pad_id = generate_kwargs.get("pad_token_id", self.base.config.eos_token_id)
        # pad_id = getattr(self.base.config, "pad_token_id", None)

        full_attn = (sequences != pad_id).long()

        # Re-run forward to get full hidden states
        out_full = self.base(
            input_ids=sequences,
            attention_mask=full_attn,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_last = out_full.hidden_states[-1]  # [B, T_total, H]

        mask = (sequences == self.answer_token_id)
        has_answer = mask.any(dim=1)

        # if some rows didn't generate <ANSWER>, you can decide a fallback:
        # e.g., use last token position (or skip those rows)
        pos = mask.float().argmax(dim=1)
        pos = torch.where(
            has_answer,
            pos,
            (full_attn.sum(dim=1) - 1).clamp(min=0)  # fallback = last non-pad token
        )

        bidx = torch.arange(hidden_last.size(0), device=hidden_last.device)
        answer_hidden = hidden_last[bidx, pos]  # [B, H]

        digit_logits = torch.stack([head(answer_hidden) for head in self.digit_heads], dim=1)
        return {
            "sequences": sequences,
            "digit_logits": digit_logits,
            "digit_preds": digit_logits.argmax(dim=-1),
        }


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
