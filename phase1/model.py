# phase1/model.py
#
# Phase 1 Coconut-style latent execution wrapper around Phase0Model.
# Adds: "latent gate" fusion so the last latent hidden state explicitly contributes
# to the digit classification head input.
#
# Fusion:
#   gate = sigmoid(W_g * h_last_latent + b_g)          # [B, H]
#   fused = h_answer + gate * h_last_latent           # [B, H]
#   logits = digit_heads(fused)                       # [B, 5, 10]
#
# Notes:
# - We define "last latent hidden" as the hidden state at position (answer_pos - 1).
#   This matches your setup where latents are placed immediately before <ANSWER>.
# - We safely handle samples with 0 latents by setting h_last_latent = 0 and gate=0
#   (so fused == h_answer).
# - Gate is initialized to zero so training starts identical to Phase0 behavior.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class Phase1ForwardDebug:
    """
    Optional debug payload (disabled by default).
    """
    max_n_latents: int
    latent_counts: List[int]


class Phase1CoconutModel(nn.Module):
    """
    Coconut-style latent execution for Phase 1.

    Wraps a trained Phase0Model and changes ONLY the reasoning pathway:
    - <|latent|> tokens are treated as "internal compute steps" by replacing their embeddings
      with the preceding hidden state (token_idx - 1) iteratively.
    - Answer extraction and digit heads are identical to Phase0Model.
    - (NEW) Latent-gate fusion: last latent hidden explicitly contributes to prediction.
    """

    def __init__(
        self,
        phase0_model: nn.Module,
        latent_token_id: int,
        *,
        enforce_single_answer_token: bool = True,
        use_latent_gate: bool = True,
        latent_gate_init_zero: bool = True,
    ):
        super().__init__()

        # Underlying Phase-0 model (contains base transformer + digit heads + answer_token_id)
        self.phase0 = phase0_model

        # Latent token id (tokenizer.convert_tokens_to_ids("<|latent|>"))
        self.latent_token_id = int(latent_token_id)

        # Safety toggle (Phase0 already enforces this; we keep it here for clarity)
        self.enforce_single_answer_token = bool(enforce_single_answer_token)

        # Cache embedding handle (for inputs_embeds construction)
        self._embedding = self.phase0.model.get_input_embeddings()

        # ─────────────────────────────────────────────────────────
        # Latent gate fusion (Option 2)
        # ─────────────────────────────────────────────────────────
        self.use_latent_gate = bool(use_latent_gate)

        # Hidden size H
        # Prefer model config if present; fallback to embedding dim.
        H = getattr(getattr(self.phase0.model, "config", None), "hidden_size", None)
        if H is None:
            H = int(self._embedding.embedding_dim)

        if self.use_latent_gate:
            param_dtype = next(self.phase0.model.parameters()).dtype
            param_device = next(self.phase0.model.parameters()).device

            self.latent_gate = nn.Linear(H, H, bias=True).to(
                dtype=param_dtype,
                device=param_device,
            )

            # Optional: initialize to zero so gate starts closed (Phase0-equivalent).
            if latent_gate_init_zero:
                nn.init.zeros_(self.latent_gate.weight)
                nn.init.zeros_(self.latent_gate.bias)
        else:
            self.latent_gate = None  # type: ignore[assignment]

    # ─────────────────────────────────────────────────────────
    # Convenience passthroughs (helps with training + saving)
    # ─────────────────────────────────────────────────────────

    @property
    def config(self):
        return getattr(self.phase0, "config", None)

    def save_pretrained(self, *args, **kwargs):
        if hasattr(self.phase0, "save_pretrained"):
            return self.phase0.save_pretrained(*args, **kwargs)
        raise AttributeError("Underlying phase0_model does not implement save_pretrained().")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "Phase1CoconutModel.from_pretrained is not implemented. "
            "Load Phase0Model.from_pretrained(...) and wrap it."
        )

    # ─────────────────────────────────────────────────────────
    # Core helpers
    # ─────────────────────────────────────────────────────────

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        B, T = attention_mask.shape
        device = attention_mask.device

        position_ids = torch.cumsum(attention_mask.to(torch.long), dim=1) - 1
        position_ids = torch.clamp(position_ids, min=0)

        if position_ids.shape != (B, T):
            position_ids = position_ids.view(B, T)

        return position_ids.to(device)

    def _latent_lists(self, input_ids: torch.Tensor) -> Tuple[List[List[int]], int]:
        B, T = input_ids.shape
        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)

        latent_lists: List[List[int]] = []
        for b in range(B):
            pos_list = [int(idx[1].item()) for idx in latent_indices if int(idx[0].item()) == b]
            pos_list.sort()
            latent_lists.append(pos_list)

        max_n_latents = max((len(lst) for lst in latent_lists), default=0)
        return latent_lists, max_n_latents

    def _replace_latent_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        hidden: torch.Tensor,
        latent_lists: List[List[int]],
        pass_idx: int,
    ) -> torch.Tensor:
        B, T, H = inputs_embeds.shape

        fill_b: List[int] = []
        fill_pos: List[int] = []
        src_pos: List[int] = []

        for b, pos_list in enumerate(latent_lists):
            if len(pos_list) > pass_idx:
                p = pos_list[pass_idx]
                if p <= 0:
                    raise RuntimeError("Found <|latent|> at position 0; cannot use token_idx-1 rule.")
                fill_b.append(b)
                fill_pos.append(p)
                src_pos.append(p - 1)

        if len(fill_b) == 0:
            return inputs_embeds

        new_embeds = inputs_embeds.clone()

        fb = torch.tensor(fill_b, device=inputs_embeds.device, dtype=torch.long)
        fp = torch.tensor(fill_pos, device=inputs_embeds.device, dtype=torch.long)
        sp = torch.tensor(src_pos, device=inputs_embeds.device, dtype=torch.long)

        new_embeds[fb, fp, :] = hidden[fb, sp, :]
        return new_embeds

    def _compute_digit_logits_from_hidden(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Phase0 digit-head extraction, with optional latent-gate fusion.

        - answer_hidden = hidden at <ANSWER>
        - latent_hidden = hidden at (<ANSWER> - 1) if that position is a <|latent|>, else zeros
        - fused = answer_hidden + sigmoid(W*latent_hidden+b) * latent_hidden
        - digit_heads(fused)
        """
        answer_token_id = getattr(self.phase0, "answer_token_id", None)
        if answer_token_id is None:
            raise RuntimeError("phase0_model must expose answer_token_id")

        answer_mask = (input_ids == int(answer_token_id))  # [B, T]

        if self.enforce_single_answer_token:
            if not torch.all(answer_mask.sum(dim=1) == 1):
                raise RuntimeError("Each sample must contain exactly one <ANSWER> token")

        idx = answer_mask.float().argmax(dim=1)  # [B]
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)

        answer_hidden = hidden[batch_idx, idx]  # [B, H]

        # Latent-gate fusion (safe even when some samples have 0 latents)
        if self.use_latent_gate:
            if self.latent_gate is None:
                raise RuntimeError("use_latent_gate=True but latent_gate is None")

            # Candidate position for "last latent" is right before <ANSWER>
            prev_idx = idx - 1  # [B]
            # Some sequences could (pathologically) place <ANSWER> at pos 0; guard.
            has_prev = prev_idx >= 0  # [B]

            # Determine whether that previous token is actually a latent token.
            # (If your format guarantees latents immediately before <ANSWER>, this will be True
            # whenever there is at least one latent.)
            prev_is_latent = torch.zeros_like(has_prev, dtype=torch.bool)
            if has_prev.any():
                safe_prev_idx = torch.clamp(prev_idx, min=0)
                prev_token = input_ids[batch_idx, safe_prev_idx]
                prev_is_latent = has_prev & (prev_token == self.latent_token_id)

            # Build latent_hidden with zeros where not applicable
            latent_hidden = torch.zeros_like(answer_hidden)  # [B, H]
            if prev_is_latent.any():
                latent_hidden[prev_is_latent] = hidden[batch_idx[prev_is_latent], prev_idx[prev_is_latent]]

            gate = torch.sigmoid(self.latent_gate(latent_hidden))  # [B, H]

            # Ensure samples with no latent do not get perturbed (gate=0, latent=0 already does it,
            # but we can be explicit and set gate=0 where prev_is_latent is False).
            gate = gate * prev_is_latent.to(gate.dtype).unsqueeze(-1)

            fused_hidden = answer_hidden + gate * latent_hidden
        else:
            fused_hidden = answer_hidden

        digit_heads = getattr(self.phase0, "digit_heads", None)
        if digit_heads is None:
            raise RuntimeError("phase0_model must expose digit_heads (nn.ModuleList)")

        logits = torch.stack([head(fused_hidden) for head in digit_heads], dim=1)  # [B, 5, 10]
        return logits

    # ─────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, T]")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be [B, T]")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have same shape")

        device = input_ids.device
        attention_mask = attention_mask.to(device)

        latent_lists, max_n_latents = self._latent_lists(input_ids)

        inputs_embeds = self._embedding(input_ids)  # [B, T, H]
        position_ids = self._build_position_ids(attention_mask)  # [B, T]

        for pass_idx in range(max_n_latents):
            outputs = self.phase0.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]  # [B, T, H]

            inputs_embeds = self._replace_latent_embeddings(
                inputs_embeds=inputs_embeds,
                hidden=hidden,
                latent_lists=latent_lists,
                pass_idx=pass_idx,
            )

        final_outputs = self.phase0.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        final_hidden = final_outputs.hidden_states[-1]  # [B, T, H]

        logits = self._compute_digit_logits_from_hidden(final_hidden, input_ids)

        out: Dict[str, torch.Tensor] = {"logits": logits}

        if return_debug:
            latent_counts = [len(x) for x in latent_lists]
            out["debug"] = Phase1ForwardDebug(max_n_latents=max_n_latents, latent_counts=latent_counts)  # type: ignore[assignment]

        return out
