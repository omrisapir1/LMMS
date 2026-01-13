# phase1/model.py
#
# Phase 1 Coconut-style latent execution wrapper around Phase0Model.
# - Keeps full gradient flow through latent feedback (no detach).
# - Uses "hidden state from token - 1" to fill each <|latent|> position.
# - Does NOT implement KV-cache optimization (we recompute full forward per latent step).
# - Produces Phase0-style digit logits: [B, 5, 10] taken from the hidden state at <ANSWER>.
#
# Extended:
# - Supports optional latent-order permutation via return_perm_out flag
#   to test/enforce causal dependence on latent computation.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class Phase1ForwardDebug:
    max_n_latents: int
    latent_counts: List[int]


class Phase1CoconutModel(nn.Module):
    def __init__(
        self,
        phase0_model: nn.Module,
        latent_token_id: int,
        *,
        enforce_single_answer_token: bool = True,
    ):
        super().__init__()

        self.phase0 = phase0_model
        self.latent_token_id = int(latent_token_id)
        self.enforce_single_answer_token = bool(enforce_single_answer_token)

        self._embedding = self.phase0.model.get_input_embeddings()

    # ─────────────────────────────────────────────
    # Convenience passthroughs
    # ─────────────────────────────────────────────

    @property
    def config(self):
        return getattr(self.phase0, "config", None)

    def save_pretrained(self, *args, **kwargs):
        if hasattr(self.phase0, "save_pretrained"):
            return self.phase0.save_pretrained(*args, **kwargs)
        raise AttributeError("Underlying phase0_model does not implement save_pretrained().")

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = torch.cumsum(attention_mask.to(torch.long), dim=1) - 1
        position_ids = torch.clamp(position_ids, min=0)
        return position_ids

    def _latent_lists(self, input_ids: torch.Tensor) -> Tuple[List[List[int]], int]:
        B, T = input_ids.shape
        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)

        latent_lists: List[List[int]] = []
        for b in range(B):
            pos_list = [int(idx[1]) for idx in latent_indices if int(idx[0]) == b]
            pos_list.sort()
            latent_lists.append(pos_list)

        max_n_latents = max(len(x) for x in latent_lists) if latent_lists else 0
        return latent_lists, max_n_latents

    def _replace_latent_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        hidden: torch.Tensor,
        latent_lists: List[List[int]],
        pass_idx: int,
        latent_order: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        B, T, H = inputs_embeds.shape

        fill_b, fill_pos, src_pos = [], [], []

        for b, pos_list in enumerate(latent_lists):
            if latent_order is None:
                latent_i = pass_idx
            else:
                if pass_idx >= len(latent_order[b]):
                    continue
                latent_i = latent_order[b][pass_idx]

            if latent_i < len(pos_list):
                p = pos_list[latent_i]
                if p <= 0:
                    raise RuntimeError("Found <|latent|> at position 0.")
                fill_b.append(b)
                fill_pos.append(p)
                src_pos.append(p - 1)

        if not fill_b:
            return inputs_embeds

        new_embeds = inputs_embeds.clone()

        fb = torch.tensor(fill_b, device=inputs_embeds.device)
        fp = torch.tensor(fill_pos, device=inputs_embeds.device)
        sp = torch.tensor(src_pos, device=inputs_embeds.device)

        new_embeds[fb, fp, :] = hidden[fb, sp, :]
        return new_embeds

    def _compute_digit_logits_from_hidden(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        answer_token_id = getattr(self.phase0, "answer_token_id", None)
        if answer_token_id is None:
            raise RuntimeError("phase0_model must expose answer_token_id")

        answer_mask = (input_ids == int(answer_token_id))

        if self.enforce_single_answer_token:
            if not torch.all(answer_mask.sum(dim=1) == 1):
                raise RuntimeError("Each sample must contain exactly one <ANSWER> token")

        idx = answer_mask.float().argmax(dim=1)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)

        answer_hidden = hidden[batch_idx, idx]

        digit_heads = getattr(self.phase0, "digit_heads", None)
        if digit_heads is None:
            raise RuntimeError("phase0_model must expose digit_heads")

        logits = torch.stack([head(answer_hidden) for head in digit_heads], dim=1)
        return logits

    # ─────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_debug: bool = False,
        return_perm_out: bool = False,
    ) -> Union[
        Dict[str, torch.Tensor],
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ]:
        device = input_ids.device
        attention_mask = attention_mask.to(device)

        latent_lists, max_n_latents = self._latent_lists(input_ids)

        position_ids = self._build_position_ids(attention_mask)

        def run_coconut(latent_order: Optional[List[List[int]]] = None) -> torch.Tensor:
            inputs_embeds = self._embedding(input_ids)

            for pass_idx in range(max_n_latents):
                outputs = self.phase0.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-1]

                inputs_embeds = self._replace_latent_embeddings(
                    inputs_embeds=inputs_embeds,
                    hidden=hidden,
                    latent_lists=latent_lists,
                    pass_idx=pass_idx,
                    latent_order=latent_order,
                )

            final_outputs = self.phase0.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            final_hidden = final_outputs.hidden_states[-1]
            return self._compute_digit_logits_from_hidden(final_hidden, input_ids)

        # ── normal run ──
        logits = run_coconut(latent_order=None)
        out = {"logits": logits}

        if return_debug:
            out["debug"] = Phase1ForwardDebug(
                max_n_latents=max_n_latents,
                latent_counts=[len(x) for x in latent_lists],
            )

        if not return_perm_out:
            return out

        # ── permuted latent order ──
        latent_order_perm: List[List[int]] = []
        for pos_list in latent_lists:
            n = len(pos_list)
            if n <= 1:
                latent_order_perm.append(list(range(n)))
            else:
                perm = torch.randperm(n).tolist()
                latent_order_perm.append(perm)

        logits_perm = run_coconut(latent_order=latent_order_perm)
        out_perm = {"logits": logits_perm}

        if return_debug:
            out_perm["debug"] = out["debug"]

        return out, out_perm
