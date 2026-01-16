# phase1/model.py
#
# Phase 1 Coconut-style latent execution wrapper around Phase0Model.
#
# Optimized:
# - Instead of recomputing the full sequence for every latent pass, we only
#   compute the minimal prefix needed to obtain hidden[p-1] for each latent position p.
# - After all latent embeddings are filled, we run ONE final full forward to read <ANSWER>.
#
# Why this is correct (for your setup):
# - Each latent embedding at position p is replaced by hidden state at p-1.
# - hidden[p-1] depends only on tokens/embeddings in positions [0..p-1].
# - Therefore we only need a prefix forward up to p to obtain hidden[p-1].
#
# Notes:
# - Works with Qwen2 (DynamicCache) because we do NOT pass legacy past_key_values tuples.
# - Supports return_perm_out by running the same logic with a different latent order.

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
        B, _T = input_ids.shape
        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)

        latent_lists: List[List[int]] = []
        for b in range(B):
            pos_list = [int(idx[1]) for idx in latent_indices if int(idx[0]) == b]
            pos_list.sort()
            latent_lists.append(pos_list)

        max_n_latents = max((len(x) for x in latent_lists), default=0)
        return latent_lists, max_n_latents

    def _bucket_fill_positions(
        self,
        latent_lists: List[List[int]],
        pass_idx: int,
        latent_order: Optional[List[List[int]]] = None,
    ) -> Dict[int, List[int]]:
        """
        Returns buckets: { latent_position_p : [batch_indices...] }
        for the current pass_idx.
        """
        buckets: Dict[int, List[int]] = {}

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
                buckets.setdefault(p, []).append(b)

        return buckets

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

        answer_hidden = hidden[batch_idx, idx]  # [B, H]

        digit_heads = getattr(self.phase0, "digit_heads", None)
        if digit_heads is None:
            raise RuntimeError("phase0_model must expose digit_heads")

        logits = torch.stack([head(answer_hidden) for head in digit_heads], dim=1)  # [B, 5, 10]
        return logits

    # ─────────────────────────────────────────────
    # Core Coconut execution (optimized)
    # ─────────────────────────────────────────────

    def _run_coconut_prefix_optimized(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        latent_lists: List[List[int]],
        max_n_latents: int,
        latent_order: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        """
        Optimized latent execution:
        - For each pass, for each latent position p, run a forward ONLY on prefix [:p]
          to get hidden[p-1], then replace embedding at p.
        - After all passes, run ONE final full forward and read <ANSWER>.
        """
        device = input_ids.device
        inputs_embeds = self._embedding(input_ids)  # [B, T, H]

        # Latent passes
        for pass_idx in range(max_n_latents):
            buckets = self._bucket_fill_positions(latent_lists, pass_idx, latent_order=latent_order)
            if not buckets:
                continue

            # Deterministic order
            for p in sorted(buckets.keys()):
                bs = buckets[p]
                bs_t = torch.tensor(bs, device=device, dtype=torch.long)

                # Forward only on prefix [:p] to obtain hidden at p-1
                # Note: no caching/past_key_values used => compatible with Qwen2 DynamicCache API.
                out_prefix = self.phase0.model(
                    inputs_embeds=inputs_embeds[bs_t, :p, :],
                    attention_mask=attention_mask[bs_t, :p],
                    position_ids=position_ids[bs_t, :p],
                    use_cache=False,
                    output_hidden_states=True,
                )
                hidden_prefix = out_prefix.hidden_states[-1]  # [len(bs), p, H]

                # Fill the latent at position p using hidden at p-1
                inputs_embeds[bs_t, p, :] = hidden_prefix[:, p - 1, :]

        # Final full forward to get hidden at <ANSWER>
        out_final = self.phase0.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        final_hidden = out_final.hidden_states[-1]
        return self._compute_digit_logits_from_hidden(final_hidden, input_ids)

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

        # ── normal run ──
        logits = self._run_coconut_prefix_optimized(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            latent_lists=latent_lists,
            max_n_latents=max_n_latents,
            latent_order=None,
        )

        out: Dict[str, torch.Tensor] = {"logits": logits}
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
                latent_order_perm.append([])
            elif n == 2:
                latent_order_perm.append([1, 0])
            else:
                latent_order_perm.append(list(reversed(range(n))))

        logits_perm = self._run_coconut_prefix_optimized(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            latent_lists=latent_lists,
            max_n_latents=max_n_latents,
            latent_order=latent_order_perm,
        )

        out_perm: Dict[str, torch.Tensor] = {"logits": logits_perm}
        if return_debug:
            out_perm["debug"] = out["debug"]

        return out, out_perm
