# phase1/model.py
#
# Phase 1 Coconut-style latent execution wrapper around Phase0Model.
# KV-cache optimized (bucketed by latent position per pass).
#
# - Keeps full gradient flow through latent feedback (no detach).
# - Uses "hidden state from token - 1" to fill each <|latent|> position.
# - Uses KV-cache to avoid recomputing stable prefixes; recomputes suffix after each latent write.
# - Produces Phase0-style digit logits: [B, 5, 10] taken from the hidden state at <ANSWER>.
#
# Notes:
# - This implementation IGNORES the permuted output path (return_perm_out is not supported).
# - Correctness first: we recompute prefix+suffix per bucket. With K<=8 and small batch sizes,
#   this is already a big win vs recomputing the full sequence for every latent step.

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
        use_kv_cache: bool = True,
    ):
        super().__init__()

        self.phase0 = phase0_model
        self.latent_token_id = int(latent_token_id)
        self.enforce_single_answer_token = bool(enforce_single_answer_token)
        self.use_kv_cache = bool(use_kv_cache)

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
        # Standard "positions for non-pad tokens" for decoder-only models
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

    def _slice_past_key_values(self, past_key_values, prefix_len: int):
        """
        Slice past_key_values to prefix_len along the sequence dimension.
        Works for common HF decoder-only layouts where k/v are:
          [B, n_heads, seq_len, head_dim]
        """
        if past_key_values is None:
            return None

        sliced = []
        for layer in past_key_values:
            # layer can be (k, v) or (k, v, ...) depending on model
            k, v = layer[0], layer[1]
            # seq dimension is usually dim=2
            k2 = k[:, :, :prefix_len, :]
            v2 = v[:, :, :prefix_len, :]
            if len(layer) > 2:
                sliced.append((k2, v2, *layer[2:]))
            else:
                sliced.append((k2, v2))
        return tuple(sliced)

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
        if return_perm_out:
            raise NotImplementedError("KV-cache implementation currently ignores permuted output. Set return_perm_out=False.")

        device = input_ids.device
        attention_mask = attention_mask.to(device)

        latent_lists, max_n_latents = self._latent_lists(input_ids)
        position_ids = self._build_position_ids(attention_mask)

        # Fast path: no latents
        if max_n_latents == 0 or not self.use_kv_cache:
            # Either no latents, or KV-cache disabled: do a single forward and read <ANSWER>
            outputs = self.phase0.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            logits = self._compute_digit_logits_from_hidden(hidden, input_ids)
            out: Dict[str, torch.Tensor] = {"logits": logits}
            if return_debug:
                out["debug"] = Phase1ForwardDebug(
                    max_n_latents=max_n_latents,
                    latent_counts=[len(x) for x in latent_lists],
                )
            return out

        # KV-cache optimized Coconut-style latent execution
        def run_coconut_kvcache(latent_order: Optional[List[List[int]]] = None) -> torch.Tensor:
            inputs_embeds = self._embedding(input_ids)  # [B, T, H]
            B, T, _H = inputs_embeds.shape

            # Iterate latent passes
            for pass_idx in range(max_n_latents):
                buckets = self._bucket_fill_positions(latent_lists, pass_idx, latent_order=latent_order)
                if not buckets:
                    continue

                # Process each bucket (same latent position p)
                # Sorting p makes behavior deterministic.
                for p in sorted(buckets.keys()):
                    bs = buckets[p]
                    bs_t = torch.tensor(bs, device=device, dtype=torch.long)

                    # ── 1) Prefix forward up to p (exclusive) to obtain:
                    #      - hidden at p-1 (for filling)
                    #      - KV cache for prefix [0..p-1]
                    prefix_out = self.phase0.model(
                        inputs_embeds=inputs_embeds[bs_t, :p, :],
                        attention_mask=attention_mask[bs_t, :p],
                        position_ids=position_ids[bs_t, :p],
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    prefix_hidden = prefix_out.hidden_states[-1]  # [len(bs), p, H]
                    prefix_past = prefix_out.past_key_values

                    # Fill latent embedding at p with hidden(p-1)
                    inputs_embeds[bs_t, p, :] = prefix_hidden[:, p - 1, :]

                    # ── 2) Suffix forward from p to end using prefix KV
                    # Attention mask should usually be full-length when using past_key_values.
                    # Position ids are for the suffix tokens only.
                    # past_key_values from prefix_out already corresponds to seq_len=p,
                    # but we slice defensively.
                    prefix_past = self._slice_past_key_values(prefix_past, prefix_len=p)

                    _ = self.phase0.model(
                        inputs_embeds=inputs_embeds[bs_t, p:, :],
                        attention_mask=attention_mask[bs_t, :],
                        position_ids=position_ids[bs_t, p:],
                        past_key_values=prefix_past,
                        use_cache=False,
                        output_hidden_states=False,
                    )
                    # We don't need suffix hidden here to fill the current latent
                    # (we used prefix_hidden[p-1]). We just need the recompute to
                    # make subsequent passes consistent because later latents depend
                    # on changed embeddings. Next pass will recompute required prefixes.

            # Final readout: one full forward with the final mutated embeddings, then read <ANSWER>
            final_out = self.phase0.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            final_hidden = final_out.hidden_states[-1]
            return self._compute_digit_logits_from_hidden(final_hidden, input_ids)

        logits = run_coconut_kvcache(latent_order=None)

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if return_debug:
            out["debug"] = Phase1ForwardDebug(
                max_n_latents=max_n_latents,
                latent_counts=[len(x) for x in latent_lists],
            )
        return out
