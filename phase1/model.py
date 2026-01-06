# phase1/model.py
#
# Phase 1 Coconut-style latent execution wrapper around Phase0Model.
# - Keeps full gradient flow through latent feedback (no detach).
# - Uses "hidden state from token - 1" to fill each <|latent|> position.
# - Does NOT implement KV-cache optimization (we recompute full forward per latent step).
# - Produces Phase0-style digit logits: [B, 5, 10] taken from the hidden state at <ANSWER>.
#
# Expected usage (in phase1/train.py):
#   phase0 = Phase0Model.from_pretrained(...)
#   model = Phase1CoconutModel(phase0_model=phase0, latent_token_id=latent_id)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

    Forward returns:
      {
        "logits": Tensor[B, 5, 10],
        "debug": Optional[Phase1ForwardDebug]   # only if return_debug=True
      }
    """

    def __init__(
        self,
        phase0_model: nn.Module,
        latent_token_id: int,
        *,
        enforce_single_answer_token: bool = True,
    ):
        super().__init__()

        # Underlying Phase-0 model (contains base transformer + digit heads + answer_token_id)
        self.phase0 = phase0_model

        # Latent token id (tokenizer.convert_tokens_to_ids("<|latent|>"))
        self.latent_token_id = int(latent_token_id)

        # Safety toggle (Phase0 already enforces this; we keep it here for clarity)
        self.enforce_single_answer_token = bool(enforce_single_answer_token)

        # Cache embedding handle (for inputs_embeds construction)
        # Phase0Model stores the base model at phase0.model
        self._embedding = self.phase0.model.get_input_embeddings()
        # Latent projection: identity-initialized linear map
        # Operates ONLY on latent reinsertion
        emb_w = self._embedding.weight
        H = self._embedding.embedding_dim

        self.latent_proj = nn.Linear(H, H, bias=True, device=emb_w.device, dtype=emb_w.dtype)

        # Identity init (must match dtype/device too)
        with torch.no_grad():
            self.latent_proj.weight.zero_()
            self.latent_proj.bias.zero_()

    # ─────────────────────────────────────────────────────────
    # Convenience passthroughs (helps with training + saving)
    # ─────────────────────────────────────────────────────────

    @property
    def config(self):
        # Delegate config for convenience
        return getattr(self.phase0, "config", None)

    def save_pretrained(self, *args, **kwargs):
        # Delegate saving to the underlying Phase0Model so HF format works.
        if hasattr(self.phase0, "save_pretrained"):
            return self.phase0.save_pretrained(*args, **kwargs)
        raise AttributeError("Underlying phase0_model does not implement save_pretrained().")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Optional: you can implement later if you want to load Phase1 directly.
        raise NotImplementedError(
            "Phase1CoconutModel.from_pretrained is not implemented. "
            "Load Phase0Model.from_pretrained(...) and wrap it."
        )

    # ─────────────────────────────────────────────────────────
    # Core helpers
    # ─────────────────────────────────────────────────────────

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Build monotonic position_ids compatible with most decoder-only LMs.
        If attention_mask is all-ones (no padding), this is just arange(T).
        """
        # attention_mask: [B, T] with 1 for tokens, 0 for padding
        B, T = attention_mask.shape
        device = attention_mask.device

        # Standard approach: cumulative sum - 1 gives positions for non-pad tokens.
        # Clamp to 0 for padded positions (won't matter if attention_mask zeros them out).
        position_ids = torch.cumsum(attention_mask.to(torch.long), dim=1) - 1
        position_ids = torch.clamp(position_ids, min=0)

        # Ensure shape [B, T]
        if position_ids.shape != (B, T):
            position_ids = position_ids.view(B, T)

        return position_ids.to(device)

    def _latent_lists(self, input_ids: torch.Tensor) -> Tuple[List[List[int]], int]:
        """
        Compute per-instance latent positions and max_n_latents in batch.
        Returns:
          latent_lists: List[List[int]] length B, each list sorted left->right
          max_n_latents: int
        """
        # input_ids: [B, T]
        B, T = input_ids.shape
        # nonzero returns indices [N, 2] where [:,0]=batch, [:,1]=pos
        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)

        latent_lists: List[List[int]] = []
        for b in range(B):
            # Collect positions for this batch item
            pos_list = [int(idx[1].item()) for idx in latent_indices if int(idx[0].item()) == b]
            pos_list.sort()
            latent_lists.append(pos_list)

        max_n_latents = 0
        for lst in latent_lists:
            if len(lst) > max_n_latents:
                max_n_latents = len(lst)

        return latent_lists, max_n_latents

    def _replace_latent_embeddings(
            self,
            inputs_embeds: torch.Tensor,
            hidden: torch.Tensor,
            latent_lists: List[List[int]],
            pass_idx: int,
            hidden_offset: int,
    ) -> torch.Tensor:
        """
        Replace the embedding at each instance's pass_idx-th latent position with
        hidden_state at (latent_pos - 1), where `hidden` corresponds to the slice
        [hidden_offset : hidden_offset + hidden.size(1)] of the full sequence.

        IMPORTANT:
        - No detach: full gradient flows through hidden -> embed -> next pass.
        - Uses clone() to avoid in-place mutation on a tensor that autograd may reuse.
        """
        B, T, H = inputs_embeds.shape

        fill_b: List[int] = []
        fill_pos: List[int] = []
        src_pos_in_hidden: List[int] = []

        hidden_T = hidden.size(1)

        for b, pos_list in enumerate(latent_lists):
            if len(pos_list) > pass_idx:
                p = pos_list[pass_idx]
                if p <= 0:
                    raise RuntimeError("Found <|latent|> at position 0; cannot use token_idx-1 rule.")

                # We need hidden at absolute position (p-1). Convert to index within hidden slice.
                abs_src = p - 1
                rel_src = abs_src - hidden_offset

                if rel_src < 0 or rel_src >= hidden_T:
                    raise RuntimeError(
                        f"Latent fill requires hidden at abs pos {abs_src}, "
                        f"but current hidden slice covers [{hidden_offset}, {hidden_offset + hidden_T})."
                    )

                fill_b.append(b)
                fill_pos.append(p)
                src_pos_in_hidden.append(rel_src)

        if len(fill_b) == 0:
            return inputs_embeds

        new_embeds = inputs_embeds.clone()

        fb = torch.tensor(fill_b, device=inputs_embeds.device, dtype=torch.long)
        fp = torch.tensor(fill_pos, device=inputs_embeds.device, dtype=torch.long)
        sp = torch.tensor(src_pos_in_hidden, device=inputs_embeds.device, dtype=torch.long)

        h = hidden[fb, sp, :]
        new_embeds[fb, fp, :] = h + self.latent_proj(h)

        return new_embeds

    def _compute_digit_logits_from_hidden(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same logic as Phase0Model.forward:
        - find <ANSWER> token position
        - gather hidden at that position
        - apply digit heads
        Returns logits [B, 5, 10]
        """
        # hidden: [B, T, H]
        # input_ids: [B, T]

        answer_token_id = getattr(self.phase0, "answer_token_id", None)
        if answer_token_id is None:
            raise RuntimeError("phase0_model must expose answer_token_id")

        answer_mask = (input_ids == int(answer_token_id))  # [B, T]

        if self.enforce_single_answer_token:
            if not torch.all(answer_mask.sum(dim=1) == 1):
                raise RuntimeError("Each sample must contain exactly one <ANSWER> token")

        # First occurrence index (since we enforce exactly one, this is the one)
        idx = answer_mask.float().argmax(dim=1)  # [B]
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)

        answer_hidden = hidden[batch_idx, idx]  # [B, H]

        # Apply digit heads from Phase0Model
        digit_heads = getattr(self.phase0, "digit_heads", None)
        if digit_heads is None:
            raise RuntimeError("phase0_model must expose digit_heads (nn.ModuleList)")

        logits = torch.stack([head(answer_hidden) for head in digit_heads], dim=1)  # [B, 5, 10]
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
        """
        Coconut-style forward:
        1) Build inputs_embeds from tokens.
        2) For each latent step k (0..max_n_latents-1):
           - Run base transformer (full sequence) to get hidden states
           - Replace k-th latent embedding per sample using hidden[token_idx-1]
        3) Final run to get hidden states after all latent replacements
        4) Extract digit logits at <ANSWER> token exactly like Phase0

        Returns:
          {"logits": [B,5,10]} (+ optional "debug")
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, T]")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be [B, T]")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have same shape")

        # Ensure tensors are on same device
        device = input_ids.device
        attention_mask = attention_mask.to(device)
        # Build initial embeddings from tokens
        inputs_embeds = self._embedding(input_ids)  # [B, T, H]
        position_ids = self._build_position_ids(attention_mask)  # [B, T]

        # Compute latent positions per sample
        latent_lists, max_n_latents = self._latent_lists(input_ids)

        T = input_ids.size(1)

        # If no latents, do a single forward and exit early
        if max_n_latents == 0:
            final_outputs = self.phase0.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            final_hidden = final_outputs.hidden_states[-1]
            logits = self._compute_digit_logits_from_hidden(final_hidden, input_ids)
            out = {"logits": logits}
            if return_debug:
                out["debug"] = Phase1ForwardDebug(max_n_latents=0,
                                                  latent_counts=[len(x) for x in latent_lists])  # type: ignore
            return out

        # --- KV-cache Coconut-style incremental compute ---
        # Start by computing up to the earliest latent position (prefix).
        all_latent_positions = [p for lst in latent_lists for p in lst]
        first_latent_pos = min(all_latent_positions)  # earliest latent across batch

        # next_compute_range = [start, end) segment to compute this pass
        next_compute_range = (0, first_latent_pos)  # prefix before earliest latent
        kv_cache = None

        # We'll update inputs_embeds by writing the k-th latent embed each pass.
        for pass_idx in range(max_n_latents):
            r0, r1 = next_compute_range

            if r1 <= r0:
                # Degenerate range; in practice first_latent_pos could be 0 (should be blocked earlier)
                raise RuntimeError(f"Invalid compute range: {next_compute_range}")

            if kv_cache is None:
                # First pass: no cache, compute prefix slice only
                outputs = self.phase0.model(
                    inputs_embeds=inputs_embeds[:, r0:r1, :],
                    attention_mask=attention_mask[:, r0:r1],
                    position_ids=position_ids[:, r0:r1],
                    use_cache=True,
                    output_hidden_states=True,
                )
                hidden_offset = 0
            else:
                # Slice cache to valid prefix length r0
                past_key_values = [
                    (k[:, :, :r0, :], v[:, :, :r0, :])
                    for (k, v) in kv_cache
                ]

                outputs = self.phase0.model(
                    inputs_embeds=inputs_embeds[:, r0:r1, :],
                    attention_mask=attention_mask[:, :r1],  # prefix + current slice
                    position_ids=position_ids[:, r0:r1],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                hidden_offset = r0

            hidden_slice = outputs.hidden_states[-1]  # [B, r1-r0, H]
            kv_cache = outputs.past_key_values

            # Replace the pass_idx-th latent per sample using hidden(token_idx-1)
            inputs_embeds = self._replace_latent_embeddings(
                inputs_embeds=inputs_embeds,
                hidden=hidden_slice,
                latent_lists=latent_lists,
                pass_idx=pass_idx,
                hidden_offset=hidden_offset,
            )

            # Advance compute range:
            # - after prefix, we incrementally compute one more token each pass until all latents filled,
            # - on the last latent pass, we jump to the full remainder in the final pass below.
            if pass_idx + 1 >= max_n_latents:
                next_compute_range = (r1, T)
            else:
                next_compute_range = (r1, min(r1 + 1, T))

        # Final pass: compute the remainder [next_compute_range[0], T)
        r0, r1 = next_compute_range
        if r0 < r1:
            if kv_cache is None:
                # Shouldn't happen here, but keep it safe
                final_outputs = self.phase0.model(
                    inputs_embeds=inputs_embeds[:, r0:r1, :],
                    attention_mask=attention_mask[:, r0:r1],
                    position_ids=position_ids[:, r0:r1],
                    use_cache=False,
                    output_hidden_states=True,
                )
                final_hidden_slice = final_outputs.hidden_states[-1]
                final_hidden_offset = r0
            else:
                past_key_values = [
                    (k[:, :, :r0, :], v[:, :, :r0, :])
                    for (k, v) in kv_cache
                ]
                final_outputs = self.phase0.model(
                    inputs_embeds=inputs_embeds[:, r0:r1, :],
                    attention_mask=attention_mask[:, :r1],
                    position_ids=position_ids[:, r0:r1],
                    past_key_values=past_key_values,
                    use_cache=False,  # no need to continue caching
                    output_hidden_states=True,
                )
                final_hidden_slice = final_outputs.hidden_states[-1]
                final_hidden_offset = r0
        else:
            raise RuntimeError("Final compute range is empty; unexpected for sequences with <ANSWER> at end.")

        # We need hidden at <ANSWER>. In your data, <ANSWER> is last ⇒ it should be in the final slice.
        answer_token_id = int(getattr(self.phase0, "answer_token_id"))
        answer_pos = (input_ids == answer_token_id).float().argmax(dim=1)  # [B]

        # Assert answer is inside final slice; if not, fallback (or raise)
        min_pos = int(answer_pos.min().item())
        max_pos = int(answer_pos.max().item())
        if min_pos < final_hidden_offset or max_pos >= final_hidden_offset + final_hidden_slice.size(1):
            raise RuntimeError(
                f"<ANSWER> positions [{min_pos},{max_pos}] not covered by final hidden slice "
                f"[{final_hidden_offset},{final_hidden_offset + final_hidden_slice.size(1)}). "
                "If this can happen, we need to assemble full hidden states or keep computing ranges longer."
            )

        # Gather answer hidden from final slice
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        rel_answer_pos = answer_pos - final_hidden_offset  # [B]
        answer_hidden = final_hidden_slice[batch_idx, rel_answer_pos, :]  # [B, H]

        # Apply digit heads (same as Phase0)
        digit_heads = getattr(self.phase0, "digit_heads", None)
        if digit_heads is None:
            raise RuntimeError("phase0_model must expose digit_heads (nn.ModuleList)")
        logits = torch.stack([head(answer_hidden) for head in digit_heads], dim=1)  # [B,5,10]

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if return_debug:
            out["debug"] = Phase1ForwardDebug(max_n_latents=max_n_latents,
                                              latent_counts=[len(x) for x in latent_lists])  # type: ignore
        return out

