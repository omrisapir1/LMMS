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
    ) -> torch.Tensor:
        """
        Replace the embedding at each instance's pass_idx-th latent position with
        hidden_state at (latent_pos - 1).

        IMPORTANT:
        - No detach: full gradient flows through hidden -> embed -> next pass.
        - Uses clone() to avoid in-place mutation on a tensor that autograd may reuse.
        """
        # inputs_embeds: [B, T, H]
        # hidden:        [B, T, H] (last layer hidden states)
        B, T, H = inputs_embeds.shape

        # Identify which (batch, pos) pairs are active for this pass_idx
        fill_b: List[int] = []
        fill_pos: List[int] = []
        src_pos: List[int] = []

        for b, pos_list in enumerate(latent_lists):
            if len(pos_list) > pass_idx:
                p = pos_list[pass_idx]
                # Latent at position 0 would have no preceding token
                if p <= 0:
                    raise RuntimeError("Found <|latent|> at position 0; cannot use token_idx-1 rule.")
                fill_b.append(b)
                fill_pos.append(p)
                src_pos.append(p - 1)

        # If no fills for this pass, return inputs_embeds unchanged
        if len(fill_b) == 0:
            return inputs_embeds

        # Clone to avoid in-place ops on a tensor that might be needed elsewhere
        new_embeds = inputs_embeds.clone()

        # Advanced indexing assignment:
        # new_embeds[fill_b, fill_pos, :] = hidden[fill_b, src_pos, :]
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

        # Compute latent positions per sample
        latent_lists, max_n_latents = self._latent_lists(input_ids)

        # Build initial embeddings from tokens
        inputs_embeds = self._embedding(input_ids)  # [B, T, H]

        # Build position_ids (some models behave better with it)
        position_ids = self._build_position_ids(attention_mask)  # [B, T]

        # Iteratively fill latent embeddings from left->right
        # Note: this is the simple (non-KV-cache) version: recompute full forward each iteration.
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

        # Final forward after all latent replacements
        final_outputs = self.phase0.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        final_hidden = final_outputs.hidden_states[-1]  # [B, T, H]

        # Compute digit logits from <ANSWER> hidden state (Phase0 logic)
        logits = self._compute_digit_logits_from_hidden(final_hidden, input_ids)

        out: Dict[str, torch.Tensor] = {"logits": logits}

        if return_debug:
            latent_counts = [len(x) for x in latent_lists]
            dbg = Phase1ForwardDebug(max_n_latents=max_n_latents, latent_counts=latent_counts)
            # Store as a python object (not a tensor); trainer/eval can ignore it safely
            out["debug"] = dbg  # type: ignore[assignment]

        return out