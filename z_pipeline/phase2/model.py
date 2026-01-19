# phase2/model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Phase2Output:
    digit_logits: torch.Tensor                 # [B, 5, 10]
    z_probs: Optional[torch.Tensor] = None     # [B, Kmax, V]
    latent_positions: Optional[torch.Tensor] = None  # [B, Kmax] (debug)


class Phase2ZModel(nn.Module):
    """
    Phase-2: Learn Z-token embeddings (inside LM embedding table) + Z-selector from digit loss ONLY.

    Batch contract (after collate/pad):
      - input_ids:      [B, T]  (suffix pad only, after <ANSWER>)
      - attention_mask: [B, T]
      - latent_states:  [B, Kmax, H]  (padded to Kmax; inactive rows can be zeros)
      - z_mask:         [B, Kmax] bool (True for active steps)
    """

    def __init__(
        self,
        *,
        base_lm: nn.Module,
        digit_heads: nn.ModuleList,
        answer_token_id: int,
        latent_token_id: int,
        z_token_ids: List[int],          # token ids for <Z_0>.. <Z_{V-1}> in order
        freeze_base: bool = True,
        freeze_digit_heads: bool = True,
        force_base_eval: bool = True,
    ):
        super().__init__()

        if not isinstance(z_token_ids, list) or len(z_token_ids) == 0:
            raise ValueError("z_token_ids must be a non-empty List[int]")

        self.base = base_lm
        self.digit_heads = digit_heads

        self.answer_token_id = int(answer_token_id)
        self.latent_token_id = int(latent_token_id)

        self.z_token_ids = [int(x) for x in z_token_ids]
        self.z_vocab_size = len(self.z_token_ids)

        # Infer hidden size
        hidden_size = getattr(getattr(self.base, "config", None), "hidden_size", None)
        if hidden_size is None:
            emb = self.base.get_input_embeddings()
            hidden_size = emb.embedding_dim
        self.hidden_size = int(hidden_size)

        # Trainable selector
        self.z_selector = nn.Linear(self.hidden_size, self.z_vocab_size, bias=True)

        # Embedding module (Z rows live here)
        self._emb = self.base.get_input_embeddings()

        # Freeze
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        if freeze_digit_heads:
            for p in self.digit_heads.parameters():
                p.requires_grad = False

        # Re-enable embedding grads + mask to Z rows only
        self._emb.weight.requires_grad = True
        z_row_mask = torch.zeros(self._emb.weight.shape[0], dtype=torch.float32)
        z_row_mask[self.z_token_ids] = 1.0
        self.register_buffer("_z_row_grad_mask", z_row_mask, persistent=False)

        def _mask_embedding_grads(grad: torch.Tensor) -> torch.Tensor:
            return grad * self._z_row_grad_mask.unsqueeze(-1).to(grad.dtype)

        self._emb.weight.register_hook(_mask_embedding_grads)

        self.force_base_eval = bool(force_base_eval)
        if self.force_base_eval:
            self.base.eval()

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.cumsum(attention_mask.to(torch.long), dim=1) - 1
        return torch.clamp(pos, min=0)

    def _infer_latent_positions(
        self,
        input_ids: torch.Tensor,   # [B,T]
        z_mask: torch.Tensor,      # [B,Kmax] bool
    ) -> torch.Tensor:
        """
        Variable-K safe:
          - each sample must have exactly K occurrences of <|latent|>, where K = sum(z_mask[b])
          - returns positions padded with -1 to Kmax
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, T]")
        if z_mask.ndim != 2:
            raise ValueError("z_mask must be [B, Kmax]")

        B, _T = input_ids.shape
        Kmax = z_mask.shape[1]
        device = input_ids.device

        positions_out = torch.full((B, Kmax), -1, device=device, dtype=torch.long)

        # per-sample scan (B is small; K is small)
        for b in range(B):
            k = int(z_mask[b].sum().item())
            pos = (input_ids[b] == self.latent_token_id).nonzero(as_tuple=False).view(-1)
            if pos.numel() != k:
                raise RuntimeError(
                    f"Phase2 error: sample {b} has {pos.numel()} '<|latent|>' placeholders "
                    f"but z_mask indicates K={k}. "
                    f"(Check batch construction: must include exactly K placeholders.)"
                )
            if k > 0:
                pos_sorted, _ = torch.sort(pos)
                positions_out[b, :k] = pos_sorted[:k]
        return positions_out

    def _get_z_embedding_matrix(self) -> torch.Tensor:
        # [V, H] from LM embedding table
        return self._emb.weight[self.z_token_ids]

    def _compute_digit_logits_from_hidden(
        self,
        hidden_last: torch.Tensor,   # [B,T,H]
        input_ids: torch.Tensor,     # [B,T]
    ) -> torch.Tensor:
        answer_mask = (input_ids == self.answer_token_id)
        if not torch.all(answer_mask.sum(dim=1) == 1):
            raise RuntimeError("Each sample must contain exactly one <ANSWER> token")

        idx = answer_mask.float().argmax(dim=1)
        bidx = torch.arange(hidden_last.size(0), device=hidden_last.device)
        answer_hidden = hidden_last[bidx, idx]  # [B,H]
        return torch.stack([head(answer_hidden) for head in self.digit_heads], dim=1)  # [B,5,10]

    # ─────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────

    def forward(
        self,
        *,
        input_ids: torch.Tensor,          # [B,T]
        attention_mask: torch.Tensor,     # [B,T]
        latent_states: torch.Tensor,      # [B,Kmax,H]  (already states[:-1], padded)
        z_mask: torch.Tensor,             # [B,Kmax] bool or 0/1
        temperature: float,
        return_z_probs: bool = False,
        return_debug: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Phase2Output]:

        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be [B, T]")
        if latent_states.ndim != 3:
            raise ValueError("latent_states must be [B, Kmax, H]")
        if z_mask.ndim != 2:
            raise ValueError("z_mask must be [B, Kmax]")

        B, T = input_ids.shape
        Kmax = z_mask.shape[1]

        if latent_states.shape[0] != B:
            raise ValueError("latent_states batch size mismatch")
        if latent_states.shape[1] != Kmax:
            raise ValueError(f"latent_states second dim must be Kmax={Kmax}, got {latent_states.shape[1]}")
        if latent_states.shape[2] != self.hidden_size:
            raise ValueError(f"latent_states hidden size mismatch: expected H={self.hidden_size}")

        # normalize mask to bool
        z_mask_bool = z_mask.bool() if z_mask.dtype != torch.bool else z_mask

        temp = float(temperature)
        if temp <= 0.0:
            raise ValueError("temperature must be > 0")

        device = input_ids.device
        attention_mask = attention_mask.to(device)

        if self.force_base_eval:
            self.base.eval()

        # infer placeholder positions for *active* steps
        latent_positions = self._infer_latent_positions(input_ids, z_mask_bool)  # [B,Kmax]

        # base embeddings
        inputs_embeds = self._emb(input_ids)  # [B,T,H]

        # selector logits only for active steps (mask others to -inf so probs ~0)
        h = latent_states.to(device=device, dtype=inputs_embeds.dtype)  # [B,Kmax,H]
        z_logits = self.z_selector(h)  # [B,Kmax,V]

        # mask inactive positions so they don't contribute to z_probs / KL
        neg_inf = torch.finfo(z_logits.dtype).min
        z_logits = torch.where(z_mask_bool.unsqueeze(-1), z_logits, neg_inf)

        z_probs = F.softmax(z_logits / temp, dim=-1)  # [B,Kmax,V]

        # soft embeddings: [B,Kmax,H] = z_probs @ Ez
        Ez = self._get_z_embedding_matrix().to(device=device, dtype=inputs_embeds.dtype)  # [V,H]
        soft_z = torch.matmul(z_probs, Ez)  # [B,Kmax,H]

        # inject only active positions (vectorized)
        # gather (b,k) where active
        active_bk = z_mask_bool.nonzero(as_tuple=False)  # [N,2] with columns [b,k]
        if active_bk.numel() > 0:
            b_idx = active_bk[:, 0]
            k_idx = active_bk[:, 1]
            p_idx = latent_positions[b_idx, k_idx]  # [N]
            if (p_idx < 0).any():
                raise RuntimeError("Found active z_mask but latent_positions == -1 (batch construction bug).")
            inputs_embeds[b_idx, p_idx, :] = soft_z[b_idx, k_idx, :]

        # LM forward
        position_ids = self._build_position_ids(attention_mask)
        out = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        hidden_last = out.hidden_states[-1]
        digit_logits = self._compute_digit_logits_from_hidden(hidden_last, input_ids)

        if return_debug:
            return Phase2Output(
                digit_logits=digit_logits,
                z_probs=z_probs if return_z_probs else None,
                latent_positions=latent_positions if return_debug else None,
            )
        if return_z_probs:
            return {'digit_logits': digit_logits, 'z_probs': z_probs}

        return {"digit_logits": digit_logits}
