# phase2/model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------

@dataclass
class Phase2Output:
    digit_logits: torch.Tensor                 # [B, 5, 10]
    z_probs: Optional[torch.Tensor] = None     # [B, Kmax, V] (one-hot)
    latent_positions: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------
# Phase-2 Z Model (hard VQ-style)
# ---------------------------------------------------------------------

class Phase2ZModel(nn.Module):
    """
    Phase-2 (hard Z version):

    latent_state → nearest centroid → Z embedding → inject → LM → digit heads

    No selector, no temperature, no softmax.
    """

    def __init__(
        self,
        *,
        base_lm: nn.Module,
        digit_heads: nn.ModuleList,
        answer_token_id: int,
        latent_token_id: int,
        z_token_ids: List[int],
        freeze_base: bool = True,
        freeze_digit_heads: bool = True,
        force_base_eval: bool = True,
    ):
        super().__init__()

        if not z_token_ids:
            raise ValueError("z_token_ids must be non-empty")

        self.base = base_lm
        self.digit_heads = digit_heads

        self.answer_token_id = int(answer_token_id)
        self.latent_token_id = int(latent_token_id)
        self.z_token_ids = list(map(int, z_token_ids))
        self.z_vocab_size = len(self.z_token_ids)

        # Infer hidden size
        hidden_size = getattr(getattr(self.base, "config", None), "hidden_size", None)
        if hidden_size is None:
            hidden_size = self.base.get_input_embeddings().embedding_dim
        self.hidden_size = int(hidden_size)

        # Embedding table (Z rows live here)
        self._emb = self.base.get_input_embeddings()

        # Freeze base LM
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # Digit heads
        for p in self.digit_heads.parameters():
            p.requires_grad = not freeze_digit_heads

        # Allow grads only on Z embedding rows
        self._emb.weight.requires_grad = True
        z_mask = torch.zeros(self._emb.weight.size(0), dtype=torch.float32)
        z_mask[self.z_token_ids] = 1.0
        self.register_buffer("_z_row_grad_mask", z_mask, persistent=False)

        def _mask_grads(grad: torch.Tensor) -> torch.Tensor:
            return grad * self._z_row_grad_mask.unsqueeze(-1).to(grad.dtype)

        self._emb.weight.register_hook(_mask_grads)

        self.force_base_eval = bool(force_base_eval)
        if self.force_base_eval:
            self.base.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.cumsum(attention_mask.to(torch.long), dim=1) - 1
        return torch.clamp(pos, min=0)

    def _get_z_embedding_matrix(self) -> torch.Tensor:
        return self._emb.weight[self.z_token_ids]  # [V, H]

    def _infer_latent_positions(
        self,
        input_ids: torch.Tensor,
        z_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Locate <|latent|> placeholders per row.
        Returns [B, Kmax] padded with -1.
        """
        B, T = input_ids.shape
        Kmax = z_mask.shape[1]
        device = input_ids.device

        out = torch.full((B, Kmax), -1, device=device, dtype=torch.long)

        for b in range(B):
            k = int(z_mask[b].sum().item())
            pos = (input_ids[b] == self.latent_token_id).nonzero(as_tuple=False).view(-1)
            if pos.numel() != k:
                raise RuntimeError(
                    f"Row {b}: found {pos.numel()} <|latent|> but z_mask says {k}"
                )
            if k > 0:
                out[b, :k] = pos.sort().values[:k]

        return out

    def _compute_digit_logits_from_hidden(
        self,
        hidden_last: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        answer_mask = (input_ids == self.answer_token_id)
        if not torch.all(answer_mask.sum(dim=1) == 1):
            raise RuntimeError("Each row must contain exactly one <ANSWER> token")

        idx = answer_mask.float().argmax(dim=1)
        bidx = torch.arange(hidden_last.size(0), device=hidden_last.device)
        h = hidden_last[bidx, idx]  # [B, H]
        return torch.stack([head(h) for head in self.digit_heads], dim=1)

    # ------------------------------------------------------------------
    # Hard Z assignment
    # ------------------------------------------------------------------

    @torch.no_grad()
    def assign_z_by_nearest(
        self,
        latent_states: torch.Tensor,  # [B, Kmax, H]
        z_mask: torch.Tensor,         # [B, Kmax]
    ) -> torch.Tensor:
        """
        Nearest-centroid assignment.
        Returns z_ids [B, Kmax].
        """
        z_mask = z_mask.bool()
        h = latent_states.to(self._emb.weight.dtype)
        Ez = self._get_z_embedding_matrix()

        # cosine distance works best here
        h = F.normalize(h, dim=-1)
        Ez = F.normalize(Ez, dim=-1)

        dists = torch.cdist(h, Ez)  # [B, K, V]
        z_ids = torch.argmin(dists, dim=-1)

        z_ids = torch.where(z_mask, z_ids, torch.zeros_like(z_ids))
        return z_ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_states: torch.Tensor,
        z_mask: torch.Tensor,
        temperature: Optional[float] = None,  # ignored (kept for API compat)
        return_z_probs: bool = False,
        return_debug: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Phase2Output]:

        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids / attention_mask must be [B, T]")
        if latent_states.ndim != 3:
            raise ValueError("latent_states must be [B, Kmax, H]")
        if z_mask.ndim != 2:
            raise ValueError("z_mask must be [B, Kmax]")

        B, T = input_ids.shape
        Kmax = z_mask.shape[1]

        if self.force_base_eval:
            self.base.eval()

        z_mask_bool = z_mask.bool()
        latent_positions = self._infer_latent_positions(input_ids, z_mask_bool)

        # Base embeddings
        inputs_embeds = self._emb(input_ids)

        # ---- HARD Z PATH ----
        z_ids = self.assign_z_by_nearest(latent_states, z_mask_bool)
        Ez = self._get_z_embedding_matrix().to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        z_emb = Ez[z_ids]  # [B, Kmax, H]

        # Inject
        active = z_mask_bool.nonzero(as_tuple=False)
        if active.numel() > 0:
            b_idx, k_idx = active[:, 0], active[:, 1]
            p_idx = latent_positions[b_idx, k_idx]
            inputs_embeds[b_idx, p_idx] = z_emb[b_idx, k_idx]

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

        z_probs = None
        if return_z_probs:
            z_probs = torch.zeros(
                B, Kmax, self.z_vocab_size,
                device=z_ids.device,
                dtype=inputs_embeds.dtype,
            )
            z_probs.scatter_(-1, z_ids.unsqueeze(-1), 1.0)

        if return_debug:
            return Phase2Output(
                digit_logits=digit_logits,
                z_probs=z_probs,
                latent_positions=latent_positions,
            )

        if return_z_probs:
            return {"digit_logits": digit_logits, "z_probs": z_probs}

        return {"digit_logits": digit_logits}

    # ------------------------------------------------------------------
    # Centroid initialization (unchanged)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def initialize_from_centroids(
            self,
            centroids: torch.Tensor,  # [V, H]
            *,
            normalize: bool = True,
            eps: float = 1e-8,
    ) -> None:

        if centroids.ndim != 2:
            raise ValueError("centroids must be [V, H]")

        V, H = centroids.shape
        if V != self.z_vocab_size:
            raise ValueError("V mismatch")
        if H != self.hidden_size:
            raise ValueError("H mismatch")

        # normalize on CPU or GPU safely
        c = centroids
        if normalize:
            c = c.float()
            c = c / torch.linalg.norm(c, dim=1, keepdim=True).clamp_min(eps)

        emb_device = self._emb.weight.device

        # move centroids to embedding device
        c = c.to(device=emb_device, dtype=self._emb.weight.dtype)

        z_ids = torch.tensor(
            self.z_token_ids,
            device=emb_device,
            dtype=torch.long,
        )

        # write Z embeddings
        self._emb.weight.index_copy_(0, z_ids, c)
