from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    def __init__(self, dim: int, vocab_size: int, ema_decay: float):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if not (0.0 < ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")

        self.dim = int(dim)
        self.vocab_size = int(vocab_size)
        self.ema_decay = float(ema_decay)

        init = torch.randn(self.vocab_size, self.dim, dtype=torch.float32)
        init = F.normalize(init, p=2, dim=-1, eps=1e-12)
        self.embeddings = nn.Parameter(init, requires_grad=False)  # [V, D]

        # Initialized to keep random vectors stable until first assignments.
        self.register_buffer(
            "ema_cluster_size",
            torch.ones(self.vocab_size, dtype=torch.float32),
        )
        self.register_buffer(
            "ema_embedding_sum",
            self.embeddings.detach().clone(),
        )

    @torch.no_grad()
    def initialize_embeddings(self, embeddings: torch.Tensor) -> None:
        if embeddings.shape != (self.vocab_size, self.dim):
            raise ValueError(
                f"Expected embeddings shape {(self.vocab_size, self.dim)}, got {tuple(embeddings.shape)}"
            )
        embeddings = embeddings.to(device=self.embeddings.device, dtype=torch.float32)
        embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-12)
        self.embeddings.copy_(embeddings)
        self.ema_embedding_sum.copy_(embeddings)
        self.ema_cluster_size.fill_(1.0)

    def forward(
        self, latents: torch.Tensor, *, return_similarity: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if latents.ndim != 2 or latents.shape[1] != self.dim:
            raise ValueError(f"latents must have shape [N, {self.dim}]")

        latents = latents.to(dtype=torch.float32)
        latents_norm = F.normalize(latents, p=2, dim=-1, eps=1e-12)
        emb_norm = F.normalize(self.embeddings, p=2, dim=-1, eps=1e-12)

        similarity = latents_norm @ emb_norm.transpose(0, 1)  # [N, V]
        z_ids = torch.argmax(similarity, dim=-1)  # [N]
        quantized = self.embeddings.index_select(0, z_ids)  # [N, D]
        if return_similarity:
            return z_ids, quantized, similarity
        return z_ids, quantized

    @torch.no_grad()
    def ema_update(
        self,
        latents: torch.Tensor,
        z_ids: torch.Tensor,
        *,
        eps: float = 1e-5,
    ) -> None:
        if latents.ndim != 2 or latents.shape[1] != self.dim:
            raise ValueError(f"latents must have shape [N, {self.dim}]")
        if z_ids.ndim != 1 or z_ids.shape[0] != latents.shape[0]:
            raise ValueError("z_ids must have shape [N] and align with latents")

        latents = latents.to(dtype=torch.float32)
        z_ids = z_ids.to(dtype=torch.long)

        counts = torch.bincount(z_ids, minlength=self.vocab_size).to(torch.float32)
        emb_sum = torch.zeros_like(self.ema_embedding_sum)
        emb_sum.index_add_(0, z_ids, latents)

        decay = self.ema_decay
        self.ema_cluster_size.mul_(decay).add_(counts, alpha=1.0 - decay)
        self.ema_embedding_sum.mul_(decay).add_(emb_sum, alpha=1.0 - decay)

        updated = self.ema_embedding_sum / (self.ema_cluster_size.unsqueeze(-1) + float(eps))
        # Keep embeddings on the unit sphere since assignment uses cosine similarity.
        updated = F.normalize(updated, p=2, dim=-1, eps=1e-12)
        self.embeddings.copy_(updated)
