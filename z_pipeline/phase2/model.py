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
        self.z_selector = nn.Linear(self.hidden_size, self.z_vocab_size, bias=True, dtype=torch.bfloat16)

        # Embedding module (Z rows live here)
        self._emb = self.base.get_input_embeddings()

        # Freeze
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        for p in self.digit_heads.parameters():
            p.requires_grad = True

        # Re-enable embedding grads + mask to Z rows only
        self._emb.weight.requires_grad = True
        z_row_mask = torch.zeros(self._emb.weight.shape[0], dtype=torch.bfloat16)
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
            input_ids: torch.Tensor,  # [B,T]
            attention_mask: torch.Tensor,  # [B,T]
            latent_states: torch.Tensor,  # [B,Kmax,H]
            z_mask: torch.Tensor,  # [B,Kmax] bool or 0/1
            temperature: float,
            z_mode: str = "soft",
            return_z_probs: bool = False,
            return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:

        if z_mode not in {"soft", "hard_argmax", "hard_sample"}:
            raise ValueError(f"Invalid z_mode: {z_mode}")

        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be [B,T]")
        if latent_states.ndim != 3:
            raise ValueError("latent_states must be [B,Kmax,H]")
        if z_mask.ndim != 2:
            raise ValueError("z_mask must be [B,Kmax]")

        B, T = input_ids.shape
        Kmax = z_mask.shape[1]
        device = input_ids.device

        z_mask_bool = z_mask.bool() if z_mask.dtype != torch.bool else z_mask

        if self.force_base_eval:
            self.base.eval()

        # --------------------------------------------------
        # Infer latent placeholder positions
        # --------------------------------------------------
        latent_positions = self._infer_latent_positions(input_ids, z_mask_bool)  # [B,Kmax]

        # --------------------------------------------------
        # Base embeddings
        # --------------------------------------------------
        inputs_embeds = self._emb(input_ids)  # [B,T,H]

        # --------------------------------------------------
        # Z selector logits (sanitize latents first!)
        # --------------------------------------------------
        h = latent_states.to(device=device, dtype=self.z_selector.weight.dtype)  # [B,K,H]
        # kill NaN/Inf anywhere (padding or buggy rows) so Linear can't produce NaNs
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        # (Optional but highly recommended) hard fail if ACTIVE positions are non-finite
        # If this triggers, you have a real dataset/collate bug.
        if not torch.isfinite(h[z_mask_bool]).all():
            raise RuntimeError("latent_states contains NaN/Inf in ACTIVE positions")

        z_logits = self.z_selector(h)  # [B,K,V]

        # --------------------------------------------------
        # Build a *finite* masked logits tensor (critical for sampling)
        # --------------------------------------------------
        # Use float32 for stable softmax/sampling even if model is bf16.
        z_logits_f32 = z_logits.to(torch.float32)
        z_logits_f32 = torch.where(
            z_mask_bool.unsqueeze(-1),
            z_logits_f32,
            torch.full_like(z_logits_f32, -1e4),  # finite "almost -inf"
        )
        z_logits_f32 = torch.nan_to_num(z_logits_f32, nan=-1e4, posinf=-1e4, neginf=-1e4)
        z_logits_f32 = torch.clamp(z_logits_f32, -50.0, 50.0)

        # --------------------------------------------------
        # Z selection + embedding
        # --------------------------------------------------
        Ez = self._get_z_embedding_matrix().to(device=device, dtype=inputs_embeds.dtype)  # [V,H]

        z_probs = None
        z_ids = None

        if z_mode == "soft":
            temp = float(temperature)
            if temp <= 0.0:
                raise ValueError("temperature must be >0 for soft mode")

            # softmax in fp32, then cast to embed dtype
            z_probs = F.softmax(z_logits_f32 / temp, dim=-1).to(inputs_embeds.dtype)  # [B,K,V]
            # ensure inactive positions are exactly 0 prob mass (nice for KL code)
            z_probs = z_probs * z_mask_bool.unsqueeze(-1).to(z_probs.dtype)

            z_emb = torch.matmul(z_probs, Ez)  # [B,K,H]

        elif z_mode == "hard_argmax":
            z_ids = torch.argmax(z_logits_f32, dim=-1)  # [B,K]
            z_ids = torch.where(z_mask_bool, z_ids, torch.zeros_like(z_ids))
            z_emb = Ez[z_ids]  # [B,K,H]

        elif z_mode == "hard_sample":
            temp = float(temperature)
            if temp <= 0.0:
                raise ValueError("temperature must be >0 for hard_sample mode")

            dist = torch.distributions.Categorical(logits=z_logits_f32 / temp)
            z_ids = dist.sample()  # [B,K]
            z_ids = torch.where(z_mask_bool, z_ids, torch.zeros_like(z_ids))
            z_emb = Ez[z_ids]  # [B,K,H]

        # --------------------------------------------------
        # Inject Z embeddings into latent positions
        # --------------------------------------------------
        active_bk = z_mask_bool.nonzero(as_tuple=False)  # [N,2]
        if active_bk.numel() > 0:
            b_idx = active_bk[:, 0]
            k_idx = active_bk[:, 1]
            p_idx = latent_positions[b_idx, k_idx]
            if (p_idx < 0).any():
                raise RuntimeError("Active z_mask but latent_positions == -1 (batch construction bug).")
            inputs_embeds[b_idx, p_idx] = z_emb[b_idx, k_idx]

        # --------------------------------------------------
        # LM forward
        # --------------------------------------------------
        position_ids = self._build_position_ids(attention_mask.to(device))
        out = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.to(device),
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )

        hidden_last = out.hidden_states[-1]
        digit_logits = self._compute_digit_logits_from_hidden(hidden_last, input_ids)

        # --------------------------------------------------
        # Outputs
        # --------------------------------------------------
        result: Dict[str, torch.Tensor] = {"digit_logits": digit_logits}

        # return_z_probs is only meaningful for soft (KL loss). For hard modes, z_probs=None.
        if return_z_probs:
            if z_probs is not None:
                result["z_probs"] = z_probs
            # Keep logits for debugging/analysis (masked fp32 is safest)
            result["z_logits"] = z_logits_f32

        if z_ids is not None:
            result["z_ids"] = z_ids

        if return_debug:
            result["latent_positions"] = latent_positions

        return result

    @torch.no_grad()
    def z_autoencoder_forward(
            self,
            *,
            latent_states: torch.Tensor,  # [B, Kmax, H]
            z_mask: torch.Tensor,  # [B, Kmax]
            temperature: float,
    ) -> torch.Tensor:
        """
        Z-only forward pass:
        returns z_probs [B, Kmax, V]
        """
        if latent_states.ndim != 3:
            raise ValueError("latent_states must be [B, Kmax, H]")
        if z_mask.ndim != 2:
            raise ValueError("z_mask must be [B, Kmax]")

        z_mask_bool = z_mask.bool()
        h = latent_states.to(dtype=self.z_selector.weight.dtype)

        z_logits = self.z_selector(h)  # [B, Kmax, V]
        z_logits = torch.clamp(z_logits, -30, 30)

        neg_inf = torch.finfo(z_logits.dtype).min
        z_logits = torch.where(z_mask_bool.unsqueeze(-1), z_logits, neg_inf)

        z_probs = F.softmax(z_logits / temperature, dim=-1)
        return z_probs

    def initialize_from_centroids(
        self,
        centroids: torch.Tensor,  # [V, H]
        *,
        normalize: bool = True,
        bias_zero: bool = True,
        selector_dtype: Optional[torch.dtype] = None,
        eps: float = 1e-8,
    ) -> None:
        """
        Initialize BOTH:
          1) Z embedding rows inside the base LM embedding table (at self.z_token_ids)
          2) z_selector weights (Linear(H -> V)) so logits ~ dot(h, centroid_j)

        centroids: Tensor[V, H] where row j is the centroid for Z_j.

        Notes:
        - If normalize=True, rows are L2-normalized before being written.
          This makes selection behave like cosine-sim (assuming h is similarly normalized).
        - selector_dtype controls the dtype written into z_selector weights.
          If None, uses self.z_selector.weight.dtype.
        """
        if not torch.is_tensor(centroids):
            raise TypeError("centroids must be a torch.Tensor")

        if centroids.ndim != 2:
            raise ValueError(f"centroids must be 2D [V,H], got shape={tuple(centroids.shape)}")

        V, H = centroids.shape
        if V != self.z_vocab_size:
            raise ValueError(
                f"centroids V mismatch: got V={V}, expected z_vocab_size={self.z_vocab_size}"
            )
        if H != self.hidden_size:
            raise ValueError(
                f"centroids H mismatch: got H={H}, expected hidden_size={self.hidden_size}"
            )

        # Optionally normalize on a safe dtype (float32) to avoid bf16 norm artifacts.
        c = centroids
        if normalize:
            c_fp32 = c.detach().to(dtype=torch.float32)
            norms = torch.linalg.norm(c_fp32, dim=1, keepdim=True).clamp_min(eps)
            c = (c_fp32 / norms).to(dtype=centroids.dtype, device=centroids.device)

        z_ids = torch.tensor(self.z_token_ids, device=c.device, dtype=torch.long)

        # ---- 1) Write centroids into the LM embedding table rows for Z tokens ----
        emb_w = self._emb.weight
        c_for_emb = c.to(device=emb_w.device, dtype=emb_w.dtype)

        with torch.no_grad():
            # emb_w[z_token_ids] = centroids
            emb_w.index_copy_(0, z_ids.to(device=emb_w.device), c_for_emb)

        # ---- 2) Write centroids into selector weights ----
        sel_w = self.z_selector.weight
        target_dtype = selector_dtype if selector_dtype is not None else sel_w.dtype
        c_for_sel = c.to(device=sel_w.device, dtype=target_dtype)

        with torch.no_grad():
            # selector.weight is [V,H]; each row corresponds to a Z token.
            sel_w.copy_(c_for_sel)

            if bias_zero and getattr(self.z_selector, "bias", None) is not None:
                self.z_selector.bias.zero_()
