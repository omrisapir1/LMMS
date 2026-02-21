from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .dataset import LATENT_TOKEN


PERM_TRUNCATE_RATIO = 0.5


@dataclass
class Phase1Forward:
    logits_orig: torch.Tensor
    logits_aux: Optional[torch.Tensor]
    aux_enabled_mask: torch.Tensor
    latent_counts: torch.Tensor


class Phase1CoconutModel(nn.Module):
    """
    Prefix-optimized Coconut latent execution.
    - For each latent slot position p: run prefix [:p], take hidden[p-1], inject at p.
    - Final full forward produces full-vocab next-token logits.
    """

    def __init__(
        self,
        *,
        base_model: nn.Module,
        latent_token_id: int,
        perm_truncate_ratio: float = PERM_TRUNCATE_RATIO,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.latent_token_id = int(latent_token_id)
        self.perm_truncate_ratio = float(perm_truncate_ratio)
        if not (0.0 <= self.perm_truncate_ratio <= 1.0):
            raise ValueError("perm_truncate_ratio must be in [0,1].")

        self._embedding = self.base_model.get_input_embeddings()

    def save_pretrained(self, save_directory: str) -> None:
        self.base_model.save_pretrained(save_directory)

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        self.base_model.resize_token_embeddings(new_num_tokens)
        self._embedding = self.base_model.get_input_embeddings()

    @property
    def config(self):
        return self.base_model.config

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.cumsum(attention_mask.to(torch.long), dim=1) - 1
        return torch.clamp(pos, min=0)

    def _assert_right_padding(self, attention_mask: torch.Tensor) -> None:
        """
        This implementation assumes right-padding:
        valid tokens (1s) must appear before padding (0s) in each row.
        """
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be rank-2.")
        attn = attention_mask.to(torch.long)
        # Invalid if any row has a 0->1 transition.
        if bool((attn[:, 1:] > attn[:, :-1]).any().item()):
            raise ValueError(
                "Left/mixed padding is not supported. Expected right-padded attention_mask."
            )

    def _extract_latent_positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[List[int]]:
        out: List[List[int]] = []
        for row, attn_row in zip(input_ids, attention_mask):
            positions = ((row == self.latent_token_id) & (attn_row == 1)).nonzero(
                as_tuple=False
            ).view(-1)
            out.append([int(x.item()) for x in positions])
        return out

    def _build_aux_orders(
        self,
        latent_positions: Sequence[Sequence[int]],
        *,
        seed: int,
    ) -> tuple[List[List[int]], torch.Tensor]:
        """
        Returns:
        - orders over latent-slot indices (not absolute token positions) per sample
        - aux_enabled_mask (n>=2)
        """
        enabled: List[bool] = []
        orders: List[List[int]] = []

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        for positions in latent_positions:
            n = len(positions)
            if n <= 1:
                enabled.append(False)
                orders.append([])
                continue
            enabled.append(True)
            do_truncate = bool(
                torch.rand((), generator=gen, device="cpu").item() < self.perm_truncate_ratio
            )
            if do_truncate:
                m = max(1, (n + 1) // 2)
                orders.append(list(range(m)))
                continue
            if n == 2:
                orders.append([1, 0])
            else:
                orders.append(list(reversed(range(n))))

        return orders, torch.tensor(enabled, dtype=torch.bool)

    def _run_path(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        latent_positions: Sequence[Sequence[int]],
        fill_orders: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """
        Executes one latent path:
        - prefix forwards only for latent fill
        - one final full forward for logits
        """
        device = input_ids.device
        inputs_embeds = self._embedding(input_ids)
        max_steps = max((len(o) for o in fill_orders), default=0)

        for step_idx in range(max_steps):
            # Bucket samples by absolute latent position p for shared prefix forwards.
            buckets: Dict[int, List[int]] = {}
            for b, order in enumerate(fill_orders):
                if step_idx >= len(order):
                    continue
                latent_slot_index = int(order[step_idx])
                if latent_slot_index >= len(latent_positions[b]):
                    continue
                p = int(latent_positions[b][latent_slot_index])
                if p <= 0:
                    raise RuntimeError(f"{LATENT_TOKEN} cannot appear at position 0.")
                buckets.setdefault(p, []).append(b)

            for p in sorted(buckets.keys()):
                rows = buckets[p]
                row_idx = torch.tensor(rows, dtype=torch.long, device=device)
                out = self.base_model(
                    inputs_embeds=inputs_embeds.index_select(0, row_idx)[:, :p, :],
                    attention_mask=attention_mask.index_select(0, row_idx)[:, :p],
                    position_ids=position_ids.index_select(0, row_idx)[:, :p],
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden = out.last_hidden_state
                inputs_embeds[row_idx, p, :] = hidden[:, p - 1, :]

        final = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return final.logits

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        compute_aux: bool = False,
        aux_seed: int = 0,
    ) -> Phase1Forward:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be rank-2 tensors.")

        attention_mask = attention_mask.to(input_ids.device)
        self._assert_right_padding(attention_mask)
        position_ids = self._build_position_ids(attention_mask)
        latent_positions = self._extract_latent_positions(input_ids, attention_mask)
        latent_counts = torch.tensor(
            [len(x) for x in latent_positions],
            dtype=torch.long,
            device=input_ids.device,
        )
        normal_orders = [list(range(len(x))) for x in latent_positions]

        logits_orig = self._run_path(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            latent_positions=latent_positions,
            fill_orders=normal_orders,
        )

        logits_aux: Optional[torch.Tensor] = None
        aux_enabled = torch.zeros_like(latent_counts, dtype=torch.bool)
        if compute_aux:
            aux_orders, aux_enabled_cpu = self._build_aux_orders(
                latent_positions=latent_positions,
                seed=int(aux_seed),
            )
            aux_enabled = aux_enabled_cpu.to(device=input_ids.device)
            if bool(aux_enabled.any().item()):
                logits_aux = self._run_path(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    latent_positions=latent_positions,
                    fill_orders=aux_orders,
                )

        return Phase1Forward(
            logits_orig=logits_orig,
            logits_aux=logits_aux,
            aux_enabled_mask=aux_enabled,
            latent_counts=latent_counts,
        )
