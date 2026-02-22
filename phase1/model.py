from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .dataset import LATENT_TOKEN


PERM_TRUNCATE_RATIO = 0.5
DO_FULL_TRUNCATE = 0.5

@dataclass
class Phase1Forward:
    logits_orig: torch.Tensor
    logits_aux: Optional[torch.Tensor]
    aux_enabled_mask: torch.Tensor
    latent_counts: torch.Tensor
    latent_vectors_orig: Optional[torch.Tensor] = None
    latent_vectors_orig_mask: Optional[torch.Tensor] = None
    latent_vectors_aux: Optional[torch.Tensor] = None
    latent_vectors_aux_mask: Optional[torch.Tensor] = None


class Phase1CoconutModel(nn.Module):
    """
    Prefix-optimized Coconut latent execution.
    - For each latent slot position p: run prefix [:p], take hidden[p-1], inject at p.
    - Final full forward computes digit-only logits [B,5,10] via restricted LM-head projection.
    """

    def __init__(
        self,
        *,
        base_model: nn.Module,
        latent_token_id: int,
        answer_token_id: Optional[int] = None,
        digit_token_ids: Optional[Sequence[int]] = None,
        perm_truncate_ratio: float = PERM_TRUNCATE_RATIO,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.latent_token_id = int(latent_token_id)
        self.answer_token_id = (
            int(answer_token_id) if answer_token_id is not None else None
        )
        self.digit_token_ids: Optional[List[int]] = None
        if digit_token_ids is not None:
            self.set_digit_token_ids(digit_token_ids)
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

    def set_digit_token_ids(self, digit_token_ids: Sequence[int]) -> None:
        ids = [int(x) for x in digit_token_ids]
        if len(ids) != 10:
            raise ValueError("digit_token_ids must contain exactly 10 ids (0-9).")
        self.digit_token_ids = ids

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
    ) -> tuple[List[List[int]], torch.Tensor, List[str]]:
        """
        Returns:
        - orders over latent-slot indices (not absolute token positions) per sample
        - aux_enabled_mask (n>=2)
        - aux_mode per sample:
          "permute" | "partial_truncate" | "full_remove_latents" | "none"
        """
        enabled: List[bool] = []
        orders: List[List[int]] = []
        modes: List[str] = []

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        for positions in latent_positions:
            n = len(positions)
            if n <= 1:
                enabled.append(True)
                orders.append([])
                modes.append("none")
                continue
            enabled.append(True)
            do_truncate = bool(
                torch.rand((), generator=gen, device="cpu").item() < self.perm_truncate_ratio
            )

            if do_truncate:
                do_full_truncate = bool(
                    torch.rand((), generator=gen, device="cpu").item() < DO_FULL_TRUNCATE
                )
                if do_full_truncate:
                    orders.append([])
                    modes.append("full_remove_latents")
                else:
                    m = max(1, (n + 1) // 2)
                    orders.append(list(range(m)))
                    modes.append("partial_truncate")
                continue
            if n == 2:
                orders.append([1, 0])
            else:
                orders.append(list(reversed(range(n))))
            modes.append("permute")

        return orders, torch.tensor(enabled, dtype=torch.bool), modes

    def _build_no_latents_aux_batch(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        digit_position_indices: torch.Tensor,
        latent_positions: Sequence[Sequence[int]],
        select_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build an auxiliary batch where latent tokens are removed from valid tokens.
        This is required because fill_orders=[] alone only skips latent *filling*; it
        does not remove latent-token embeddings from the effective sequence.

        Returns:
        - aux_input_ids [B2,S2]
        - aux_attention_mask [B2,S2]
        - aux_digit_position_indices [B2,5] remapped to new positions
        - row_map [B2] mapping aux rows back to original batch rows
        """
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be rank-2.")
        if digit_position_indices.ndim != 2 or digit_position_indices.shape[1] != 5:
            raise ValueError("digit_position_indices must have shape [B,5].")
        if select_mask.ndim != 1 or select_mask.shape[0] != input_ids.shape[0]:
            raise ValueError("select_mask must have shape [B].")

        device = input_ids.device
        row_map = torch.nonzero(select_mask.to(torch.bool), as_tuple=False).squeeze(1)
        if row_map.numel() == 0:
            raise ValueError("_build_no_latents_aux_batch called with empty select_mask.")

        aux_rows_tokens: List[List[int]] = []
        aux_rows_digit_pos: List[List[int]] = []
        answer_counts: List[int] = []

        for b_t in row_map.tolist():
            b = int(b_t)
            row_ids = input_ids[b]
            row_attn = attention_mask[b].to(torch.long)
            row_digit_pos = digit_position_indices[b].to(torch.long)
            latent_set = {int(p) for p in latent_positions[b]}
            valid_idx = torch.nonzero(row_attn == 1, as_tuple=False).squeeze(1).tolist()

            keep_old_idx: List[int] = []
            for old_i in valid_idx:
                if int(old_i) in latent_set:
                    continue
                keep_old_idx.append(int(old_i))
            if not keep_old_idx:
                raise RuntimeError("No valid non-latent tokens remained after latent removal.")

            old_to_new = {old_i: new_i for new_i, old_i in enumerate(keep_old_idx)}

            remapped_digit_pos: List[int] = []
            for old_p_t in row_digit_pos.tolist():
                old_p = int(old_p_t)
                if old_p not in old_to_new:
                    raise RuntimeError(
                        "Digit supervision position was removed while dropping latent tokens."
                    )
                remapped_digit_pos.append(int(old_to_new[old_p]))

            kept_tokens = [int(row_ids[old_i].item()) for old_i in keep_old_idx]
            answer_count = sum(1 for tid in kept_tokens if self.answer_token_id is not None and tid == int(self.answer_token_id))
            answer_counts.append(answer_count)
            aux_rows_tokens.append(kept_tokens)
            aux_rows_digit_pos.append(remapped_digit_pos)

        if self.answer_token_id is not None:
            bad = [i for i, c in enumerate(answer_counts) if c != 1]
            if bad:
                raise RuntimeError(
                    "No-latents aux batch must contain exactly one <ANSWER> per sample."
                )

        pad_token_id = int(getattr(self.base_model.config, "pad_token_id", 0) or 0)
        b2 = int(row_map.numel())
        s2 = max(len(x) for x in aux_rows_tokens)
        aux_input_ids = torch.full((b2, s2), pad_token_id, dtype=torch.long, device=device)
        aux_attention_mask = torch.zeros((b2, s2), dtype=torch.long, device=device)
        aux_digit_pos = torch.full((b2, 5), -1, dtype=torch.long, device=device)

        for i in range(b2):
            n = len(aux_rows_tokens[i])
            aux_input_ids[i, :n] = torch.tensor(aux_rows_tokens[i], dtype=torch.long, device=device)
            aux_attention_mask[i, :n] = 1
            aux_digit_pos[i] = torch.tensor(aux_rows_digit_pos[i], dtype=torch.long, device=device)

        # Sanity: valid tokens in no-latents aux must not contain latent token ids.
        has_latent = ((aux_input_ids == int(self.latent_token_id)) & (aux_attention_mask == 1)).any()
        if bool(has_latent.item()):
            raise RuntimeError("No-latents aux batch still contains latent tokens in valid positions.")
        if bool((aux_digit_pos < 0).any().item()):
            raise RuntimeError("Remapped digit positions contain invalid entries.")
        max_valid = aux_attention_mask.sum(dim=1).to(torch.long) - 1
        if bool((aux_digit_pos > max_valid.unsqueeze(1)).any().item()):
            raise RuntimeError("Remapped digit positions exceed no-latents sequence length.")

        return aux_input_ids, aux_attention_mask, aux_digit_pos, row_map

    def _get_lm_head(self) -> nn.Module:
        if hasattr(self.base_model, "get_output_embeddings"):
            head = self.base_model.get_output_embeddings()
            if head is not None and hasattr(head, "weight"):
                return head
        head = getattr(self.base_model, "lm_head", None)
        if head is None or not hasattr(head, "weight"):
            raise RuntimeError("Could not resolve lm_head/output embeddings on base_model.")
        return head

    def _resolve_digit_token_ids(
        self,
        *,
        device: torch.device,
        digit_token_ids: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        ids = [int(x) for x in digit_token_ids] if digit_token_ids is not None else self.digit_token_ids
        if ids is None:
            raise ValueError(
                "digit_token_ids were not provided. Pass digit_token_ids to the model "
                "constructor or to forward(...)."
            )
        if len(ids) != 10:
            raise ValueError("digit_token_ids must contain exactly 10 ids (0-9).")
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _resolve_digit_position_indices(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        digit_position_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        if digit_position_indices is not None:
            if digit_position_indices.ndim != 2 or digit_position_indices.shape != (bsz, 5):
                raise ValueError("digit_position_indices must have shape [B,5].")
            pos = digit_position_indices.to(device=device, dtype=torch.long)
            if bool((pos < 0).any().item()) or bool((pos >= seqlen).any().item()):
                raise ValueError("digit_position_indices contains out-of-range values.")
            attn_at_pos = torch.gather(attention_mask.to(torch.long), 1, pos)
            if bool((attn_at_pos == 0).any().item()):
                raise ValueError("digit_position_indices points to padding tokens.")
            return pos

        if self.answer_token_id is None:
            raise ValueError(
                "digit_position_indices not provided and answer_token_id is unset on model."
            )
        answer_mask = (input_ids == int(self.answer_token_id)) & (attention_mask == 1)
        if not bool((answer_mask.sum(dim=1) == 1).all().item()):
            raise RuntimeError("Each sample must contain exactly one <ANSWER> token.")

        answer_pos = answer_mask.to(torch.float).argmax(dim=1).to(torch.long)
        offsets = torch.arange(5, dtype=torch.long, device=device).unsqueeze(0)
        pos = answer_pos.unsqueeze(1) + offsets
        if bool((pos >= seqlen).any().item()):
            raise RuntimeError("Digit supervision positions exceed sequence length.")
        attn_at_pos = torch.gather(attention_mask.to(torch.long), 1, pos)
        if bool((attn_at_pos == 0).any().item()):
            raise RuntimeError("Digit supervision positions point to padding tokens.")
        return pos

    def _project_hidden_to_token_subset(
        self,
        *,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden: [B,S,H]
        token_ids: [K]
        returns: [B,S,K]
        """
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B,S,H].")
        if token_ids.ndim != 1:
            raise ValueError("token_ids must have shape [K].")

        lm_head = self._get_lm_head()
        w = lm_head.weight.index_select(0, token_ids.to(torch.long))
        logits = hidden @ w.transpose(0, 1)
        bias = getattr(lm_head, "bias", None)
        if bias is not None:
            logits = logits + bias.index_select(0, token_ids.to(torch.long)).view(1, 1, -1)
        return logits

    def _run_path(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        digit_position_indices: torch.Tensor,
        digit_token_ids: torch.Tensor,
        latent_positions: Sequence[Sequence[int]],
        fill_orders: Sequence[Sequence[int]],
        collect_latents: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Executes one latent path:
        - prefix forwards only for latent fill
        - one final full forward for hidden states, then restricted projection to digit logits
        """
        device = input_ids.device
        inputs_embeds = self._embedding(input_ids)
        bsz, _, hidden_size = inputs_embeds.shape
        max_steps = max((len(o) for o in fill_orders), default=0)
        max_latents_in_batch = max((len(x) for x in latent_positions), default=0)

        latent_vecs: Optional[torch.Tensor] = None
        latent_vec_mask: Optional[torch.Tensor] = None
        if collect_latents:
            latent_vecs = inputs_embeds.new_zeros((bsz, max_latents_in_batch, hidden_size))
            latent_vec_mask = torch.zeros(
                (bsz, max_latents_in_batch),
                dtype=torch.bool,
                device=device,
            )

        # Use backbone forward to get last_hidden_state directly without
        # requesting the full hidden-state stack from CausalLM outputs.
        backbone = getattr(self.base_model, "model", None)
        if backbone is None:
            backbone = getattr(self.base_model, "transformer", None)
        if backbone is None or not callable(backbone):
            raise RuntimeError(
                "Could not resolve transformer backbone on base_model. "
                "Expected `.model` or `.transformer` for prefix latent execution."
            )

        for step_idx in range(max_steps):
            # Bucket samples by absolute latent position p for shared prefix forwards.
            buckets: Dict[int, List[tuple[int, int]]] = {}
            for b, order in enumerate(fill_orders):
                if step_idx >= len(order):
                    continue
                latent_slot_index = int(order[step_idx])
                if latent_slot_index >= len(latent_positions[b]):
                    continue
                p = int(latent_positions[b][latent_slot_index])
                if p <= 0:
                    raise RuntimeError(f"{LATENT_TOKEN} cannot appear at position 0.")
                buckets.setdefault(p, []).append((b, latent_slot_index))

            for p in sorted(buckets.keys()):
                pairs = buckets[p]
                rows = [b for b, _ in pairs]
                slot_indices = [slot_idx for _, slot_idx in pairs]
                row_idx = torch.tensor(rows, dtype=torch.long, device=device)
                prefix_embeds = inputs_embeds.index_select(0, row_idx)[:, :p, :]
                prefix_mask = attention_mask.index_select(0, row_idx)[:, :p]
                prefix_pos = position_ids.index_select(0, row_idx)[:, :p]

                out = backbone(
                    inputs_embeds=prefix_embeds,
                    attention_mask=prefix_mask,
                    position_ids=prefix_pos,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden = out.last_hidden_state
                fill_vecs = hidden[:, p - 1, :]
                inputs_embeds[row_idx, p, :] = fill_vecs
                if collect_latents and latent_vecs is not None and latent_vec_mask is not None:
                    slot_idx_t = torch.tensor(slot_indices, dtype=torch.long, device=device)
                    latent_vecs[row_idx, slot_idx_t, :] = fill_vecs
                    latent_vec_mask[row_idx, slot_idx_t] = True

        final = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_last = final.last_hidden_state

        batch_idx = torch.arange(bsz, dtype=torch.long, device=device).unsqueeze(1).expand(-1, 5)
        digit_hidden = hidden_last[batch_idx, digit_position_indices]
        digit_logits = self._project_hidden_to_token_subset(
            hidden=digit_hidden,
            token_ids=digit_token_ids,
        )
        return digit_logits, latent_vecs, latent_vec_mask

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        digit_position_indices: Optional[torch.Tensor] = None,
        digit_token_ids: Optional[Sequence[int]] = None,
        compute_aux: bool = False,
        aux_seed: int = 0,
        collect_latents: bool = False,
    ) -> Phase1Forward:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be rank-2 tensors.")

        attention_mask = attention_mask.to(input_ids.device)
        self._assert_right_padding(attention_mask)
        position_ids = self._build_position_ids(attention_mask)
        digit_pos = self._resolve_digit_position_indices(
            input_ids=input_ids,
            attention_mask=attention_mask,
            digit_position_indices=digit_position_indices,
        )
        digit_ids_t = self._resolve_digit_token_ids(
            device=input_ids.device,
            digit_token_ids=digit_token_ids,
        )
        latent_positions = self._extract_latent_positions(input_ids, attention_mask)
        latent_counts = torch.tensor(
            [len(x) for x in latent_positions],
            dtype=torch.long,
            device=input_ids.device,
        )
        normal_orders = [list(range(len(x))) for x in latent_positions]

        logits_orig, latent_vecs_orig, latent_vecs_orig_mask = self._run_path(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            digit_position_indices=digit_pos,
            digit_token_ids=digit_ids_t,
            latent_positions=latent_positions,
            fill_orders=normal_orders,
            collect_latents=collect_latents,
        )

        logits_aux: Optional[torch.Tensor] = None
        latent_vecs_aux: Optional[torch.Tensor] = None
        latent_vecs_aux_mask: Optional[torch.Tensor] = None
        aux_enabled = torch.zeros_like(latent_counts, dtype=torch.bool)
        if compute_aux:
            aux_orders, aux_enabled_cpu, aux_modes = self._build_aux_orders(
                latent_positions=latent_positions,
                seed=int(aux_seed),
            )
            aux_enabled = aux_enabled_cpu.to(device=input_ids.device)
            if bool(aux_enabled.any().item()):
                device = input_ids.device
                full_remove_mask = torch.tensor(
                    [m == "full_remove_latents" for m in aux_modes],
                    dtype=torch.bool,
                    device=device,
                ) & aux_enabled
                normal_mask = aux_enabled & (~full_remove_mask)

                logits_aux_full = logits_orig.new_zeros(logits_orig.shape)
                computed_mask = torch.zeros_like(aux_enabled, dtype=torch.bool)

                if bool(normal_mask.any().item()):
                    row_idx = torch.nonzero(normal_mask, as_tuple=False).squeeze(1)
                    sub_input_ids = input_ids.index_select(0, row_idx)
                    sub_attention_mask = attention_mask.index_select(0, row_idx)
                    sub_position_ids = position_ids.index_select(0, row_idx)
                    sub_digit_pos = digit_pos.index_select(0, row_idx)
                    sub_latent_positions = [latent_positions[int(i)] for i in row_idx.tolist()]
                    sub_orders = [aux_orders[int(i)] for i in row_idx.tolist()]
                    logits_sub, latent_vecs_sub, latent_vecs_sub_mask = self._run_path(
                        input_ids=sub_input_ids,
                        attention_mask=sub_attention_mask,
                        position_ids=sub_position_ids,
                        digit_position_indices=sub_digit_pos,
                        digit_token_ids=digit_ids_t,
                        latent_positions=sub_latent_positions,
                        fill_orders=sub_orders,
                        collect_latents=collect_latents,
                    )
                    logits_aux_full.index_copy_(0, row_idx, logits_sub)
                    computed_mask[row_idx] = True

                    if collect_latents:
                        hidden_size = int(self._embedding.weight.shape[1])
                        max_latents_all = max((len(x) for x in latent_positions), default=0)
                        if latent_vecs_aux is None:
                            latent_vecs_aux = logits_orig.new_zeros((input_ids.size(0), max_latents_all, hidden_size))
                            latent_vecs_aux_mask = torch.zeros(
                                (input_ids.size(0), max_latents_all),
                                dtype=torch.bool,
                                device=device,
                            )
                        if latent_vecs_sub is not None and latent_vecs_sub_mask is not None:
                            lat_n = int(latent_vecs_sub.shape[1])
                            if lat_n > 0:
                                latent_vecs_aux[row_idx, :lat_n, :] = latent_vecs_sub
                                latent_vecs_aux_mask[row_idx, :lat_n] = latent_vecs_sub_mask

                if bool(full_remove_mask.any().item()):
                    no_lat_input_ids, no_lat_attention_mask, no_lat_digit_pos, row_map = self._build_no_latents_aux_batch(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        digit_position_indices=digit_pos,
                        latent_positions=latent_positions,
                        select_mask=full_remove_mask,
                    )
                    no_lat_position_ids = self._build_position_ids(no_lat_attention_mask)
                    no_lat_latent_positions = self._extract_latent_positions(no_lat_input_ids, no_lat_attention_mask)
                    no_lat_orders = [[] for _ in no_lat_latent_positions]
                    logits_sub, _, _ = self._run_path(
                        input_ids=no_lat_input_ids,
                        attention_mask=no_lat_attention_mask,
                        position_ids=no_lat_position_ids,
                        digit_position_indices=no_lat_digit_pos,
                        digit_token_ids=digit_ids_t,
                        latent_positions=no_lat_latent_positions,
                        fill_orders=no_lat_orders,
                        collect_latents=False,
                    )
                    logits_aux_full.index_copy_(0, row_map, logits_sub)
                    computed_mask[row_map] = True

                if bool(computed_mask.any().item()):
                    logits_aux = logits_aux_full
                    aux_enabled = computed_mask

        return Phase1Forward(
            logits_orig=logits_orig,
            logits_aux=logits_aux,
            aux_enabled_mask=aux_enabled,
            latent_counts=latent_counts,
            latent_vectors_orig=latent_vecs_orig,
            latent_vectors_orig_mask=latent_vecs_orig_mask,
            latent_vectors_aux=latent_vecs_aux,
            latent_vectors_aux_mask=latent_vecs_aux_mask,
        )
