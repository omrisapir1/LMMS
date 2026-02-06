from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

from .utils import safe_softmax


# ============================================================
# Bundle
# ============================================================

@dataclass
class Phase23Bundle:
    tokenizer: Any
    model: "UnifiedZSoftModel"


# ============================================================
# Logits processor (Phase3-style masking, no lm_head replacement)
# ============================================================

class AllowedTokensOnly(LogitsProcessor):
    """
    Masks logits so that ONLY allowed_token_ids may be generated.
    """
    def __init__(self, allowed_token_ids: List[int], fill_value: float = -1e4):
        self.allowed = torch.tensor(allowed_token_ids, dtype=torch.long)
        self.fill_value = float(fill_value)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        device = scores.device
        allowed = self.allowed.to(device)

        masked = scores.new_full(scores.shape, self.fill_value)
        masked.index_copy_(1, allowed, scores.index_select(1, allowed))
        return masked


# ============================================================
# Unified Phase-2.5 Model
# ============================================================

class UnifiedZSoftModel(nn.Module):
    """
    Unified Phase-2.5 model:

    - Soft latent execution with expected Z embeddings
    - Z distributions come from the LM head (restricted rows)
    - Digit heads predict final answer
    - Generation uses standard HF .generate() with logits masking
    """

    def __init__(
        self,
        *,
        base_lm: nn.Module,
        digit_heads: nn.ModuleList,
        latent_token_id: int,
        answer_token_id: int,
        z_token_ids: List[int],
    ) -> None:
        super().__init__()

        self.base = base_lm
        self.digit_heads = digit_heads

        self.latent_token_id = int(latent_token_id)
        self.answer_token_id = int(answer_token_id)
        self.z_token_ids = list(map(int, z_token_ids))

        self._embedding = self.base.get_input_embeddings()

    # ============================================================
    # Loading from pretrained repo/checkpoint
    # ============================================================

    @classmethod
    def _build_from_phase1(
        cls,
        repo_or_dir: str,
        *,
        v_z: int,
        device: torch.device,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
    ) -> Phase23Bundle:
        from z_pipeline.shared.load_model_phase1 import load_phase1, LATENT_TOKEN

        tokenizer, phase1_model, _meta = load_phase1(
            repo_or_dir=repo_or_dir,
            device=str(device),
            torch_dtype=torch_dtype,
        )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer has no pad or eos token; set pad token explicitly.")
            tokenizer.pad_token = tokenizer.eos_token

        latent_token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
        answer_token_id = tokenizer.convert_tokens_to_ids("<ANSWER>")
        if latent_token_id is None or latent_token_id < 0:
            raise RuntimeError(f"Could not resolve latent token id for '{LATENT_TOKEN}'")
        if answer_token_id is None or answer_token_id < 0:
            raise RuntimeError("Could not resolve answer token id '<ANSWER>'")

        phase0 = phase1_model.phase0
        base_lm = phase0.model
        digit_heads = phase0.digit_heads

        z_tokens = [f"<Z_{i}>" for i in range(int(v_z))]
        existing = set(tokenizer.get_vocab().keys())
        to_add = [t for t in z_tokens if t not in existing]
        if to_add:
            tokenizer.add_tokens(to_add, special_tokens=False)
        base_lm.resize_token_embeddings(len(tokenizer))

        z_token_ids: List[int] = []
        for i in range(int(v_z)):
            tok = f"<Z_{i}>"
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid < 0:
                raise RuntimeError(f"Failed to convert token to id: {tok}")
            z_token_ids.append(int(tid))

        model = cls(
            base_lm=base_lm,
            digit_heads=digit_heads,
            answer_token_id=int(answer_token_id),
            latent_token_id=int(latent_token_id),
            z_token_ids=z_token_ids,
        )
        model.to(device)
        model.eval()

        return Phase23Bundle(tokenizer=tokenizer, model=model)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        device: torch.device,
        v_z: int = 512,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
    ) -> "UnifiedZSoftModel":
        bundle = cls._build_from_phase1(
            repo_or_dir=repo_id,
            v_z=v_z,
            device=device,
            torch_dtype=torch_dtype,
        )
        return bundle.model

    @classmethod
    def from_phase1(
        cls,
        phase1_dir: str,
        v_z: int,
        device: torch.device | str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> Phase23Bundle:
        return cls._build_from_phase1(
            repo_or_dir=phase1_dir,
            v_z=v_z,
            device=torch.device(device),
            torch_dtype=torch_dtype,
        )

    # ============================================================
    # Helpers
    # ============================================================

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.cumsum(attention_mask.to(torch.long), dim=1) - 1
        return torch.clamp(pos, min=0)

    def _latent_lists(self, input_ids: torch.Tensor) -> Tuple[List[List[int]], int]:
        B, _ = input_ids.shape
        idx = (input_ids == self.latent_token_id).nonzero(as_tuple=False)

        lists: List[List[int]] = []
        for b in range(B):
            ps = [int(i[1]) for i in idx if int(i[0]) == b]
            ps.sort()
            lists.append(ps)

        return lists, max((len(x) for x in lists), default=0)

    def _bucket_fill_positions(
        self,
        latent_lists: List[List[int]],
        pass_idx: int,
    ) -> Dict[int, List[int]]:
        buckets: Dict[int, List[int]] = {}
        for b, ps in enumerate(latent_lists):
            if pass_idx < len(ps):
                p = ps[pass_idx]
                if p <= 0:
                    raise RuntimeError("<|latent|> at position 0")
                buckets.setdefault(p, []).append(b)
        return buckets

    def _digit_logits_from_hidden(
        self,
        hidden_last: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        mask = (input_ids == self.answer_token_id)
        if not torch.all(mask.sum(dim=1) == 1):
            raise RuntimeError("Each sample must contain exactly one <ANSWER>")

        idx = mask.float().argmax(dim=1)
        bidx = torch.arange(hidden_last.size(0), device=hidden_last.device)
        h = hidden_last[bidx, idx]

        return torch.stack([head(h) for head in self.digit_heads], dim=1)

    def _get_lm_head(self) -> nn.Module:
        if hasattr(self.base, "get_output_embeddings"):
            head = self.base.get_output_embeddings()
            if head is not None:
                return head
        head = getattr(self.base, "lm_head", None)
        if head is None:
            raise RuntimeError("Base LM has no lm_head")
        return head

    def _z_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self._get_lm_head()
        W = head.weight[self.z_token_ids]          # [Vz, H]
        logits = hidden @ W.t()
        b = getattr(head, "bias", None)
        if b is not None:
            logits = logits + b[self.z_token_ids]
        return logits

    def _z_embeddings(self) -> torch.Tensor:
        return self.base.get_input_embeddings().weight[self.z_token_ids]

    # ============================================================
    # Training forward (soft-embedding latent execution)
    # ============================================================

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_self_teacher: bool = True,
        tau_teacher: float = 1.0,
        tau_student: float = 1.0,
        return_distributions: bool = False,
    ) -> Dict[str, torch.Tensor]:

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = self._build_position_ids(attention_mask)
        inputs_embeds = self._embedding(input_ids)

        latent_lists, Kmax = self._latent_lists(input_ids)
        B = input_ids.size(0)
        Vz = len(self.z_token_ids)

        slot_mask = torch.zeros((B, Kmax), device=input_ids.device)
        for b, ps in enumerate(latent_lists):
            slot_mask[b, : len(ps)] = 1.0

        p_student = None
        q_teacher = None
        if return_distributions or use_self_teacher:
            p_student = torch.zeros((B, Kmax, Vz), device=input_ids.device, dtype=torch.float32)
        if use_self_teacher:
            q_teacher = torch.zeros((B, Kmax, Vz), device=input_ids.device, dtype=torch.float32)

        z_emb = self._z_embeddings()

        for pass_idx in range(Kmax):
            buckets = self._bucket_fill_positions(latent_lists, pass_idx)
            if not buckets:
                continue

            for p in sorted(buckets):
                bs = torch.tensor(buckets[p], device=input_ids.device)
                out = self.base.model(
                    inputs_embeds=inputs_embeds[bs, :p],
                    attention_mask=attention_mask[bs, :p],
                    position_ids=position_ids[bs, :p],
                    use_cache=False,
                    output_hidden_states=True,
                )
                u = out.hidden_states[-1][:, p - 1]

                s_logits = self._z_logits_from_hidden(u)
                p_s = safe_softmax(s_logits, tau=tau_student, dim=-1).to(torch.float32)

                inputs_embeds[bs, p] = (p_s.to(inputs_embeds.dtype)) @ z_emb
                p_student[bs, pass_idx] = p_s

                if use_self_teacher:
                    # Teacher is self-teacher computed from the same logits, pre-injection.
                    with torch.no_grad():
                        q_teacher[bs, pass_idx] = safe_softmax(
                            s_logits, tau=tau_teacher, dim=-1
                        ).to(torch.float32)

        out_final = self.base.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        hidden_last = out_final.hidden_states[-1]
        digit_logits = self._digit_logits_from_hidden(hidden_last, input_ids)

        out = {
            "digit_logits": digit_logits,
            "slot_mask": slot_mask,
        }
        if return_distributions:
            out["p_student"] = p_student
            out["q_teacher"] = q_teacher

        return out

    # ============================================================
    # Forward with externally provided Z distributions
    # ============================================================

    def forward_with_fixed_z_distributions(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        p_z: torch.Tensor,  # [B,Kmax,Vz]
    ) -> torch.Tensor:
        """
        Inject fixed per-slot Z mixtures and return digit logits only.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = self._build_position_ids(attention_mask)
        inputs_embeds = self._embedding(input_ids)
        latent_lists, Kmax = self._latent_lists(input_ids)

        if p_z.dim() != 3:
            raise RuntimeError("p_z must have shape [B,Kmax,Vz]")
        if p_z.size(0) != input_ids.size(0):
            raise RuntimeError("p_z batch size must match input_ids")
        if p_z.size(1) < Kmax:
            raise RuntimeError("p_z K dimension smaller than required latent count")
        if p_z.size(2) != len(self.z_token_ids):
            raise RuntimeError("p_z V dimension must equal number of Z tokens")

        z_emb = self._z_embeddings()

        for pass_idx in range(Kmax):
            buckets = self._bucket_fill_positions(latent_lists, pass_idx)
            if not buckets:
                continue
            for p in sorted(buckets):
                bs = torch.tensor(buckets[p], device=input_ids.device)
                p_s = p_z[bs, pass_idx].to(dtype=inputs_embeds.dtype)
                inputs_embeds[bs, p] = p_s @ z_emb

        out_final = self.base.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        hidden_last = out_final.hidden_states[-1]
        return self._digit_logits_from_hidden(hidden_last, input_ids)

    # ============================================================
    # Generation (Phase3-style, but lm_head intact)
    # ============================================================

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        fill_value: float = -1e4,
        **kwargs,
    ) -> torch.Tensor:

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        allowed = self.z_token_ids + [self.answer_token_id]
        lp = LogitsProcessorList([AllowedTokensOnly(allowed, fill_value)])

        return self.base.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.answer_token_id,
            logits_processor=lp,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # ============================================================
    # Generation + digits
    # ============================================================

    @torch.no_grad()
    def generate_with_digits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        sequences = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        if pad_token_id is None:
            pad_token_id = getattr(self.base.config, "pad_token_id", self.answer_token_id)

        full_attn = (sequences != pad_token_id).long()

        out = self.base(
            input_ids=sequences,
            attention_mask=full_attn,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_last = out.hidden_states[-1]

        mask = (sequences == self.answer_token_id)
        pos = torch.where(
            mask.any(dim=1),
            mask.float().argmax(dim=1),
            (full_attn.sum(dim=1) - 1).clamp(min=0),
        )

        bidx = torch.arange(hidden_last.size(0), device=hidden_last.device)
        h = hidden_last[bidx, pos]
        digit_logits = torch.stack([head(h) for head in self.digit_heads], dim=1)

        return {
            "sequences": sequences,
            "digit_logits": digit_logits,
            "digit_preds": digit_logits.argmax(dim=-1),
        }
