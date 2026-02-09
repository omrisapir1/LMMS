from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

from .utils import safe_softmax


@dataclass
class Phase23Bundle:
    tokenizer: Any
    model: "UnifiedZSoftModel"


class AllowedTokensOnly(LogitsProcessor):
    """Masks logits so that only allowed token ids can be generated."""

    def __init__(self, allowed_token_ids: List[int], fill_value: float = -1e4) -> None:
        self.allowed = torch.tensor(allowed_token_ids, dtype=torch.long)
        self.fill_value = float(fill_value)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        del input_ids
        allowed = self.allowed.to(scores.device)
        masked = scores.new_full(scores.shape, self.fill_value)
        masked.index_copy_(1, allowed, scores.index_select(1, allowed))
        return masked


class UnifiedZSoftModel(nn.Module):
    """
    Phase23 merged model.

    GS-forward / soft-backprop latent execution:
    - At each <|latent|> slot, compute Z logits from LM head rows.
    - If use_gs=True: inject hard one-hot sample via Gumbel-ST.
    - If use_gs=False: inject deterministic argmax Z.

    No teacher pathway in Phase23-GS.
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
        self.z_token_ids = [int(x) for x in z_token_ids]

        self._embedding = self.base.get_input_embeddings()

    @staticmethod
    def _ensure_causal_lm(base_lm: nn.Module) -> nn.Module:
        """Ensure a CausalLM wrapper with lm_head and .generate exists."""
        has_output_head = False
        if hasattr(base_lm, "get_output_embeddings"):
            try:
                has_output_head = base_lm.get_output_embeddings() is not None
            except Exception:
                has_output_head = False
        has_lm_head = getattr(base_lm, "lm_head", None) is not None

        if hasattr(base_lm, "generate") and (has_output_head or has_lm_head):
            return base_lm

        # Phase1 currently yields Qwen backbone (Qwen2Model). Re-wrap into causal LM.
        if base_lm.__class__.__name__ == "Qwen2Model":
            from transformers import Qwen2ForCausalLM

            wrapped = Qwen2ForCausalLM(base_lm.config)
            wrapped.model.load_state_dict(base_lm.state_dict(), strict=True)
            try:
                p = next(base_lm.parameters())
                wrapped = wrapped.to(dtype=p.dtype)
            except StopIteration:
                pass

            # Initialize lm_head from token embeddings for a stable starting point.
            if wrapped.lm_head.weight.shape == wrapped.model.embed_tokens.weight.shape:
                wrapped.lm_head.weight.data.copy_(wrapped.model.embed_tokens.weight.data)
            return wrapped

        raise RuntimeError(
            f"Unsupported base LM type for Phase23: {type(base_lm)}. "
            "Expected CausalLM or Qwen2Model backbone."
        )

    def _core_model(self) -> nn.Module:
        return self.base.model if hasattr(self.base, "model") else self.base

    @classmethod
    def _build_from_phase1(
        cls,
        repo_or_dir: str,
        *,
        v_z: int,
        device: torch.device,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
        z_prefix: str = "Z_",
        latent_token: str = "<|latent|>",
        answer_token: str = "<ANSWER>",
    ) -> Phase23Bundle:
        from z_pipeline.shared.load_model_phase1 import load_phase1

        tokenizer, phase1_model, _meta = load_phase1(
            repo_or_dir=repo_or_dir,
            device=str(device),
            torch_dtype=torch_dtype,
        )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer has no pad/eos token.")
            tokenizer.pad_token = tokenizer.eos_token

        latent_token_id = tokenizer.convert_tokens_to_ids(latent_token)
        answer_token_id = tokenizer.convert_tokens_to_ids(answer_token)
        if latent_token_id is None or int(latent_token_id) < 0:
            raise RuntimeError(f"Could not resolve latent token id for '{latent_token}'")
        if answer_token_id is None or int(answer_token_id) < 0:
            raise RuntimeError(f"Could not resolve answer token id '{answer_token}'")

        phase0 = phase1_model.phase0
        base_lm = cls._ensure_causal_lm(phase0.model)
        digit_heads = phase0.digit_heads

        z_tokens = [f"<{z_prefix}{i}>" for i in range(int(v_z))]
        vocab = tokenizer.get_vocab()
        to_add = [tok for tok in z_tokens if tok not in vocab]
        if to_add:
            tokenizer.add_tokens(to_add, special_tokens=False)
        base_lm.resize_token_embeddings(len(tokenizer))

        z_token_ids: List[int] = []
        for i in range(int(v_z)):
            tok = f"<{z_prefix}{i}>"
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or int(tid) < 0:
                raise RuntimeError(f"Failed to resolve token id for {tok}")
            z_token_ids.append(int(tid))

        model = cls(
            base_lm=base_lm,
            digit_heads=digit_heads,
            latent_token_id=int(latent_token_id),
            answer_token_id=int(answer_token_id),
            z_token_ids=z_token_ids,
        )
        model.to(device)
        model.eval()
        return Phase23Bundle(tokenizer=tokenizer, model=model)

    @classmethod
    def from_phase1(
        cls,
        phase1_dir: str,
        v_z: int,
        device: Union[torch.device, str] = "cuda",
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
        z_prefix: str = "Z_",
        latent_token: str = "<|latent|>",
        answer_token: str = "<ANSWER>",
    ) -> Phase23Bundle:
        return cls._build_from_phase1(
            repo_or_dir=phase1_dir,
            v_z=v_z,
            device=torch.device(device),
            torch_dtype=torch_dtype,
            z_prefix=z_prefix,
            latent_token=latent_token,
            answer_token=answer_token,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_or_dir: str,
        *,
        device: Union[torch.device, str],
        v_z: int = 512,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
        z_prefix: str = "Z_",
        latent_token: str = "<|latent|>",
        answer_token: str = "<ANSWER>",
    ) -> "UnifiedZSoftModel":
        """
        Load Phase23 model either from:
        1) phase23 checkpoint directory (phase23_state.pt + config.json + tokenizer files), or
        2) Phase1 repo/directory (build fresh phase23 wrapper from Phase1).
        """
        device_t = torch.device(device)
        state_path = os.path.join(repo_or_dir, "phase23_state.pt")
        config_path = os.path.join(repo_or_dir, "config.json")

        if os.path.isdir(repo_or_dir) and os.path.exists(state_path):
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Missing config for phase23 checkpoint: {config_path}"
                )
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            phase1_dir = cfg.get("model", {}).get("phase1_dir")
            if not phase1_dir:
                raise RuntimeError("config.json missing model.phase1_dir required for restore")
            ckpt_vz = int(cfg.get("model", {}).get("v_z", v_z))
            ckpt_z_prefix = cfg.get("model", {}).get("z_prefix", z_prefix)
            ckpt_latent_token = cfg.get("model", {}).get("latent_token", latent_token)
            ckpt_answer_token = cfg.get("model", {}).get("answer_token", answer_token)

            bundle = cls.from_phase1(
                phase1_dir=phase1_dir,
                v_z=ckpt_vz,
                device=device_t,
                torch_dtype=torch_dtype,
                z_prefix=ckpt_z_prefix,
                latent_token=ckpt_latent_token,
                answer_token=ckpt_answer_token,
            )
            state = torch.load(state_path, map_location="cpu")
            bundle.model.load_state_dict(state, strict=True)
            bundle.model.to(device_t)
            bundle.model.eval()
            return bundle.model

        bundle = cls.from_phase1(
            phase1_dir=repo_or_dir,
            v_z=v_z,
            device=device_t,
            torch_dtype=torch_dtype,
            z_prefix=z_prefix,
            latent_token=latent_token,
            answer_token=answer_token,
        )
        return bundle.model

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.cumsum(attention_mask.to(torch.long), dim=1) - 1
        return torch.clamp(pos, min=0)

    def _latent_lists(self, input_ids: torch.Tensor) -> Tuple[List[List[int]], int]:
        bsz = input_ids.size(0)
        idx = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        latent_lists: List[List[int]] = []
        for b in range(bsz):
            pos = [int(i[1]) for i in idx if int(i[0]) == b]
            pos.sort()
            latent_lists.append(pos)
        return latent_lists, max((len(x) for x in latent_lists), default=0)

    def _bucket_fill_positions(
        self,
        latent_lists: List[List[int]],
        pass_idx: int,
    ) -> Dict[int, List[int]]:
        buckets: Dict[int, List[int]] = {}
        for b, pos_list in enumerate(latent_lists):
            if pass_idx < len(pos_list):
                p = pos_list[pass_idx]
                if p <= 0:
                    raise RuntimeError("Found <|latent|> at sequence position 0")
                buckets.setdefault(p, []).append(b)
        return buckets

    def _find_answer_pos(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids == self.answer_token_id)
        if not torch.all(mask.sum(dim=1) == 1):
            raise RuntimeError("Each sample must contain exactly one <ANSWER> token")
        return mask.float().argmax(dim=1)

    def _get_lm_head(self) -> nn.Module:
        if hasattr(self.base, "get_output_embeddings"):
            head = self.base.get_output_embeddings()
            if head is not None:
                return head
        head = getattr(self.base, "lm_head", None)
        if head is None:
            raise RuntimeError("Base LM has no output head")
        return head

    def _lm_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self._get_lm_head()
        logits = hidden @ head.weight.t()
        bias = getattr(head, "bias", None)
        if bias is not None:
            logits = logits + bias
        return logits

    def _z_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self._get_lm_head()
        w = head.weight[self.z_token_ids]
        logits = hidden @ w.t()
        bias = getattr(head, "bias", None)
        if bias is not None:
            logits = logits + bias[self.z_token_ids]
        return logits

    def _z_embeddings(self) -> torch.Tensor:
        return self.base.get_input_embeddings().weight[self.z_token_ids]

    def _digit_logits_from_hidden(self, hidden_last: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        h_answer = self._answer_hidden_from_hidden(hidden_last, input_ids)
        return torch.stack([head(h_answer) for head in self.digit_heads], dim=1)

    def _answer_hidden_from_hidden(self, hidden_last: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        answer_pos = self._find_answer_pos(input_ids)
        bidx = torch.arange(hidden_last.size(0), device=hidden_last.device)
        return hidden_last[bidx, answer_pos]

    def _answer_next_logits_from_hidden(self, hidden_last: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        answer_pos = self._find_answer_pos(input_ids)
        prev_pos = (answer_pos - 1).clamp(min=0)
        bidx = torch.arange(hidden_last.size(0), device=hidden_last.device)
        h_prev = hidden_last[bidx, prev_pos]
        return self._lm_logits_from_hidden(h_prev)

    def _build_additive_causal_mask_with_answer_z_bias(
        self,
        *,
        input_ids: torch.Tensor,        # [B,T]
        attention_mask: torch.Tensor,   # [B,T]
        cf_bias_scale: float,
        cf_attention_bias_strength: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build compact additive bias [B,1,1,T] and answer query positions [B].
        Bias is injected only at the answer query row inside attention hooks.
        """
        bsz, t = input_ids.shape
        device = input_ids.device
        valid = attention_mask.to(torch.bool)
        bias = torch.zeros((bsz, 1, 1, t), device=device, dtype=torch.float32)
        answer_pos = self._find_answer_pos(input_ids)  # [B]

        if cf_bias_scale <= 0.0 or cf_attention_bias_strength == 0.0:
            return bias, answer_pos

        # These are the sequence slots where latent Z embeddings are executed.
        z_slot_pos_mask = (input_ids == self.latent_token_id) & valid
        if not torch.all(z_slot_pos_mask.any(dim=1)):
            raise RuntimeError("CF bias expects latent slots to be present in input_ids.")
        bias_delta = float(cf_bias_scale) * float(cf_attention_bias_strength)
        for b in range(bsz):
            key_mask = z_slot_pos_mask[b]
            if key_mask.any():
                bias[b, 0, 0, key_mask] += bias_delta
        return bias, answer_pos

    def _run_core_with_attention_score_bias(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        answer_pos: Optional[torch.Tensor] = None,
    ):
        core = self._core_model()
        if attn_bias is None:
            return core(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        bias = attn_bias
        handles = []
        applied = {"ok": False}
        modified_mask_ptrs = set()

        def _pre_hook(module, args, kwargs):
            am = kwargs.get("attention_mask", None)
            if isinstance(am, torch.Tensor) and am.dim() == 4:
                local = bias.to(device=am.device, dtype=am.dtype)
                k_len = am.size(-1)
                if local.size(-2) != 1 or k_len != local.size(-1):
                    return args, kwargs
                if answer_pos is None:
                    raise RuntimeError("answer_pos must be provided when CF attention bias is enabled.")

                ptr = am.data_ptr()
                if ptr not in modified_mask_ptrs:
                    q_len = am.size(-2)
                    bsz = am.size(0)
                    if answer_pos.numel() != bsz:
                        raise RuntimeError("answer_pos batch size mismatch during CF bias injection.")
                    for b in range(bsz):
                        q = int(answer_pos[b].item())
                        if 0 <= q < q_len:
                            am[b, :, q, :] = am[b, :, q, :] + local[b, :, 0, :]
                    modified_mask_ptrs.add(ptr)
                    applied["ok"] = True
            return args, kwargs

        try:
            for mod in core.modules():
                name = mod.__class__.__name__.lower()
                if "attention" in name:
                    handles.append(mod.register_forward_pre_hook(_pre_hook, with_kwargs=True))

            out = core(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if not applied["ok"]:
                raise RuntimeError(
                    "CF attention bias was requested but could not be applied to attention scores "
                    "(no compatible 4D attention_mask observed inside attention modules)."
                )
            return out
        finally:
            for h in handles:
                h.remove()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gumbel_tau: float = 1.0,
        use_gs: bool = True,
        return_distributions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = self._build_position_ids(attention_mask)
        inputs_embeds = self._embedding(input_ids)

        latent_lists, kmax = self._latent_lists(input_ids)
        bsz = input_ids.size(0)
        vz = len(self.z_token_ids)

        slot_mask = torch.zeros((bsz, kmax), device=input_ids.device, dtype=torch.bool)
        for b, positions in enumerate(latent_lists):
            slot_mask[b, : len(positions)] = True

        p_student: Optional[torch.Tensor] = None
        if return_distributions:
            p_student = torch.zeros((bsz, kmax, vz), device=input_ids.device, dtype=inputs_embeds.dtype)

        z_emb = self._z_embeddings().to(dtype=inputs_embeds.dtype)

        tau = max(float(gumbel_tau), 1e-6)
        for pass_idx in range(kmax):
            buckets = self._bucket_fill_positions(latent_lists, pass_idx)
            if not buckets:
                continue

            for p in sorted(buckets):
                bs = torch.tensor(buckets[p], device=input_ids.device, dtype=torch.long)

                def _prefix_last_hidden(
                    emb: torch.Tensor,
                    attn: torch.Tensor,
                    pos: torch.Tensor,
                ) -> torch.Tensor:
                    out_prefix = self._core_model()(
                        inputs_embeds=emb,
                        attention_mask=attn,
                        position_ids=pos,
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    return out_prefix.last_hidden_state

                prefix_emb = inputs_embeds[bs, :p]
                prefix_attn = attention_mask[bs, :p]
                prefix_pos = position_ids[bs, :p]
                if self.training:
                    hidden_prefix = activation_checkpoint(
                        _prefix_last_hidden,
                        prefix_emb,
                        prefix_attn,
                        prefix_pos,
                        use_reentrant=False,
                    )
                else:
                    hidden_prefix = _prefix_last_hidden(prefix_emb, prefix_attn, prefix_pos)
                u = hidden_prefix[:, p - 1]

                s_logits = self._z_logits_from_hidden(u).to(torch.float32)

                if use_gs:
                    # GS-ST with explicit soft sample tracking:
                    # forward uses hard one-hot; backward flows through y_soft.
                    u_rand = torch.rand_like(s_logits).clamp_(1e-6, 1.0 - 1e-6)
                    g_noise = -torch.log(-torch.log(u_rand))
                    y_soft = safe_softmax(s_logits + g_noise, tau=tau, dim=-1)
                    z_idx = y_soft.argmax(dim=-1)
                    y_hard = F.one_hot(z_idx, num_classes=vz).to(torch.float32)
                    z_st = y_hard - y_soft.detach() + y_soft
                    p_slot = y_soft
                else:
                    # Deterministic evaluation path: no gumbel noise.
                    z_idx = s_logits.argmax(dim=-1)
                    z_st = F.one_hot(z_idx, num_classes=vz).to(torch.float32)
                    p_slot = z_st

                e_latent = torch.matmul(z_st.to(z_emb.dtype), z_emb).to(inputs_embeds.dtype)
                inputs_embeds[bs, p] = e_latent

                if p_student is not None:
                    p_student[bs, pass_idx] = p_slot.to(dtype=p_student.dtype)

        out_final = self._core_model()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_last = out_final.hidden_states[-1]

        digit_logits = self._digit_logits_from_hidden(hidden_last, input_ids)
        answer_next_logits = self._answer_next_logits_from_hidden(hidden_last, input_ids)

        out: Dict[str, torch.Tensor] = {
            "digit_logits": digit_logits,
            "answer_next_logits": answer_next_logits,
            "slot_mask": slot_mask,
        }
        if return_distributions:
            out["p_student"] = p_student
        return out

    def forward_with_fixed_z_distributions(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        p_z: Optional[torch.Tensor] = None,  # [B,Kmax,Vz]
        p_z_idx: Optional[torch.Tensor] = None,  # [B,Kmax]
        cf_bias_scale: float = 0.0,
        apply_cf_answer_z_bias: bool = False,
        cf_attention_bias_strength: float = 0.0,
        return_answer_hidden: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Inject fixed per-slot Z mixtures (no GS sampling).
        Returns digit logits [B,5,10], and optionally <ANSWER> hidden state [B,H].
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if p_z is None and p_z_idx is None:
            raise RuntimeError("One of p_z or p_z_idx must be provided.")
        if p_z is not None and p_z_idx is not None:
            raise RuntimeError("Provide only one of p_z or p_z_idx, not both.")

        position_ids = self._build_position_ids(attention_mask)
        inputs_embeds = self._embedding(input_ids)

        latent_lists, kmax = self._latent_lists(input_ids)
        if p_z is not None:
            if p_z.dim() != 3:
                raise RuntimeError("p_z must have shape [B,Kmax,Vz]")
            if p_z.size(0) != input_ids.size(0):
                raise RuntimeError("p_z batch dimension must match input_ids")
            if p_z.size(1) < kmax:
                raise RuntimeError("p_z slot dimension is smaller than required latent slots")
            if p_z.size(2) != len(self.z_token_ids):
                raise RuntimeError("p_z last dimension must equal len(z_token_ids)")
        if p_z_idx is not None:
            if p_z_idx.dim() != 2:
                raise RuntimeError("p_z_idx must have shape [B,Kmax]")
            if p_z_idx.size(0) != input_ids.size(0):
                raise RuntimeError("p_z_idx batch dimension must match input_ids")
            if p_z_idx.size(1) < kmax:
                raise RuntimeError("p_z_idx slot dimension is smaller than required latent slots")

        z_emb = self._z_embeddings().to(dtype=inputs_embeds.dtype)

        for pass_idx in range(kmax):
            buckets = self._bucket_fill_positions(latent_lists, pass_idx)
            if not buckets:
                continue
            for p in sorted(buckets):
                bs = torch.tensor(buckets[p], device=input_ids.device, dtype=torch.long)
                if p_z_idx is not None:
                    slot_idx = p_z_idx[bs, pass_idx].to(dtype=torch.long)
                    inputs_embeds[bs, p] = z_emb.index_select(0, slot_idx).to(inputs_embeds.dtype)
                else:
                    slot_p = p_z[bs, pass_idx].to(dtype=inputs_embeds.dtype)
                    inputs_embeds[bs, p] = torch.matmul(slot_p, z_emb).to(inputs_embeds.dtype)

        if apply_cf_answer_z_bias and cf_bias_scale > 0.0:
            attn_bias, answer_pos = self._build_additive_causal_mask_with_answer_z_bias(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cf_bias_scale=float(cf_bias_scale),
                cf_attention_bias_strength=float(cf_attention_bias_strength),
            )
            out_final = self._run_core_with_attention_score_bias(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                attn_bias=attn_bias,
                answer_pos=answer_pos,
            )
        else:
            out_final = self._run_core_with_attention_score_bias(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                attn_bias=None,
                answer_pos=None,
            )
        hidden_last = out_final.hidden_states[-1]
        h_answer = self._answer_hidden_from_hidden(hidden_last, input_ids)
        digit_logits = torch.stack([head(h_answer) for head in self.digit_heads], dim=1)
        if return_answer_hidden:
            return {
                "digit_logits": digit_logits,
                "h_answer": h_answer,
            }
        return digit_logits

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
        processors = LogitsProcessorList([AllowedTokensOnly(allowed, fill_value=fill_value)])

        return self.base.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.answer_token_id,
            logits_processor=processors,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    @torch.no_grad()
    def generate_with_digits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        gen_out = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        if isinstance(gen_out, torch.Tensor):
            sequences = gen_out
        elif hasattr(gen_out, "sequences"):
            sequences = gen_out.sequences
        else:
            raise TypeError(f"Unexpected generate() output type: {type(gen_out)}")

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
