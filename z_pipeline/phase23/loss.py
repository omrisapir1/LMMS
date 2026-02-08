from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import safe_softmax


@dataclass
class LossOutputs:
    total: torch.Tensor
    answer: torch.Tensor
    answer_sft: torch.Tensor
    softz: torch.Tensor
    counterfactual: torch.Tensor
    usage: torch.Tensor


class AnswerDigitLoss(nn.Module):
    """Digit CE with optional per-position keep-prob masking for label==0."""

    def __init__(self, keep_prob: Optional[Union[Mapping[int, float], Sequence[float]]] = None) -> None:
        super().__init__()
        self.keep_prob = self._normalize_keep_prob(keep_prob)
        self._warned_fallback = False

    @staticmethod
    def _normalize_keep_prob(
        keep_prob: Optional[Union[Mapping[int, float], Sequence[float]]]
    ) -> Sequence[float]:
        if keep_prob is None:
            return [1.0] * 5

        if isinstance(keep_prob, Mapping):
            keys = set(int(k) for k in keep_prob.keys())
            if keys == {0, 1, 2, 3, 4}:
                vals = [float(keep_prob[i]) for i in range(5)]
            elif keys == {1, 2, 3, 4, 5}:
                vals = [float(keep_prob[i]) for i in range(1, 6)]
            else:
                raise ValueError("keep_prob mapping keys must be 0..4 or 1..5")
        else:
            vals = [float(x) for x in keep_prob]
            if len(vals) != 5:
                raise ValueError("keep_prob sequence must have length 5")

        for p in vals:
            if not (0.0 <= p <= 1.0):
                raise ValueError("keep_prob values must be in [0,1]")
        return vals

    def forward(
        self,
        digit_logits: torch.Tensor,  # [B,5,10]
        digit_labels: torch.Tensor,  # [B,5]
    ) -> torch.Tensor:
        if digit_logits.ndim != 3 or digit_logits.shape[1:] != (5, 10):
            raise ValueError("digit_logits must be [B,5,10]")
        if digit_labels.ndim != 2 or digit_labels.shape[1] != 5:
            raise ValueError("digit_labels must be [B,5]")

        total = digit_logits.new_zeros(())
        contributed = 0
        for i in range(5):
            logits_i = digit_logits[:, i, :]
            labels_i = digit_labels[:, i]
            is_zero = (labels_i == 0)

            kp = float(self.keep_prob[i])
            if kp >= 1.0:
                include_zero = torch.ones_like(is_zero, dtype=torch.bool)
            elif kp <= 0.0:
                include_zero = torch.zeros_like(is_zero, dtype=torch.bool)
            else:
                probs = torch.full(labels_i.shape, kp, device=labels_i.device, dtype=torch.float)
                include_zero = torch.zeros_like(is_zero, dtype=torch.bool)
                if is_zero.any():
                    include_zero[is_zero] = torch.bernoulli(probs[is_zero]).bool()

            include = (~is_zero) | (is_zero & include_zero)
            if include.any():
                total = total + F.cross_entropy(logits_i[include], labels_i[include], reduction="mean")
                contributed += 1

        if contributed == 0:
            if not self._warned_fallback:
                warnings.warn(
                    "AnswerDigitLoss fallback activated: no digit contributed after keep-prob mask.",
                    stacklevel=2,
                )
                self._warned_fallback = True
            per_digit = [
                F.cross_entropy(digit_logits[:, i, :], digit_labels[:, i], reduction="mean")
                for i in range(5)
            ]
            return torch.stack(per_digit).mean()

        return total / float(contributed)


class AnswerTokenSFTLoss(nn.Module):
    """CE on the next-token logits right before <ANSWER>, target is <ANSWER>."""

    def __init__(self, answer_token_id: int) -> None:
        super().__init__()
        self.answer_token_id = int(answer_token_id)

    def forward(self, answer_next_logits: torch.Tensor) -> torch.Tensor:
        if answer_next_logits.ndim != 2:
            raise ValueError("answer_next_logits must be [B,V]")
        labels = torch.full(
            (answer_next_logits.size(0),),
            self.answer_token_id,
            dtype=torch.long,
            device=answer_next_logits.device,
        )
        return F.cross_entropy(answer_next_logits, labels)


def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


class CounterfactualAnswerLoss(nn.Module):
    def __init__(
        self,
        *,
        permute_prob: Dict[int, float],
        digit_temperature: float = 1.0,
        seed: Optional[int] = None,
        debug_every: int = 0,
    ) -> None:
        super().__init__()
        self.permute_prob = {int(k): float(v) for k, v in permute_prob.items()}
        self.digit_temperature = float(digit_temperature)
        self.seed = seed
        self.debug_every = int(debug_every)

    def _deterministic_pz(
        self,
        p_z: torch.Tensor,  # [B,Kmax,V]
    ) -> torch.Tensor:
        vocab = p_z.size(-1)
        idx = p_z.argmax(dim=-1)
        return F.one_hot(idx, num_classes=vocab).to(dtype=p_z.dtype)

    def _deterministic_z_idx(
        self,
        p_z: torch.Tensor,  # [B,Kmax,V]
    ) -> torch.Tensor:
        return p_z.argmax(dim=-1)

    def _build_counterfactual_pz(
        self,
        p_z: torch.Tensor,      # [B,Kmax,V]
        k_vals: torch.Tensor,   # [B]
        cf_mode: str,
    ) -> torch.Tensor:
        bsz, kmax, vocab = p_z.shape
        device = p_z.device
        if cf_mode == "det":
            base = self._deterministic_pz(p_z)
        elif cf_mode == "gs":
            base = p_z
        else:
            raise ValueError(f"Unknown cf_mode: {cf_mode}")

        out = base.clone()

        gen = torch.Generator(device=device)
        if self.seed is not None:
            gen.manual_seed(int(self.seed))

        for b in range(bsz):
            k = int(k_vals[b].item())
            if k <= 0:
                continue
            if k > kmax:
                raise RuntimeError(f"k_vals[{b}]={k} exceeds p_z slots={kmax}")

            prob = self.permute_prob.get(k, 0.0)

            do_permute = (k > 1 and torch.rand((), device=device, generator=gen) < prob)
            if do_permute:
                perm = torch.randperm(k, device=device, generator=gen)
                if cf_mode == "det":
                    active_idx = base[b, :k].argmax(dim=-1)
                    cf_idx = active_idx[perm]
                    out[b, :k] = F.one_hot(cf_idx, num_classes=vocab).to(dtype=p_z.dtype)
                else:
                    out[b, :k] = base[b, :k][perm]
            else:
                if cf_mode == "det":
                    cf_idx = torch.randint(
                        low=0,
                        high=vocab,
                        size=(k,),
                        device=device,
                        generator=gen,
                    )
                    out[b, :k] = F.one_hot(cf_idx, num_classes=vocab).to(dtype=p_z.dtype)
                else:
                    # Structure-preserving GS perturbation:
                    # temperature-warp current policy distributions (no fresh random simplex samples).
                    active = base[b, :k].clamp_min(1e-8)
                    tau = torch.empty((k, 1), device=device, dtype=active.dtype)
                    tau.uniform_(0.7, 1.6, generator=gen)
                    warped = torch.softmax(active.log() / tau, dim=-1)
                    out[b, :k] = warped.to(dtype=p_z.dtype)

        return out

    def _build_counterfactual_z_idx(
        self,
        z_idx: torch.Tensor,    # [B,Kmax]
        k_vals: torch.Tensor,   # [B]
        vocab_size: int,
    ) -> torch.Tensor:
        bsz, kmax = z_idx.shape
        device = z_idx.device
        out = z_idx.clone()

        gen = torch.Generator(device=device)
        if self.seed is not None:
            gen.manual_seed(int(self.seed))

        for b in range(bsz):
            k = int(k_vals[b].item())
            if k <= 0:
                continue
            if k > kmax:
                raise RuntimeError(f"k_vals[{b}]={k} exceeds z_idx slots={kmax}")

            prob = self.permute_prob.get(k, 0.0)
            do_permute = (k > 1 and torch.rand((), device=device, generator=gen) < prob)
            if do_permute:
                perm = torch.randperm(k, device=device, generator=gen)
                out[b, :k] = z_idx[b, :k][perm]
            else:
                if vocab_size <= 0:
                    raise RuntimeError("Cannot sample deterministic CF indices: empty latent vocabulary.")
                out[b, :k] = torch.randint(
                    low=0,
                    high=vocab_size,
                    size=(k,),
                    device=device,
                    generator=gen,
                )
        return out

    def forward(
            self,
            *,
            model,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            p_z: Optional[torch.Tensor] = None,
            p_z_idx_det: Optional[torch.Tensor] = None,
            k_vals: torch.Tensor,
            cf_bias_scale: float = 0.0,
            cf_attention_bias_strength: float = 0.0,
            apply_cf_answer_z_bias: bool = False,
            cf_mode: str = "gs",
            global_step: Optional[int] = None,
            stage_name: str = "main",
            return_details: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if cf_mode not in {"gs", "det"}:
            raise ValueError("cf_mode must be one of {'gs','det'}")
        is_det = (cf_mode == "det")

        if is_det:
            if p_z_idx_det is None:
                if p_z is None:
                    raise ValueError("DET mode requires p_z_idx_det or p_z.")
                p_ref_idx = self._deterministic_z_idx(p_z)
            else:
                p_ref_idx = p_z_idx_det
            cf_bias_scale = 0.0
            apply_cf_answer_z_bias = False
        else:
            if p_z is None:
                raise ValueError("GS mode requires p_z.")
            p_ref_z = p_z

        # ----------------------------
        # Reference
        # ----------------------------

        if is_det:
            ref_out = model.forward_with_fixed_z_distributions(
                input_ids=input_ids,
                attention_mask=attention_mask,
                p_z_idx=p_ref_idx.detach(),
                cf_bias_scale=0.0,
                apply_cf_answer_z_bias=False,
                cf_attention_bias_strength=0.0,
                return_answer_hidden=True,
            )
        else:
            ref_out = model.forward_with_fixed_z_distributions(
                input_ids=input_ids,
                attention_mask=attention_mask,
                p_z=p_ref_z.detach(),
                cf_bias_scale=0.0,
                apply_cf_answer_z_bias=False,
                cf_attention_bias_strength=0.0,
                return_answer_hidden=True,
            )
        digit_logits_ref_det = ref_out["digit_logits"]
        h_ans_ref = ref_out["h_answer"]
        p_ref = safe_softmax(
            digit_logits_ref_det, tau=self.digit_temperature, dim=-1
        ).detach()  # [B,5,10]

        # ----------------------------
        # Counterfactual
        # ----------------------------
        if is_det:
            p_cf_idx = self._build_counterfactual_z_idx(
                p_ref_idx,
                k_vals,
                vocab_size=len(model.z_token_ids),
            )

            cf_out = model.forward_with_fixed_z_distributions(
                input_ids=input_ids,
                attention_mask=attention_mask,
                p_z_idx=p_cf_idx,
                cf_bias_scale=0.0,
                apply_cf_answer_z_bias=False,
                cf_attention_bias_strength=0.0,
                return_answer_hidden=True,
            )
        else:
            p_cf_z = self._build_counterfactual_pz(p_ref_z, k_vals, cf_mode=cf_mode)
            cf_out = model.forward_with_fixed_z_distributions(
                input_ids=input_ids,
                attention_mask=attention_mask,
                p_z=p_cf_z,
                cf_bias_scale=float(cf_bias_scale),
                apply_cf_answer_z_bias=bool(apply_cf_answer_z_bias),
                cf_attention_bias_strength=float(cf_attention_bias_strength),
                return_answer_hidden=True,
            )
        digit_logits_cf = cf_out["digit_logits"]
        h_ans_cf = cf_out["h_answer"]
        p_cf = safe_softmax(
            digit_logits_cf, tau=self.digit_temperature, dim=-1
        )  # [B,5,10]

        # ----------------------------
        # Losses
        # ----------------------------
        js = js_divergence(p_ref, p_cf).mean()
        loss_cf = torch.log(torch.tensor(2.0, device=js.device, dtype=js.dtype)) - js

        ref_norm = F.normalize(h_ans_ref.detach().float(), p=2, dim=-1)
        cf_norm = F.normalize(h_ans_cf.float(), p=2, dim=-1)
        cos_sim = (ref_norm * cf_norm).sum(dim=-1).mean()
        # Use dependence loss only for GS-mode objective to avoid fighting deterministic CF.
        if cf_mode == "gs":
            # Minimize cosine similarity to enforce answer-state dependence on latent Z.
            # (equivalent up to constant shift: cos_sim - 1.0)
            loss_dep = cos_sim - 1.0
        else:
            loss_dep = torch.zeros((), device=cos_sim.device, dtype=cos_sim.dtype)
        dep_norm = (h_ans_ref.detach().float() - h_ans_cf.float()).norm(dim=-1).mean()

        if self.debug_every > 0 and global_step is not None and (int(global_step) % self.debug_every == 0):
            with torch.no_grad():
                delta_logits = (digit_logits_ref_det - digit_logits_cf).abs().mean()
                pred_ref = digit_logits_ref_det.argmax(dim=-1)  # [B,5]
                pred_cf = digit_logits_cf.argmax(dim=-1)        # [B,5]
                # Per-digit flip rate over [B,5].
                flip = (pred_ref != pred_cf).float().mean()

                conf_ref = p_ref.max(dim=-1).values.mean()
                conf_cf = p_cf.max(dim=-1).values.mean()

                top2_ref = torch.topk(p_ref, k=2, dim=-1).values
                top2_cf = torch.topk(p_cf, k=2, dim=-1).values
                margin_ref = (top2_ref[..., 0] - top2_ref[..., 1]).mean()
                margin_cf = (top2_cf[..., 0] - top2_cf[..., 1]).mean()

                h_ref = (-(p_ref.clamp_min(1e-8) * p_ref.clamp_min(1e-8).log()).sum(dim=-1)).mean()
                h_cf = (-(p_cf.clamp_min(1e-8) * p_cf.clamp_min(1e-8).log()).sum(dim=-1)).mean()

                print(
                    f"cf_debug step={int(global_step)} stage={stage_name} mode={cf_mode} "
                    f"bias_scale={float(cf_bias_scale):.3f} "
                    f"|dlogits|={float(delta_logits):.4f} "
                    f"JS={float(js):.4f} "
                    f"flip={float(flip):.4f} "
                    f"cos_sim={float(cos_sim):.4f} dep_norm={float(dep_norm):.4f} "
                    f"conf_ref={float(conf_ref):.4f} conf_cf={float(conf_cf):.4f} "
                    f"margin_ref={float(margin_ref):.4f} margin_cf={float(margin_cf):.4f} "
                    f"H_ref={float(h_ref):.4f} H_cf={float(h_cf):.4f}"
                )

        if return_details:
            return {
                "loss_cf": loss_cf,
                "loss_dep": loss_dep,
            }
        return loss_cf


def usage_shaping_loss_stub(*, device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)
