# phase3/loss.py
#
# Phase-3 losses:
# 1) Answer loss (digit heads) — reused exactly from Phase-2
# 2) SFT loss — standard causal LM cross entropy
# 3) KL diversity loss on digit distributions via Z-sequence perturbations
#
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from z_pipeline.phase2.loss import AnswerLoss


# -------------------------------------------------
# 2) Standard SFT loss
# -------------------------------------------------

class SFTLoss:
    """
    Standard causal LM cross-entropy loss.
    """

    def __init__(self, ignore_index: int = -100):
        self.ignore_index = int(ignore_index)

    def compute(
        self,
        *,
        logits: torch.Tensor,          # [B,T,V]
        input_ids: torch.Tensor,       # [B,T]
        attention_mask: torch.Tensor,  # [B,T]
    ) -> torch.Tensor:
        B, T, V = logits.shape

        shift_logits = logits[:, :-1, :]        # [B,T-1,V]
        shift_labels = input_ids[:, 1:]         # [B,T-1]
        shift_mask = attention_mask[:, 1:]      # [B,T-1]

        labels = shift_labels.clone()
        labels[shift_mask == 0] = self.ignore_index

        loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )
        return loss


# -------------------------------------------------
# 3) KL diversity loss on digits
# -------------------------------------------------

class DigitKLDiversityLoss:
    """
    KL loss that encourages *different* digit distributions when Z sequence is perturbed.

    Alternative Z sequence is chosen per-sample based on Z-length-dependent probability:
      P(reverse | length=K)
      otherwise -> random Z replacement (same length)
    """

    def __init__(
        self,
        *,
        z_token_ids: List[int],
        answer_token_id: int,
        length_to_reverse_prob: Dict[int, float],  # K -> prob(reverse)
        digit_temperature: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        self.z_token_ids = set(int(x) for x in z_token_ids)
        self.answer_token_id = int(answer_token_id)
        self.length_to_reverse_prob = dict(length_to_reverse_prob)
        self.digit_temperature = float(digit_temperature)
        self.random_seed = random_seed

    # -------------------------
    # helpers
    # -------------------------

    def _digit_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.digit_temperature, dim=-1)

    def _find_answer_pos(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids == self.answer_token_id)
        if not torch.all(mask.sum(dim=1) == 1):
            raise RuntimeError("Each sample must contain exactly one <ANSWER> token")
        return mask.float().argmax(dim=1)

    def _extract_z_positions(
        self,
        input_ids: torch.Tensor,
        answer_pos: torch.Tensor,
    ) -> List[List[int]]:
        ids = input_ids.detach().cpu().tolist()
        ans = answer_pos.detach().cpu().tolist()

        out = []
        for row, ap in zip(ids, ans):
            out.append([i for i, t in enumerate(row[:ap]) if t in self.z_token_ids])
        return out

    # -------------------------
    # build alternative
    # -------------------------

    def _build_alternative_input_ids(
        self,
        input_ids: torch.Tensor,
        z_positions: List[List[int]],
    ) -> torch.Tensor:
        device = input_ids.device
        alt = input_ids.clone()

        gen = torch.Generator(device=device)
        if self.random_seed is not None:
            gen.manual_seed(self.random_seed)

        z_token_tensor = torch.tensor(list(self.z_token_ids), device=device)

        for b, pos in enumerate(z_positions):
            K = len(pos)
            if K <= 1:
                continue

            p_reverse = self.length_to_reverse_prob.get(K, 0.0)
            do_reverse = torch.rand((), generator=gen, device=device) < p_reverse

            if do_reverse:
                vals = alt[b, pos].clone()
                alt[b, pos] = torch.flip(vals, dims=[0])
            else:
                idx = torch.randint(
                    low=0,
                    high=z_token_tensor.numel(),
                    size=(K,),
                    generator=gen,
                    device=device,
                )
                alt[b, torch.tensor(pos, device=device)] = z_token_tensor[idx]

        return alt

    # -------------------------
    # main compute
    # -------------------------

    def compute(
        self,
        *,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        digit_logits_ref: torch.Tensor,  # [B,5,10]
    ) -> torch.Tensor:
        """
        Returns scalar KL loss (NEGATIVE KL → maximize divergence).
        """

        p_ref = self._digit_probs(digit_logits_ref).detach()

        answer_pos = self._find_answer_pos(input_ids)
        z_positions = self._extract_z_positions(input_ids, answer_pos)

        alt_ids = self._build_alternative_input_ids(input_ids, z_positions)

        out_alt = model(
            input_ids=alt_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        digit_logits_alt = out_alt.digit_logits
        p_alt = self._digit_probs(digit_logits_alt)

        eps = 1e-12
        p_ref = p_ref.clamp_min(eps)
        p_alt = p_alt.clamp_min(eps)

        kl = (p_ref * (p_ref.log() - p_alt.log())).sum(dim=-1)  # [B,5]
        kl = kl.sum(dim=-1).mean()  # scalar

        # NEGATIVE → maximize KL
        return -kl


# -------------------------------------------------
# Loss orchestrator
# -------------------------------------------------

class Phase3Loss:
    """
    Combines:
      - AnswerLoss
      - SFT loss
      - KL diversity loss
    """

    def __init__(
        self,
        *,
        answer_loss: AnswerLoss,
        sft_loss: SFTLoss,
        kl_loss: DigitKLDiversityLoss,
        lambda_answer: float,
        lambda_sft: float,
        lambda_kl: float,
    ):
        self.answer_loss = answer_loss
        self.sft_loss = sft_loss
        self.kl_loss = kl_loss
        self.lambda_answer = float(lambda_answer)
        self.lambda_sft = float(lambda_sft)
        self.lambda_kl = float(lambda_kl)

    def compute(
        self,
        *,
        model,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        digit_logits: torch.Tensor,
        digit_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        loss_answer = self.answer_loss.compute(digit_logits, digit_labels)
        loss_sft = self.sft_loss.compute(
            logits=logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        loss_kl = self.kl_loss.compute(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            digit_logits_ref=digit_logits,
        )

        loss_total = (
            self.lambda_answer * loss_answer
            + self.lambda_sft * loss_sft
            + self.lambda_kl * loss_kl
        )

        return {
            "loss_total": loss_total,
            "loss_answer": loss_answer,
            "loss_sft": loss_sft,
            "loss_kl": loss_kl,
        }
