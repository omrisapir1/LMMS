# phase3/eval.py
#
# Phase-3 evaluation:
# - Uses Phase3ZModel.generate_with_digits
# - Two modes: greedy + sampling
# - Returns per-sample JSON rows (no aggregation)
#
from __future__ import annotations

from typing import Dict, List, Optional
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
FIRST_PART_PROMPT = '''<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n'''
SECOND_PART_PROMPT = '''<|im_end|>\n<|im_start|>assistant\n'''

def build_prompt(tokenizer, question: str) -> str:
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return FIRST_PART_PROMPT + question + SECOND_PART_PROMPT

class Phase3Evaluator:
    def __init__(
        self,
        *,
        tokenizer,
        dataset_name: str = "omrisap/phase_3",
        split: str = "eval",
        batch_size: int,
        answer_token_id: int,
        device: Optional[torch.device] = None,
    ):
        self.tokenizer = tokenizer
        self.answer_token_id = int(answer_token_id)
        self.batch_size = int(batch_size)
        self.device = device

        # ---------------------------------------------
        # Load dataset ONCE
        # ---------------------------------------------
        self.ds = load_dataset(dataset_name, split=split)
        # self.ds = self.ds[:100]

        # ---------------------------------------------
        # Build dataloader ONCE
        # ---------------------------------------------
        self.loader = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )

    # --------------------------------------------------
    # Collate
    # --------------------------------------------------
    def _collate(self, batch):
        input_ids = []
        attention_mask = []
        problems = []
        final_answers = []

        for ex in batch:
            prompt = build_prompt(tokenizer=self.tokenizer, question=ex["problem"])

            enc = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_ids.append(torch.tensor(enc, dtype=torch.long))
            attention_mask.append(torch.ones(len(enc), dtype=torch.long))
            problems.append(ex["problem"])
            final_answers.append(ex["final_answer"])

        max_len = max(x.size(0) for x in input_ids)

        def pad(x, pad_val):
            return torch.cat(
                [x, torch.full((max_len - x.size(0),), pad_val, dtype=x.dtype)],
                dim=0,
            )

        input_ids = torch.stack(
            [pad(x, self.tokenizer.pad_token_id) for x in input_ids]
        )
        attention_mask = torch.stack([pad(x, 0) for x in attention_mask])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "problem": problems,
            "final_answer": final_answers,
        }

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _truncate_at_answer(self, token_ids: List[int]) -> List[int]:
        """
        Truncate sequence at first <ANSWER> (inclusive).
        If <ANSWER> not present, return full sequence.
        """
        if self.answer_token_id in token_ids:
            idx = token_ids.index(self.answer_token_id)
            return token_ids[: idx + 1]
        return token_ids

    def _decode_tokens(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        *,
        model,
        max_generation_tokens: int,
        sampling_temperature: float,
        top_p: float,
        top_k: int,
    ) -> List[Dict]:

        if self.device is None:
            self.device = next(model.parameters()).device

        model.eval()

        # Store intermediate results by global index
        rows: Dict[int, Dict] = {}

        # ==================================================
        # Pass 1: Greedy
        # ==================================================
        global_idx = 0
        for batch in self.loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            out = model.generate_with_digits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_generation_tokens,
                do_sample=False,
            )

            sequences = out["sequences"]
            digit_preds = out["digit_preds"]  # [B,5]

            B = sequences.size(0)

            for b in range(B):
                try:
                    seq_ids = sequences[b].tolist()
                except Exception as e:
                    print(sequences[b])
                    print(f"Error on batch {b}: {e}")
                    continue
                seq_ids = self._truncate_at_answer(seq_ids)
                decoded = self._decode_tokens(seq_ids)

                has_answer = self.answer_token_id in seq_ids
                digit_answer = (
                    "".join(str(d) for d in digit_preds[b].tolist())
                    if has_answer
                    else None
                )

                rows[global_idx] = {
                    "problem": batch["problem"][b],
                    "final_answer": batch["final_answer"][b],
                    "greedy_tokens": decoded,
                    "greedy_digit_answer": digit_answer,
                }
                global_idx += 1

        # ==================================================
        # Pass 2: Sampling
        # ==================================================
        global_idx = 0
        for batch in self.loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            out = model.generate_with_digits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_generation_tokens,
                do_sample=True,
                temperature=sampling_temperature,
                top_p=top_p,
                top_k=top_k,
            )

            sequences = out["sequences"]
            digit_preds = out["digit_preds"]

            B = sequences.size(0)

            for b in range(B):
                seq_ids = sequences[b].tolist()
                seq_ids = self._truncate_at_answer(seq_ids)
                decoded = self._decode_tokens(seq_ids)

                has_answer = self.answer_token_id in seq_ids
                digit_answer = (
                    "".join(str(d) for d in digit_preds[b].tolist())
                    if has_answer
                    else None
                )

                rows[global_idx].update(
                    {
                        "sample_tokens": decoded,
                        "sample_digit_answer": digit_answer,
                    }
                )
                global_idx += 1

        # --------------------------------------------------
        # Return ordered list
        # --------------------------------------------------
        return [rows[i] for i in range(len(rows))]


__all__ = ["Phase3Evaluator"]
