from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence

import torch
from transformers import LogitsProcessor, LogitsProcessorList


class _AllowedTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: Sequence[int]) -> None:
        self.allowed_token_ids = [int(x) for x in allowed_token_ids]
        self._mask = None
        self._vocab_size = None
        self._device = None

    def _ensure_mask(self, scores: torch.Tensor) -> None:
        vocab_size = int(scores.shape[-1])
        device = scores.device
        if self._mask is not None and self._vocab_size == vocab_size and self._device == device:
            return
        mask = torch.full((vocab_size,), float("-inf"), device=device, dtype=scores.dtype)
        ids = torch.tensor(self.allowed_token_ids, dtype=torch.long, device=device)
        valid = ids[(ids >= 0) & (ids < vocab_size)]
        mask[valid] = 0.0
        self._mask = mask
        self._vocab_size = vocab_size
        self._device = device

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._ensure_mask(scores)
        return scores + self._mask


class _AdditiveLogitBiasProcessor(LogitsProcessor):
    def __init__(self, logit_bias: Dict[int, float]) -> None:
        self.logit_bias = {int(k): float(v) for k, v in dict(logit_bias).items()}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if not self.logit_bias:
            return scores
        for tok_id, bias in self.logit_bias.items():
            if 0 <= int(tok_id) < int(scores.shape[-1]):
                scores[:, int(tok_id)] = scores[:, int(tok_id)] + float(bias)
        return scores


class HFRolloutEngine:
    def __init__(
        self,
        *,
        tokenizer,
        answer_token_id: int,
        z_allowed_token_ids: Sequence[int],
        digit_allowed_token_ids: Sequence[int],
        sync_every: int,
        logger,
        verify_allowed_token_ids: Sequence[int] = (),
        finalize_token_id: Optional[int] = None,
        retry_token_id: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.answer_token_id = int(answer_token_id)
        self.z_allowed_token_ids = [int(x) for x in z_allowed_token_ids]
        self.digit_allowed_token_ids = [int(x) for x in digit_allowed_token_ids]
        self.verify_allowed_token_ids = [int(x) for x in verify_allowed_token_ids]
        self.finalize_token_id = int(finalize_token_id) if finalize_token_id is not None else None
        self.retry_token_id = int(retry_token_id) if retry_token_id is not None else None
        if len(self.verify_allowed_token_ids) > 0:
            if self.finalize_token_id is None or self.retry_token_id is None:
                raise RuntimeError("HF rollout verify ids are required when verify_allowed_token_ids is set")
            if sorted(set(self.verify_allowed_token_ids)) != sorted({self.finalize_token_id, self.retry_token_id}):
                raise RuntimeError("HF rollout verify allowlist must contain exactly <FINALIZE> and <RETRY>")
        if (self.finalize_token_id is None) != (self.retry_token_id is None):
            raise RuntimeError("HF rollout finalize/retry ids must be set together")
        self.sync_every = max(1, int(sync_every))
        self._log = logger
        self._lock = threading.RLock()
        self._model = None
        self._device = None

    def close(self) -> None:
        return

    def supports_prompt_token_ids(self) -> bool:
        return True

    def maybe_sync_from_torch(self, model, tokenizer, update_idx: int) -> bool:
        del tokenizer
        with self._lock:
            self._model = model
            try:
                self._device = next(model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        should_sync = int(update_idx) == 1 or (int(update_idx) % self.sync_every == 0)
        return bool(should_sync)

    def _build_inputs(self, prompts: Optional[Sequence[str]], prompt_token_ids: Optional[Sequence[Sequence[int]]]) -> List[List[int]]:
        if prompt_token_ids is not None:
            return [list(map(int, p)) for p in prompt_token_ids]
        if prompts is None:
            raise RuntimeError("prompts must be provided when prompt_token_ids is None")
        rows: List[List[int]] = []
        for text in prompts:
            pack = self.tokenizer(str(text), add_special_tokens=False, return_attention_mask=False)
            rows.append([int(x) for x in list(pack["input_ids"])])
        return rows

    def _run_generate_batch(
        self,
        *,
        input_ids_rows: List[List[int]],
        allowed_token_ids: Sequence[int],
        max_new_tokens: int,
        min_new_tokens: Optional[int],
        temperature: float,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        greedy: bool,
        eos_token_id: Optional[int],
        num_return_sequences: int = 1,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> List[List[int]]:
        if self._model is None:
            raise RuntimeError("HF rollout model is not set; call maybe_sync_from_torch first")
        if len(input_ids_rows) == 0:
            return []

        do_sample = not bool(greedy)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = int(self.answer_token_id)

        logits_processors: List[LogitsProcessor] = [_AllowedTokenLogitsProcessor(allowed_token_ids)]
        if logit_bias is not None and len(logit_bias) > 0:
            logits_processors.append(_AdditiveLogitBiasProcessor(logit_bias))
        logits_processor = LogitsProcessorList(logits_processors)
        lengths = [len(x) for x in input_ids_rows]
        max_len = max(lengths)
        batch_size = len(input_ids_rows)
        inpt = torch.full(
            (batch_size, max_len),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=self._device,
        )
        attn = torch.zeros((batch_size, max_len), dtype=torch.long, device=self._device)
        for i, row in enumerate(input_ids_rows):
            n = len(row)
            if n == 0:
                continue
            inpt[i, :n] = torch.tensor(row, dtype=torch.long, device=self._device)
            attn[i, :n] = 1

        gen_kwargs: Dict[str, object] = {
            "input_ids": inpt,
            "attention_mask": attn,
            "logits_processor": logits_processor,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "pad_token_id": int(pad_token_id),
            "eos_token_id": None if eos_token_id is None else int(eos_token_id),
            "num_return_sequences": int(num_return_sequences),
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
            gen_kwargs["min_p"] = float(min_p)
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)
        if min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = int(min_new_tokens)

        try:
            out = self._model.generate(**gen_kwargs)
        except TypeError:
            # Backward compatibility for transformers builds without min_p support.
            gen_kwargs.pop("min_p", None)
            out = self._model.generate(**gen_kwargs)
        out_cpu = out.detach().to("cpu")
        prompt_len = int(inpt.shape[1])
        rows: List[List[int]] = []
        total_rows = int(out_cpu.shape[0])
        for i in range(total_rows):
            full = out_cpu[i].tolist()
            gen = [int(x) for x in full[prompt_len:]]
            rows.append(gen)
        return rows

    def generate_z(
        self,
        prompts: Optional[Sequence[str]] = None,
        prompt_token_ids: Optional[Sequence[Sequence[int]]] = None,
        *,
        num_samples_per_prompt: int = 1,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        greedy: bool = False,
    ) -> List[Dict[str, object]]:
        inputs = self._build_inputs(prompts, prompt_token_ids)
        rows: List[Dict[str, object]] = []

        with self._lock:
            if self._model is None:
                raise RuntimeError("HF rollout model is not set")
            was_training = bool(self._model.training)
            self._model.eval()
            try:
                with torch.no_grad():
                    z_allowed = sorted(set(int(x) for x in self.z_allowed_token_ids) | {int(self.answer_token_id)})
                    n = max(1, int(num_samples_per_prompt))
                    gens = self._run_generate_batch(
                        input_ids_rows=inputs,
                        allowed_token_ids=z_allowed,
                        max_new_tokens=int(max_new_tokens),
                        min_new_tokens=None,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        min_p=float(min_p),
                        repetition_penalty=float(repetition_penalty),
                        greedy=bool(greedy),
                        eos_token_id=int(self.answer_token_id),
                        num_return_sequences=n,
                    )
                    for gen in gens:
                        ended_on_answer = len(gen) > 0 and int(gen[-1]) == int(self.answer_token_id)
                        rows.append(
                            {
                                "token_ids": [int(x) for x in gen],
                                "token_logprobs": None,
                                "stop_reason": int(self.answer_token_id) if ended_on_answer else None,
                                "finish_reason": "stop" if ended_on_answer else "length",
                            }
                        )
            finally:
                if was_training:
                    self._model.train()
        return rows

    def generate_digits(
        self,
        prompts: Optional[Sequence[str]] = None,
        prompt_token_ids: Optional[Sequence[Sequence[int]]] = None,
        *,
        num_samples_per_prompt: int = 1,
        num_digits: int = 5,
        temperature: float,
        top_p: float,
        greedy: bool,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> List[List[int]]:
        inputs = self._build_inputs(prompts, prompt_token_ids)
        rows: List[List[int]] = []
        k = int(num_digits)
        if k < 1 or k > 5:
            raise RuntimeError(f"num_digits must be in [1, 5], got {k}")

        with self._lock:
            if self._model is None:
                raise RuntimeError("HF rollout model is not set")
            was_training = bool(self._model.training)
            self._model.eval()
            try:
                with torch.no_grad():
                    gens = self._run_generate_batch(
                        input_ids_rows=inputs,
                        allowed_token_ids=self.digit_allowed_token_ids,
                        max_new_tokens=k,
                        min_new_tokens=k,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        min_p=float(min_p),
                        repetition_penalty=float(repetition_penalty),
                        greedy=bool(greedy),
                        eos_token_id=None,
                        num_return_sequences=max(1, int(num_samples_per_prompt)),
                    )
                    for gen in gens:
                        if len(gen) != k:
                            raise RuntimeError(f"HF digit generation must return exactly {k} tokens, got {len(gen)}")
                        rows.append([int(x) for x in gen])
            finally:
                if was_training:
                    self._model.train()
        return rows

    def generate_verify(
        self,
        prompts: Optional[Sequence[str]] = None,
        prompt_token_ids: Optional[Sequence[Sequence[int]]] = None,
        *,
        num_samples_per_prompt: int = 1,
        temperature: float,
        top_p: float,
        greedy: bool,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> List[List[int]]:
        inputs = self._build_inputs(prompts, prompt_token_ids)
        rows: List[List[int]] = []

        with self._lock:
            if self._model is None:
                raise RuntimeError("HF rollout model is not set")
            was_training = bool(self._model.training)
            self._model.eval()
            try:
                with torch.no_grad():
                    gens = self._run_generate_batch(
                        input_ids_rows=inputs,
                        allowed_token_ids=self.verify_allowed_token_ids,
                        max_new_tokens=1,
                        min_new_tokens=1,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        min_p=float(min_p),
                        repetition_penalty=float(repetition_penalty),
                        greedy=bool(greedy),
                        eos_token_id=None,
                        num_return_sequences=max(1, int(num_samples_per_prompt)),
                        logit_bias=logit_bias,
                    )
                    for gen in gens:
                        if len(gen) != 1:
                            raise RuntimeError(f"HF verify generation must return exactly 1 token, got {len(gen)}")
                        rows.append([int(gen[0])])
            finally:
                if was_training:
                    self._model.train()
        return rows
