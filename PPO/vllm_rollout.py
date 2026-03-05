from __future__ import annotations

import gc
import inspect
import os
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

import torch


class VLLMRolloutEngine:
    def __init__(
        self,
        *,
        init_ckpt: str,
        tokenizer,
        answer_token_id: int,
        z_allowed_token_ids: Sequence[int],
        digit_allowed_token_ids: Sequence[int],
        trust_remote_code: bool,
        engine_kwargs: Dict[str, Any],
        output_dir: str,
        tmp_ckpt_dir: str,
        sync_every: int,
        seed: int,
        logger,
    ) -> None:
        self.tokenizer = tokenizer
        self.answer_token_id = int(answer_token_id)
        self.z_allowed_token_ids = [int(x) for x in z_allowed_token_ids]
        self.digit_allowed_token_ids = [int(x) for x in digit_allowed_token_ids]
        self.trust_remote_code = bool(trust_remote_code)
        self.engine_kwargs = dict(engine_kwargs)
        self.output_dir = output_dir
        self.tmp_ckpt_dir = tmp_ckpt_dir
        self.sync_every = max(1, int(sync_every))
        self.seed = int(seed)
        self._log = logger
        self._model_ref = init_ckpt
        self._llm = None
        self._sampling_params_cls = None

        self._init_engine(model_ref=init_ckpt)

    def close(self) -> None:
        self._destroy_engine()

    def _init_engine(self, *, model_ref: str) -> None:
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError(
                "vLLM rollout is enabled but vllm is not importable. "
                "Install a compatible vllm version or set rollout.vllm_enabled=False"
            ) from exc

        kwargs = dict(self.engine_kwargs)
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("trust_remote_code", self.trust_remote_code)

        self._sampling_params_cls = SamplingParams
        self._llm = LLM(model=model_ref, tokenizer=model_ref, **kwargs)
        self._model_ref = model_ref

    def _destroy_engine(self) -> None:
        if self._llm is None:
            return
        try:
            del self._llm
        finally:
            self._llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build_sampling_params(
        self,
        *,
        allowed_token_ids: Sequence[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        greedy: bool,
        min_tokens: Optional[int] = None,
        stop_on_answer: bool = False,
    ):
        if self._sampling_params_cls is None:
            raise RuntimeError("vLLM SamplingParams class is not initialized")
        sig = inspect.signature(self._sampling_params_cls.__init__)
        params = sig.parameters

        kwargs: Dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "temperature": 0.0 if greedy else float(temperature),
            "top_p": 1.0 if greedy else float(top_p),
            "n": 1,
        }
        if min_tokens is not None and "min_tokens" in params:
            kwargs["min_tokens"] = int(min_tokens)
        if stop_on_answer:
            if "stop_token_ids" in params:
                kwargs["stop_token_ids"] = [int(self.answer_token_id)]
            elif "stop" in params:
                kwargs["stop"] = [str(self.tokenizer.convert_ids_to_tokens(int(self.answer_token_id)))]

        # Version-adaptive allowed-token restriction. We intentionally avoid logits processors.
        if "allowed_token_ids" in params:
            kwargs["allowed_token_ids"] = list(int(x) for x in allowed_token_ids)
        elif "allowed_token_ids_list" in params:
            kwargs["allowed_token_ids_list"] = [list(int(x) for x in allowed_token_ids)]
        else:
            raise RuntimeError(
                "This vLLM version does not expose allowed-token-id sampling params; "
                "cannot enforce PPO FSM without logits processors."
            )

        return self._sampling_params_cls(**kwargs)

    def supports_prompt_token_ids(self) -> bool:
        if self._llm is None:
            return False
        try:
            sig = inspect.signature(self._llm.generate)
        except Exception:
            return False
        return "prompt_token_ids" in sig.parameters

    def _try_runtime_weight_sync(self, model) -> bool:
        if self._llm is None:
            return False

        # Best-effort runtime sync across vLLM variants.
        candidate_calls = []
        if hasattr(self._llm, "load_weights"):
            candidate_calls.append((self._llm, "load_weights"))
        engine = getattr(self._llm, "llm_engine", None)
        if engine is not None:
            for obj_name in ("model_executor", "executor"):
                obj = getattr(engine, obj_name, None)
                if obj is None:
                    continue
                for meth in ("load_weights", "set_weights", "update_weights"):
                    if hasattr(obj, meth):
                        candidate_calls.append((obj, meth))

        if not candidate_calls:
            return False

        state = model.state_dict()
        for obj, meth in candidate_calls:
            fn = getattr(obj, meth)
            for payload in (state, state.items(), list(state.items())):
                try:
                    fn(payload)
                    self._log(f"vLLM in-place weight sync succeeded via {obj.__class__.__name__}.{meth}")
                    return True
                except Exception:
                    continue
        return False

    def _atomic_export_checkpoint(self, model, tokenizer, *, update_idx: int) -> str:
        base = self.tmp_ckpt_dir or os.path.join(self.output_dir, "vllm_ckpt_latest")
        parent = os.path.dirname(base) or "."
        os.makedirs(parent, exist_ok=True)

        stage = f"{base}.staging_u{int(update_idx)}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        model.save_pretrained(stage)
        tokenizer.save_pretrained(stage)

        live = base
        old = f"{base}.old"
        if os.path.exists(old):
            shutil.rmtree(old, ignore_errors=True)
        if os.path.exists(live):
            os.rename(live, old)
        os.rename(stage, live)
        if os.path.exists(old):
            shutil.rmtree(old, ignore_errors=True)
        return live

    def _rebuild_from_checkpoint(self, ckpt_dir: str) -> None:
        self._destroy_engine()
        self._init_engine(model_ref=ckpt_dir)

    def maybe_sync_from_torch(self, model, tokenizer, update_idx: int) -> bool:
        should_sync = int(update_idx) == 1 or (int(update_idx) % self.sync_every == 0)
        if not should_sync:
            return False

        if self._try_runtime_weight_sync(model):
            return True

        ckpt_dir = self._atomic_export_checkpoint(model, tokenizer, update_idx=int(update_idx))
        self._rebuild_from_checkpoint(ckpt_dir)
        self._log(f"vLLM engine rebuilt from latest policy checkpoint: {ckpt_dir}")
        return True

    def generate_z(
        self,
        prompts: Optional[Sequence[str]] = None,
        prompt_token_ids: Optional[Sequence[Sequence[int]]] = None,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[List[int]]:
        if self._llm is None:
            raise RuntimeError("vLLM engine is not initialized")
        sp = self._build_sampling_params(
            allowed_token_ids=self.z_allowed_token_ids,
            max_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            greedy=False,
            stop_on_answer=True,
        )
        if prompt_token_ids is not None and self.supports_prompt_token_ids():
            outs = self._llm.generate(prompt_token_ids=[list(map(int, x)) for x in prompt_token_ids], sampling_params=sp)
        else:
            if prompts is None:
                raise RuntimeError("generate_z requires text prompts when prompt_token_ids are unsupported")
            outs = self._llm.generate(list(prompts), sp)
        return [list(getattr(o.outputs[0], "token_ids", []) or []) for o in outs]

    def generate_digits(
        self,
        prompts: Optional[Sequence[str]] = None,
        prompt_token_ids: Optional[Sequence[Sequence[int]]] = None,
        *,
        temperature: float,
        top_p: float,
        greedy: bool,
    ) -> List[List[int]]:
        if self._llm is None:
            raise RuntimeError("vLLM engine is not initialized")
        sp = self._build_sampling_params(
            allowed_token_ids=self.digit_allowed_token_ids,
            max_tokens=5,
            min_tokens=5,
            temperature=float(temperature),
            top_p=float(top_p),
            greedy=bool(greedy),
        )
        if prompt_token_ids is not None and self.supports_prompt_token_ids():
            outs = self._llm.generate(prompt_token_ids=[list(map(int, x)) for x in prompt_token_ids], sampling_params=sp)
        else:
            if prompts is None:
                raise RuntimeError("generate_digits requires text prompts when prompt_token_ids are unsupported")
            outs = self._llm.generate(list(prompts), sp)
        return [list(getattr(o.outputs[0], "token_ids", []) or []) for o in outs]
