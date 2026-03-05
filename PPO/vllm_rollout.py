from __future__ import annotations

import gc
import os
import shutil
import threading
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
        # NEW: serialize vLLM generate/swap/rebuild operations.
        self._vllm_lock = threading.RLock()

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
        prev = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            self._llm = LLM(model=model_ref, tokenizer=model_ref, **kwargs)
        finally:
            if prev is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
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

        kwargs: Dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "temperature": 0.0 if greedy else float(temperature),
            "top_p": 1.0 if greedy else float(top_p),
            "n": 1,
        }
        if min_tokens is not None:
            kwargs["min_tokens"] = int(min_tokens)
        kwargs_with_stop = dict(kwargs)
        if stop_on_answer:
            try:
                kwargs_with_stop["stop_token_ids"] = [int(self.answer_token_id)]
                _ = self._sampling_params_cls(**kwargs_with_stop)
            except TypeError:
                kwargs_with_stop = dict(kwargs)
                kwargs_with_stop["stop"] = [str(self.tokenizer.convert_ids_to_tokens(int(self.answer_token_id)))]
                try:
                    _ = self._sampling_params_cls(**kwargs_with_stop)
                except TypeError:
                    kwargs_with_stop = dict(kwargs)

        try:
            return self._sampling_params_cls(
                **kwargs_with_stop,
                allowed_token_ids=[int(x) for x in allowed_token_ids],
            )
        except TypeError:
            try:
                return self._sampling_params_cls(
                    **kwargs_with_stop,
                    allowed_token_ids_list=[[int(x) for x in allowed_token_ids]],
                )
            except TypeError as exc:
                raise RuntimeError(
                    "This vLLM version does not expose allowed-token-id sampling params; "
                    "cannot enforce PPO FSM without logits processors."
                ) from exc

    def supports_prompt_token_ids(self) -> bool:
        # vLLM 0.14 accepts token prompts via TokensPrompt dicts in prompts list,
        # even when generate() signature does not expose prompt_token_ids kwarg.
        return True

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

    # NEW: best-effort hot swap using vLLM engine reload/load_weights hooks.
    def _hot_swap_from_hf_checkpoint(self, ckpt_dir: str) -> bool:
        with self._vllm_lock:
            if self._llm is None:
                return False
            engine = getattr(self._llm, "llm_engine", None)
            if engine is None:
                return False

            candidates = []
            candidates.append((engine, "reload_model"))
            for obj_name in ("model_executor", "executor"):
                obj = getattr(engine, obj_name, None)
                if obj is None:
                    continue
                candidates.append((obj, "reload_model"))
            for obj_name in ("model_executor", "executor"):
                obj = getattr(engine, obj_name, None)
                if obj is None:
                    continue
                candidates.append((obj, "load_weights"))

            last_error = None
            last_attempt = None
            for obj, meth in candidates:
                fn = getattr(obj, meth, None)
                if fn is None:
                    continue
                # Signature differs across vLLM versions/builds; try a few simple forms.
                for args, kwargs in (
                    ((ckpt_dir,), {}),
                    (tuple(), {"model": ckpt_dir}),
                    (tuple(), {"model_path": ckpt_dir}),
                    (tuple(), {"checkpoint": ckpt_dir}),
                    (tuple(), {"checkpoint_path": ckpt_dir}),
                ):
                    try:
                        fn(*args, **kwargs)
                        self._log(f"vLLM hot swap succeeded via {obj.__class__.__name__}.{meth}")
                        return True
                    except TypeError:
                        last_attempt = f"{obj.__class__.__name__}.{meth}"
                        last_error = TypeError(
                            f"TypeError for args={args}, kwargs={kwargs}"
                        )
                        continue
                    except Exception:
                        last_attempt = f"{obj.__class__.__name__}.{meth}"
                        last_error = Exception(
                            f"Exception for args={args}, kwargs={kwargs}"
                        )
                        continue
            if last_error is not None:
                self._log(f"vLLM hot swap failed; last attempt={last_attempt}; error={last_error}")
        return False

    def maybe_sync_from_torch(self, model, tokenizer, update_idx: int) -> bool:
        should_sync = int(update_idx) == 1 or (int(update_idx) % self.sync_every == 0)
        if not should_sync:
            return False

        # NEW: export synchronously to avoid save_pretrained() on a background thread.
        self._log("vLLM checkpoint export started")
        ckpt_dir = self._atomic_export_checkpoint(model, tokenizer, update_idx=int(update_idx))
        self._log("vLLM checkpoint ready")

        # NEW: swap only at rollout boundary; fallback to rebuild on failure.
        if self._hot_swap_from_hf_checkpoint(ckpt_dir):
            self._log("vLLM hot swap succeeded")
            return True

        with self._vllm_lock:
            self._rebuild_from_checkpoint(ckpt_dir)
        self._log(f"vLLM fallback engine rebuild: {ckpt_dir}")
        return True

    def generate_z(
        self,
        prompts: Optional[Sequence[str]] = None,
        prompt_token_ids: Optional[Sequence[Sequence[int]]] = None,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[Dict[str, object]]:
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
            token_prompts = [{"prompt_token_ids": list(map(int, x))} for x in prompt_token_ids]
            with self._vllm_lock:
                outs = self._llm.generate(token_prompts, sp, use_tqdm=False)
        else:
            if prompts is None:
                raise RuntimeError("generate_z requires text prompts when prompt_token_ids are unsupported")
            with self._vllm_lock:
                outs = self._llm.generate(list(prompts), sp, use_tqdm=False)
        rows: List[Dict[str, object]] = []
        for o in outs:
            out0 = o.outputs[0]
            rows.append(
                {
                    "token_ids": list(getattr(out0, "token_ids", []) or []),
                    "stop_reason": getattr(out0, "stop_reason", None),
                    "finish_reason": getattr(out0, "finish_reason", None),
                }
            )
        return rows

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
            token_prompts = [{"prompt_token_ids": list(map(int, x))} for x in prompt_token_ids]
            with self._vllm_lock:
                outs = self._llm.generate(token_prompts, sp, use_tqdm=False)
        else:
            if prompts is None:
                raise RuntimeError("generate_digits requires text prompts when prompt_token_ids are unsupported")
            with self._vllm_lock:
                outs = self._llm.generate(list(prompts), sp, use_tqdm=False)
        return [list(getattr(o.outputs[0], "token_ids", []) or []) for o in outs]
