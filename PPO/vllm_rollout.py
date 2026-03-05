from __future__ import annotations

import gc
import importlib
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
        # NEW: vLLM worker-native weight transfer state.
        self._wt_inited = False
        self._wt_obj = None
        self._wt_obj_label = None
        self._wt_update_method = None
        self._wt_debug_logged = False

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
        self._debug_list_weight_sync_apis()
        self._maybe_init_weight_transfer()

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

    def _iter_weight_sync_objects(self):
        if self._llm is None:
            return
        engine = getattr(self._llm, "llm_engine", None)
        if engine is None:
            return
        seen = set()

        def _add(label: str, obj):
            if obj is None:
                return
            oid = id(obj)
            if oid in seen:
                return
            seen.add(oid)
            yield (label, obj)

        roots = [("llm_engine", engine)]
        for attr in ("model_executor", "executor", "core_client"):
            roots.append((f"llm_engine.{attr}", getattr(engine, attr, None)))

        queue = []
        for label, obj in roots:
            for item in _add(label, obj):
                queue.append(item)
                yield item

        for label, obj in list(queue):
            for child_attr in (
                "model_executor",
                "executor",
                "core_client",
                "engine_core",
                "model_runner",
                "worker",
                "workers",
            ):
                child = getattr(obj, child_attr, None)
                for item in _add(f"{label}.{child_attr}", child):
                    yield item

    def _debug_list_weight_sync_apis(self) -> None:
        if self._wt_debug_logged:
            return
        self._wt_debug_logged = True
        matches = ("weight", "transfer", "update", "reload", "sync")
        for label, obj in self._iter_weight_sync_objects() or []:
            try:
                attrs = [
                    name
                    for name in dir(obj)
                    if any(tok in name.lower() for tok in matches)
                ]
            except Exception:
                continue
            if attrs:
                attrs_str = ", ".join(sorted(attrs))
                self._log(f"vLLM weight-sync APIs on {label} ({obj.__class__.__name__}): {attrs_str}")

    def _maybe_init_weight_transfer(self) -> None:
        self._wt_inited = False
        self._wt_obj = None
        self._wt_obj_label = None
        self._wt_update_method = None

        candidates = list(self._iter_weight_sync_objects() or [])
        if not candidates:
            self._log("vLLM weight transfer unavailable: llm_engine not found")
            return

        for label, obj in candidates:
            init_fn = getattr(obj, "init_weight_transfer_engine", None)
            update_name = None
            if callable(getattr(obj, "update_weights", None)):
                update_name = "update_weights"
            elif callable(getattr(obj, "reload_weights", None)):
                update_name = "reload_weights"
            if update_name is None:
                continue

            if callable(init_fn):
                init_ok = False
                last_error = None
                for args, kwargs in (
                    (tuple(), {}),
                    (tuple(), {"model": self._model_ref}),
                    (tuple(), {"model_path": self._model_ref}),
                    (tuple(), {"checkpoint_path": self._model_ref}),
                ):
                    try:
                        init_fn(*args, **kwargs)
                        init_ok = True
                        self._log(f"vLLM weight transfer initialized via {label}.init_weight_transfer_engine")
                        break
                    except TypeError as exc:
                        last_error = f"{type(exc).__name__}: {exc}; args={args}, kwargs={kwargs}"
                        continue
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}; args={args}, kwargs={kwargs}"
                        continue
                if not init_ok:
                    self._log(
                        f"vLLM init_weight_transfer_engine failed on {label}; "
                        f"last_error={last_error}"
                    )
                    continue
            else:
                self._log(f"vLLM weight transfer uses {label}.{update_name} without explicit init")

            self._wt_inited = True
            self._wt_obj = obj
            self._wt_obj_label = label
            self._wt_update_method = update_name
            self._log(f"vLLM weight transfer ready: {label}.{update_name}")
            return

        self._log("vLLM weight transfer unavailable: no update_weights/reload_weights target found")

    def _build_weight_update_payload(self, model):
        helper_modules = (
            "vllm.examples.offline_inference.rlhf_utils",
            "vllm.examples.rlhf_utils",
            "examples.offline_inference.rlhf_utils",
        )
        helper_function_names = (
            "build_weight_update_payload",
            "prepare_weight_update_payload",
            "prepare_weight_update",
            "pack_weights_for_update",
            "get_weight_update_payload",
        )
        for mod_name in helper_modules:
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            for fn_name in helper_function_names:
                fn = getattr(mod, fn_name, None)
                if not callable(fn):
                    continue
                for arg in (model, model.state_dict()):
                    try:
                        payload = fn(arg)
                        self._log(f"vLLM weight payload prepared via {mod_name}.{fn_name}")
                        return payload
                    except Exception:
                        continue

        # Fallback: explicit named tensor payload for update_weights-style APIs.
        return [(k, v.detach().cpu()) for k, v in model.state_dict().items()]

    # NEW: in-place hot-swap from live torch model using vLLM 0.16 worker-native APIs.
    def _hot_swap_from_torch_model(self, model) -> bool:
        with self._vllm_lock:
            if self._llm is None or not self._wt_inited or self._wt_obj is None:
                return False
            update_name = self._wt_update_method
            if not update_name:
                return False
            fn = getattr(self._wt_obj, update_name, None)
            if not callable(fn):
                self._log(f"vLLM weight transfer object lost callable method: {self._wt_obj_label}.{update_name}")
                return False

            payload = self._build_weight_update_payload(model)
            attempts = (
                ((payload,), {}),
                (tuple(), {"weights": payload}),
                (tuple(), {"named_tensors": payload}),
                (tuple(), {"state_dict": model.state_dict()}),
                (tuple(), {"model": model}),
            )

            last_error = None
            for args, kwargs in attempts:
                try:
                    fn(*args, **kwargs)
                    self._log(
                        "vLLM hot swap succeeded via "
                        f"{self._wt_obj.__class__.__name__}.{update_name}"
                    )
                    return True
                except TypeError as exc:
                    last_error = f"{type(exc).__name__}: {exc}; args={args}, kwargs={kwargs}"
                    continue
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}; args={args}, kwargs={kwargs}"
                    continue
            self._log(
                "vLLM hot swap failed via "
                f"{self._wt_obj_label}.{update_name}; last_error={last_error}"
            )
        return False

    def maybe_sync_from_torch(self, model, tokenizer, update_idx: int) -> bool:
        should_sync = int(update_idx) == 1 or (int(update_idx) % self.sync_every == 0)
        if not should_sync:
            return False

        # Preferred path: in-place worker-native weight update (no engine rebuild).
        if self._hot_swap_from_torch_model(model):
            return True

        # Fallback path: checkpoint export + full rebuild.
        self._log("vLLM checkpoint export started")
        ckpt_dir = self._atomic_export_checkpoint(model, tokenizer, update_idx=int(update_idx))
        self._log("vLLM checkpoint ready")

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
