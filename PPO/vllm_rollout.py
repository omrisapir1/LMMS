from __future__ import annotations

import threading
import socket
import inspect
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        self.output_dir = output_dir
        self.tmp_ckpt_dir = tmp_ckpt_dir
        self.sync_every = max(1, int(sync_every))
        self.seed = int(seed)
        self._log = logger
        self._model_ref = str(init_ckpt)

        self._lock = threading.RLock()
        self._llm = None

        # Weight transfer runtime state.
        self._wt_ready = False
        self._wt_group = None
        self._wt_master_address = None
        self._wt_master_port = None
        self._wt_world_size = None
        self._wt_rank_offset = 1
        self._update_fail_streak = 0
        self._debug_api_logged = False

        self._llm_cls = None
        self._sampling_params_cls = None
        self._weight_transfer_init_request_cls = None
        self._weight_transfer_update_request_cls = None
        self._nccl_engine_cls = None
        self._nccl_trainer_send_weights_args_cls = None
        self._nccl_weight_transfer_init_info_cls = None
        self._nccl_weight_transfer_update_info_cls = None

        self._engine_kwargs = dict(engine_kwargs)
        # Weight-transfer safety knobs.
        self._wt_packed = bool(self._engine_kwargs.pop("weight_transfer_packed", False))
        self._wt_transfer_device = str(self._engine_kwargs.pop("weight_transfer_device", "cuda:0"))
        self._wt_rank_offset = int(self._engine_kwargs.pop("weight_transfer_rank_offset", 1))
        self._cuda_visible_devices = self._engine_kwargs.pop("cuda_visible_devices", None)
        self._init_engine(self._model_ref)

    def _init_engine(self, init_ckpt: str) -> None:
        try:
            from vllm import LLM, SamplingParams
            from vllm.config import WeightTransferConfig
            from vllm.distributed.weight_transfer.base import (
                WeightTransferInitRequest,
                WeightTransferUpdateRequest,
            )
            from vllm.distributed.weight_transfer import nccl_engine as _nccl_engine_mod
        except Exception as exc:
            raise RuntimeError(
                "vLLM rollout is enabled but vLLM weight-sync APIs are unavailable"
            ) from exc

        self._llm_cls = LLM
        self._sampling_params_cls = SamplingParams
        self._weight_transfer_init_request_cls = WeightTransferInitRequest
        self._weight_transfer_update_request_cls = WeightTransferUpdateRequest
        self._nccl_engine_cls = getattr(_nccl_engine_mod, "NCCLWeightTransferEngine", None)
        self._nccl_weight_transfer_init_info_cls = getattr(_nccl_engine_mod, "NCCLWeightTransferInitInfo", None)
        self._nccl_weight_transfer_update_info_cls = getattr(_nccl_engine_mod, "NCCLWeightTransferUpdateInfo", None)
        self._nccl_trainer_send_weights_args_cls = getattr(_nccl_engine_mod, "NCCLTrainerSendWeightsArgs", None)
        if self._nccl_engine_cls is None:
            raise RuntimeError("vLLM NCCLWeightTransferEngine is not available")
        if self._nccl_weight_transfer_init_info_cls is None:
            raise RuntimeError("vLLM NCCLWeightTransferInitInfo is not available")
        if self._nccl_weight_transfer_update_info_cls is None:
            raise RuntimeError("vLLM NCCLWeightTransferUpdateInfo is not available")
        if self._nccl_trainer_send_weights_args_cls:
            self._log("vLLM weight sync: using legacy trainer_send_weights signature (no NCCLTrainerSendWeightsArgs)")
        else:
            self._log("vLLM weight sync: using NCCLTrainerSendWeightsArgs signature")

        kwargs = dict(self._engine_kwargs)
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("trust_remote_code", self.trust_remote_code)
        kwargs.setdefault("distributed_executor_backend", "ray")
        kwargs.setdefault("load_format", "dummy")
        kwargs.setdefault("weight_transfer_config", WeightTransferConfig(backend="nccl"))
        if self._cuda_visible_devices is not None and str(self._cuda_visible_devices).strip():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._cuda_visible_devices)
            self._log(f"vLLM init with CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

        try:
            self._llm = LLM(model=init_ckpt, tokenizer=init_ckpt, **kwargs)
        except Exception as first_exc:
            try:
                import ray

                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self._llm = LLM(model=init_ckpt, tokenizer=init_ckpt, **kwargs)
            except Exception as second_exc:
                raise RuntimeError(
                    "Failed to initialize vLLM LLM with Ray executor backend"
                ) from second_exc

        self._log("vLLM engine initialized")
        self._log_weight_sync_apis_once()
        self._init_weight_transfer_once()

    def _destroy_engine(self) -> None:
        self._wt_ready = False
        self._wt_group = None
        self._wt_master_address = None
        self._wt_master_port = None
        self._wt_world_size = None
        if self._llm is not None:
            try:
                del self._llm
            finally:
                self._llm = None

    def close(self) -> None:
        with self._lock:
            self._destroy_engine()

    def _log_weight_sync_apis_once(self) -> None:
        if self._debug_api_logged or self._llm is None:
            return
        self._debug_api_logged = True

        engine = getattr(self._llm, "llm_engine", None)
        if engine is None:
            self._log("vLLM llm_engine not found for weight-sync API debug")
            return

        def _log_obj_attrs(label: str, obj: Any) -> None:
            if obj is None:
                return
            attrs = dir(obj)
            filtered = [
                x
                for x in attrs
                if (
                    "weight" in x.lower()
                    or "transfer" in x.lower()
                    or "update" in x.lower()
                    or "sync" in x.lower()
                    or "rpc" in x.lower()
                    or "client" in x.lower()
                )
            ]
            self._log(f"vLLM API debug {label}: {sorted(filtered)}")

        _log_obj_attrs("llm", self._llm)
        _log_obj_attrs("llm_engine", engine)
        _log_obj_attrs("llm_engine.model_executor", getattr(engine, "model_executor", None))
        _log_obj_attrs("llm_engine.executor", getattr(engine, "executor", None))
        _log_obj_attrs("llm_engine.engine_core_client", getattr(engine, "engine_core_client", None))

    def _resolve_world_size(self) -> int:
        if self._llm is None:
            raise RuntimeError("vLLM engine is not initialized")

        # Explicit API path in vLLM LLM class.
        if hasattr(self._llm, "get_world_size") and callable(getattr(self._llm, "get_world_size")):
            val = int(self._llm.get_world_size())
            if val > 0:
                return val

        # Fallback explicit path through collective RPC.
        engine = getattr(self._llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "collective_rpc") and callable(getattr(engine, "collective_rpc")):
            vals = engine.collective_rpc("get_world_size")
            if isinstance(vals, list) and len(vals) > 0:
                val = int(vals[0])
                if val > 0:
                    return val

        raise RuntimeError("Unable to resolve vLLM worker world size")

    def _init_weight_transfer_once(self) -> None:
        if self._wt_ready:
            return
        if self._llm is None:
            raise RuntimeError("vLLM engine is not initialized")

        if not hasattr(self._llm, "init_weight_transfer_engine"):
            raise RuntimeError("vLLM LLM.init_weight_transfer_engine is not available")

        self._wt_master_address = str(self._resolve_master_ip())
        self._wt_master_port = int(self._resolve_open_port())
        worker_world_size = int(self._resolve_world_size())
        self._wt_world_size = int(worker_world_size + self._wt_rank_offset)

        trainer_init_info = self._build_nccl_init_info(
            master_address=self._wt_master_address,
            master_port=self._wt_master_port,
            world_size=self._wt_world_size,
            rank_offset=0,
        )

        worker_init_info = self._build_nccl_init_info(
            master_address=self._wt_master_address,
            master_port=self._wt_master_port,
            rank_offset=self._wt_rank_offset,
            world_size=self._wt_world_size,
        )
        trainer_exc: List[BaseException] = []

        def _trainer_init_call() -> None:
            try:
                self._wt_group = self._nccl_engine_cls.trainer_init(asdict(trainer_init_info))
            except BaseException as exc:
                trainer_exc.append(exc)

        # Some vLLM versions block trainer_init until workers join.
        trainer_thread = threading.Thread(target=_trainer_init_call, daemon=True)
        trainer_thread.start()
        init_req = self._weight_transfer_init_request_cls(init_info=asdict(worker_init_info))
        self._llm.init_weight_transfer_engine(init_req)
        trainer_thread.join()
        if len(trainer_exc) > 0:
            raise RuntimeError(
                f"vLLM trainer_init failed during weight transfer setup: "
                f"{type(trainer_exc[0]).__name__}: {trainer_exc[0]}"
            ) from trainer_exc[0]
        if self._wt_group is None:
            raise RuntimeError("vLLM trainer_init did not produce a valid NCCL group")

        self._wt_ready = True
        self._log("weight transfer initialized")

    def _build_nccl_init_info(
        self,
        *,
        master_address: str,
        master_port: int,
        world_size: int,
        rank_offset: int,
    ):
        cls = self._nccl_weight_transfer_init_info_cls
        if cls is None:
            raise RuntimeError("vLLM NCCLWeightTransferInitInfo is not available")
        params = inspect.signature(cls).parameters
        kwargs: Dict[str, Any] = {
            "master_address": str(master_address),
            "master_port": int(master_port),
            "world_size": int(world_size),
        }
        if "rank_offset" in params:
            kwargs["rank_offset"] = int(rank_offset)
        return cls(**kwargs)

    @staticmethod
    def _resolve_master_ip() -> str:
        # vLLM moved network helpers across versions; use whichever exists first.
        try:
            from vllm.utils import get_ip as _get_ip  # type: ignore
            return str(_get_ip())
        except Exception:
            pass
        try:
            from vllm.utils.network import get_ip as _get_ip  # type: ignore
            return str(_get_ip())
        except Exception:
            pass

        # Last-resort fallback when vLLM helper APIs are unavailable.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            return str(ip) if ip else "127.0.0.1"
        except Exception:
            return "127.0.0.1"
        finally:
            sock.close()

    @staticmethod
    def _resolve_open_port() -> int:
        try:
            from vllm.utils import get_open_port as _get_open_port  # type: ignore
            return int(_get_open_port())
        except Exception:
            pass
        try:
            from vllm.utils.network import get_open_port as _get_open_port  # type: ignore
            return int(_get_open_port())
        except Exception:
            pass

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])
        finally:
            sock.close()

    def _build_sampling_params(
        self,
        *,
        allowed_token_ids: Sequence[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        greedy: bool,
        n: int = 1,
        min_tokens: Optional[int] = None,
        stop_on_answer: bool = False,
    ):
        if self._sampling_params_cls is None:
            raise RuntimeError("vLLM SamplingParams class is not initialized")

        kwargs: Dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "temperature": 0.0 if greedy else float(temperature),
            "top_p": 1.0 if greedy else float(top_p),
            "min_p": 0.0 if greedy else float(min_p),
            "repetition_penalty": float(repetition_penalty),
            "n": int(n),
        }
        if min_tokens is not None:
            kwargs["min_tokens"] = int(min_tokens)
        if stop_on_answer:
            kwargs["stop_token_ids"] = [int(self.answer_token_id)]

        try:
            return self._sampling_params_cls(
                **kwargs,
                allowed_token_ids=[int(x) for x in allowed_token_ids],
            )
        except TypeError:
            return self._sampling_params_cls(
                **kwargs,
                allowed_token_ids_list=[[int(x) for x in allowed_token_ids]],
            )

    def supports_prompt_token_ids(self) -> bool:
        return True

    def _snapshot_named_weights(self, model) -> List[Tuple[str, torch.Tensor]]:
        sd_cpu = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        return list(sd_cpu.items())

    @staticmethod
    def _canonical_dtype_name(dtype: torch.dtype) -> str:
        mapping = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
            torch.float64: "float64",
            torch.int64: "int64",
            torch.int32: "int32",
            torch.int16: "int16",
            torch.int8: "int8",
            torch.uint8: "uint8",
            torch.bool: "bool",
        }
        if dtype not in mapping:
            raise RuntimeError(f"Unsupported tensor dtype for weight sync: {dtype}")
        return mapping[dtype]

    def _update_weights_in_place_once(self, named_weights: List[Tuple[str, torch.Tensor]]) -> bool:
        if self._llm is None:
            return False
        if not self._wt_ready:
            self._init_weight_transfer_once()

        packed = bool(self._wt_packed)
        staged_named: List[Tuple[str, torch.Tensor]] = []
        for name, tensor in named_weights:
            try:
                staged = tensor.to(device=self._wt_transfer_device, non_blocking=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed staging tensor {name!r} to {self._wt_transfer_device}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            staged_named.append((name, staged))

        total_bytes = int(sum(t.numel() * t.element_size() for _, t in staged_named))
        first_device = str(staged_named[0][1].device) if len(staged_named) > 0 else "n/a"
        # self._log(
        #     "vLLM weight sync send: "
        #     f"num_tensors={len(staged_named)} total_bytes={total_bytes} "
        #     f"first_device={first_device} packed={packed}"
        # )

        update_info = self._nccl_weight_transfer_update_info_cls(
            names=[name for name, _ in staged_named],
            dtype_names=[self._canonical_dtype_name(t.dtype) for _, t in staged_named],
            shapes=[tuple(int(x) for x in t.shape) for _, t in staged_named],
            packed=packed,
        )
        update_req = self._weight_transfer_update_request_cls(update_info=asdict(update_info))

        update_exc: List[BaseException] = []

        def _driver_update_call() -> None:
            try:
                self._llm.update_weights(update_req)
            except BaseException as exc:  # capture and re-raise in caller thread
                update_exc.append(exc)

        worker_thread = threading.Thread(target=_driver_update_call, daemon=True)
        worker_thread.start()

        if self._nccl_trainer_send_weights_args_cls is not None:
            send_args = self._nccl_trainer_send_weights_args_cls(group=self._wt_group, packed=packed)
            self._nccl_engine_cls.trainer_send_weights(staged_named, trainer_args=send_args)
        else:
            sent = False
            last_exc: Optional[Exception] = None
            for args, kwargs in (
                ((staged_named,), {"group": self._wt_group, "packed": packed}),
                ((staged_named, self._wt_group), {"packed": packed}),
                ((staged_named, self._wt_group, packed), {}),
                (tuple(), {"weights": staged_named, "group": self._wt_group, "packed": packed}),
            ):
                try:
                    self._nccl_engine_cls.trainer_send_weights(*args, **kwargs)
                    sent = True
                    break
                except Exception as exc:
                    last_exc = exc
                    continue
            if not sent:
                raise RuntimeError(
                    "vLLM trainer_send_weights call failed for all known signatures"
                ) from last_exc

        worker_thread.join()
        if len(update_exc) > 0:
            raise update_exc[0]

        return True

    def _rebuild_engine_once(self) -> None:
        self._log("vLLM full rebuild fallback triggered")
        self._destroy_engine()
        self._init_engine(str(self._model_ref))

    def maybe_sync_from_torch(self, model, tokenizer, update_idx: int) -> bool:
        del tokenizer
        should_sync = int(update_idx) == 1 or (int(update_idx) % self.sync_every == 0)
        if not should_sync:
            return False

        named = self._snapshot_named_weights(model)
        with self._lock:
            try:
                self._update_weights_in_place_once(named)
                self._update_fail_streak = 0
                # self._log("vLLM weights updated in-place via LLM.update_weights")
                return True
            except Exception as exc1:
                self._update_fail_streak += 1
                self._log(f"vLLM in-place update failed: {type(exc1).__name__}: {exc1}")

            try:
                self._update_weights_in_place_once(named)
                self._update_fail_streak = 0
                # self._log("vLLM weights updated in-place via LLM.update_weights")
                return True
            except Exception as exc2:
                self._update_fail_streak += 1
                self._log(f"vLLM in-place update retry failed: {type(exc2).__name__}: {exc2}")

            if self._update_fail_streak >= 2:
                self._rebuild_engine_once()
                self._update_weights_in_place_once(named)
                self._update_fail_streak = 0
                self._log("vLLM weights updated in-place after rebuild fallback")
                return True

        return False

    def smoke_test_weight_sync(self, model) -> None:
        named = self._snapshot_named_weights(model)
        try:
            with self._lock:
                self._update_weights_in_place_once(named)
                if self._llm is None:
                    raise RuntimeError("vLLM engine is not initialized during smoke test")
                sp = self._sampling_params_cls(
                    max_tokens=4,
                    temperature=0.0,
                    top_p=1.0,
                    n=1,
                )
                outs = self._llm.generate(["Compute 1+1."], sp, use_tqdm=False)
                _ = outs[0].outputs[0]
        except Exception as exc:
            raise RuntimeError(f"vLLM smoke_test_weight_sync failed: {type(exc).__name__}: {exc}") from exc

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
        if self._llm is None:
            raise RuntimeError("vLLM engine is not initialized")
        n = max(1, int(num_samples_per_prompt))
        sp = self._build_sampling_params(
            allowed_token_ids=self.z_allowed_token_ids,
            max_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            min_p=float(min_p),
            repetition_penalty=float(repetition_penalty),
            greedy=bool(greedy),
            n=n,
            stop_on_answer=True,
        )

        with self._lock:
            if prompt_token_ids is not None and self.supports_prompt_token_ids():
                token_prompts = [{"prompt_token_ids": list(map(int, x))} for x in prompt_token_ids]
                outs = self._llm.generate(token_prompts, sp, use_tqdm=False)
            else:
                if prompts is None:
                    raise RuntimeError("generate_z requires text prompts when prompt_token_ids are unsupported")
                outs = self._llm.generate(list(prompts), sp, use_tqdm=False)

        rows: List[Dict[str, object]] = []
        for o in outs:
            for out_j in list(getattr(o, "outputs", []) or []):
                rows.append(
                    {
                        "token_ids": list(getattr(out_j, "token_ids", []) or []),
                        "stop_reason": getattr(out_j, "stop_reason", None),
                        "finish_reason": getattr(out_j, "finish_reason", None),
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
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> List[List[int]]:
        if self._llm is None:
            raise RuntimeError("vLLM engine is not initialized")
        sp = self._build_sampling_params(
            allowed_token_ids=self.digit_allowed_token_ids,
            max_tokens=5,
            min_tokens=5,
            temperature=float(temperature),
            top_p=float(top_p),
            min_p=float(min_p),
            repetition_penalty=float(repetition_penalty),
            greedy=bool(greedy),
        )

        with self._lock:
            if prompt_token_ids is not None and self.supports_prompt_token_ids():
                token_prompts = [{"prompt_token_ids": list(map(int, x))} for x in prompt_token_ids]
                outs = self._llm.generate(token_prompts, sp, use_tqdm=False)
            else:
                if prompts is None:
                    raise RuntimeError("generate_digits requires text prompts when prompt_token_ids are unsupported")
                outs = self._llm.generate(list(prompts), sp, use_tqdm=False)

        return [list(getattr(o.outputs[0], "token_ids", []) or []) for o in outs]
