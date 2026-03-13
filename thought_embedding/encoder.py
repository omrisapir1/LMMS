from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

from transformers import AutoTokenizer

from thought_embedding.config import ThoughtEmbeddingConfig


class EmbedderError(RuntimeError):
    pass


class Embedder(ABC):
    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class VLLMEmbedder(Embedder):
    def __init__(self, cfg: ThoughtEmbeddingConfig) -> None:
        try:
            from vllm import LLM
        except Exception as exc:  # pragma: no cover - depends on environment
            raise EmbedderError(
                "Failed to import vLLM. Install a compatible vLLM version to use backend='vllm'."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        try:
            self._llm = LLM(
                model=cfg.model_name,
                task="embed",
                dtype=cfg.dtype,
                max_model_len=cfg.max_model_len,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                max_num_seqs=cfg.max_num_seqs,
            )
        except TypeError:
            # Some vLLM versions do not expose task/max_num_seqs in constructor.
            self._llm = LLM(
                model=cfg.model_name,
                dtype=cfg.dtype,
                max_model_len=cfg.max_model_len,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
            )

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        outputs = None
        if hasattr(self._llm, "embed"):
            outputs = self._llm.embed(texts)
        elif hasattr(self._llm, "encode"):
            outputs = self._llm.encode(texts)
        else:  # pragma: no cover - defensive
            raise EmbedderError("This vLLM version does not expose embed/encode APIs.")

        vectors = [self._extract_embedding(o) for o in outputs]
        if len(vectors) != len(texts):
            raise EmbedderError(
                f"Embedding count mismatch: got {len(vectors)} vectors for {len(texts)} texts."
            )
        return vectors

    def _extract_embedding(self, output: Any) -> list[float]:
        # Common object-style shapes.
        for attr in ("embedding",):
            if hasattr(output, attr):
                emb = getattr(output, attr)
                if emb is not None:
                    return _to_float_list(emb)

        # Shapes like output.outputs[0].embedding or output.outputs.embedding
        if hasattr(output, "outputs"):
            outs = getattr(output, "outputs")
            if isinstance(outs, Sequence) and outs:
                first = outs[0]
                if hasattr(first, "embedding"):
                    return _to_float_list(first.embedding)
                if isinstance(first, dict) and "embedding" in first:
                    return _to_float_list(first["embedding"])
            elif hasattr(outs, "embedding"):
                return _to_float_list(outs.embedding)

        # Dict-style fallback.
        if isinstance(output, dict):
            if "embedding" in output:
                return _to_float_list(output["embedding"])
            if "data" in output and output["data"]:
                first = output["data"][0]
                if isinstance(first, dict) and "embedding" in first:
                    return _to_float_list(first["embedding"])

        raise EmbedderError(f"Unable to extract embedding vector from output of type {type(output)!r}.")


def _to_float_list(x: Any) -> list[float]:
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return [float(v) for v in x]
    raise EmbedderError(f"Embedding is not iterable: {type(x)!r}")
