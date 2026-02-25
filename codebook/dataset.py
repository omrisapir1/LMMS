from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch


@dataclass
class LatentRow:
    qid: str
    question: str
    answer_int: int
    answer_digits: List[int]
    k_star: int
    k_max: int
    latent_vectors: np.ndarray  # [K_star, dim], float32


@dataclass
class SequenceBatch:
    latents: torch.Tensor  # [N, dim], float32
    sequence_count: int
    vector_count: int


def list_parquet_shards(input_dir: str) -> List[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    shards = sorted(root.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in: {input_dir}")
    return shards


def iter_parquet_rows(
    shard_path: Path,
    *,
    read_batch_size: int = 256,
) -> Iterator[Dict]:
    parquet = pq.ParquetFile(str(shard_path))
    for record_batch in parquet.iter_batches(batch_size=read_batch_size):
        cols = record_batch.to_pydict()
        for i in range(record_batch.num_rows):
            yield {k: cols[k][i] for k in cols}


def _coerce_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_digits(value: object) -> List[int]:
    if value is None:
        return []
    if not isinstance(value, Sequence):
        return []
    out: List[int] = []
    for x in value:
        try:
            out.append(int(x))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return []
    return out


def validate_latent_vectors(
    latent_vectors: object,
    *,
    k_star: int,
    dim: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if latent_vectors is None:
        return None, "latent_vectors is null"
    if k_star < 0:
        return None, "K_star is negative"
    if not isinstance(latent_vectors, Sequence):
        return None, "latent_vectors is not a sequence"
    if len(latent_vectors) != k_star:
        return None, f"len(latent_vectors)={len(latent_vectors)} != K_star={k_star}"
    if k_star == 0:
        return np.zeros((0, dim), dtype=np.float32), None

    try:
        arr = np.asarray(latent_vectors, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - defensive parsing
        return None, f"failed to convert latent_vectors to float32: {exc}"

    if arr.ndim != 2:
        return None, f"latent_vectors rank={arr.ndim}, expected 2"
    if arr.shape[0] != k_star:
        return None, f"latent_vectors first dim={arr.shape[0]} != K_star={k_star}"
    if arr.shape[1] != dim:
        return None, f"latent_vectors dim={arr.shape[1]} != expected dim={dim}"
    return arr, None


def parse_latent_row(row: Dict, *, dim: int) -> Tuple[Optional[LatentRow], Optional[str]]:
    k_star = _coerce_int(row.get("K_star"), default=-1)
    latents, err = validate_latent_vectors(row.get("latent_vectors"), k_star=k_star, dim=dim)
    if err is not None or latents is None:
        return None, err or "invalid latent_vectors"

    parsed = LatentRow(
        qid=str(row.get("qid", "")),
        question=str(row.get("question", "")),
        answer_int=_coerce_int(row.get("answer_int"), default=0),
        answer_digits=_coerce_digits(row.get("answer_digits")),
        k_star=k_star,
        k_max=_coerce_int(row.get("k_max"), default=0),
        latent_vectors=latents,
    )
    return parsed, None


def iter_valid_latent_rows(
    input_dir: str,
    *,
    dim: int,
    read_batch_size: int = 256,
) -> Iterator[LatentRow]:
    for shard in list_parquet_shards(input_dir):
        for row in iter_parquet_rows(shard, read_batch_size=read_batch_size):
            parsed, _ = parse_latent_row(row, dim=dim)
            if parsed is None:
                continue
            yield parsed


def iter_sequence_batches(
    input_dir: str,
    *,
    batch_size: int,
    dim: int,
    read_batch_size: int = 256,
) -> Iterator[SequenceBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    seq_latents: List[np.ndarray] = []
    for row in iter_valid_latent_rows(input_dir, dim=dim, read_batch_size=read_batch_size):
        seq_latents.append(row.latent_vectors)
        if len(seq_latents) < batch_size:
            continue
        yield _pack_batch(seq_latents, dim=dim)
        seq_latents.clear()

    if seq_latents:
        yield _pack_batch(seq_latents, dim=dim)


def _pack_batch(seq_latents: List[np.ndarray], *, dim: int) -> SequenceBatch:
    if not seq_latents:
        empty = torch.zeros((0, dim), dtype=torch.float32)
        return SequenceBatch(latents=empty, sequence_count=0, vector_count=0)

    flat = np.concatenate(seq_latents, axis=0)
    if flat.ndim != 2 or flat.shape[1] != dim:
        raise RuntimeError(f"Invalid packed latent shape: {tuple(flat.shape)}")

    latents = torch.from_numpy(flat.astype(np.float32, copy=False))
    return SequenceBatch(
        latents=latents,
        sequence_count=len(seq_latents),
        vector_count=int(flat.shape[0]),
    )
