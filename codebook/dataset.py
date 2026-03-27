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
    max_k_in_batch: int
    avg_k_in_batch: float
    sequence_lengths: List[int]


def list_parquet_shards(input_dir: str) -> List[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    shards = sorted(root.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in: {input_dir}")
    return shards


def _is_local_parquet_dir(input_ref: str) -> bool:
    return Path(input_ref).exists() and Path(input_ref).is_dir()


def _parse_hf_dataset_ref(input_ref: str) -> Tuple[str, str]:
    # Supports "repo_id" (defaults to train split) or "repo_id:split".
    if ":" in input_ref:
        dataset_id, split = input_ref.rsplit(":", 1)
        dataset_id = dataset_id.strip()
        split = split.strip()
        if dataset_id and split:
            return dataset_id, split
    return input_ref.strip(), "train"


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


def iter_hf_rows(
    input_ref: str,
    *,
    read_batch_size: int = 256,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
) -> Iterator[Dict]:
    del read_batch_size  # Streaming rows are handled by datasets' internal iterators.
    if shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size must be > 0")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "The 'datasets' package is required for Hugging Face dataset input. "
            "Install it (e.g. pip install datasets)."
        ) from exc

    dataset_id, split = _parse_hf_dataset_ref(input_ref)
    if not dataset_id:
        raise FileNotFoundError(f"Input reference is empty: {input_ref!r}")

    try:
        ds = load_dataset(dataset_id, split=split, streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        raise FileNotFoundError(
            f"Input path does not exist locally and could not be loaded from Hugging Face: "
            f"ref={input_ref!r}, dataset_id={dataset_id!r}, split={split!r}"
        ) from exc

    for row in ds:
        if isinstance(row, dict):
            yield row
        else:  # pragma: no cover - defensive
            yield dict(row)


def _coerce_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _first_present(row: Dict, keys: Sequence[str]) -> object:
    for key in keys:
        if key in row:
            return row[key]
    return None


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
    if not isinstance(latent_vectors, Sequence) and not isinstance(latent_vectors, np.ndarray):
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
    raw_latents = _first_present(row, ("latent_vectors", "state_vectors"))
    raw_k_star = _first_present(row, ("K_star", "k_star", "num_thoughts"))
    k_star = _coerce_int(raw_k_star, default=-1)
    if k_star < 0 and raw_latents is not None:
        try:
            k_star = int(len(raw_latents))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            k_star = -1

    latents, err = validate_latent_vectors(raw_latents, k_star=k_star, dim=dim)
    if err is not None or latents is None:
        return None, err or "invalid latent_vectors"

    raw_qid = _first_present(row, ("qid", "id"))
    raw_question = _first_present(row, ("question", "problem"))
    raw_k_max = _first_present(row, ("k_max", "K_max", "num_thoughts", "K_star", "k_star"))
    raw_answer_int = _first_present(row, ("expected_answer", "answer_int", "answer"))

    parsed = LatentRow(
        qid=str(raw_qid if raw_qid is not None else ""),
        question=str(raw_question if raw_question is not None else ""),
        answer_int=_coerce_int(raw_answer_int, default=0),
        answer_digits=_coerce_digits(row.get("answer_digits")),
        k_star=k_star,
        k_max=_coerce_int(raw_k_max, default=k_star),
        latent_vectors=latents,
    )
    return parsed, None


def iter_valid_latent_rows(
    input_dir: str,
    *,
    dim: int,
    read_batch_size: int = 256,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    delete_input_files: bool = False,
) -> Iterator[LatentRow]:
    valid_count = 0
    invalid_count = 0
    first_error: Optional[str] = None

    def _on_invalid(err: Optional[str]) -> None:
        nonlocal invalid_count, first_error
        invalid_count += 1
        if first_error is None and err:
            first_error = err

    if _is_local_parquet_dir(input_dir):
        for shard in list_parquet_shards(input_dir):
            for row in iter_parquet_rows(shard, read_batch_size=read_batch_size):
                parsed, err = parse_latent_row(row, dim=dim)
                if parsed is None:
                    _on_invalid(err)
                    continue
                valid_count += 1
                yield parsed
            if delete_input_files:
                try:
                    shard.unlink()
                except FileNotFoundError:
                    pass
        if valid_count == 0:
            raise RuntimeError(
                f"No valid latent rows found in local parquet dir={input_dir!r} for dim={dim}. "
                f"invalid_rows={invalid_count}, first_error={first_error!r}. "
                "Expected row keys include one of {latent_vectors,state_vectors} and one of {K_star,k_star,num_thoughts}."
            )
        return

    for row in iter_hf_rows(
        input_dir,
        read_batch_size=read_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    ):
        parsed, err = parse_latent_row(row, dim=dim)
        if parsed is None:
            _on_invalid(err)
            continue
        valid_count += 1
        yield parsed
    if valid_count == 0:
        raise RuntimeError(
            f"No valid latent rows found in HF input={input_dir!r} for dim={dim}. "
            f"invalid_rows={invalid_count}, first_error={first_error!r}. "
            "Expected row keys include one of {latent_vectors,state_vectors} and one of {K_star,k_star,num_thoughts}."
        )


def iter_sequence_batches(
    input_dir: str,
    *,
    max_vectors_per_batch: int,
    dim: int,
    read_batch_size: int = 256,
    max_sequences_per_batch: int | None = None,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    delete_input_files: bool = False,
) -> Iterator[SequenceBatch]:
    if max_vectors_per_batch <= 0:
        raise ValueError("max_vectors_per_batch must be > 0")
    if max_sequences_per_batch is not None and max_sequences_per_batch <= 0:
        raise ValueError("max_sequences_per_batch must be > 0 when provided")

    seq_latents: List[np.ndarray] = []
    seq_lengths: List[int] = []
    vector_count = 0
    for row in iter_valid_latent_rows(
        input_dir,
        dim=dim,
        read_batch_size=read_batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        delete_input_files=delete_input_files,
    ):
        row_latents = row.latent_vectors
        row_vectors = int(row_latents.shape[0])

        if seq_latents:
            would_exceed_vectors = vector_count + row_vectors > max_vectors_per_batch
            reached_sequence_cap = (
                max_sequences_per_batch is not None and len(seq_latents) >= max_sequences_per_batch
            )
            if would_exceed_vectors or reached_sequence_cap:
                yield _pack_batch(seq_latents, seq_lengths, dim=dim)
                seq_latents.clear()
                seq_lengths.clear()
                vector_count = 0

        seq_latents.append(row_latents)
        seq_lengths.append(row_vectors)
        vector_count += row_vectors

    if seq_latents:
        yield _pack_batch(seq_latents, seq_lengths, dim=dim)


def _pack_batch(seq_latents: List[np.ndarray], seq_lengths: List[int], *, dim: int) -> SequenceBatch:
    if not seq_latents:
        empty = torch.zeros((0, dim), dtype=torch.float32)
        return SequenceBatch(
            latents=empty,
            sequence_count=0,
            vector_count=0,
            max_k_in_batch=0,
            avg_k_in_batch=0.0,
            sequence_lengths=[],
        )

    flat = np.concatenate(seq_latents, axis=0)
    if flat.ndim != 2 or flat.shape[1] != dim:
        raise RuntimeError(f"Invalid packed latent shape: {tuple(flat.shape)}")

    vector_count = int(flat.shape[0])
    sequence_count = len(seq_latents)
    max_k_in_batch = max(seq_lengths) if seq_lengths else 0
    avg_k_in_batch = float(vector_count / sequence_count) if sequence_count > 0 else 0.0
    latents = torch.from_numpy(flat.astype(np.float32, copy=False))
    return SequenceBatch(
        latents=latents,
        sequence_count=sequence_count,
        vector_count=vector_count,
        max_k_in_batch=max_k_in_batch,
        avg_k_in_batch=avg_k_in_batch,
        sequence_lengths=list(seq_lengths),
    )
