from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, load_dataset

from thought_embedding.config import ThoughtEmbeddingConfig, validate_config
from thought_embedding.encoder import Embedder, VLLMEmbedder
from thought_embedding.io_utils import (
    ensure_output_dir,
    list_existing_shards,
    load_manifest,
    save_manifest,
    write_parquet_shard,
)
from thought_embedding.prompts import PromptBuildResult, PromptError, build_state_prompt_for_thought
from thought_embedding.split_logic import split_thoughts

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    row_key: str
    thought_idx: int
    prompt: str


@dataclass
class RowBuffer:
    key: str
    row_index: int
    question: str
    answer: Any
    thoughts: list[str]
    state_vectors: list[Optional[list[float]]]
    was_truncated: bool = False
    num_previous_thoughts_kept: list[int] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    source_id: Optional[str] = None
    solution: Optional[str] = None


def run_pipeline(
    cfg: ThoughtEmbeddingConfig,
    *,
    dataset: Optional[Dataset] = None,
    embedder: Optional[Embedder] = None,
) -> dict[str, Any]:
    validate_config(cfg)
    _configure_logging()

    output_dir = ensure_output_dir(cfg.output_dir)
    _prepare_output_dir(cfg, output_dir)

    manifest = load_manifest(output_dir)
    completed_keys = set(manifest.get("completed_keys", []))

    if dataset is None:
        dataset = load_dataset(cfg.dataset_name, split=cfg.train_split)

    _validate_dataset_columns(dataset, cfg)

    local_embedder = embedder or VLLMEmbedder(cfg)

    pending: list[PendingRequest] = []
    row_buffers: dict[str, RowBuffer] = {}
    output_rows: list[dict[str, Any]] = []

    processed_rows = int(manifest.get("processed_rows", 0))
    written_rows = int(manifest.get("written_rows", 0))
    next_shard_idx = int(manifest.get("next_shard_idx", 0))
    skipped_rows = 0
    failed_rows = 0

    for row_idx, row in enumerate(dataset):
        processed_rows += 1
        try:
            key = _row_key(row, row_idx, cfg)
            if cfg.resume and key in completed_keys:
                continue

            question = row[cfg.input_question_field]
            solution = row[cfg.input_solution_field]
            if not isinstance(question, str) or not question.strip():
                logger.warning("Skipping row %s: question is missing or empty", row_idx)
                skipped_rows += 1
                continue
            if not isinstance(solution, str) or not solution.strip():
                logger.warning("Skipping row %s: solution is missing or empty", row_idx)
                skipped_rows += 1
                continue

            thoughts = split_thoughts(solution)
            thoughts = [t.strip() for t in thoughts]
            if cfg.drop_empty_thoughts:
                thoughts = [t for t in thoughts if t]

            if cfg.max_thoughts_per_example is not None and len(thoughts) > cfg.max_thoughts_per_example:
                thoughts = thoughts[: cfg.max_thoughts_per_example]

            if len(thoughts) < cfg.min_thoughts:
                logger.warning(
                    "Skipping row %s: only %s thoughts after splitting (min=%s)",
                    row_idx,
                    len(thoughts),
                    cfg.min_thoughts,
                )
                skipped_rows += 1
                continue

            answer = row.get(cfg.input_answer_field) if cfg.input_answer_field else None
            source_id = None
            if cfg.input_id_field and row.get(cfg.input_id_field) is not None:
                source_id = str(row[cfg.input_id_field])
            elif cfg.input_qid_field and row.get(cfg.input_qid_field) is not None:
                source_id = str(row[cfg.input_qid_field])

            buffer = RowBuffer(
                key=key,
                row_index=row_idx,
                question=question,
                answer=answer,
                thoughts=thoughts,
                state_vectors=[None] * len(thoughts),
                source_id=source_id,
                solution=solution if cfg.keep_solution else None,
            )
            row_buffers[key] = buffer

            row_failed = False
            for thought_idx in range(len(thoughts)):
                try:
                    prompt_result = build_state_prompt_for_thought(
                        cfg,
                        question,
                        thoughts,
                        thought_idx,
                        local_embedder.tokenizer,
                    )
                except PromptError as exc:
                    if cfg.skip_overlong_examples:
                        logger.warning(
                            "Skipping row %s due to prompt construction error at thought %s: %s",
                            row_idx,
                            thought_idx,
                            exc,
                        )
                        row_failed = True
                        break
                    raise

                buffer.was_truncated = buffer.was_truncated or prompt_result.was_truncated
                buffer.num_previous_thoughts_kept.append(prompt_result.num_previous_thoughts_kept)
                buffer.token_counts.append(prompt_result.token_count)
                pending.append(
                    PendingRequest(
                        row_key=key,
                        thought_idx=thought_idx,
                        prompt=prompt_result.text,
                    )
                )

            if row_failed:
                row_buffers.pop(key, None)
                skipped_rows += 1
                continue

            while len(pending) >= cfg.batch_size:
                _flush_pending_batch(
                    cfg,
                    pending,
                    row_buffers,
                    output_rows,
                    completed_keys,
                    local_embedder,
                )

            if processed_rows % cfg.log_every_n_examples == 0:
                logger.info(
                    "Processed %s rows | completed=%s | skipped=%s | pending=%s",
                    processed_rows,
                    len(completed_keys),
                    skipped_rows,
                    len(pending),
                )

            if len(output_rows) >= cfg.shard_size:
                write_parquet_shard(output_rows, output_dir, next_shard_idx)
                next_shard_idx += 1
                written_rows += len(output_rows)
                output_rows.clear()

            if (written_rows + len(output_rows)) % cfg.save_every_n_examples == 0:
                manifest = {
                    "completed_keys": sorted(completed_keys),
                    "processed_rows": processed_rows,
                    "written_rows": written_rows,
                    "next_shard_idx": next_shard_idx,
                }
                save_manifest(output_dir, manifest)

        except Exception as exc:
            logger.exception("Row %s failed: %s", row_idx, exc)
            failed_rows += 1
            continue

    while pending:
        _flush_pending_batch(
            cfg,
            pending,
            row_buffers,
            output_rows,
            completed_keys,
            local_embedder,
        )

    if output_rows:
        write_parquet_shard(output_rows, output_dir, next_shard_idx)
        next_shard_idx += 1
        written_rows += len(output_rows)
        output_rows.clear()

    manifest = {
        "completed_keys": sorted(completed_keys),
        "processed_rows": processed_rows,
        "written_rows": written_rows,
        "next_shard_idx": next_shard_idx,
    }
    save_manifest(output_dir, manifest)

    summary = {
        "output_dir": str(output_dir),
        "num_shards": len(list_existing_shards(output_dir)),
        "processed_rows": processed_rows,
        "written_rows": written_rows,
        "skipped_rows": skipped_rows,
        "failed_rows": failed_rows,
        "manifest": manifest,
    }
    logger.info("Run complete: %s", summary)
    return summary


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def _prepare_output_dir(cfg: ThoughtEmbeddingConfig, output_dir: Path) -> None:
    if cfg.overwrite:
        for shard in output_dir.glob("part-*.parquet"):
            shard.unlink()
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()


def _validate_dataset_columns(dataset: Dataset, cfg: ThoughtEmbeddingConfig) -> None:
    columns = set(dataset.column_names)
    required = {cfg.input_question_field, cfg.input_solution_field}
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns for this config: "
            f"{missing}. Available columns: {sorted(columns)}"
        )


def _row_key(row: dict[str, Any], row_idx: int, cfg: ThoughtEmbeddingConfig) -> str:
    if cfg.input_id_field and row.get(cfg.input_id_field) is not None:
        return f"id:{row[cfg.input_id_field]}"
    if cfg.input_qid_field and row.get(cfg.input_qid_field) is not None:
        return f"qid:{row[cfg.input_qid_field]}"
    return f"row:{row_idx}"


def _flush_pending_batch(
    cfg: ThoughtEmbeddingConfig,
    pending: list[PendingRequest],
    row_buffers: dict[str, RowBuffer],
    output_rows: list[dict[str, Any]],
    completed_keys: set[str],
    embedder: Embedder,
) -> None:
    batch = pending[: cfg.batch_size]
    del pending[: cfg.batch_size]

    prompts = [b.prompt for b in batch]
    vectors = embedder.embed_texts(prompts)
    if len(vectors) != len(batch):
        raise ValueError(f"Expected {len(batch)} vectors but got {len(vectors)}.")

    for req, vector in zip(batch, vectors):
        row_buf = row_buffers.get(req.row_key)
        if row_buf is None:
            continue
        row_buf.state_vectors[req.thought_idx] = _cast_vector_dtype(vector, cfg.save_float_dtype)

        if all(v is not None for v in row_buf.state_vectors):
            output_row = _finalize_row(cfg, row_buf)
            output_rows.append(output_row)
            completed_keys.add(row_buf.key)
            row_buffers.pop(row_buf.key, None)


def _cast_vector_dtype(vector: list[float], dtype: str) -> list[float]:
    if dtype == "float32":
        return [float(v) for v in vector]

    # float16 path
    try:
        import numpy as np

        return np.asarray(vector, dtype=np.float16).astype(float).tolist()
    except Exception:
        # Fallback when numpy is unavailable.
        return [float(v) for v in vector]


def _finalize_row(cfg: ThoughtEmbeddingConfig, row_buf: RowBuffer) -> dict[str, Any]:
    state_vectors: list[list[float]] = [v for v in row_buf.state_vectors if v is not None]

    if not row_buf.question:
        raise ValueError(f"Row {row_buf.row_index} has empty question")
    if not row_buf.thoughts:
        raise ValueError(f"Row {row_buf.row_index} has no thoughts")
    if len(state_vectors) != len(row_buf.thoughts):
        raise ValueError(
            f"Row {row_buf.row_index} vector count mismatch: "
            f"{len(state_vectors)} != {len(row_buf.thoughts)}"
        )

    dim = len(state_vectors[0])
    if any(len(v) != dim for v in state_vectors):
        raise ValueError(f"Row {row_buf.row_index} has inconsistent embedding dimensions")

    out = {
        "id": row_buf.source_id,
        "question": row_buf.question,
        "answer": row_buf.answer,
        "thoughts": row_buf.thoughts,
        "num_thoughts": len(row_buf.thoughts),
        "state_vectors": state_vectors,
        "embedding_dim": dim,
        "model_name": cfg.model_name,
        "prompt_version": cfg.prompt_version,
        "was_truncated": row_buf.was_truncated,
        "num_previous_thoughts_kept": row_buf.num_previous_thoughts_kept,
        "prompt_token_counts": row_buf.token_counts,
    }
    if cfg.keep_solution:
        out["solution"] = row_buf.solution
    return out
