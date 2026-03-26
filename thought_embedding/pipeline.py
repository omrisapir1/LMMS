from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from thought_embedding.config import ThoughtEmbeddingConfig, validate_config
from thought_embedding.io_utils import (
    ensure_output_dir,
    list_existing_shards,
    load_manifest,
    save_manifest,
    write_parquet_shard,
)

logger = logging.getLogger(__name__)


@dataclass
class PreparedExample:
    key: str
    row_index: int
    input_ids: list[int]
    thought_token_end_positions: list[int]
    problem: str
    thoughts: list[str]
    answer: Any = None
    expected_answer: Any = None
    source_id: Optional[str] = None
    source_qid: Optional[str] = None
    solution: Optional[str] = None


@dataclass
class BatchState:
    examples: list[PreparedExample]
    total_tokens: int


class PipelineError(RuntimeError):
    pass


def run_pipeline(
    cfg: ThoughtEmbeddingConfig,
    *,
    dataset: Optional[Dataset] = None,
    tokenizer: Any = None,
    model: Any = None,
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

    local_tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    local_model = model or _load_model(cfg)

    sep_token_id = _separator_token_id(local_tokenizer, cfg.separator_text)

    output_rows: list[dict[str, Any]] = []
    batch = BatchState(examples=[], total_tokens=0)

    processed_rows = int(manifest.get("processed_rows", 0))
    written_rows = int(manifest.get("written_rows", 0))
    next_shard_idx = int(manifest.get("next_shard_idx", 0))
    skipped_rows = 0
    failed_rows = 0

    for row_idx, row in enumerate(dataset):
        if cfg.max_samples > 0 and row_idx >= cfg.max_samples:
            break

        processed_rows += 1
        try:
            key = _row_key(row, row_idx, cfg)
            if cfg.resume and key in completed_keys:
                continue

            prepared = _prepare_example_row(cfg, local_tokenizer, row, row_idx, key, sep_token_id)
            if prepared is None:
                skipped_rows += 1
                continue

            # Primary limit: total tokens in a model forward. Secondary: number of examples.
            if _would_overflow_batch(cfg, batch, prepared):
                _flush_batch(cfg, local_model, batch, output_rows, completed_keys)
                if len(output_rows) >= cfg.shard_size:
                    write_parquet_shard(output_rows, output_dir, next_shard_idx)
                    next_shard_idx += 1
                    written_rows += len(output_rows)
                    output_rows.clear()

            batch.examples.append(prepared)
            batch.total_tokens += len(prepared.input_ids)

            if _is_batch_full(cfg, batch):
                _flush_batch(cfg, local_model, batch, output_rows, completed_keys)

            if len(output_rows) >= cfg.shard_size:
                write_parquet_shard(output_rows, output_dir, next_shard_idx)
                next_shard_idx += 1
                written_rows += len(output_rows)
                output_rows.clear()

            if processed_rows % cfg.log_every_n_examples == 0:
                logger.info(
                    "Processed %s rows | completed=%s | skipped=%s | pending_batch_examples=%s",
                    processed_rows,
                    len(completed_keys),
                    skipped_rows,
                    len(batch.examples),
                )

            if (written_rows + len(output_rows)) % cfg.save_every_n_examples == 0:
                save_manifest(
                    output_dir,
                    {
                        "completed_keys": sorted(completed_keys),
                        "processed_rows": processed_rows,
                        "written_rows": written_rows,
                        "next_shard_idx": next_shard_idx,
                    },
                )

        except Exception as exc:
            logger.exception("Row %s failed: %s", row_idx, exc)
            failed_rows += 1
            continue

    if batch.examples:
        _flush_batch(cfg, local_model, batch, output_rows, completed_keys)

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
    required = {cfg.input_problem_field, cfg.input_thoughts_field}
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


def _torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def _load_model(cfg: ThoughtEmbeddingConfig) -> Any:
    if not torch.cuda.is_available():
        raise PipelineError("CUDA is required for this phase. No CPU fallback is implemented.")

    kwargs = {
        "torch_dtype": _torch_dtype(cfg.dtype),
        "trust_remote_code": True,
    }

    try:
        model = AutoModel.from_pretrained(cfg.model_name, **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)

    model = model.to("cuda")
    model.eval()
    return model


def _separator_token_id(tokenizer: Any, separator_text: str) -> int:
    ids = tokenizer.encode(separator_text, add_special_tokens=False)
    if len(ids) != 1:
        raise PipelineError(
            f"separator_text={separator_text!r} must tokenize to exactly one token, got {len(ids)}."
        )
    return int(ids[0])


def _prepare_example_row(
    cfg: ThoughtEmbeddingConfig,
    tokenizer: Any,
    row: dict[str, Any],
    row_idx: int,
    key: str,
    sep_token_id: int,
) -> Optional[PreparedExample]:
    problem = row[cfg.input_problem_field]
    if not isinstance(problem, str) or not problem.strip():
        logger.warning("Skipping row %s: problem is missing or empty", row_idx)
        return None

    raw_thoughts = row[cfg.input_thoughts_field]
    if not isinstance(raw_thoughts, list):
        logger.warning("Skipping row %s: %s must be a list", row_idx, cfg.input_thoughts_field)
        return None

    thoughts: list[str] = []
    if any(not isinstance(t, str) for t in raw_thoughts):
        logger.warning("Skipping row %s: %s must contain only strings", row_idx, cfg.input_thoughts_field)
        return None

    for t in raw_thoughts:
        cleaned = t.strip()
        if cleaned or not cfg.drop_empty_thoughts:
            thoughts.append(cleaned)

    if cfg.drop_empty_thoughts:
        thoughts = [t for t in thoughts if t]

    if len(thoughts) < cfg.min_thoughts:
        logger.warning(
            "Skipping row %s: only %s thoughts after filtering (min=%s)",
            row_idx,
            len(thoughts),
            cfg.min_thoughts,
        )
        return None

    user_prompt = cfg.user_prompt_template.format(problem=problem)
    assistant_content, sep_positions_in_assistant = _build_assistant_content_and_positions(
        tokenizer,
        thoughts,
        cfg.separator_text,
    )

    final_ids, sep_positions_final = _build_chat_ids_and_positions(
        tokenizer,
        user_prompt,
        assistant_content,
        sep_positions_in_assistant,
    )

    for pos in sep_positions_final:
        if pos < 0 or pos >= len(final_ids):
            raise PipelineError(f"Invalid separator token position {pos} for sequence length {len(final_ids)}")
        if int(final_ids[pos]) != sep_token_id:
            raise PipelineError(
                "Tracked separator position does not point to separator token after chat template. "
                f"pos={pos}, token_id={final_ids[pos]}, expected={sep_token_id}."
            )

    if len(final_ids) > cfg.max_model_len:
        logger.warning(
            "Skipping row %s: input length %s exceeds max_model_len=%s",
            row_idx,
            len(final_ids),
            cfg.max_model_len,
        )
        return None

    answer = row.get(cfg.input_answer_field) if cfg.input_answer_field else None
    expected_answer = row.get(cfg.input_expected_answer_field) if cfg.input_expected_answer_field else None
    source_id = str(row[cfg.input_id_field]) if cfg.input_id_field and row.get(cfg.input_id_field) is not None else None
    source_qid = str(row[cfg.input_qid_field]) if cfg.input_qid_field and row.get(cfg.input_qid_field) is not None else None

    solution = None
    if cfg.keep_solution and cfg.input_solution_field and row.get(cfg.input_solution_field) is not None:
        solution = str(row[cfg.input_solution_field])

    return PreparedExample(
        key=key,
        row_index=row_idx,
        input_ids=[int(x) for x in final_ids],
        thought_token_end_positions=sep_positions_final,
        problem=problem,
        thoughts=thoughts,
        answer=answer,
        expected_answer=expected_answer,
        source_id=source_id,
        source_qid=source_qid,
        solution=solution,
    )


def _build_assistant_content_and_positions(
    tokenizer: Any,
    thoughts: list[str],
    separator: str,
) -> tuple[str, list[int]]:
    text = ""
    token_len = 0
    sep_positions: list[int] = []

    # Must start exactly with separator.
    text, token_len = _append_segment(tokenizer, text, token_len, separator)

    for thought in thoughts:
        text, token_len = _append_segment(tokenizer, text, token_len, thought)
        text, token_len = _append_segment(tokenizer, text, token_len, separator)
        sep_positions.append(token_len - 1)

    return text, sep_positions


def _append_segment(tokenizer: Any, current_text: str, current_len: int, segment: str) -> tuple[str, int]:
    new_text = current_text + segment
    new_len = len(tokenizer.encode(new_text, add_special_tokens=False))
    if new_len < current_len:
        raise PipelineError("Token length decreased while constructing assistant content.")
    return new_text, new_len


def _build_chat_ids_and_positions(
    tokenizer: Any,
    user_prompt: str,
    assistant_content: str,
    sep_positions_in_assistant: list[int],
) -> tuple[list[int], list[int]]:
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]
    try:
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as exc:
        raise PipelineError("Failed to apply chat template for this tokenizer.") from exc

    assistant_start_char = chat_text.rfind(assistant_content)
    if assistant_start_char < 0:
        raise PipelineError("Failed to locate assistant content inside chat template output.")

    prefix_text = chat_text[:assistant_start_char]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    full_ids = tokenizer.encode(chat_text, add_special_tokens=False)

    assistant_start_token_idx = len(prefix_ids)
    sep_positions_final = [assistant_start_token_idx + p for p in sep_positions_in_assistant]

    return [int(x) for x in full_ids], sep_positions_final


def _would_overflow_batch(cfg: ThoughtEmbeddingConfig, batch: BatchState, example: PreparedExample) -> bool:
    if not batch.examples:
        return False

    if len(batch.examples) >= cfg.max_examples_per_batch:
        return True

    return (batch.total_tokens + len(example.input_ids)) > cfg.max_tokens_per_batch


def _is_batch_full(cfg: ThoughtEmbeddingConfig, batch: BatchState) -> bool:
    if not batch.examples:
        return False
    if len(batch.examples) >= cfg.max_examples_per_batch:
        return True
    return batch.total_tokens >= cfg.max_tokens_per_batch


def _flush_batch(
    cfg: ThoughtEmbeddingConfig,
    model: Any,
    batch: BatchState,
    output_rows: list[dict[str, Any]],
    completed_keys: set[str],
) -> None:
    if not batch.examples:
        return

    pad_token_id = _pad_token_id(model, batch.examples)

    max_len = max(len(x.input_ids) for x in batch.examples)
    bsz = len(batch.examples)

    device = _model_device(model)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

    for i, ex in enumerate(batch.examples):
        seq = torch.tensor(ex.input_ids, dtype=torch.long, device=device)
        input_ids[i, : len(ex.input_ids)] = seq
        attention_mask[i, : len(ex.input_ids)] = 1

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden = outputs.hidden_states[-1]

    for i, ex in enumerate(batch.examples):
        vectors: list[list[float]] = []
        for pos in ex.thought_token_end_positions:
            vec = hidden[i, pos, :].detach().to(dtype=torch.float32).cpu().tolist()
            vectors.append(_cast_vector_dtype(vec, cfg.save_float_dtype))

        if len(vectors) != len(ex.thoughts):
            raise PipelineError(
                f"Row {ex.row_index} vector count mismatch: {len(vectors)} != {len(ex.thoughts)}"
            )

        dim = len(vectors[0])
        out = {
            "problem": ex.problem,
            "thoughts": ex.thoughts,
            "num_thoughts": len(ex.thoughts),
            "state_vectors": vectors,
            "embedding_dim": dim,
            "model_name": cfg.model_name,
            "thought_token_end_positions": ex.thought_token_end_positions,
            "input_token_count": len(ex.input_ids),
            "was_truncated": False,
        }
        if ex.answer is not None:
            out["answer"] = ex.answer
        if ex.expected_answer is not None:
            out["expected_answer"] = ex.expected_answer
        if ex.source_id is not None:
            out["id"] = ex.source_id
        if ex.source_qid is not None:
            out["qid"] = ex.source_qid
        if cfg.keep_solution and ex.solution is not None:
            out["solution"] = ex.solution

        output_rows.append(out)
        completed_keys.add(ex.key)

    batch.examples.clear()
    batch.total_tokens = 0


def _pad_token_id(model: Any, examples: list[PreparedExample]) -> int:
    model_cfg = getattr(model, "config", None)
    if model_cfg is not None and getattr(model_cfg, "pad_token_id", None) is not None:
        return int(model_cfg.pad_token_id)
    if model_cfg is not None and getattr(model_cfg, "eos_token_id", None) is not None:
        return int(model_cfg.eos_token_id)

    return int(examples[0].input_ids[0])


def _model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for p in model.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cast_vector_dtype(vector: list[float], dtype: str) -> list[float]:
    if dtype == "float32":
        return [float(v) for v in vector]

    try:
        import numpy as np

        return np.asarray(vector, dtype=np.float16).astype(float).tolist()
    except Exception:
        return [float(v) for v in vector]
