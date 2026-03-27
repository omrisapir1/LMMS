from __future__ import annotations

import logging
from collections import Counter
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

_MAP_TOKENIZER_CACHE: dict[str, Any] = {}
_PRINTED_SEPARATOR_DEBUG = False
_PRINTED_NONSTANDALONE_NOTICE = False
_CANONICAL_SEPARATOR_MARKER_TOKEN_ID = 271


@dataclass
class PreTokenizedExample:
    key: str
    row_index: int
    input_ids: list[int]
    input_token_count: int
    thought_separator_positions: list[int]
    problem: str
    thoughts: list[str]
    answer: Any = None
    expected_answer: Any = None
    source: Any = None
    source_id: Optional[str] = None
    source_qid: Optional[str] = None
    solution: Optional[str] = None


@dataclass
class BatchState:
    examples: list[PreTokenizedExample]
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
    _print_startup_config_checks(cfg)

    output_dir = ensure_output_dir(cfg.output_dir)
    _prepare_output_dir(cfg, output_dir)

    manifest = load_manifest(output_dir)
    completed_keys = set(manifest.get("completed_keys", []))

    if dataset is None:
        dataset = load_dataset(cfg.dataset_name, split=cfg.train_split)
    loaded_rows = len(dataset)

    rows_after_source_filter = loaded_rows
    if cfg.source_filter:
        if not cfg.input_source_field:
            raise PipelineError("source_filter is set but input_source_field is not configured.")
        if cfg.input_source_field not in set(dataset.column_names):
            raise PipelineError(
                f"source_filter is set but '{cfg.input_source_field}' column is missing from dataset."
            )
        wanted_source = str(cfg.source_filter)
        dataset = dataset.filter(
            lambda x: str(x.get(cfg.input_source_field)) == wanted_source,
            desc=f"Filter source={wanted_source}",
        )
        rows_after_source_filter = len(dataset)

    dataset = dataset.shuffle(seed=cfg.seed)
    rows_after_shuffle = len(dataset)

    if cfg.max_samples > 0:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))
    rows_after_sampling = len(dataset)

    run_rows = rows_after_sampling
    processed_rows = int(manifest.get("processed_rows", 0)) + run_rows
    written_rows = int(manifest.get("written_rows", 0))
    next_shard_idx = int(manifest.get("next_shard_idx", 0))
    failed_rows = 0
    completed_since_manifest = 0

    local_tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    local_model = model or _load_model(cfg)

    sep_token_id = _separator_token_id(local_tokenizer, cfg.separator_text)
    print(f"separator_token_ids={local_tokenizer.encode(cfg.separator_text, add_special_tokens=False)}")

    pretokenized_ds = _pretokenize_dataset(
        cfg,
        dataset,
        tokenizer=local_tokenizer,
        separator_token_id=sep_token_id,
    )
    pretokenized_rows = len(pretokenized_ds)
    resume_removed_count = 0

    if cfg.resume and completed_keys:
        before_resume = len(pretokenized_ds)
        keep_indices = [
            i
            for i, row_key in enumerate(pretokenized_ds["__row_key"])
            if row_key not in completed_keys
        ]
        pretokenized_ds = pretokenized_ds.select(keep_indices)
        resume_removed_count = before_resume - len(pretokenized_ds)
    after_resume_rows = len(pretokenized_ds)

    print(f"raw_dataset_len={loaded_rows}")
    print(f"len_after_source_filter={rows_after_source_filter}")
    print(f"len_after_resume_filter={after_resume_rows}")
    print(f"len_pretokenized_ds={pretokenized_rows}")

    invalid_count = sum(1 for v in pretokenized_ds["is_valid"] if not v)
    overlong_count = sum(1 for v in pretokenized_ds["is_overlong"] if v)
    valid_count = after_resume_rows - invalid_count
    print(f"invalid_count={invalid_count}")
    print(f"overlong_count={overlong_count}")
    print(f"Counter(is_valid)={Counter(pretokenized_ds['is_valid'])}")
    print(f"Counter(is_overlong)={Counter(pretokenized_ds['is_overlong'])}")
    _print_filter_reasons(pretokenized_ds)

    ready_indices = [
        i
        for i, (is_valid, is_overlong) in enumerate(
            zip(pretokenized_ds["is_valid"], pretokenized_ds["is_overlong"])
        )
        if bool(is_valid) and (not bool(is_overlong))
    ]
    filtered_ds = pretokenized_ds.select(ready_indices)
    print(f"len_filtered_ds={len(filtered_ds)}")

    logger.info(
        "Preprocess stats | loaded=%s | after_source_filter=%s | after_shuffle=%s | "
        "after_sampling=%s | pretokenized=%s | resume_removed=%s | after_resume=%s | "
        "valid=%s | invalid=%s | overlong=%s | ready=%s",
        loaded_rows,
        rows_after_source_filter,
        rows_after_shuffle,
        run_rows,
        pretokenized_rows,
        resume_removed_count,
        after_resume_rows,
        valid_count,
        invalid_count,
        overlong_count,
        len(filtered_ds),
    )

    _print_startup_example(filtered_ds)

    if cfg.sort_by_length and len(filtered_ds) > 0:
        filtered_ds = filtered_ds.sort("input_token_count")

    pad_token_id = _pad_token_id(local_tokenizer, local_model)

    output_rows: list[dict[str, Any]] = []
    batch = BatchState(examples=[], total_tokens=0)

    for row in filtered_ds:
        try:
            ex = _row_to_pretokenized_example(row)

            if _would_overflow_batch(cfg, batch, ex):
                completed_now = _flush_batch(
                    cfg,
                    local_model,
                    pad_token_id,
                    batch,
                    output_rows,
                    completed_keys,
                )
                completed_since_manifest += completed_now

                if len(output_rows) >= cfg.shard_size:
                    rows_written, next_shard_idx = _write_shard(output_rows, output_dir, next_shard_idx)
                    written_rows += rows_written
                    _save_manifest(output_dir, completed_keys, processed_rows, written_rows, next_shard_idx)
                    completed_since_manifest = 0

            batch.examples.append(ex)
            batch.total_tokens += ex.input_token_count

            if _is_batch_full(cfg, batch):
                completed_now = _flush_batch(
                    cfg,
                    local_model,
                    pad_token_id,
                    batch,
                    output_rows,
                    completed_keys,
                )
                completed_since_manifest += completed_now

            if len(output_rows) >= cfg.shard_size:
                rows_written, next_shard_idx = _write_shard(output_rows, output_dir, next_shard_idx)
                written_rows += rows_written
                _save_manifest(output_dir, completed_keys, processed_rows, written_rows, next_shard_idx)
                completed_since_manifest = 0

            if len(completed_keys) % cfg.log_every_n_examples == 0 and len(completed_keys) > 0:
                logger.info(
                    "Completed=%s | skipped_invalid=%s | skipped_overlong=%s | pending_batch_examples=%s",
                    len(completed_keys),
                    invalid_count,
                    overlong_count,
                    len(batch.examples),
                )

            if completed_since_manifest >= cfg.save_every_n_examples:
                _save_manifest(output_dir, completed_keys, processed_rows, written_rows, next_shard_idx)
                completed_since_manifest = 0

        except Exception as exc:
            logger.exception("Pre-tokenized row failed: %s", exc)
            failed_rows += 1
            continue

    if batch.examples:
        completed_now = _flush_batch(
            cfg,
            local_model,
            pad_token_id,
            batch,
            output_rows,
            completed_keys,
        )
        completed_since_manifest += completed_now

    if output_rows:
        rows_written, next_shard_idx = _write_shard(output_rows, output_dir, next_shard_idx)
        written_rows += rows_written
        _save_manifest(output_dir, completed_keys, processed_rows, written_rows, next_shard_idx)
        completed_since_manifest = 0

    manifest = _save_manifest(output_dir, completed_keys, processed_rows, written_rows, next_shard_idx)

    summary = {
        "output_dir": str(output_dir),
        "num_shards": len(list_existing_shards(output_dir)),
        "processed_rows": processed_rows,
        "written_rows": written_rows,
        "skipped_rows": invalid_count + overlong_count + resume_removed_count,
        "invalid_rows": invalid_count,
        "overlong_rows": overlong_count,
        "resume_removed_rows": resume_removed_count,
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

    model_dtype = _torch_dtype(cfg.dtype)
    kwargs = {
        "dtype": model_dtype,
        "trust_remote_code": True,
    }

    try:
        model = AutoModel.from_pretrained(cfg.model_name, **kwargs)
    except TypeError:
        legacy_kwargs = {
            "torch_dtype": model_dtype,
            "trust_remote_code": True,
        }
        try:
            model = AutoModel.from_pretrained(cfg.model_name, **legacy_kwargs)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **legacy_kwargs)
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


def _pretokenize_dataset(
    cfg: ThoughtEmbeddingConfig,
    dataset: Dataset,
    *,
    tokenizer: Any,
    separator_token_id: int,
) -> Dataset:
    ds = dataset.map(
        _map_add_row_keys,
        with_indices=True,
        batched=True,
        batch_size=cfg.pretokenize_batch_size,
        load_from_cache_file=False,
        fn_kwargs={
            "input_id_field": cfg.input_id_field,
            "input_qid_field": cfg.input_qid_field,
        },
        desc="Attach row keys",
    )

    map_num_proc = cfg.pretokenize_num_proc
    if tokenizer is not None and map_num_proc > 1:
        # Injected tokenizers (e.g., tests) may not be serializable across worker processes.
        map_num_proc = 1

    if map_num_proc > 1:
        ds = ds.map(
            _map_pretokenize_parallel,
            batched=True,
            batch_size=cfg.pretokenize_batch_size,
            num_proc=map_num_proc,
            load_from_cache_file=False,
            fn_kwargs={
                "model_name": cfg.model_name,
                "input_problem_field": cfg.input_problem_field,
                "input_thoughts_field": cfg.input_thoughts_field,
                "input_answer_field": cfg.input_answer_field,
                "input_expected_answer_field": cfg.input_expected_answer_field,
                "input_solution_field": cfg.input_solution_field,
                "input_source_field": cfg.input_source_field,
                "input_id_field": cfg.input_id_field,
                "input_qid_field": cfg.input_qid_field,
                "drop_empty_thoughts": cfg.drop_empty_thoughts,
                "min_thoughts": cfg.min_thoughts,
                "user_prompt_template": cfg.user_prompt_template,
                "separator_text": cfg.separator_text,
                "separator_token_id": separator_token_id,
                "max_model_len": cfg.max_model_len,
                "keep_solution": cfg.keep_solution,
            },
            desc="Pre-tokenize/alignment",
        )
    else:
        ds = ds.map(
            _map_pretokenize_serial,
            batched=True,
            batch_size=cfg.pretokenize_batch_size,
            load_from_cache_file=False,
            fn_kwargs={
                "tokenizer": tokenizer,
                "input_problem_field": cfg.input_problem_field,
                "input_thoughts_field": cfg.input_thoughts_field,
                "input_answer_field": cfg.input_answer_field,
                "input_expected_answer_field": cfg.input_expected_answer_field,
                "input_solution_field": cfg.input_solution_field,
                "input_source_field": cfg.input_source_field,
                "input_id_field": cfg.input_id_field,
                "input_qid_field": cfg.input_qid_field,
                "drop_empty_thoughts": cfg.drop_empty_thoughts,
                "min_thoughts": cfg.min_thoughts,
                "user_prompt_template": cfg.user_prompt_template,
                "separator_text": cfg.separator_text,
                "separator_token_id": separator_token_id,
                "max_model_len": cfg.max_model_len,
                "keep_solution": cfg.keep_solution,
            },
            desc="Pre-tokenize/alignment",
        )

    return ds


def _map_add_row_keys(
    batch: dict[str, list[Any]],
    indices: list[int],
    *,
    input_id_field: Optional[str],
    input_qid_field: Optional[str],
) -> dict[str, list[Any]]:
    keys: list[str] = []
    for i, row_idx in enumerate(indices):
        source_id = _get_optional_value(batch, input_id_field, i)
        source_qid = _get_optional_value(batch, input_qid_field, i)
        if source_id is not None:
            keys.append(f"id:{source_id}")
        elif source_qid is not None:
            keys.append(f"qid:{source_qid}")
        else:
            keys.append(f"row:{row_idx}")

    return {
        "__row_key": keys,
        "__row_index": [int(i) for i in indices],
    }


def _map_pretokenize_parallel(
    batch: dict[str, list[Any]],
    *,
    model_name: str,
    input_problem_field: str,
    input_thoughts_field: str,
    input_answer_field: Optional[str],
    input_expected_answer_field: Optional[str],
    input_solution_field: Optional[str],
    input_source_field: Optional[str],
    input_id_field: Optional[str],
    input_qid_field: Optional[str],
    drop_empty_thoughts: bool,
    min_thoughts: int,
    user_prompt_template: str,
    separator_text: str,
    separator_token_id: int,
    max_model_len: int,
    keep_solution: bool,
) -> dict[str, list[Any]]:
    tokenizer = _get_map_tokenizer(model_name)
    return _map_pretokenize_impl(
        batch,
        tokenizer=tokenizer,
        input_problem_field=input_problem_field,
        input_thoughts_field=input_thoughts_field,
        input_answer_field=input_answer_field,
        input_expected_answer_field=input_expected_answer_field,
        input_solution_field=input_solution_field,
        input_source_field=input_source_field,
        input_id_field=input_id_field,
        input_qid_field=input_qid_field,
        drop_empty_thoughts=drop_empty_thoughts,
        min_thoughts=min_thoughts,
        user_prompt_template=user_prompt_template,
        separator_text=separator_text,
        separator_token_id=separator_token_id,
        max_model_len=max_model_len,
        keep_solution=keep_solution,
    )


def _map_pretokenize_serial(
    batch: dict[str, list[Any]],
    *,
    tokenizer: Any,
    input_problem_field: str,
    input_thoughts_field: str,
    input_answer_field: Optional[str],
    input_expected_answer_field: Optional[str],
    input_solution_field: Optional[str],
    input_source_field: Optional[str],
    input_id_field: Optional[str],
    input_qid_field: Optional[str],
    drop_empty_thoughts: bool,
    min_thoughts: int,
    user_prompt_template: str,
    separator_text: str,
    separator_token_id: int,
    max_model_len: int,
    keep_solution: bool,
) -> dict[str, list[Any]]:
    return _map_pretokenize_impl(
        batch,
        tokenizer=tokenizer,
        input_problem_field=input_problem_field,
        input_thoughts_field=input_thoughts_field,
        input_answer_field=input_answer_field,
        input_expected_answer_field=input_expected_answer_field,
        input_solution_field=input_solution_field,
        input_source_field=input_source_field,
        input_id_field=input_id_field,
        input_qid_field=input_qid_field,
        drop_empty_thoughts=drop_empty_thoughts,
        min_thoughts=min_thoughts,
        user_prompt_template=user_prompt_template,
        separator_text=separator_text,
        separator_token_id=separator_token_id,
        max_model_len=max_model_len,
        keep_solution=keep_solution,
    )


def _map_pretokenize_impl(
    batch: dict[str, list[Any]],
    *,
    tokenizer: Any,
    input_problem_field: str,
    input_thoughts_field: str,
    input_answer_field: Optional[str],
    input_expected_answer_field: Optional[str],
    input_solution_field: Optional[str],
    input_source_field: Optional[str],
    input_id_field: Optional[str],
    input_qid_field: Optional[str],
    drop_empty_thoughts: bool,
    min_thoughts: int,
    user_prompt_template: str,
    separator_text: str,
    separator_token_id: int,
    max_model_len: int,
    keep_solution: bool,
) -> dict[str, list[Any]]:
    batch_size = len(batch["__row_key"])

    out: dict[str, list[Any]] = {
        "problem": [],
        "thoughts": [],
        "num_thoughts": [],
        "full_text": [],
        "input_ids": [],
        "input_token_count": [],
        "thought_separator_positions": [],
        "is_overlong": [],
        "is_valid": [],
        "filter_reason": [],
        "answer": [],
        "expected_answer": [],
        "source": [],
        "id": [],
        "qid": [],
        "solution": [],
    }

    for i in range(batch_size):
        problem = _get_optional_value(batch, input_problem_field, i)
        thoughts_raw = _get_optional_value(batch, input_thoughts_field, i)

        answer = _get_optional_value(batch, input_answer_field, i)
        expected_answer = _get_optional_value(batch, input_expected_answer_field, i)
        source = _get_optional_value(batch, input_source_field, i)
        source_id = _get_optional_value(batch, input_id_field, i)
        source_qid = _get_optional_value(batch, input_qid_field, i)
        solution = _get_optional_value(batch, input_solution_field, i) if keep_solution else None

        out["answer"].append(answer)
        out["expected_answer"].append(expected_answer)
        out["source"].append(source)
        out["id"].append(None if source_id is None else str(source_id))
        out["qid"].append(None if source_qid is None else str(source_qid))
        out["solution"].append(None if solution is None else str(solution))

        if not isinstance(problem, str) or not problem.strip():
            _append_invalid_row(out, problem=None, reason="invalid_problem")
            continue
        if not isinstance(thoughts_raw, list):
            _append_invalid_row(out, problem=problem, reason="invalid_splitted_solution_not_list")
            continue
        if any(not isinstance(t, str) for t in thoughts_raw):
            _append_invalid_row(out, problem=problem, reason="invalid_splitted_solution_non_string_item")
            continue

        thoughts = [t.strip() for t in thoughts_raw]
        if drop_empty_thoughts:
            thoughts = [t for t in thoughts if t]

        if len(thoughts) < min_thoughts:
            _append_invalid_row(out, problem=problem, reason="invalid_min_thoughts_not_met")
            continue

        user_prompt = _render_user_prompt(user_prompt_template, problem)
        assistant_content, separator_positions_in_assistant, separator_char_spans_in_assistant = (
            _build_assistant_content_and_positions_with_spans(
                tokenizer,
                thoughts,
                separator_text,
            )
        )

        input_ids, separator_positions, alignment_debug = _build_chat_ids_and_positions(
            tokenizer,
            user_prompt,
            assistant_content,
            separator_positions_in_assistant,
            separator_char_spans_in_assistant,
            separator_text,
        )

        valid = True
        try:
            _assert_shifted_positions_in_bounds(
                input_ids=input_ids,
                shifted_positions=separator_positions,
            )
        except AssertionError:
            valid = False

        if len(separator_positions) != len(thoughts):
            valid = False

        if not valid:
            _debug_separator_alignment_failure(
                tokenizer=tokenizer,
                separator_text=separator_text,
                separator_token_id=separator_token_id,
                thoughts=thoughts,
                assistant_text=assistant_content,
                separator_positions_in_assistant=separator_positions_in_assistant,
                input_ids=input_ids,
                shifted_positions=separator_positions,
                alignment_debug=alignment_debug,
            )
            _append_invalid_row(out, problem=problem, reason="invalid_separator_alignment")
            continue

        exact_match_count = sum(1 for p in separator_positions if int(input_ids[p]) == int(separator_token_id))
        if exact_match_count != len(separator_positions):
            _debug_separator_nonstandalone_notice(
                separator_text=separator_text,
                separator_token_id=separator_token_id,
                exact_match_count=exact_match_count,
                expected_count=len(separator_positions),
            )

        input_ids = _overwrite_canonical_marker_positions(
            input_ids=input_ids,
            positions=separator_positions,
            marker_token_id=_CANONICAL_SEPARATOR_MARKER_TOKEN_ID,
        )

        out["problem"].append(problem)
        out["thoughts"].append(thoughts)
        out["num_thoughts"].append(len(thoughts))
        out["full_text"].append(_build_chat_text(tokenizer, user_prompt, assistant_content))
        out["input_ids"].append([int(x) for x in input_ids])
        out["input_token_count"].append(len(input_ids))
        out["thought_separator_positions"].append([int(x) for x in separator_positions])
        out["is_overlong"].append(len(input_ids) > max_model_len)
        out["is_valid"].append(True)
        out["filter_reason"].append("" if len(input_ids) <= max_model_len else "overlong_input")

    return out


def _append_invalid_row(out: dict[str, list[Any]], *, problem: Optional[str], reason: str) -> None:
    out["problem"].append(problem)
    out["thoughts"].append([])
    out["num_thoughts"].append(0)
    out["full_text"].append("")
    out["input_ids"].append([])
    out["input_token_count"].append(0)
    out["thought_separator_positions"].append([])
    out["is_overlong"].append(False)
    out["is_valid"].append(False)
    out["filter_reason"].append(reason)


def _get_map_tokenizer(model_name: str) -> Any:
    tok = _MAP_TOKENIZER_CACHE.get(model_name)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        _MAP_TOKENIZER_CACHE[model_name] = tok
    return tok


def _get_optional_value(batch: dict[str, list[Any]], field: Optional[str], idx: int) -> Any:
    if not field:
        return None
    values = batch.get(field)
    if values is None:
        return None
    return values[idx]


def _render_user_prompt(user_prompt_template: str, problem: str) -> str:
    try:
        return user_prompt_template.format(problem=problem)
    except Exception as exc:
        raise PipelineError(
            "Invalid user_prompt_template for str.format(...). "
            "If you need literal braces (e.g., '\\boxed{}'), escape them as '{{' and '}}'."
        ) from exc


def _row_to_pretokenized_example(row: dict[str, Any]) -> PreTokenizedExample:
    return PreTokenizedExample(
        key=row["__row_key"],
        row_index=int(row.get("__row_index", -1)),
        input_ids=[int(x) for x in row["input_ids"]],
        input_token_count=int(row["input_token_count"]),
        thought_separator_positions=[int(x) for x in row["thought_separator_positions"]],
        problem=row["problem"],
        thoughts=list(row["thoughts"]),
        answer=row.get("answer"),
        expected_answer=row.get("expected_answer"),
        source=row.get("source"),
        source_id=row.get("id"),
        source_qid=row.get("qid"),
        solution=row.get("solution"),
    )



def _build_assistant_content_and_positions(
    tokenizer: Any,
    thoughts: list[str],
    separator: str,
) -> tuple[str, list[int]]:
    text, positions, _ = _build_assistant_content_and_positions_with_spans(
        tokenizer=tokenizer,
        thoughts=thoughts,
        separator=separator,
    )
    return text, positions


def _build_assistant_content_and_positions_with_spans(
    tokenizer: Any,
    thoughts: list[str],
    separator: str,
) -> tuple[str, list[int], list[tuple[int, int]]]:
    text = ""
    separator_char_spans: list[tuple[int, int]] = []

    # Canonical format: separator is inserted before each thought, with no trailing separator.
    # This guarantees exactly one separator token per thought.
    for thought in thoughts:
        sep_start = len(text)
        text = text + separator
        sep_end = len(text)
        separator_char_spans.append((sep_start, sep_end))
        text = text + thought

    separator_positions = _resolve_separator_positions(
        tokenizer,
        text=text,
        separator=separator,
        separator_char_spans=separator_char_spans,
    )
    return text, separator_positions, separator_char_spans


def _resolve_separator_positions(
    tokenizer: Any,
    *,
    text: str,
    separator: str,
    separator_char_spans: list[tuple[int, int]],
) -> list[int]:
    # Fast-tokenizer path: resolve separator positions from final token offsets.
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        positions: list[int] = []
        for sep_start, sep_end in separator_char_spans:
            overlapping_positions = [
                i
                for i, (tok_start, tok_end) in enumerate(offsets)
                if tok_end > sep_start and tok_start < sep_end
            ]
            if not overlapping_positions:
                raise PipelineError(
                    f"Could not resolve separator token span ({sep_start}, {sep_end}) in assistant text."
                )
            # Canonical extraction index: last token overlapping the separator span.
            positions.append(int(overlapping_positions[-1]))
        return positions
    except Exception:
        # Fallback for tokenizers without offset mapping support.
        sep_ids = tokenizer.encode(separator, add_special_tokens=False)
        if len(sep_ids) != 1:
            raise PipelineError(
                f"separator_text={separator!r} must tokenize to exactly one token, got {len(sep_ids)}."
            )
        sep_id = int(sep_ids[0])
        ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        all_sep_positions = [i for i, tok_id in enumerate(ids) if tok_id == sep_id]
        if len(all_sep_positions) != len(separator_char_spans):
            raise PipelineError(
                "Unable to align separator positions in assistant text: "
                f"expected {len(separator_char_spans)} separators, found {len(all_sep_positions)}."
            )
        return all_sep_positions


def _overwrite_canonical_marker_positions(
    *,
    input_ids: list[int],
    positions: list[int],
    marker_token_id: int,
) -> list[int]:
    out = [int(x) for x in input_ids]
    for pos in positions:
        out[pos] = int(marker_token_id)
    return out


def _build_chat_ids_and_positions(
    tokenizer: Any,
    user_prompt: str,
    assistant_content: str,
    separator_positions_in_assistant: list[int],
    separator_char_spans_in_assistant: list[tuple[int, int]],
    separator_text: str,
) -> tuple[list[int], list[int], dict[str, Any]]:
    chat_text = _build_chat_text(tokenizer, user_prompt, assistant_content)

    assistant_start_char = chat_text.rfind(assistant_content)
    if assistant_start_char < 0:
        raise PipelineError("Failed to locate assistant content inside chat template output.")

    prefix_text = chat_text[:assistant_start_char]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    assistant_ids = tokenizer.encode(assistant_content, add_special_tokens=False)
    full_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    absolute_separator_spans = [
        (assistant_start_char + start, assistant_start_char + end)
        for start, end in separator_char_spans_in_assistant
    ]

    separator_positions_final = _resolve_separator_positions(
        tokenizer,
        text=chat_text,
        separator=separator_text,
        separator_char_spans=absolute_separator_spans,
    )

    debug = {
        "prefix_len": len(prefix_ids),
        "assistant_len": len(assistant_ids),
        "assistant_positions": [int(x) for x in separator_positions_in_assistant],
        "shifted_positions": [int(x) for x in separator_positions_final],
        "assistant_start_char": assistant_start_char,
    }
    return [int(x) for x in full_ids], separator_positions_final, debug


def _build_chat_text(tokenizer: Any, user_prompt: str, assistant_content: str) -> str:
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as exc:
        raise PipelineError("Failed to apply chat template for this tokenizer.") from exc


def _print_startup_example(ds: Dataset) -> None:
    if len(ds) == 0:
        print("Startup example: no rows ready for inference after preprocessing/resume filtering.")
        return

    row = ds[0]
    input_ids = [int(x) for x in row["input_ids"]]
    positions = [int(x) for x in row["thought_separator_positions"]]
    extracted_token_ids = [input_ids[p] for p in positions]

    print("Startup example full text sequence:")
    print(row.get("full_text", ""))
    print("Startup example input ids:")
    print(input_ids)
    print("Startup example extraction positions:")
    print(positions)
    print("Startup example token ids at extraction positions:")
    print(extracted_token_ids)


def _print_filter_reasons(ds: Dataset, max_examples_per_reason: int = 5) -> None:
    if len(ds) == 0:
        print("Filter reasons: dataset is empty before preprocessing output analysis.")
        return

    if "filter_reason" not in set(ds.column_names):
        print(
            "Filter reasons: unavailable (missing filter_reason column in pretokenized dataset). "
            "This usually indicates a stale cached map artifact or an older pipeline version."
        )
        return

    reasons: dict[str, list[str]] = {}
    for row in ds:
        reason = row.get("filter_reason", "")
        if not reason:
            continue
        key = str(row.get("__row_key", "unknown"))
        reasons.setdefault(reason, []).append(key)

    if not reasons:
        print("Filter reasons: none (all rows passed preprocessing filters).")
        return

    print("Filter reasons summary:")
    for reason, keys in sorted(reasons.items(), key=lambda x: (-len(x[1]), x[0])):
        sample = keys[:max_examples_per_reason]
        print(f"- {reason}: {len(keys)} rows | sample_keys={sample}")

    _print_one_filtered_example(ds)


def _print_startup_config_checks(cfg: ThoughtEmbeddingConfig) -> None:
    print(f"cfg.separator_text repr: {repr(cfg.separator_text)}")
    boxed_ok = "\\boxed{{}}" in cfg.user_prompt_template
    print(f"user_prompt_template_contains_escaped_boxed={{}}: {boxed_ok}")


def _print_one_filtered_example(ds: Dataset) -> None:
    for row in ds:
        reason = row.get("filter_reason", "")
        if not reason:
            continue
        print("One filtered example:")
        print(f"  row_key={row.get('__row_key')}")
        print(f"  reason={reason}")
        print(f"  problem={repr(row.get('problem'))}")
        print(f"  num_thoughts={row.get('num_thoughts')}")
        print(f"  input_token_count={row.get('input_token_count')}")
        print(f"  source={row.get('source')}")
        print(f"  id={row.get('id')} qid={row.get('qid')}")
        return
    print("One filtered example: none")


def _assert_shifted_positions_in_bounds(
    *,
    input_ids: list[int],
    shifted_positions: list[int],
) -> None:
    for pos in shifted_positions:
        assert 0 <= pos < len(input_ids), f"Shifted separator position out of range: pos={pos}, len={len(input_ids)}"


def _debug_separator_alignment_failure(
    *,
    tokenizer: Any,
    separator_text: str,
    separator_token_id: int,
    thoughts: list[str],
    assistant_text: str,
    separator_positions_in_assistant: list[int],
    input_ids: list[int],
    shifted_positions: list[int],
    alignment_debug: dict[str, Any],
) -> None:
    global _PRINTED_SEPARATOR_DEBUG
    if _PRINTED_SEPARATOR_DEBUG:
        return
    _PRINTED_SEPARATOR_DEBUG = True

    separator_ids = tokenizer.encode(separator_text, add_special_tokens=False)
    assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
    assistant_sep_count = sum(1 for t in assistant_ids if int(t) == int(separator_token_id))
    shifted_values = [int(input_ids[p]) for p in shifted_positions if 0 <= p < len(input_ids)]

    print("DEBUG separator alignment failure:")
    print(f"- repr(separator_text)={repr(separator_text)}")
    print(f"- tokenizer.encode(separator_text)={separator_ids}")
    print(f"- num_thoughts={len(thoughts)}")
    print(f"- expected_num_separator_positions={len(thoughts)}")
    print("- assistant_text_first_300:")
    print(assistant_text[:300])
    print(f"- repr(assistant_text[:100])={repr(assistant_text[:100])}")
    print(f"- assistant_separator_positions={separator_positions_in_assistant}")
    print(f"- assistant_ids_len={len(assistant_ids)}")
    print(f"- assistant_separator_token_count={assistant_sep_count}")
    print(f"- prefix_ids_len={alignment_debug.get('prefix_len')}")
    print(f"- assistant_ids_len_from_shift_debug={alignment_debug.get('assistant_len')}")
    print(f"- shifted_separator_positions={shifted_positions}")
    print(f"- final_ids_at_shifted_positions={shifted_values}")


def _debug_separator_nonstandalone_notice(
    *,
    separator_text: str,
    separator_token_id: int,
    exact_match_count: int,
    expected_count: int,
) -> None:
    global _PRINTED_NONSTANDALONE_NOTICE
    if exact_match_count == expected_count:
        return
    if _PRINTED_NONSTANDALONE_NOTICE:
        return
    _PRINTED_NONSTANDALONE_NOTICE = True
    print(
        "NOTE: separator token is not standalone in all contexts; using span-aligned boundary positions. "
        f"repr(separator_text)={repr(separator_text)} separator_token_id={separator_token_id} "
        f"exact_token_matches={exact_match_count}/{expected_count}"
    )


def _would_overflow_batch(cfg: ThoughtEmbeddingConfig, batch: BatchState, example: PreTokenizedExample) -> bool:
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
    pad_token_id: int,
    batch: BatchState,
    output_rows: list[dict[str, Any]],
    completed_keys: set[str],
) -> int:
    if not batch.examples:
        return 0

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

    completed_now = 0
    for i, ex in enumerate(batch.examples):
        vectors: list[list[float]] = []
        for pos in ex.thought_separator_positions:
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
            "thought_separator_positions": ex.thought_separator_positions,
            "input_token_count": len(ex.input_ids),
            "was_truncated": False,
        }
        if ex.answer is not None:
            out["answer"] = ex.answer
        if ex.expected_answer is not None:
            out["expected_answer"] = ex.expected_answer
        if ex.source is not None:
            out["source"] = ex.source
        if ex.source_id is not None:
            out["id"] = ex.source_id
        if ex.source_qid is not None:
            out["qid"] = ex.source_qid
        if cfg.keep_solution and ex.solution is not None:
            out["solution"] = ex.solution

        output_rows.append(out)
        completed_keys.add(ex.key)
        completed_now += 1

    batch.examples.clear()
    batch.total_tokens = 0
    return completed_now


def _pad_token_id(tokenizer: Any, model: Any) -> int:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)

    model_cfg = getattr(model, "config", None)
    if model_cfg is not None and getattr(model_cfg, "pad_token_id", None) is not None:
        return int(model_cfg.pad_token_id)
    if model_cfg is not None and getattr(model_cfg, "eos_token_id", None) is not None:
        return int(model_cfg.eos_token_id)

    raise PipelineError(
        "Unable to determine a padding token id. Provide tokenizer.pad_token_id/eos_token_id "
        "or model.config.pad_token_id/eos_token_id."
    )


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


def _write_shard(
    output_rows: list[dict[str, Any]],
    output_dir: Path,
    next_shard_idx: int,
) -> tuple[int, int]:
    rows_written = len(output_rows)
    if rows_written == 0:
        return 0, next_shard_idx
    write_parquet_shard(output_rows, output_dir, next_shard_idx)
    output_rows.clear()
    return rows_written, next_shard_idx + 1


def _save_manifest(
    output_dir: Path,
    completed_keys: set[str],
    processed_rows: int,
    written_rows: int,
    next_shard_idx: int,
) -> dict[str, Any]:
    manifest = {
        "completed_keys": sorted(completed_keys),
        "processed_rows": processed_rows,
        "written_rows": written_rows,
        "next_shard_idx": next_shard_idx,
    }
    save_manifest(output_dir, manifest)
    return manifest
