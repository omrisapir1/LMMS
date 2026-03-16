from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

try:
    from .dataset import (
        iter_hf_rows,
        iter_parquet_rows,
        list_parquet_shards,
        parse_latent_row,
    )
    from .model import Codebook
    from .quantize_repr import normalize_quantize_mode, transform_sequence_np
except ImportError:
    from dataset import iter_hf_rows, iter_parquet_rows, list_parquet_shards, parse_latent_row  # type: ignore
    from model import Codebook  # type: ignore
    from quantize_repr import normalize_quantize_mode, transform_sequence_np  # type: ignore


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("qid", pa.string()),
        pa.field("question", pa.string()),
        pa.field("answer_int", pa.int32()),
        pa.field("answer_digits", pa.list_(pa.int32())),
        pa.field("K_star", pa.int32()),
        pa.field("k_max", pa.int32()),
        pa.field("z_ids", pa.list_(pa.int32())),
    ]
)


def _load_codebook(codebook_path: str, *, device: torch.device) -> tuple[Codebook, Dict]:
    ckpt = torch.load(codebook_path, map_location="cpu")
    embeddings = ckpt["embeddings"].to(torch.float32)
    vocab_size, dim = embeddings.shape
    ema_decay = float(ckpt.get("ema_decay", 0.995))

    model = Codebook(dim=dim, vocab_size=vocab_size, ema_decay=ema_decay).to(device)
    model.initialize_embeddings(embeddings.to(device))

    if "ema_cluster_size" in ckpt and "ema_embedding_sum" in ckpt:
        model.ema_cluster_size.copy_(ckpt["ema_cluster_size"].to(device=device, dtype=torch.float32))
        model.ema_embedding_sum.copy_(ckpt["ema_embedding_sum"].to(device=device, dtype=torch.float32))
        updated = model.ema_embedding_sum / (model.ema_cluster_size.unsqueeze(-1) + 1e-5)
        # Match training-time unit-norm embedding behavior for cosine assignment.
        updated = F.normalize(updated, p=2, dim=-1, eps=1e-12)
        model.embeddings.copy_(updated)

    model.eval()
    return model, ckpt


@torch.no_grad()
def _quantize_flat(
    *,
    model: Codebook,
    flat_latents: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    if flat_latents.ndim != 2 or flat_latents.shape[1] != model.dim:
        raise ValueError(f"Expected flat_latents shape [N, {model.dim}]")
    if flat_latents.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    out = np.empty((flat_latents.shape[0],), dtype=np.int32)
    start = 0
    while start < flat_latents.shape[0]:
        end = min(start + chunk_size, flat_latents.shape[0])
        chunk = torch.from_numpy(flat_latents[start:end]).to(device=device, dtype=torch.float32)
        z_ids, _ = model(chunk)
        out[start:end] = z_ids.cpu().numpy().astype(np.int32, copy=False)
        start = end
    return out


def _rows_to_table(rows: List[Dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)


def export_dataset(
    *,
    input_dir: str,
    output_dir: str,
    codebook_path: str,
    dim: int = 1024,
    read_batch_size: int = 256,
    quantize_chunk_size: int = 16_384,
    quantize_mode: str | None = None,
    skip_invalid_rows: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_codebook(codebook_path, device=device)
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    inferred_mode = "delta"
    if isinstance(ckpt_cfg, dict):
        inferred_mode = str(ckpt_cfg.get("quantize_mode", inferred_mode))
    mode = normalize_quantize_mode(quantize_mode if quantize_mode is not None else inferred_mode)
    print(f"[export] quantize_mode={mode}")

    total_rows = 0
    total_shards = 0

    input_path = Path(input_dir)
    if input_path.exists() and input_path.is_dir():
        sources = [(shard.name, iter_parquet_rows(shard, read_batch_size=read_batch_size)) for shard in list_parquet_shards(input_dir)]
    else:
        # HF input is streamed and exported as a single parquet shard.
        sources = [("hf_stream.parquet", iter_hf_rows(input_dir, read_batch_size=read_batch_size))]

    for source_name, row_iter in sources:
        out_path = Path(output_dir) / source_name
        writer = pq.ParquetWriter(str(out_path), OUTPUT_SCHEMA, compression="zstd")
        shard_rows = 0
        shard_invalid_rows = 0

        try:
            batch_rows: List[Dict] = []
            batch_latents: List[np.ndarray] = []
            batch_offsets: List[tuple[int, int]] = []

            for row in row_iter:
                parsed, err = parse_latent_row(row, dim=dim)
                if parsed is None:
                    if skip_invalid_rows:
                        shard_invalid_rows += 1
                        print(
                            f"[warn] skipping invalid row shard={source_name} "
                            f"qid={row.get('qid', '')!r}: {err}"
                        )
                        continue
                    raise RuntimeError(
                        f"Invalid row in shard={source_name} qid={row.get('qid', '')!r}: {err}"
                    )

                start = 0 if not batch_offsets else batch_offsets[-1][1]
                end = start + parsed.k_star
                batch_offsets.append((start, end))
                batch_latents.append(transform_sequence_np(parsed.latent_vectors, mode=mode))
                batch_rows.append(
                    {
                        "qid": parsed.qid,
                        "question": parsed.question,
                        "answer_int": int(parsed.answer_int),
                        "answer_digits": [int(x) for x in parsed.answer_digits],
                        "K_star": int(parsed.k_star),
                        "k_max": int(parsed.k_max),
                    }
                )

                if len(batch_rows) < read_batch_size:
                    continue

                flat = np.concatenate(batch_latents, axis=0)
                flat = np.ascontiguousarray(flat, dtype=np.float32)
                z_flat = _quantize_flat(
                    model=model,
                    flat_latents=flat,
                    device=device,
                    chunk_size=quantize_chunk_size,
                )
                out_rows: List[Dict] = []
                for item, (s, e) in zip(batch_rows, batch_offsets):
                    out_item = dict(item)
                    out_item["z_ids"] = z_flat[s:e].tolist()
                    out_rows.append(out_item)

                writer.write_table(_rows_to_table(out_rows))
                shard_rows += len(out_rows)
                batch_rows.clear()
                batch_latents.clear()
                batch_offsets.clear()

            if batch_rows:
                flat = np.concatenate(batch_latents, axis=0)
                flat = np.ascontiguousarray(flat, dtype=np.float32)
                z_flat = _quantize_flat(
                    model=model,
                    flat_latents=flat,
                    device=device,
                    chunk_size=quantize_chunk_size,
                )
                out_rows = []
                for item, (s, e) in zip(batch_rows, batch_offsets):
                    out_item = dict(item)
                    out_item["z_ids"] = z_flat[s:e].tolist()
                    out_rows.append(out_item)
                writer.write_table(_rows_to_table(out_rows))
                shard_rows += len(out_rows)
        finally:
            writer.close()

        total_rows += shard_rows
        total_shards += 1
        print(
            f"[export] shard={source_name} rows={shard_rows} invalid_rows={shard_invalid_rows} -> {out_path}"
        )

    print(f"[done] exported shards={total_shards} rows={total_rows} to {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Phase1 latent dataset into z_ids using trained codebook")
    p.add_argument("--input_dir", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--codebook_path", required=True, type=str)
    p.add_argument("--dim", default=1024, type=int)
    p.add_argument("--read_batch_size", default=256, type=int)
    p.add_argument("--quantize_chunk_size", default=16384, type=int)
    p.add_argument("--quantize_mode", default=None, choices=["raw", "delta"], type=str)
    p.add_argument("--skip_invalid_rows", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        codebook_path=args.codebook_path,
        dim=args.dim,
        read_batch_size=args.read_batch_size,
        quantize_chunk_size=args.quantize_chunk_size,
        quantize_mode=args.quantize_mode,
        skip_invalid_rows=bool(args.skip_invalid_rows),
    )


if __name__ == "__main__":
    main()
