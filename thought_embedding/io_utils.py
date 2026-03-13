from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


MANIFEST_FILENAME = "manifest.json"


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def shard_path(output_dir: str | Path, shard_idx: int) -> Path:
    return Path(output_dir) / f"part-{shard_idx:05d}.parquet"


def write_parquet_shard(rows: list[dict[str, Any]], output_dir: str | Path, shard_idx: int) -> Path:
    if not rows:
        raise ValueError("Refusing to write empty shard.")
    path = shard_path(output_dir, shard_idx)
    ds = Dataset.from_list(rows)
    ds.to_parquet(str(path))
    return path


def manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / MANIFEST_FILENAME


def load_manifest(output_dir: str | Path) -> dict[str, Any]:
    path = manifest_path(output_dir)
    if not path.exists():
        return {
            "completed_keys": [],
            "processed_rows": 0,
            "written_rows": 0,
            "next_shard_idx": 0,
        }
    return json.loads(path.read_text())


def save_manifest(output_dir: str | Path, data: dict[str, Any]) -> Path:
    path = manifest_path(output_dir)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path


def list_existing_shards(output_dir: str | Path) -> list[Path]:
    return sorted(Path(output_dir).glob("part-*.parquet"))
