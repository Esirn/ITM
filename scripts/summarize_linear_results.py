#!/usr/bin/env python
"""Summarize linear baseline CSV/JSONL outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more summary.csv or summary.jsonl files.",
    )
    parser.add_argument(
        "--sort-key",
        default="eval_mse",
        help="Numeric column used for sorting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.paths:
        rows.extend(_read_rows(path))
    rows.sort(key=lambda row: float(row[args.sort_key]))
    print("variant,eval_mse,eval_mae,train_mse,train_mae,feature_dim,source")
    for row in rows:
        print(
            f"{row['variant']},{float(row['eval_mse']):.6f},"
            f"{float(row['eval_mae']):.6f},{float(row['train_mse']):.6f},"
            f"{float(row['train_mae']):.6f},{int(float(row['feature_dim']))},"
            f"{row['_source']}"
        )
    return 0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError(f"Unsupported summary format: {path}")
    for row in rows:
        row["_source"] = str(path)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
