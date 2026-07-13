#!/usr/bin/env python
"""Recompute per-run metrics for completed MDM-control experiment suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from itm.experiments.mdm_control_suite import compute_run_metrics, write_summary_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Experiment roots to scan for results.npz files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written = 0
    for root_arg in args.roots:
        root = Path(root_arg)
        for result_path in sorted(root.glob("**/results.npz")):
            metrics = compute_run_metrics(result_path)
            metrics_path = result_path.with_name("metrics.json")
            summary_path = result_path.with_name("summary.md")
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            write_summary_markdown(metrics, summary_path)
            written += 1
            print(f"metrics: {metrics_path}")
    print(f"recomputed_runs: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
