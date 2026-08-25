#!/usr/bin/env python3
"""Render best and worst direct-IMU test-suite examples for manual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itm.visualization.skeleton import save_motion_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--text-only", type=Path, required=True)
    parser.add_argument("--text-head", type=Path, required=True)
    parser.add_argument("--text-wrists", type=Path, required=True)
    parser.add_argument("--per-side", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text())
    spec = json.loads(Path(result["protocol"]["spec"]).read_text())
    arrays = {
        "text_only": np.load(args.text_only)["joints"],
        "head": np.load(args.text_head)["joints"],
        "wrists": np.load(args.text_wrists)["joints"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for config in ("head", "wrists"):
        ranked = sorted(
            enumerate(result["records"]),
            key=lambda item: item[1]["configs"][config]["active_error_change_m"],
        )
        for category, values in (
            ("best", ranked[: args.per_side]),
            ("worst", ranked[-args.per_side :]),
        ):
            for index, record in values:
                length = int(record["length"])
                target = np.load(spec["ground_truth_paths"][index]).astype(np.float32)[:length]
                output = args.output_dir / f"{config}_{category}_{record['motion_id']}.gif"
                save_motion_comparison(
                    output,
                    [
                        target,
                        arrays["text_only"][index, :length],
                        arrays[config][index, :length],
                    ],
                    ["Ground truth", "Text only", f"Text + {config} IMU"],
                    record["caption"],
                    fps=20,
                )
                selected.append(
                    {
                        "config": config,
                        "category": category,
                        "motion_id": record["motion_id"],
                        "active_error_change_m": record["configs"][config]["active_error_change_m"],
                        "caption": record["caption"],
                        "gif": str(output.resolve()),
                    }
                )
    (args.output_dir / "index.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    lines = ["# Manual Review Selection", ""]
    for item in selected:
        lines.append(
            f"- {item['config']} {item['category']} `{item['motion_id']}`: "
            f"change `{item['active_error_change_m']:+.4f} m`, `{item['gif']}`"
        )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(selected, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
