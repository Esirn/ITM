#!/usr/bin/env python
"""Build an index for generated ITM qualitative artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    entries = []
    for result in sorted(directory.glob("*_results.npz")):
        with np.load(result) as data:
            metadata = json.loads(str(data["metadata"]))
        stem = result.name.removesuffix("_results.npz")
        gif = directory / f"{stem}_comparison.gif"
        entries.append({
            "name": stem,
            "results": result.name,
            "visualization": gif.name if gif.exists() else None,
            **metadata,
        })
    index = directory / "index.json"
    index.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"index: {index} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
