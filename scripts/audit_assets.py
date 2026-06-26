#!/usr/bin/env python3
"""Audit local assets needed for ITM experiments.

The script is deliberately non-mutating. It checks path existence, reports
likely dataset contents, and flags hard-coded paths in Ego4o configs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


HARDCODED_PATH_RE = re.compile(r"['\"](/(?:CT|scratch|home/jianwang)[^'\"]+)['\"]")


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def path_status(path: str) -> str:
    if not path:
        return "missing-value"
    if path.startswith("/home/a200/mount/"):
        return "deferred mount check"
    p = Path(path)
    if p.exists():
        if p.is_symlink():
            return "exists symlink"
        if p.is_dir():
            return "exists dir"
        return "exists file"
    return "missing"


def count_files(
    root: Path,
    suffixes: Iterable[str],
    limit: int = 500,
    max_depth: int = 4,
    max_dirs: int = 2_000,
) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    suffixes = tuple(suffixes)
    total = 0
    root_depth = len(root.parts)
    visited_dirs = 0
    for current, dirs, files in os.walk(root):
        visited_dirs += 1
        if visited_dirs > max_dirs:
            return total
        current_depth = len(Path(current).parts) - root_depth
        if current_depth >= max_depth:
            dirs[:] = []
        for name in files:
            if name.endswith(suffixes):
                total += 1
                if total >= limit:
                    return total
    return total


def count_label(count: int, limit: int = 500) -> str:
    if count >= limit:
        return f">={limit}"
    return str(count)


def print_path_section(title: str, values: dict[str, str]) -> None:
    print(f"\n## {title}", flush=True)
    for key, value in values.items():
        print(f"- {key}: {value} [{path_status(value)}]", flush=True)


def audit_datasets(values: dict[str, str]) -> None:
    print("\n## Dataset Probes", flush=True)
    probes = {
        "AMASS .npz": ("amass", (".npz",)),
        "HumanML3D text .txt": ("humanml3d_texts", (".txt",)),
        "motion split .txt": ("motion_splits", (".txt",)),
        "SMPL model files": ("smpl", (".pkl", ".npz")),
    }
    for label, (key, suffixes) in probes.items():
        root = Path(values.get(key, ""))
        count = count_files(root, suffixes)
        print(f"- {label}: {count_label(count)} under {root}", flush=True)


def audit_related_code(values: dict[str, str]) -> None:
    print("\n## Related Code Probes", flush=True)
    for key in ["ego4o", "motionlab", "motiondiffuse", "motiongpt", "transpose", "dip"]:
        root = Path(values.get(key, ""))
        git_dir = root / ".git"
        py_count = count_files(root, (".py",), limit=500, max_depth=6)
        print(
            f"- {key}: {path_status(str(root))}, "
            f"git={git_dir.exists()}, python_files={count_label(py_count)}",
            flush=True,
        )


def audit_ego4o_hardcoded_paths(ego4o_mocap: Path, max_files: int = 30) -> None:
    print("\n## Ego4o Hard-Coded Path Audit", flush=True)
    if not ego4o_mocap.exists():
        print(f"- skipped: missing {ego4o_mocap}", flush=True)
        return
    config_root = ego4o_mocap / "configs"
    matches: dict[str, set[str]] = {}
    for path in config_root.rglob("*.py"):
        text = path.read_text(errors="ignore")
        found = set(HARDCODED_PATH_RE.findall(text))
        if found:
            matches[str(path.relative_to(ego4o_mocap))] = found
    if not matches:
        print("- no hard-coded historical paths found", flush=True)
        return
    missing_paths: set[str] = set()
    for paths in matches.values():
        for value in paths:
            if path_status(value) == "missing":
                missing_paths.add(value)

    for idx, (rel_path, paths) in enumerate(sorted(matches.items())):
        if idx >= max_files:
            remaining = len(matches) - max_files
            print(f"- ... {remaining} more config files omitted", flush=True)
            break
        print(f"- {rel_path}", flush=True)
        for value in sorted(paths):
            print(f"  - {value} [{path_status(value)}]", flush=True)

    print("\n## Ego4o Missing Path Summary", flush=True)
    print(f"- config_files_with_hardcoded_paths: {len(matches)}", flush=True)
    print(f"- unique_missing_hardcoded_paths: {len(missing_paths)}", flush=True)
    for value in sorted(missing_paths)[:40]:
        print(f"  - {value}", flush=True)
    if len(missing_paths) > 40:
        print(f"  - ... {len(missing_paths) - 40} more missing paths omitted", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paths.toml"))
    parser.add_argument("--max-ego4o-files", type=int, default=30)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print("# ITM Asset Audit", flush=True)
    print_path_section("Datasets", config.get("datasets", {}))
    print_path_section("Related Papers", config.get("related_papers", {}))
    print_path_section("Related Code", config.get("related_code", {}))
    audit_datasets(config.get("datasets", {}))
    audit_related_code(config.get("related_code", {}))
    audit_ego4o_hardcoded_paths(
        Path(config["related_code"]["ego4o_mocap"]),
        max_files=args.max_ego4o_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
