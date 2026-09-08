#!/usr/bin/env python3
"""Extract a clean HY-Motion source snapshot without modifying its mounted repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


TEXT_ENCODER = Path("hymotion/network/text_encoders/text_encoder.py")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/hymotion/runtime_root"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    git_dir = source / ".git"
    if not git_dir.is_dir():
        raise FileNotFoundError(f"HY-Motion git metadata is missing: {git_dir}")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"output exists; pass --force to replace it: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    commit = subprocess.check_output(
        ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"], text=True
    ).strip()
    clean_text = subprocess.check_output(
        ["git", f"--git-dir={git_dir}", "show", f"{commit}:{TEXT_ENCODER.as_posix()}"]
    )
    mounted_text = (source / TEXT_ENCODER).read_bytes()
    process = subprocess.Popen(
        ["git", f"--git-dir={git_dir}", "archive", "--format=tar", commit, "hymotion"],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        root = args.output.resolve()
        for member in archive:
            target = (root / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.name}")
            archive.extract(member, root)
    if process.wait() != 0:
        raise RuntimeError("git archive failed")
    extracted_text = (args.output / TEXT_ENCODER).read_bytes()
    if extracted_text != clean_text:
        raise RuntimeError("extracted source does not match the requested commit")
    metadata = {
        "source": str(source),
        "commit": commit,
        "clean_text_encoder_sha256": sha256_bytes(clean_text),
        "mounted_text_encoder_sha256": sha256_bytes(mounted_text),
        "mounted_text_encoder_matches_commit": mounted_text == clean_text,
        "scope": "official committed hymotion Python package only",
        "assets_policy": "checkpoints, stats, and wooden body assets remain read-only in source",
    }
    (args.output / "workspace.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
