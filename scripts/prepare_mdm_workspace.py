#!/usr/bin/env python
"""Link an external official MDM checkpoint into a writable ITM run directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--output-dir", default="outputs/mdm/checkpoints/humanml_trans_enc_512")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _replace_link(output / Path(args.checkpoint).name, Path(args.checkpoint).resolve())
    _replace_link(output / "args.json", Path(args.args_json).resolve())
    print(f"model_path: {output / Path(args.checkpoint).name}")
    return 0


def _replace_link(link: Path, target: Path) -> None:
    if not target.exists():
        raise FileNotFoundError(target)
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    link.symlink_to(target)


if __name__ == "__main__":
    raise SystemExit(main())
