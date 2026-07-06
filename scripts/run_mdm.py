#!/usr/bin/env python
"""Run the external official MDM code with modern NumPy compatibility."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys

import numpy as np


MODULES = {"sample": "sample.generate", "evaluate": "eval.eval_humanml"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODULES)
    parser.add_argument(
        "--mdm-root",
        default="/home/a200/mount/a40/relatedworks/mdm/motion-diffusion-model",
    )
    parser.add_argument(
        "--external-render",
        action="store_true",
        help="Use MDM's legacy Matplotlib renderer instead of saving results only.",
    )
    args, forwarded = parser.parse_known_args()
    root = Path(args.mdm_root).resolve()
    if not (root / "model/mdm.py").exists():
        raise FileNotFoundError(f"Invalid MDM root: {root}")
    _enable_legacy_numpy_aliases()
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = [MODULES[args.mode], *forwarded]
    if args.mode == "sample" and not args.external_render:
        module = importlib.import_module(MODULES[args.mode])
        module.plot_3d_motion = lambda *unused_args, **unused_kwargs: None
        module.save_multiple_samples = lambda *unused_args, **unused_kwargs: []
        module.main()
    else:
        runpy.run_module(MODULES[args.mode], run_name="__main__")
    return 0


def _enable_legacy_numpy_aliases() -> None:
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
