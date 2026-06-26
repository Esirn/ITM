"""Project configuration helpers."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def load_toml(path: str | Path) -> dict:
    """Load a TOML config file."""

    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def dataset_paths(config: dict) -> dict:
    """Return the dataset path section from a loaded config."""

    return config.get("datasets", {})

