"""Text embedding cache utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TextEmbeddingCache:
    """Embeddings keyed by motion ID, with provenance metadata."""

    embeddings: Mapping[str, np.ndarray]
    captions: Mapping[str, str]
    metadata: dict


def save_text_embedding_cache(
    output_path: str | Path,
    motion_ids: list[str],
    captions: list[str],
    embeddings: np.ndarray,
    *,
    metadata: dict,
) -> None:
    """Save one fixed-size text embedding per motion record."""

    if len(motion_ids) != len(captions) or len(motion_ids) != len(embeddings):
        raise ValueError("motion_ids, captions, and embeddings must have equal length")
    if len(set(motion_ids)) != len(motion_ids):
        raise ValueError("motion_ids must be unique")
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"embeddings must have shape (N, D), got {array.shape}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        motion_ids=np.asarray(motion_ids),
        captions=np.asarray(captions),
        embeddings=array,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def load_text_embedding_cache(path: str | Path) -> TextEmbeddingCache:
    """Load and validate a text embedding cache."""

    with np.load(path) as data:
        motion_ids = [str(value) for value in data["motion_ids"]]
        captions = [str(value) for value in data["captions"]]
        embeddings = data["embeddings"].astype(np.float32, copy=False)
        metadata = json.loads(str(data["metadata"]))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(motion_ids):
        raise ValueError("Invalid text embedding cache dimensions")
    if len(set(motion_ids)) != len(motion_ids):
        raise ValueError("Duplicate motion IDs in text embedding cache")
    return TextEmbeddingCache(
        embeddings={key: value for key, value in zip(motion_ids, embeddings)},
        captions={key: value for key, value in zip(motion_ids, captions)},
        metadata=metadata,
    )

