"""Backbone-neutral interfaces used by ITM sampling and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MotionConditions:
    """Backbone-neutral generation conditions.

    Backbone-specific tensors belong in ``extra``. IMU preprocessing and
    experiment provenance intentionally remain outside the backbone adapter.
    """

    text: list[str]
    lengths: Any
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MotionBackbone(Protocol):
    def encode_text(self, text: list[str]) -> Any: ...

    def predict(self, x_t: Any, timestep: Any, conditions: MotionConditions) -> Any: ...

    def sample(self, conditions: MotionConditions, *, seed: int) -> Any: ...

    def decode_motion(self, motion: Any) -> Any: ...
