"""Interchangeable Text-to-Motion backbone adapters."""

from itm.backbones.base import MotionBackbone, MotionConditions
from itm.backbones.mdm import MDMBackbone
from itm.backbones.hymotion import HYMotionBackbone
from itm.backbones.motionlab import MotionLabBackbone

__all__ = [
    "HYMotionBackbone",
    "MDMBackbone",
    "MotionBackbone",
    "MotionConditions",
    "MotionLabBackbone",
]
