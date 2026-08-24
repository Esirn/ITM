"""Interchangeable Text-to-Motion backbone adapters."""

from itm.backbones.base import MotionBackbone, MotionConditions
from itm.backbones.mdm import MDMBackbone
from itm.backbones.motionlab import MotionLabBackbone

__all__ = ["MDMBackbone", "MotionBackbone", "MotionConditions", "MotionLabBackbone"]
