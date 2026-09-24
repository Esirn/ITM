"""Strict caption intervals; these checks do not certify semantic/world alignment."""
from dataclasses import dataclass
import math

import numpy as np

from itm.data.physical_time import resample_positions_at_times

VERSION = "caption-interval-v1"


@dataclass(frozen=True)
class TimedCaption:
    text: str
    tokens: str
    start: float
    end: float


def parse_timed_caption(line: str) -> TimedCaption:
    """Reject missing/malformed tags instead of silently treating them as full clip."""
    parts = line.strip().split("#")
    if len(parts) != 4 or not parts[0].strip():
        raise ValueError("Nonempty caption and exactly four fields required")
    try:
        start, end = float(parts[2]), float(parts[3])
    except ValueError as exc:
        raise ValueError("Numeric caption tags required") from exc
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < 0:
        raise ValueError("Finite nonnegative caption tags required")
    if (start, end) != (0., 0.) and end <= start:
        raise ValueError("Tagged caption end must exceed start")
    return TimedCaption(parts[0].strip(), parts[1].strip(), start, end)


def caption_interval(caption: TimedCaption, motion_frames: int, *, fps=20.):
    """Map tags to a half-open feature slice using the local MDM int(tag*fps) rule.

    (0, 0) is the full-clip annotation sentinel, not a zero-duration observation.
    Out-of-range tags are rejected before slicing; Python's silent clamp is forbidden.
    """
    if motion_frames < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Nonempty motion and positive FPS required")
    if (not caption.text or not math.isfinite(caption.start) or not math.isfinite(caption.end)
            or caption.start < 0 or caption.end < 0):
        raise ValueError("Invalid caption")
    full = caption.start == caption.end == 0.
    if not full and (caption.end <= caption.start or caption.end > motion_frames / fps):
        raise ValueError("Caption interval outside feature support or reversed")
    start, stop = (0, motion_frames) if full else (int(caption.start * fps), int(caption.end * fps))
    if not 0 <= start < stop <= motion_frames:
        raise ValueError("Caption interval is empty or outside feature support")
    return {
        "version": VERSION, "annotation_kind": "full_clip_sentinel" if full else "tagged_interval",
        "raw_tags_seconds": [caption.start, caption.end], "frame_slice": [start, stop],
        "slice_convention": "start_inclusive_stop_exclusive", "fps": fps, "frames": stop - start,
        "source_sample_interval_seconds": [start / fps, (stop - 1) / fps],
        "source_slice_bounds_seconds": [start / fps, stop / fps],
        "tag_quantization_seconds": [0., 0.] if full else [start / fps - caption.start, stop / fps - caption.end],
        "clock": "HumanML_clip_local_seconds", "world_alignment_certified": False,
        "semantic_alignment_certified": False,
    }


def plan_caption_crop(caption, motion_frames, source_times, *, max_frames=196, fps=20.):
    """Require a complete annotation interval within model and IMU support.

    IMU must be sampled at source_times_seconds, NOT at the rebased model time.
    Cropped HumanML features retain their original representation; this function
    does not recanonicalize roots, recover a lost heading, or approve a witness.
    """
    interval = caption_interval(caption, motion_frames, fps=fps)
    if max_frames < 1 or interval["frames"] > max_frames:
        raise ValueError("Complete caption interval exceeds model limit; explicit semantic review required")
    start, stop = interval["frame_slice"]
    source = np.asarray(source_times, dtype=np.float64)
    times = np.arange(start, stop, dtype=np.float64) / fps
    # Reuse the public range/finite/monotonic checks without loading sensor arrays.
    resample_positions_at_times(np.zeros((len(source), 1)), source, times)
    return {**interval, "source_times_seconds": times,
            "model_times_seconds": np.arange(stop - start, dtype=np.float64) / fps,
            "source_time_origin_seconds": start / fps,
            "training_admissible": False,
            "remaining_checks": ["world_and_skeleton_alignment", "caption_semantic_review",
                                 "method_crop_integration_and_witness_revalidation"]}
