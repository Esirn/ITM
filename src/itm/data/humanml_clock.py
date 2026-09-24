"""Diagnostic replay of the local HumanML raw-pose notebook's index clock.

This is NOT an approved input transform or a replacement for physical resampling.
An index match does not prove historical artifact provenance or semantic alignment.
"""
import math

import numpy as np

VERSION = "humanml-notebook-clock-diagnostic-v1"
PREFIX_TRIM_SECONDS = {
    "Eyes_Japan_Dataset": 3., "MPI_HDM05": 3., "TotalCapture": 1.,
    "MPI_Limits": 1., "Transitions_mocap": .5,
}


def notebook_source_indices(*, source_path, source_fps, source_frames, start_frame, end_frame):
    """Replay [::int(fps/20)], dataset prefix trim, then [start:end].

    Returns both the nominal 20 Hz clock and the physical source clock. Python
    end clamping is reported because this function diagnoses notebook behavior;
    it never creates a cache or promotes the result to an admissible input.
    """
    if not math.isfinite(source_fps) or source_fps < 20 or source_frames < 1 or start_frame < 0:
        raise ValueError("Valid source metadata with FPS >= 20 required")
    if end_frame <= start_frame:
        raise ValueError("Explicit nonempty index interval required")
    step = int(source_fps / 20.)
    parts = str(source_path).replace("\\", "/").split("/")
    trim = sum(seconds for dataset, seconds in PREFIX_TRIM_SECONDS.items() if dataset in parts)
    trim_frames = int(trim * 20)
    downsampled_count = (source_frames + step - 1) // step
    available = max(0, downsampled_count - trim_frames)
    stop = min(end_frame, available)
    selected = np.arange(min(start_frame, available), stop, dtype=np.int64)
    raw_indices = (selected + trim_frames) * step
    source_times = raw_indices.astype(np.float64) / source_fps
    return {
        "version": VERSION, "integer_stride": step, "dataset_prefix_trim_nominal_seconds": trim,
        "downsampled_frames_before_trim": downsampled_count, "frames_after_prefix_trim": available,
        "notebook_end_would_clamp": end_frame > available,
        "raw_source_indices": raw_indices, "source_times_seconds": source_times,
        "nominal_clip_times_seconds": np.arange(len(raw_indices), dtype=np.float64) / 20.,
        "predicted_feature_frames": max(0, len(raw_indices) - 1),
        "physical_sample_rate_hz": source_fps / step,
        "nominal_to_physical_duration_ratio": source_fps / (20. * step),
        "permitted_as_deployable_input_transform": False,
    }
