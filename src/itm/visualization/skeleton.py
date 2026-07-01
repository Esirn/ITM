"""Matplotlib animation for side-by-side HumanML3D skeletons."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import numpy as np

from itm.data.humanml import T2M_KINEMATIC_CHAIN


def save_motion_comparison(
    output_path: str | Path,
    motions: list[np.ndarray],
    labels: list[str],
    caption: str,
    *,
    fps: int = 20,
) -> None:
    """Save synchronized skeleton motions as GIF or MP4."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if len(motions) != len(labels) or not motions:
        raise ValueError("motions and labels must have the same non-zero length")
    arrays = [np.asarray(motion, dtype=np.float32) for motion in motions]
    frame_count = min(len(motion) for motion in arrays)
    if frame_count == 0:
        raise ValueError("Cannot render an empty motion")

    colors = ("#247BA0", "#F18F01", "#2E8B57", "#7A5195")
    fig = plt.figure(figsize=(4.2 * len(arrays), 4.5))
    axes = [fig.add_subplot(1, len(arrays), index + 1, projection="3d") for index in range(len(arrays))]
    fig.suptitle(fill(caption, 90), fontsize=11)

    all_points = np.concatenate([motion[:frame_count] for motion in arrays], axis=0)
    height_min = float(all_points[..., 1].min())
    horizontal_span = max(
        float(np.ptp(all_points[..., 0])), float(np.ptp(all_points[..., 2])), 2.0
    )
    radius = horizontal_span * 0.32

    def update(frame: int):
        artists = []
        for panel, (axis, motion, label) in enumerate(zip(axes, arrays, labels)):
            axis.clear()
            root = motion[frame, 0]
            axis.set_xlim(root[0] - radius, root[0] + radius)
            axis.set_ylim(root[2] - radius, root[2] + radius)
            axis.set_zlim(height_min, height_min + 2.2)
            axis.set_box_aspect((1, 1, 1.25))
            axis.view_init(elev=18, azim=-70)
            axis.set_title(label, fontsize=10)
            axis.set_axis_off()
            for chain in T2M_KINEMATIC_CHAIN:
                line = axis.plot(
                    motion[frame, chain, 0],
                    motion[frame, chain, 2],
                    motion[frame, chain, 1],
                    color=colors[panel % len(colors)],
                    linewidth=2.6,
                )[0]
                artists.append(line)
        return artists

    animation = FuncAnimation(
        fig, update, frames=frame_count, interval=1000 / fps, blit=False, repeat=False
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        animation.save(output, writer=PillowWriter(fps=fps))
    else:
        animation.save(output, fps=fps)
    plt.close(fig)
