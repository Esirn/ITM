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
    imu_acceleration: np.ndarray | None = None,
    imu_orientation: np.ndarray | None = None,
    imu_title: str | None = None,
) -> None:
    """Save synchronized skeleton motions and an optional IMU trace."""

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
    has_imu = imu_acceleration is not None or imu_orientation is not None
    fig = plt.figure(figsize=(4.2 * len(arrays), 7.0 if has_imu else 4.5))
    if has_imu:
        grid = fig.add_gridspec(2, len(arrays), height_ratios=(3.2, 1.25))
        axes = [
            fig.add_subplot(grid[0, index], projection="3d")
            for index in range(len(arrays))
        ]
        acceleration_axis = fig.add_subplot(grid[1, : len(arrays) // 2])
        orientation_axis = fig.add_subplot(grid[1, len(arrays) // 2 :])
    else:
        axes = [
            fig.add_subplot(1, len(arrays), index + 1, projection="3d")
            for index in range(len(arrays))
        ]
        acceleration_axis = None
        orientation_axis = None
    fig.suptitle(fill(caption, 90), fontsize=11)

    all_points = np.concatenate([motion[:frame_count] for motion in arrays], axis=0)
    height_min = float(all_points[..., 1].min())
    horizontal_span = max(
        float(np.ptp(all_points[..., 0])), float(np.ptp(all_points[..., 2])), 2.0
    )
    radius = horizontal_span * 0.32
    cursor_lines = []
    if has_imu:
        time = np.arange(frame_count, dtype=np.float32) / fps
        _plot_imu_signal(
            acceleration_axis,
            imu_acceleration,
            time,
            "Acceleration proxy (position/frame^2)",
        )
        _plot_imu_signal(
            orientation_axis,
            imu_orientation,
            time,
            "Limb orientation unit vector",
        )
        fig.text(
            0.5,
            0.34,
            imu_title or "Displayed IMU",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="semibold",
        )
        for axis in (acceleration_axis, orientation_axis):
            cursor_lines.append(axis.axvline(0.0, color="#222222", linewidth=1.2))
        fig.subplots_adjust(top=0.86, bottom=0.09, hspace=0.18, wspace=0.08)

    def update(frame: int):
        artists = list(cursor_lines)
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
        for cursor in cursor_lines:
            cursor.set_xdata([frame / fps, frame / fps])
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


def _plot_imu_signal(axis, signal, time: np.ndarray, title: str) -> None:
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("Time (s)", fontsize=8)
    axis.grid(True, color="#DDDDDD", linewidth=0.6)
    axis.tick_params(labelsize=7)
    if signal is None:
        axis.text(0.5, 0.5, "Not available", ha="center", va="center", transform=axis.transAxes)
        return
    values = np.asarray(signal, dtype=np.float32)
    usable = min(len(values), len(time))
    for component, color in zip(range(3), ("#D1495B", "#2E8B57", "#247BA0")):
        axis.plot(time[:usable], values[:usable, component], color=color, linewidth=1.0, label="xyz"[component])
    axis.legend(loc="upper right", ncol=3, fontsize=7, frameon=False)


def save_control_comparison(
    output_path: str | Path,
    motions: list[np.ndarray],
    labels: list[str],
    captions: list[str],
    imu_accelerations: list[np.ndarray],
    imu_orientations: list[np.ndarray],
    imu_titles: list[str],
    *,
    fps: int = 20,
) -> None:
    """Render per-panel motion and synchronized target IMU traces."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    count = len(motions)
    if not all(len(values) == count for values in (labels, captions, imu_accelerations, imu_orientations, imu_titles)):
        raise ValueError("All control comparison inputs must have equal lengths")
    arrays = [np.asarray(motion, dtype=np.float32) for motion in motions]
    frame_count = min(len(value) for value in arrays)
    fig = plt.figure(figsize=(4.4 * count, 8.0))
    grid = fig.add_gridspec(3, count, height_ratios=(3.3, 1.0, 1.0))
    motion_axes = [fig.add_subplot(grid[0, i], projection="3d") for i in range(count)]
    acceleration_axes = [fig.add_subplot(grid[1, i]) for i in range(count)]
    orientation_axes = [fig.add_subplot(grid[2, i]) for i in range(count)]
    time = np.arange(frame_count, dtype=np.float32) / fps
    cursors = []
    for index in range(count):
        acceleration_axes[index].set_title(imu_titles[index], fontsize=8)
        _plot_imu_signal(acceleration_axes[index], imu_accelerations[index], time, "Acceleration (m/s^2)")
        _plot_imu_signal(orientation_axes[index], imu_orientations[index], time, "Orientation first column")
        cursors.extend(
            [
                acceleration_axes[index].axvline(0, color="#222222", linewidth=1),
                orientation_axes[index].axvline(0, color="#222222", linewidth=1),
            ]
        )
    all_points = np.concatenate([motion[:frame_count] for motion in arrays], axis=0)
    height_min = float(all_points[..., 1].min())
    horizontal_span = max(float(np.ptp(all_points[..., 0])), float(np.ptp(all_points[..., 2])), 2.0)
    radius = horizontal_span * 0.32
    colors = ("#247BA0", "#F18F01", "#2E8B57", "#7A5195", "#D1495B")

    def update(frame):
        artists = list(cursors)
        for panel, (axis, motion) in enumerate(zip(motion_axes, arrays)):
            axis.clear()
            root = motion[frame, 0]
            axis.set_xlim(root[0] - radius, root[0] + radius)
            axis.set_ylim(root[2] - radius, root[2] + radius)
            axis.set_zlim(height_min, height_min + 2.2)
            axis.set_box_aspect((1, 1, 1.25))
            axis.view_init(elev=18, azim=-70)
            axis.set_title(f"{labels[panel]}\n{fill(captions[panel], 38)}", fontsize=9)
            axis.set_axis_off()
            for chain in T2M_KINEMATIC_CHAIN:
                artists.append(axis.plot(
                    motion[frame, chain, 0], motion[frame, chain, 2], motion[frame, chain, 1],
                    color=colors[panel % len(colors)], linewidth=2.6,
                )[0])
        for cursor in cursors:
            cursor.set_xdata([frame / fps, frame / fps])
        return artists

    fig.subplots_adjust(top=0.94, bottom=0.07, hspace=0.45, wspace=0.22)
    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps, blit=False, repeat=False)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
