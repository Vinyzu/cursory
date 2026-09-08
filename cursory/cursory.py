import math
from collections.abc import Sequence

import numpy as np

from .trajectory_selection import (
    Point,
    find_trajectory,
    jitter_trajectory,
    knot_trajectory,
    morph_trajectory,
)


def _sample_timings(
    timings: Sequence[int | float],
    frequency: int,
    frequency_randomizer: int,
    rng: np.random.Generator,
) -> list[int]:
    total_time = max(1, round(timings[-1] - timings[0]))
    interval_count = max(1, round(total_time * frequency / 1000))

    recorded_intervals = np.diff(timings)
    recorded_intervals = recorded_intervals[recorded_intervals > 0]
    if not len(recorded_intervals):
        recorded_intervals = np.ones(1)

    # Time-warp the selected recording's interval sequence to the requested sample count.
    recorded_intervals = np.roll(recorded_intervals, -int(rng.integers(len(recorded_intervals))))
    source_indices = np.floor(
        (np.arange(interval_count) + 0.5) * len(recorded_intervals) / interval_count,
    ).astype(int)
    sampled_intervals = recorded_intervals[source_indices].astype(float)
    sampled_intervals *= total_time / sampled_intervals.sum()
    if frequency_randomizer:
        sampled_intervals += rng.uniform(-frequency_randomizer, frequency_randomizer, size=interval_count)
        sampled_intervals = np.maximum(sampled_intervals, np.finfo(float).eps)
        sampled_intervals *= total_time / sampled_intervals.sum()

    sampled_timings = [
        int(timing) for timing in np.rint(np.concatenate(([0.0], np.cumsum(sampled_intervals)))).astype(int)
    ]
    sampled_timings[-1] = total_time
    return sampled_timings


def generate_trajectory(
    target_start: Point,
    target_end: Point,
    frequency: int = 60,
    frequency_randomizer: int = 1,
    seed: int | None = None,
    directness: float = 0.65,
) -> tuple[list[Point], list[int]]:
    """Generate a realistic mouse trajectory from start to end points.

    Args:
        target_start (Point): Starting point of the trajectory.
        target_end (Point): Ending point of the trajectory.
        frequency (int): Number of samples per second (hz).
        frequency_randomizer (int): Max jitter in ms to apply to each sample time.
        seed (int | None): Optional seed for reproducible trajectory generation.
        directness (float): Relative preference for shorter, straighter paths, from 0 to 1.

    Returns:
        Tuple[List[Point], List[int]]:
            - List of points representing the trajectory.
            - List of timings (in ms) corresponding to each point.
    """
    if frequency <= 0:
        raise ValueError("frequency must be greater than zero")
    if frequency_randomizer < 0:
        raise ValueError("frequency_randomizer must not be negative")
    if not 0 <= directness <= 1:
        raise ValueError("directness must be between zero and one")
    if target_start == target_end:
        return [target_start], [0]

    rng = np.random.default_rng(seed)

    # Generate a new, non-timed trajectory
    trajectory_points, timings = find_trajectory(target_start, target_end, directness=directness, rng=rng)

    # Normalize timings to start at 0
    timings = [t - timings[0] for t in timings]
    sampled_timings = _sample_timings(timings, frequency, frequency_randomizer, rng)

    sampled_points: list[Point] = []

    # Sample the trajectory using the selected human timing pattern.
    for sample_time in sampled_timings:
        # Find surrounding keyframes for the sample time
        prev_idx = max(i for i, t in enumerate(timings) if t <= sample_time)
        next_idx = min(prev_idx + 1, len(timings) - 1)

        prev_point, prev_time = trajectory_points[prev_idx], timings[prev_idx]
        next_point, next_time = trajectory_points[next_idx], timings[next_idx]

        # Interpolation factor
        alpha = (sample_time - prev_time) / (next_time - prev_time) if next_time != prev_time else 0.0

        # Linear interpolation to get position at jittered time
        point_x = prev_point[0] + alpha * (next_point[0] - prev_point[0])
        point_y = prev_point[1] + alpha * (next_point[1] - prev_point[1])

        # Save interpolated position
        sampled_points.append((point_x, point_y))

    trajectory_length = math.sqrt((target_end[0] - target_start[0]) ** 2 + (target_end[1] - target_start[1]) ** 2)
    # Knot the newly sampled points
    sampled_knotted_points = knot_trajectory(sampled_points, target_start, target_end, rng=rng)
    # Apply jitter to the generated points
    sampled_jittered_points = jitter_trajectory(sampled_knotted_points, trajectory_length, rng=rng)

    # Morph the trajectory to fit exactly between start and end
    dx_tar = target_end[0] - target_start[0]
    dy_tar = target_end[1] - target_start[1]
    len_tar = math.hypot(dx_tar, dy_tar)
    sampled_morphed_points = morph_trajectory(
        sampled_jittered_points,
        target_start,
        target_end,
        dx_tar,
        dy_tar,
        len_tar,
    )
    output_points = [(point[0], point[1]) for point in sampled_morphed_points]
    return output_points, sampled_timings
