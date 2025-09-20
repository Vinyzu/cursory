import math
import random

from .trajectory_selection import (
    Point,
    find_trajectory,
    jitter_trajectory,
    knot_trajectory,
    morph_trajectory,
)


def generate_trajectory(
    target_start: Point,
    target_end: Point,
    frequency: int = 60,
    frequency_randomizer: int = 1,
) -> tuple[list[Point], list[int]]:
    """Generate a realistic mouse trajectory from start to end points.

    Args:
        target_start (Point): Starting point of the trajectory.
        target_end (Point): Ending point of the trajectory.
        frequency (int): Number of samples per second (hz).
        frequency_randomizer (int): Max jitter in ms to apply to each sample time.

    Returns:
        Tuple[List[Point], List[int]]:
            - List of points representing the trajectory.
            - List of timings (in ms) corresponding to each point.
    """
    # Generate a new, non-timed trajectory
    trajectory_points, timings = find_trajectory(target_start, target_end)

    # Normalize timings to start at 0
    timings = [t - timings[0] for t in timings]
    total_time = timings[-1]
    # Milliseconds between samples
    base_step = 1000 // frequency

    sampled_points: list[Point] = []
    sampled_timings: list[int] = []

    # Sample the trajectory at regular intervals with jitter
    current_time = 0
    while current_time <= total_time:
        # Apply jitter to the current sampling time
        jitter_scale = max(1.5, frequency_randomizer)
        jitter = int(random.gauss(0, frequency_randomizer / jitter_scale))
        jitter = max(-frequency_randomizer, min(frequency_randomizer, jitter))
        # Clamp the jittered time within the trajectory duration
        sample_time = max(0, min(total_time, current_time + jitter))

        # Find surrounding keyframes for the jittered time
        prev_idx = max(i for i, t in enumerate(timings) if t <= sample_time)
        next_idx = min(prev_idx + 1, len(timings) - 1)

        prev_point, prev_time = trajectory_points[prev_idx], timings[prev_idx]
        next_point, next_time = trajectory_points[next_idx], timings[next_idx]

        # Interpolation factor
        alpha = (sample_time - prev_time) / (next_time - prev_time) if next_time != prev_time else 0.0

        # Linear interpolation to get position at jittered time
        point_x = prev_point[0] + alpha * (next_point[0] - prev_point[0])
        point_y = prev_point[1] + alpha * (next_point[1] - prev_point[1])

        # Save interpolated position and the actual jittered time
        sampled_points.append((point_x, point_y))
        sampled_timings.append(sample_time)

        # Step forward by the base interval (without jitter here)
        current_time += base_step

    trajectory_length = math.sqrt((target_end[0] - target_start[0]) ** 2 + (target_end[1] - target_start[1]) ** 2)
    # Knot the newly sampled points
    sampled_knotted_points = knot_trajectory(sampled_points, target_start, target_end)
    # Apply jitter to the generated points
    sampled_jittered_points = jitter_trajectory(sampled_knotted_points, trajectory_length)

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
    return sampled_morphed_points, sampled_timings
