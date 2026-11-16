import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Define type aliases for better readability
Point = tuple[float, float]
Trajectory = dict[str, Any]

# Load Trajectory Data
cursory_directory = Path(__file__).resolve().parent
with open(cursory_directory / "trajectories.json.gz", "rb") as f, gzip.open(f, "rt", encoding="utf-8") as gz:
    LOADED_TRAJECTORIES: list[Trajectory] = json.load(gz)

# Extract trajectory features as numpy arrays for vectorized calculations
TRAJECTORIES_DX = np.array(
    [traj["points"][-1][0] - traj["points"][0][0] for traj in LOADED_TRAJECTORIES],
)
TRAJECTORIES_DY = np.array(
    [traj["points"][-1][1] - traj["points"][0][1] for traj in LOADED_TRAJECTORIES],
)
TRAJECTORIES_DISPLACEMENTS = np.hypot(TRAJECTORIES_DX, TRAJECTORIES_DY)
_TRAJECTORIES_DISPLACEMENTS_F32 = TRAJECTORIES_DISPLACEMENTS.astype(np.float32)
TRAJECTORIES_LENGTHS = np.array([traj["length"] for traj in LOADED_TRAJECTORIES])
TRAJECTORIES_EFFICIENCIES = np.divide(
    TRAJECTORIES_DISPLACEMENTS,
    TRAJECTORIES_LENGTHS,
    out=np.ones_like(TRAJECTORIES_DISPLACEMENTS),
    where=TRAJECTORIES_LENGTHS > 0,
)
_SORTED_TRAJECTORY_EFFICIENCIES = np.sort(TRAJECTORIES_EFFICIENCIES)
# Normalized directions of the trajectories, handling zero-length trajectories
_TRAJECTORIES_NORM_DX = np.divide(
    TRAJECTORIES_DX,
    TRAJECTORIES_DISPLACEMENTS,
    out=np.zeros_like(TRAJECTORIES_DX, dtype=float),
    where=TRAJECTORIES_DISPLACEMENTS != 0,
).astype(np.float32)
_TRAJECTORIES_NORM_DY = np.divide(
    TRAJECTORIES_DY,
    TRAJECTORIES_DISPLACEMENTS,
    out=np.zeros_like(TRAJECTORIES_DY, dtype=float),
    where=TRAJECTORIES_DISPLACEMENTS != 0,
).astype(np.float32)


def find_nearest_trajectory(
    target_start: Point,
    target_end: Point,
    direction_weight: float = 0.8,
    length_weight: float = 0.2,
    top_n: int = 5,
) -> list[Trajectory]:
    """Find the top N nearest trajectories to the target start and end points based on direction and length similarity.

    Args:
        target_start (Point): (x, y) coordinates of the target start point.
        target_end (Point): (x, y) coordinates of the target end point.
        direction_weight (float, optional): Weight for direction similarity in the combined score. Defaults to 0.5.
        length_weight (float, optional): Weight for length similarity in the combined score. Defaults to 0.5.
        top_n (int, optional): Number of top nearest trajectories to return. Defaults to 5.

    Returns:
        List[Trajectory]: A list of the top N nearest trajectories based on the combined score.
    """
    return find_nearest_trajectories(target_start, [target_end], direction_weight, length_weight, top_n)


def find_nearest_trajectories(
    target_start: Point,
    target_ends: list[Point],
    direction_weight: float = 0.8,
    length_weight: float = 0.2,
    top_n: int = 5,
    prefilter_size: int | None = None,
) -> list[Trajectory]:
    """Vectorized find_nearest_trajectory for several target ends at once.

    Args:
        target_start (Point): (x, y) coordinates of the target start point.
        target_ends (List[Point]): (x, y) coordinates of the target end points.
        direction_weight (float, optional): Weight for direction similarity in the combined score. Defaults to 0.8.
        length_weight (float, optional): Weight for length similarity in the combined score. Defaults to 0.2.
        top_n (int, optional): Number of top nearest trajectories to return per target end. Defaults to 5.
        prefilter_size (int | None, optional): If set, only the first target end is scored against all trajectories;
            the other target ends are scored against its prefilter_size best matches. This is much faster and
            almost always gives the same result when the other target ends lie close to the first one.

    Returns:
        List[Trajectory]: The top N nearest trajectories of each target end, concatenated in target end order.
    """
    if prefilter_size is None or len(target_ends) == 1 or not top_n < prefilter_size < len(LOADED_TRAJECTORIES):
        scores, zero_length = _trajectory_scores(target_start, target_ends, direction_weight, length_weight)
        sorted_indices = _lowest_score_indices(scores, top_n)
    else:
        first_scores, first_zero_length = _trajectory_scores(
            target_start, target_ends[:1], direction_weight, length_weight
        )
        subset = np.sort(np.argpartition(first_scores[0], prefilter_size - 1)[:prefilter_size])
        rest_scores, rest_zero_length = _trajectory_scores(
            target_start, target_ends[1:], direction_weight, length_weight, subset
        )
        sorted_indices = np.vstack(
            (_lowest_score_indices(first_scores, top_n), subset[_lowest_score_indices(rest_scores, top_n)])
        )
        zero_length = np.concatenate((first_zero_length, rest_zero_length))

    # Handle zero-length target vector case: Get top N shortest trajectories
    if zero_length.any():
        sorted_indices[zero_length] = np.argsort(TRAJECTORIES_LENGTHS)[:top_n]
    return [LOADED_TRAJECTORIES[i] for i in sorted_indices.ravel().tolist()]


def _trajectory_scores(
    target_start: Point,
    target_ends: list[Point],
    direction_weight: float,
    length_weight: float,
    subset: NDArray[np.intp] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Score trajectories (all, or only those in subset) against each target end. Lower is better."""
    norm_dx = _TRAJECTORIES_NORM_DX if subset is None else _TRAJECTORIES_NORM_DX[subset]
    norm_dy = _TRAJECTORIES_NORM_DY if subset is None else _TRAJECTORIES_NORM_DY[subset]
    displacements = _TRAJECTORIES_DISPLACEMENTS_F32 if subset is None else _TRAJECTORIES_DISPLACEMENTS_F32[subset]

    # Calculate target vector components and length
    dx_tar = np.array([target_end[0] - target_start[0] for target_end in target_ends], dtype=float)
    dy_tar = np.array([target_end[1] - target_start[1] for target_end in target_ends], dtype=float)
    len_tar = np.array([math.hypot(dx, dy) for dx, dy in zip(dx_tar.tolist(), dy_tar.tolist(), strict=True)])
    zero_length = len_tar == 0
    safe_len_tar = np.where(zero_length, 1.0, len_tar)

    # Normalize target vector to get direction
    norm_dx_tar = (dx_tar / safe_len_tar).astype(np.float32)[:, np.newaxis]
    norm_dy_tar = (dy_tar / safe_len_tar).astype(np.float32)[:, np.newaxis]

    # The score is computed in place in two buffers to avoid allocating a temporary array per operation.
    # Compute cosine similarity of directions: 1 for same, -1 for opposite, 0 for perpendicular
    combined_score = norm_dx * norm_dx_tar
    buffer = np.multiply(norm_dy, norm_dy_tar)
    combined_score += buffer
    # Convert similarity to distance (0 is best)
    np.subtract(1, combined_score, out=combined_score)

    # Calculate the ratio of length difference to the target length (normalized length difference)
    np.subtract(displacements, len_tar.astype(np.float32)[:, np.newaxis], out=buffer)
    np.abs(buffer, out=buffer)
    buffer /= np.maximum(len_tar, 1).astype(np.float32)[:, np.newaxis]

    # Combine direction distance and length difference ratio to get a combined score
    combined_score *= direction_weight
    buffer *= length_weight
    combined_score += buffer
    return combined_score, zero_length


def _lowest_score_indices(scores: NDArray[np.float32], top_n: int) -> NDArray[np.intp]:
    """Column indices of the top N lowest scores (best matches) of each row, sorted by score."""
    if top_n < scores.shape[1]:
        candidate_indices = np.argpartition(scores, top_n - 1, axis=1)[:, :top_n]
        candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)
        return np.take_along_axis(candidate_indices, np.argsort(candidate_scores, axis=1), axis=1)
    return np.argsort(scores, axis=1)[:, :top_n]


def find_closest_trajectory(
    target_start: Point,
    target_end: Point,
    num_nearest_to_sample: int = 5,
    random_sample_iterations: int = 20,
    directness: float = 0.65,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[Trajectory, float, float, float]:
    """Find a direction-, distance-, and path-efficiency-appropriate trajectory.

    Randomly selects from the nearest candidates, with a soft preference for the
    requested path directness. Unusually direct and indirect paths remain possible.

    Args:
        target_start (Point): (x, y) coordinates of the target start point.
        target_end (Point): (x, y) coordinates of the target end point.
        num_nearest_to_sample (int, optional): Number of nearest trajectories to consider initially. Defaults to 5.
        random_sample_iterations (int, optional): Number of random perturbations to refine selection. Defaults to 20.
        directness (float, optional): Relative preference for shorter, straighter paths, from 0 to 1. Default: 0.65.
        rng (np.random.Generator | None): Random number generator to use.

    Returns:
        Tuple[Trajectory, float, float, float]: A tuple containing:
            - The selected closest trajectory (Trajectory dictionary).
            - Target x-direction difference (dx_tar).
            - Target y-direction difference (dy_tar).
            - Target trajectory length (len_tar).
    """
    dx_tar = target_end[0] - target_start[0]
    dy_tar = target_end[1] - target_start[1]
    len_tar = math.hypot(dx_tar, dy_tar)
    if not 0 <= directness <= 1:
        raise ValueError("directness must be between zero and one")
    if rng is None:
        rng = np.random.default_rng()

    # Select top trajectories for the target end and for randomly perturbed target ends
    perturbations = rng.uniform(-len_tar * 0.1, len_tar * 0.1, size=(random_sample_iterations, 2)).tolist()
    target_ends = [target_end]
    target_ends += [(target_end[0] + offset_x, target_end[1] + offset_y) for offset_x, offset_y in perturbations]
    # The perturbed target ends lie close to the target end, so their nearest trajectories are (almost always)
    # among the nearest trajectories of the target end. Prefiltering to those makes the search much faster.
    top_trajectories = find_nearest_trajectories(
        target_start,
        target_ends,
        top_n=num_nearest_to_sample,
        prefilter_size=128,
    )

    # Compare path length with endpoint displacement so unusually straight and
    # unusually indirect recordings do not dominate generated paths.
    top_lengths = np.array([traj["length"] for traj in top_trajectories])
    top_displacements = np.array(
        [
            math.hypot(
                traj["points"][-1][0] - traj["points"][0][0],
                traj["points"][-1][1] - traj["points"][0][1],
            )
            for traj in top_trajectories
        ],
    )
    top_efficiencies = np.divide(
        top_displacements,
        top_lengths,
        out=np.ones_like(top_displacements),
        where=top_lengths > 0,
    )
    # Rank efficiency against all source recordings so the preference has stable
    # semantics across directions and distances. The small tail lift restores
    # some very direct and indirect paths without making either the default.
    efficiency_ranks = np.searchsorted(
        _SORTED_TRAJECTORY_EFFICIENCIES,
        top_efficiencies,
        side="right",
    ) / len(_SORTED_TRAJECTORY_EFFICIENCIES)
    preferred_rank = 0.2 + (0.75 * directness)
    core_weights = np.exp(-0.5 * ((efficiency_ranks - preferred_rank) / 0.2) ** 2)
    centered_ranks = (2 * efficiency_ranks) - 1
    tail_floor = 0.05 * (1 + (0.1 * np.abs(centered_ranks) ** 3))
    weights = core_weights + tail_floor
    weights /= np.sum(weights)

    # Randomly select a trajectory index based on calculated weights (biased towards shorter trajectories)
    selected_trajectory_index = rng.choice(len(top_trajectories), p=weights)
    selected_trajectory = top_trajectories[selected_trajectory_index]

    return selected_trajectory, dx_tar, dy_tar, len_tar


def morph_trajectory(
    points: list[Point],
    target_start: Point,
    target_end: Point,
    dx_tar: float,
    dy_tar: float,
    len_tar: float,
) -> list[Point]:
    """Morph a given trajectory to match the target start and end points by scaling, rotating, and translating.

    Args:
        points (List[Point]): The list of points representing the trajectory.
        target_start (Point): (x, y) coordinates of the target start point.
        target_end (Point): (x, y) coordinates of the target end point.
        dx_tar (float): Target x-direction displacement.
        dy_tar (float): Target y-direction displacement.
        len_tar (float): Target trajectory length.

    Returns:
        List[Point]: Morphed points of the trajectory, transformed to fit target start and end points.
    """
    # Convert points to numpy array
    original_points = np.array(points)
    start = original_points[0]
    end = original_points[-1]

    # Calculate original trajectory's displacement and length
    dx_orig, dy_orig = end[0] - start[0], end[1] - start[1]
    len_orig = np.linalg.norm([dx_orig, dy_orig])

    # Calculate scaling factor based on target and original lengths
    scale_factor = len_tar / len_orig if len_orig != 0 else 1.0

    # Calculate rotation angle needed to align original trajectory direction with target direction
    angle_orig = np.arctan2(dy_orig, dx_orig)
    angle_tar = np.arctan2(dy_tar, dx_tar)
    rotation_angle = angle_tar - angle_orig

    # Create rotation matrix using pre-calculated cosine and sine for efficiency
    cos_a, sin_a = math.cos(rotation_angle), math.sin(rotation_angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Apply transformation: Scale, Rotate, Translate
    morphed_points = (original_points - start) * scale_factor
    morphed_points = morphed_points @ rotation_matrix.T
    morphed_points += np.array(target_start)

    # Ensure exact start and end point matching
    morphed_points[0] = np.array(target_start)
    morphed_points[-1] = np.array(target_end)

    morphed_points_list: list[Point] = morphed_points.tolist()
    return morphed_points_list


def _endpoint_weights(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fade added displacement to zero with zero slope at both ends of the path."""
    progress = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if progress[-1] == 0:
        return np.zeros(len(points))
    progress /= progress[-1]
    return (4 * progress * (1 - progress)) ** 2


def jitter_trajectory(
    points: list[Point],
    trajectory_length: float,
    scale: float = 0.01,
    rng: np.random.Generator | None = None,
) -> list[Point]:
    """Applies a smooth jitter to the trajectory points, scaling jitter based on distance and trajectory length.

    Args:
        points (List[Point]): List of trajectory points as (x, y) tuples.
        trajectory_length (float): The length of the trajectory, used for adaptive scaling of jitter.
        scale (float, optional): Base scale factor for the jitter effect. Defaults to 1.0.
        rng (np.random.Generator | None): Random number generator to use.

    Returns:
        List[Point]: A new list of jittered trajectory points.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert points to numpy array
    points_np = np.array(points, dtype=float)
    # Scale jitter based on trajectory length
    length_scale = min(1.0, trajectory_length / 400)

    # Calculate distances between consecutive points to adapt jitter scale
    distances_prev = np.zeros(len(points))
    distances_next = np.zeros(len(points))

    # Calculate distances between consecutive points
    diffs = np.diff(points_np, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))

    # Assign distances to previous and next arrays for averaging
    distances_prev[1:] = distances
    distances_next[:-1] = distances

    # Average distances to determine adaptive jitter scale
    avg_distances = (distances_prev + distances_next) / 2
    adaptive_scales = scale * (avg_distances / np.maximum(avg_distances, 1)) * length_scale
    adaptive_scales *= _endpoint_weights(points_np)

    # Use correlated lateral drift instead of forcing every sample to reverse direction.
    tangents = np.empty_like(points_np)
    tangents[0] = points_np[min(1, len(points_np) - 1)] - points_np[0]
    tangents[-1] = points_np[-1] - points_np[max(0, len(points_np) - 2)]
    if len(points_np) > 2:
        tangents[1:-1] = points_np[2:] - points_np[:-2]
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    normals = np.zeros_like(tangents)
    moving = tangent_lengths > np.finfo(float).eps
    normals[moving, 0] = -tangents[moving, 1] / tangent_lengths[moving]
    normals[moving, 1] = tangents[moving, 0] / tangent_lengths[moving]

    innovations = rng.normal(size=len(points)).tolist()
    lateral_offsets = [innovations[0]]
    correlation = 0.75
    innovation_scale = math.sqrt(1 - correlation**2)
    for innovation in innovations[1:]:
        lateral_offsets.append(correlation * lateral_offsets[-1] + innovation_scale * innovation)

    jittered_points = points_np + normals * (np.array(lateral_offsets) * adaptive_scales)[:, np.newaxis]
    jittered_points_list: list[Point] = jittered_points.tolist()
    return jittered_points_list


def generate_middle_biased_point(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    bias_factor: float = 2.0,
    *,
    rng: np.random.Generator | None = None,
) -> Point:
    """Generates a random point within a rectangle defined by two corner points,
    with a Gaussian bias towards the center of the rectangle.

    Args:
        x1 (float): x-coordinate of the first corner of the rectangle.
        y1 (float): y-coordinate of the first corner of the rectangle.
        x2 (float): x-coordinate of the second corner of the rectangle.
        y2 (float): y-coordinate of the second corner of the rectangle.
        bias_factor (float, optional): Factor controlling bias strength. Defaults to 2.0.
        rng (np.random.Generator | None): Random number generator to use.

    Returns:
        Point: A tuple (x, y) representing the generated point, biased towards the center.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine rectangle bounds
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)

    # Calculate center and dimensions of the rectangle
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    # Generate Gaussian offsets centered at 0, scaled by rectangle dimensions and bias factor
    offset_x = rng.normal(0, width / (2 * bias_factor))
    offset_y = rng.normal(0, height / (2 * bias_factor))

    # Calculate final point coordinates, clamped to stay within the rectangle bounds
    x = max(min_x, min(max_x, center_x + offset_x))
    y = max(min_y, min(max_y, center_y + offset_y))

    return x, y


def knot_trajectory(
    traj_points: list[Point],
    target_start: Point,
    target_end: Point,
    num_knots: int = 5,
    knot_strength: float = 0.15,
    *,
    rng: np.random.Generator | None = None,
) -> list[Point]:
    """Introduces "knots" into a trajectory by displacing points towards randomly generated biased points.

    Args:
        traj_points (List[Point]): List of trajectory points to be knotted.
        target_start (Point): Target start point of the trajectory (used to define knot generation area).
        target_end (Point): Target end point of the trajectory (used to define knot generation area).
        num_knots (int, optional): Number of knot points to generate and apply. Defaults to 3.
        knot_strength (float, optional): Scaling factor to control the intensity of knotting. Defaults to 0.5.
        rng (np.random.Generator | None): Random number generator to use.

    Returns:
        List[Point]: List of knotted trajectory points.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert trajectory points to numpy array
    traj_points_np = np.array(traj_points, dtype=float)
    knot_points = np.zeros_like(traj_points_np, dtype=float)

    # Generate biased points within the bounding box of target start and end
    knots = np.array(
        [
            generate_middle_biased_point(target_start[0], target_start[1], target_end[0], target_end[1], rng=rng)
            for _ in range(num_knots)
        ],
        dtype=float,
    ).reshape(-1, 2)

    # Calculate vectors from each trajectory point to each knot point
    diff_vectors = knots[:, np.newaxis, :] - traj_points_np
    distances = np.linalg.norm(diff_vectors, axis=2)
    max_distances = np.max(distances, axis=1, initial=0.0)

    # Skip knots whose max distance is too small (avoid division issues)
    active = max_distances >= 1e-6
    if active.any():
        # Normalize difference vectors and scale by proximity and knot strength
        proximity = 1.0 - (distances[active] / max_distances[active, np.newaxis])
        scaling_factor = proximity * knot_strength
        knot_offsets = diff_vectors[active] * scaling_factor[:, :, np.newaxis]
        knot_points += np.sum(knot_offsets, axis=0)

    # Keep knot displacements small and fade them out at the endpoints.
    offsets = np.copysign(np.sqrt(np.abs(knot_points)), knot_points)
    offsets *= _endpoint_weights(traj_points_np)[:, np.newaxis]
    knotted_points = traj_points_np + offsets
    knotted_points_list: list[Point] = knotted_points.tolist()
    return knotted_points_list


def find_trajectory(
    target_start: Point,
    target_end: Point,
    directness: float = 0.65,
    rng: np.random.Generator | None = None,
) -> tuple[list[Point], list[int | float]]:
    """Generates a new trajectory by selecting, morphing, jittering, and knotting a base trajectory.

    Args:
        target_start (Point): The desired starting point for the generated trajectory.
        target_end (Point): The desired ending point for the generated trajectory.
        directness (float): Relative preference for shorter, straighter paths, from 0 to 1.
        rng (np.random.Generator | None): Random number generator to use.

    Returns:
        Tuple[List[Point], List[Point]]: A tuple containing:
            - Original points of the selected base trajectory.
            - Knotted points of the modified trajectory, after morphing and jittering.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Find the closest base trajectory to the target start and end points
    selected_trajectory, dx_tar, dy_tar, len_tar = find_closest_trajectory(
        target_start,
        target_end,
        directness=directness,
        rng=rng,
    )

    # Shorten the recorded duration when morphing a trajectory to a shorter distance.
    # Square-root scaling keeps very short movements representable at typical sampling frequencies.
    recorded_length = selected_trajectory["length"]
    timings: list[int | float] = list(selected_trajectory["timing"])
    if 0 < len_tar < recorded_length:
        time_scale = math.sqrt(len_tar / recorded_length)
        start_time = timings[0]
        timings = [start_time + ((timing - start_time) * time_scale) for timing in timings]

    # Apply noise in target coordinates so source screen position cannot affect the shape.
    morphed_points = morph_trajectory(selected_trajectory["points"], target_start, target_end, dx_tar, dy_tar, len_tar)
    jittered_points = jitter_trajectory(morphed_points, len_tar, rng=rng)
    # Apply knotting to the jittered trajectory to further modify its shape
    knotted_points = knot_trajectory(jittered_points, target_start, target_end, rng=rng)
    return knotted_points, timings
