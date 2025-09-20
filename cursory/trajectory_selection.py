import gzip
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

# Define type aliases for better readability
Point = tuple[float, float]
Trajectory = dict[str, Any]

# Load Trajectory Data
cursory_directory = Path(__file__).resolve().parent
with open(cursory_directory / "trajectories.json.gz", "rb") as f, gzip.open(f, "rt", encoding="utf-8") as gz:
    LOADED_TRAJECTORIES: list[Trajectory] = json.load(gz)

# Extract trajectory features as numpy arrays for vectorized calculations
TRAJECTORIES_DX = np.array(
    [
        math.sqrt(traj["dx"][0] ** 2 + traj["dx"][1] ** 2) if isinstance(traj["dx"], list) else traj["dx"]
        for traj in LOADED_TRAJECTORIES
    ],
)
TRAJECTORIES_DY = np.array(
    [
        math.sqrt(traj["dy"][0] ** 2 + traj["dy"][1] ** 2) if isinstance(traj["dy"], list) else traj["dy"]
        for traj in LOADED_TRAJECTORIES
    ],
)
TRAJECTORIES_LENGTHS = np.array([traj["length"] for traj in LOADED_TRAJECTORIES])


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
    # Calculate target vector components and length
    dx_tar = target_end[0] - target_start[0]
    dy_tar = target_end[1] - target_start[1]
    len_tar = math.hypot(dx_tar, dy_tar)

    # Handle zero-length target vector case
    if len_tar == 0:
        lengths = np.array([traj["length"] for traj in LOADED_TRAJECTORIES])
        # Get top N shortest trajectories
        sorted_indices = np.argsort(lengths)[:top_n]
        return [LOADED_TRAJECTORIES[i] for i in sorted_indices]

    # Normalize target vector to get direction
    norm_dx_tar = dx_tar / len_tar
    norm_dy_tar = dy_tar / len_tar

    # Calculate normalized directions for trajectories, handling zero-length trajectories
    norm_dx = np.where(TRAJECTORIES_LENGTHS != 0, TRAJECTORIES_DX / TRAJECTORIES_LENGTHS, 0)
    norm_dy = np.where(TRAJECTORIES_LENGTHS != 0, TRAJECTORIES_DY / TRAJECTORIES_LENGTHS, 0)

    # Compute cosine similarity of directions: 1 for same, -1 for opposite, 0 for perpendicular
    direction_similarity = norm_dx * norm_dx_tar + norm_dy * norm_dy_tar
    # Convert similarity to distance (0 is best)
    direction_distance = 1 - direction_similarity

    # Calculate the ratio of length difference to the target length (normalized length difference)
    length_diff_ratio = np.abs(TRAJECTORIES_LENGTHS - len_tar) / np.maximum(len_tar, 1)

    # Combine direction distance and length difference ratio to get a combined score
    combined_score = (direction_weight * direction_distance) + (length_weight * length_diff_ratio)

    # Get indices of trajectories with the lowest combined scores (best matches)
    sorted_indices = np.argsort(combined_score)[:top_n]
    return [LOADED_TRAJECTORIES[i] for i in sorted_indices]


def find_closest_trajectory(
    target_start: Point,
    target_end: Point,
    num_nearest_to_sample: int = 5,
    random_sample_iterations: int = 20,
    length_preference_power: int = 2,
) -> tuple[Trajectory, float, float, float]:
    """Find the closest trajectory to the target start and end points, with a preference for shorter trajectories.
    Randomly selects from top candidates, heavily biased towards shorter trajectories through weighted random sampling.

    Args:
        target_start (Point): (x, y) coordinates of the target start point.
        target_end (Point): (x, y) coordinates of the target end point.
        num_nearest_to_sample (int, optional): Number of nearest trajectories to consider initially. Defaults to 5.
        random_sample_iterations (int, optional): Number of random perturbations to refine selection. Defaults to 20.
        length_preference_power (int, optional): Power to raise inverse length for selection bias. Default: 2.

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

    # Initial selection of top trajectories
    top_trajectories = find_nearest_trajectory(target_start, target_end, top_n=num_nearest_to_sample)

    # Iteratively refine top trajectories by perturbing target end and re-querying
    for _ in range(random_sample_iterations):
        perturbed_target_end = (
            target_end[0] + random.uniform(-len_tar * 0.1, len_tar * 0.1),
            target_end[1] + random.uniform(-len_tar * 0.1, len_tar * 0.1),
        )
        top_trajectories.extend(
            find_nearest_trajectory(target_start, perturbed_target_end, top_n=num_nearest_to_sample),
        )

    # Extract lengths of top trajectories
    top_lengths = np.array([traj["length"] for traj in top_trajectories])

    # Calculate weights based on inverse length, powered to strongly prefer shorter trajectories
    epsilon = 1e-10  # Small value to avoid division by zero
    inverse_lengths = 1.0 / ((top_lengths + epsilon) ** length_preference_power)
    weights = inverse_lengths / np.sum(inverse_lengths)  # Normalize weights to sum to 1

    # Randomly select a trajectory index based on calculated weights (biased towards shorter trajectories)
    rng = np.random.default_rng()
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
    cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
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


def jitter_trajectory(
    points: list[Point],
    trajectory_length: float,
    scale: float = 0.01,
) -> list[Point]:
    """Applies a smooth jitter to the trajectory points, scaling jitter based on distance and trajectory length.

    Args:
        points (List[Point]): List of trajectory points as (x, y) tuples.
        trajectory_length (float): The length of the trajectory, used for adaptive scaling of jitter.
        scale (float, optional): Base scale factor for the jitter effect. Defaults to 1.0.

    Returns:
        List[Point]: A new list of jittered trajectory points.
    """
    # Convert points to numpy array
    points_np = np.array(points)
    # Scale jitter based on trajectory length
    length_scale = min(1.0, trajectory_length / 400)

    # Initialize jittered points array
    jittered_points = np.zeros_like(points_np)
    jittered_points[0] = points_np[0]
    direction = 1

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

    # Generate jitter values using uniform distribution and adaptive scales
    rng = np.random.default_rng()
    jitter_values = rng.uniform(0.5, 1, size=(len(points), 2)) * adaptive_scales[:, np.newaxis]
    directions = np.array([direction if i % 2 == 0 else -direction for i in range(len(points))])
    jitter_x = jitter_values[:, 0] * directions
    jitter_y = jitter_values[:, 1] * directions

    # Apply jitter to original points
    jittered_points = points_np + np.stack([jitter_x, jitter_y], axis=-1)
    jittered_points_list: list[Point] = jittered_points.tolist()
    return jittered_points_list


def generate_middle_biased_point(x1: float, y1: float, x2: float, y2: float, bias_factor: float = 2.0) -> Point:
    """Generates a random point within a rectangle defined by two corner points,
    with a Gaussian bias towards the center of the rectangle.

    Args:
        x1 (float): x-coordinate of the first corner of the rectangle.
        y1 (float): y-coordinate of the first corner of the rectangle.
        x2 (float): x-coordinate of the second corner of the rectangle.
        y2 (float): y-coordinate of the second corner of the rectangle.
        bias_factor (float, optional): Factor controlling bias strength. Defaults to 2.0.

    Returns:
        Point: A tuple (x, y) representing the generated point, biased towards the center.
    """
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
    offset_x = random.gauss(0, width / (2 * bias_factor))
    offset_y = random.gauss(0, height / (2 * bias_factor))

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
) -> list[Point]:
    """Introduces "knots" into a trajectory by displacing points towards randomly generated biased points.

    Args:
        traj_points (List[Point]): List of trajectory points to be knotted.
        target_start (Point): Target start point of the trajectory (used to define knot generation area).
        target_end (Point): Target end point of the trajectory (used to define knot generation area).
        num_knots (int, optional): Number of knot points to generate and apply. Defaults to 3.
        knot_strength (float, optional): Scaling factor to control the intensity of knotting. Defaults to 0.5.

    Returns:
        List[Point]: List of knotted trajectory points.
    """
    # Convert trajectory points to numpy array
    traj_points_np = np.array(traj_points)
    knot_points = np.zeros_like(traj_points_np, dtype=float)

    for _ in range(num_knots):
        # Generate a biased point within the bounding box of target start and end
        knot = generate_middle_biased_point(target_start[0], target_start[1], target_end[0], target_end[1])
        knot_np = np.array(knot)

        # Calculate vectors from each trajectory point to the knot point
        diff_vectors = knot_np - traj_points_np
        distances = np.linalg.norm(diff_vectors, axis=1)
        max_distance = np.max(distances)

        # Skip adjustment if max distance is too small (avoid division issues)
        if max_distance < 1e-6:
            continue

        # Normalize difference vectors and scale by proximity and knot strength
        proximity = 1.0 - (distances / max_distance)
        scaling_factor = proximity * knot_strength
        knot_offsets = diff_vectors * scaling_factor[:, np.newaxis]
        knot_points += knot_offsets

    # Apply square-root scaling to knot displacements for smoother effect
    def sqrt_anything_np(x):  # type: ignore[no-untyped-def] # noqa: ANN202,ANN001
        # Element-wise square root with sign preservation
        return np.copysign(np.sqrt(np.abs(x)), x)

    # Apply displacements to trajectory points
    knotted_points = traj_points_np + sqrt_anything_np(knot_points)  # type: ignore[no-untyped-call]
    knotted_points_list: list[Point] = knotted_points.tolist()
    return knotted_points_list


def find_trajectory(target_start: Point, target_end: Point) -> tuple[list[Point], list[int]]:
    """Generates a new trajectory by selecting, morphing, jittering, and knotting a base trajectory.

    Args:
        target_start (Point): The desired starting point for the generated trajectory.
        target_end (Point): The desired ending point for the generated trajectory.

    Returns:
        Tuple[List[Point], List[Point]]: A tuple containing:
            - Original points of the selected base trajectory.
            - Knotted points of the modified trajectory, after morphing and jittering.
    """
    # Find the closest base trajectory to the target start and end points
    selected_trajectory, dx_tar, dy_tar, len_tar = find_closest_trajectory(target_start, target_end)

    # Apply jitter to the morphed trajectory to add variation
    jittered_points = jitter_trajectory(selected_trajectory["points"], len_tar)
    # Apply knotting to the jittered trajectory to further modify its shape
    knotted_points = knot_trajectory(jittered_points, target_start, target_end)
    # Final morphing to ensure exact fit between target start and end points
    morphed_points = morph_trajectory(knotted_points, target_start, target_end, dx_tar, dy_tar, len_tar)
    return morphed_points, selected_trajectory["timing"]
