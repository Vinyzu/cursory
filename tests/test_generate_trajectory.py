import math
from itertools import pairwise

import cursory.cursory as cursory_module
import numpy as np
import pytest
from cursory import generate_trajectory, trajectory_selection
from cursory.trajectory_selection import (
    LOADED_TRAJECTORIES,
    TRAJECTORIES_DISPLACEMENTS,
    TRAJECTORIES_DX,
    TRAJECTORIES_DY,
    find_closest_trajectory,
    jitter_trajectory,
)

BASE_POINTS = [(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]
BASE_TIMINGS = [0, 100, 200]


@pytest.fixture
def _patch_trajectory(monkeypatch):
    monkeypatch.setattr(cursory_module, "find_trajectory", lambda *_, **__: (BASE_POINTS, BASE_TIMINGS))
    monkeypatch.setattr(cursory_module, "knot_trajectory", lambda points, *_, **__: points)
    monkeypatch.setattr(cursory_module, "jitter_trajectory", lambda points, *_, **__: points)
    monkeypatch.setattr(cursory_module, "morph_trajectory", lambda points, *_, **__: points)


@pytest.mark.usefixtures("_patch_trajectory")
def test_high_frequency_still_generates_samples():
    points, timings = generate_trajectory((0.0, 0.0), (10.0, 10.0), frequency=2000, seed=1)

    assert len(points) == len(timings) == 401
    assert timings == sorted(timings)
    assert any(first == second for first, second in pairwise(points))


@pytest.mark.usefixtures("_patch_trajectory")
def test_timings_never_go_backwards_under_large_jitter():
    _points, timings = generate_trajectory(
        (0.0, 0.0),
        (10.0, 10.0),
        frequency=1000,
        frequency_randomizer=10,
        seed=1,
    )

    assert timings[0] == 0
    assert timings[-1] == BASE_TIMINGS[-1]
    assert timings == sorted(timings)


@pytest.mark.usefixtures("_patch_trajectory")
def test_resampling_does_not_force_points_onto_half_pixel_grid():
    points, _timings = generate_trajectory(
        (0.0, 0.0),
        (10.0, 10.0),
        frequency=70,
        frequency_randomizer=0,
        seed=1,
    )

    assert any(coordinate * 2 != round(coordinate * 2) for point in points[1:-1] for coordinate in point)


@pytest.mark.parametrize("directness", [-0.01, 1.01])
def test_directness_must_be_between_zero_and_one(directness):
    with pytest.raises(ValueError, match="directness"):
        generate_trajectory((0.0, 0.0), (10.0, 10.0), directness=directness)


def test_seed_reproduces_trajectory_and_preserves_endpoints():
    start = (0.1, 0.1)
    end = (258.2, 100.3)
    first = generate_trajectory(start, end, frequency=100, seed=42)
    second = generate_trajectory(start, end, frequency=100, seed=42)

    assert first == second
    assert first[0][0] == start
    assert first[0][-1] == end


def test_zero_distance_returns_one_stationary_point(monkeypatch):
    def fail_selection(*_args, **_kwargs):
        raise AssertionError("selection called for a stationary trajectory")

    monkeypatch.setattr(cursory_module, "find_trajectory", fail_selection)

    assert generate_trajectory((100.0, 100.0), (100.0, 100.0)) == ([(100.0, 100.0)], [0])


def test_duplicate_samples_keep_their_timing(monkeypatch):
    source_points = [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]
    source_timings = [0, 10, 20]
    monkeypatch.setattr(cursory_module, "find_trajectory", lambda *_, **__: (source_points, source_timings))
    monkeypatch.setattr(cursory_module, "knot_trajectory", lambda points, *_, **__: points)
    monkeypatch.setattr(cursory_module, "jitter_trajectory", lambda points, *_, **__: points)
    monkeypatch.setattr(cursory_module, "morph_trajectory", lambda points, *_, **__: points)

    assert generate_trajectory(
        (0.0, 0.0),
        (1.0, 1.0),
        frequency=100,
        frequency_randomizer=0,
        seed=1,
    ) == (source_points, source_timings)


@pytest.mark.parametrize(("distance", "duration"), [(25.0, 250), (200.0, 500)])
def test_duration_scales_only_for_shorter_movements(monkeypatch, distance, duration):
    recorded = {
        "points": [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)],
        "timing": [100, 350, 600],
        "length": 100.0,
    }

    def select(target_start, target_end, directness=0.65, rng=None):
        del directness, rng
        dx = target_end[0] - target_start[0]
        dy = target_end[1] - target_start[1]
        return recorded, dx, dy, math.hypot(dx, dy)

    monkeypatch.setattr(trajectory_selection, "find_closest_trajectory", select)
    _points, timings = trajectory_selection.find_trajectory((0.0, 0.0), (distance, 0.0))

    assert timings[-1] - timings[0] == duration


def test_selection_uses_endpoint_displacement():
    expected_dx = np.array(
        [trajectory["points"][-1][0] - trajectory["points"][0][0] for trajectory in LOADED_TRAJECTORIES],
    )
    expected_dy = np.array(
        [trajectory["points"][-1][1] - trajectory["points"][0][1] for trajectory in LOADED_TRAJECTORIES],
    )

    np.testing.assert_array_equal(TRAJECTORIES_DX, expected_dx)
    np.testing.assert_array_equal(TRAJECTORIES_DY, expected_dy)
    np.testing.assert_allclose(TRAJECTORIES_DISPLACEMENTS, np.hypot(expected_dx, expected_dy))


def test_directness_shifts_selection_without_excluding_extremes(monkeypatch):
    candidates = [
        {"points": [(0.0, 0.0), (100.0, 0.0)], "timing": [0, 10], "length": 100.01},
        {"points": [(0.0, 0.0), (90.0, 20.0), (100.0, 0.0)], "timing": [0, 5, 10], "length": 112.0},
        {"points": [(0.0, 0.0), (0.0, 100.0), (100.0, 0.0)], "timing": [0, 5, 10], "length": 241.0},
    ]
    monkeypatch.setattr(trajectory_selection, "find_nearest_trajectory", lambda *_args, **_kwargs: candidates)

    low_selections = []
    high_selections = []
    default_selections = set()
    for seed in range(300):
        for selections, directness in ((low_selections, 0.0), (high_selections, 1.0)):
            trajectory, *_ = find_closest_trajectory(
                (0.0, 0.0),
                (100.0, 0.0),
                random_sample_iterations=0,
                directness=directness,
                rng=np.random.default_rng(seed),
            )
            selections.append(candidates.index(trajectory))
        trajectory, *_ = find_closest_trajectory(
            (0.0, 0.0),
            (100.0, 0.0),
            random_sample_iterations=0,
            rng=np.random.default_rng(seed),
        )
        default_selections.add(candidates.index(trajectory))

    assert np.mean(high_selections) < np.mean(low_selections)
    assert default_selections == {0, 1, 2}


def test_jitter_is_lateral_and_not_forced_to_alternate():
    points = [(float(index), 0.0) for index in range(20)]
    jittered = np.asarray(jitter_trajectory(points, 400.0, scale=1.0, rng=np.random.default_rng(1)))

    np.testing.assert_allclose(jittered[:, 0], np.arange(20))
    assert np.any(jittered[1:, 1] * jittered[:-1, 1] > 0)
