import pytest
import cursory.cursory as cursory_module
from cursory.cursory import generate_trajectory
import itertools

BASE_POINTS = [(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]
BASE_TIMINGS = [0, 100, 200]


@pytest.fixture(autouse=True)
def patch_trajectory(monkeypatch):
    monkeypatch.setattr(cursory_module, "find_trajectory", lambda *_, **__: (BASE_POINTS, BASE_TIMINGS))
    monkeypatch.setattr(cursory_module, "knot_trajectory", lambda points, *_, **__: points)
    monkeypatch.setattr(cursory_module, "jitter_trajectory", lambda points, *_, **__: points)
    monkeypatch.setattr(cursory_module, "morph_trajectory", lambda points, *_, **__: points)
    monkeypatch.setattr(cursory_module.random, "gauss", lambda *_, **__: next(itertools.cycle([10, -10])))


def test_high_frequency_still_generates_samples():
    points, timings = generate_trajectory((0.0, 0.0), (10.0, 10.0), frequency=2000)

    assert len(points) > 0
    assert len(points) == len(timings)
    assert timings == sorted(timings)


def test_timings_never_go_backwards_under_large_jitter():
    _points, timings = generate_trajectory(
        (0.0, 0.0),
        (10.0, 10.0),
        frequency=1000,
        frequency_randomizer=10,
    )

    assert timings == sorted(timings)