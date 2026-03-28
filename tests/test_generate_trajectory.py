import unittest
from unittest.mock import patch

from cursory.cursory import generate_trajectory


BASE_POINTS = [(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]
BASE_TIMINGS = [0, 100, 200]


class GenerateTrajectoryTests(unittest.TestCase):
    @patch("cursory.cursory.morph_trajectory", side_effect=lambda points, *_args: points)
    @patch("cursory.cursory.jitter_trajectory", side_effect=lambda points, *_args: points)
    @patch("cursory.cursory.knot_trajectory", side_effect=lambda points, *_args: points)
    @patch("cursory.cursory.find_trajectory", return_value=(BASE_POINTS, BASE_TIMINGS))
    def test_high_frequency_still_generates_samples(self, _find, _knot, _jitter, _morph):
        points, timings = generate_trajectory((0.0, 0.0), (10.0, 10.0), frequency=2000)

        self.assertGreater(len(points), 0)
        self.assertEqual(len(points), len(timings))
        self.assertEqual(timings, sorted(timings))

    @patch("cursory.cursory.morph_trajectory", side_effect=lambda points, *_args: points)
    @patch("cursory.cursory.jitter_trajectory", side_effect=lambda points, *_args: points)
    @patch("cursory.cursory.knot_trajectory", side_effect=lambda points, *_args: points)
    @patch("cursory.cursory.find_trajectory", return_value=(BASE_POINTS, BASE_TIMINGS))
    @patch("cursory.cursory.random.gauss", side_effect=[10, -10] * 200)
    def test_timings_never_go_backwards_under_large_jitter(
        self,
        _gauss,
        _find,
        _knot,
        _jitter,
        _morph,
    ):
        _points, timings = generate_trajectory(
            (0.0, 0.0),
            (10.0, 10.0),
            frequency=1000,
            frequency_randomizer=10,
        )

        self.assertEqual(timings, sorted(timings))


if __name__ == "__main__":
    unittest.main()
