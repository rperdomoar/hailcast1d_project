# ================================================================
# tests/test_wind_profile.py
# ================================================================
import unittest

import numpy as np

from hailcast2d.wind_profile import HorizontalWindProfile


class TestHorizontalWindProfile(unittest.TestCase):

    def test_constant_profile(self):
        wind = HorizontalWindProfile.constant(speed_ms=12.0)

        self.assertAlmostEqual(wind.velocity(0.0), 12.0)
        self.assertAlmostEqual(wind.velocity(6000.0), 12.0)
        self.assertAlmostEqual(wind.velocity(12000.0), 12.0)

    def test_linear_profile(self):
        wind = HorizontalWindProfile.linear(
            u_surface_ms=5.0,
            u_top_ms=25.0,
            z_top_m=10000.0,
        )

        self.assertAlmostEqual(wind.velocity(0.0), 5.0)
        self.assertAlmostEqual(wind.velocity(5000.0), 15.0)
        self.assertAlmostEqual(wind.velocity(10000.0), 25.0)

    def test_array_input(self):
        wind = HorizontalWindProfile.linear(
            u_surface_ms=0.0,
            u_top_ms=20.0,
            z_top_m=10000.0,
        )

        result = wind.velocity(np.array([0.0, 2500.0, 5000.0, 10000.0]))

        np.testing.assert_allclose(result, [0.0, 5.0, 10.0, 20.0])

    def test_values_outside_profile_use_nearest_endpoint(self):
        wind = HorizontalWindProfile.linear(
            u_surface_ms=4.0,
            u_top_ms=18.0,
            z_top_m=10000.0,
        )

        self.assertAlmostEqual(wind.velocity(-100.0), 4.0)
        self.assertAlmostEqual(wind.velocity(15000.0), 18.0)

    def test_rejects_unsorted_heights(self):
        with self.assertRaises(ValueError):
            HorizontalWindProfile(
                heights_m=[0.0, 5000.0, 4000.0],
                speeds_ms=[5.0, 15.0, 20.0],
            )

    def test_rejects_different_lengths(self):
        with self.assertRaises(ValueError):
            HorizontalWindProfile(
                heights_m=[0.0, 5000.0, 10000.0],
                speeds_ms=[5.0, 15.0],
            )


if __name__ == "__main__":
    unittest.main()