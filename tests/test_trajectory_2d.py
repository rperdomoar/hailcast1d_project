# ================================================================
# tests/test_trajectory_2d.py
# ================================================================
import unittest

import numpy as np

from cloud_model import CloudProfile
from hail_growth import simulate_trajectory
from hailcast2d.trajectory import simulate_trajectory_2d
from hailcast2d.wind_profile import HorizontalWindProfile
from hailcast2d.trajectory import (
    gaussian_updraft_factor,
    simulate_trajectory_2d,
)


class TestTrajectory2D(unittest.TestCase):

    def setUp(self):
        self.profile = CloudProfile(updraft_type="strong")

    def test_zero_wind_preserves_zero_horizontal_motion(self):
        wind = HorizontalWindProfile.constant(0.0)

        result = simulate_trajectory_2d(
            profile=self.profile,
            wind_profile=wind,
            D0_mm=5.0,
            z0=5000.0,
            T0_K=263.15,
            x0=0.0,
            vx0=0.0,
            t_max=1.0,
            dt=0.1,
        )

        np.testing.assert_allclose(result["x"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["vx"], 0.0, atol=1e-12)

    def test_vertical_solution_matches_hailcast1d(self):
        wind = HorizontalWindProfile.constant(10.0)

        result_1d = simulate_trajectory(
            profile=self.profile,
            D0_mm=5.0,
            z0=5000.0,
            T0_K=263.15,
            t_max=1.0,
            dt=0.1,
        )

        result_2d = simulate_trajectory_2d(
            profile=self.profile,
            wind_profile=wind,
            D0_mm=5.0,
            z0=5000.0,
            T0_K=263.15,
            t_max=1.0,
            dt=0.1,
        )

        np.testing.assert_allclose(
            result_2d["height"],
            result_1d["height"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result_2d["mass"],
            result_1d["mass"],
            rtol=0.0,
            atol=1e-12,
        )

    def test_particle_following_wind_keeps_constant_vx(self):
        wind = HorizontalWindProfile.constant(5.0)

        result = simulate_trajectory_2d(
            profile=self.profile,
            wind_profile=wind,
            D0_mm=5.0,
            z0=5000.0,
            T0_K=263.15,
            x0=2.0,
            vx0=5.0,
            t_max=1.0,
            dt=0.1,
        )

        np.testing.assert_allclose(result["vx"], 5.0, atol=1e-10)
        np.testing.assert_allclose(
            result["x"],
            2.0 + 5.0 * result["time"],
            atol=1e-10,
        )

    def test_short_simulation_has_no_ground_impact(self):
        wind = HorizontalWindProfile.constant(10.0)

        result = simulate_trajectory_2d(
            profile=self.profile,
            wind_profile=wind,
            D0_mm=5.0,
            z0=5000.0,
            T0_K=263.15,
            t_max=1.0,
            dt=0.1,
        )

        self.assertFalse(result["impact"]["impacted_ground"])

    def test_small_embryo_is_stable_with_automatic_substeps(self):
        """Los embriones pequeños deben permanecer numéricamente estables."""

        wind = HorizontalWindProfile.constant(15.0)

        result = simulate_trajectory_2d(
            profile=self.profile,
            wind_profile=wind,
            D0_mm=0.9,
            z0=4000.0,
            T0_K=263.15,
            t_max=2.0,
            dt=0.5,
        )

        variables = (
            "x",
            "vx",
            "height",
            "velocity",
            "mass",
            "radius",
        )

        for variable in variables:
            self.assertTrue(
                np.all(np.isfinite(result[variable])),
                msg=f"{variable} contiene valores no finitos",
            )

    def test_gaussian_updraft_factor(self):
        radius = 3000.0

        self.assertAlmostEqual(
            gaussian_updraft_factor(0.0, 0.0, radius),
            1.0,
        )
        self.assertAlmostEqual(
            gaussian_updraft_factor(radius, 0.0, radius),
            np.exp(-1.0),
        )
        self.assertAlmostEqual(
            gaussian_updraft_factor(-radius, 0.0, radius),
            np.exp(-1.0),
        )
        self.assertLess(
            gaussian_updraft_factor(
                2.0 * radius,
                0.0,
                radius,
            ),
            0.02,
        )

        with self.assertRaises(ValueError):
            gaussian_updraft_factor(
                0.0,
                0.0,
                0.0,
            )

    def test_finite_updraft_uses_coupled_stable_integration(self):
        wind = HorizontalWindProfile.constant(15.0)

        result = simulate_trajectory_2d(
            profile=self.profile,
            wind_profile=wind,
            D0_mm=0.9,
            z0=4000.0,
            T0_K=263.15,
            dt=0.5,
            t_max=10.0,
            updraft_center_x_m=0.0,
            updraft_radius_m=3000.0,
        )

        self.assertEqual(
            result["coupling"],
            "two_way_finite_updraft",
        )

        for variable in (
            "x",
            "vx",
            "height",
            "velocity",
            "mass",
            "radius",
            "updraft_factor",
            "vertical_air_velocity",
        ):
            self.assertTrue(
                np.all(np.isfinite(result[variable])),
                msg=f"{variable} contiene valores no finitos",
            )

        self.assertAlmostEqual(
            result["updraft_factor"][0],
            1.0,
        )
        self.assertLess(
            result["updraft_factor"][-1],
            result["updraft_factor"][0],
        )
        self.assertGreater(
            result["x"][-1],
            result["x"][0],
        )

if __name__ == "__main__":
    unittest.main()