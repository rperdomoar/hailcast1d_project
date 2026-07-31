import unittest
from unittest.mock import patch

import numpy as np

from hailcast2d.ensemble import run_ensemble
from hailcast2d.wind_profile import HorizontalWindProfile


def fake_trajectory(**kwargs):
    """Resultado mínimo que imita una trayectoria 2D."""

    impacted = kwargs["D0_mm"] >= 5.0

    return {
        "time": np.array([0.0, 1.0]),
        "height": np.array([100.0, 0.0]),
        "x": np.array([kwargs["x0"], kwargs["x0"] + 1.0]),
        "vx": np.array([kwargs["vx0"], kwargs["vx0"]]),
        "impact": {
            "impacted_ground": impacted,
            "x_m": kwargs["x0"] + 1.0,
            "diameter_mm": kwargs["D0_mm"],
        },
    }


class TestEnsemble(unittest.TestCase):

    def setUp(self):
        self.wind = HorizontalWindProfile(
            heights_m=[0.0, 5000.0, 10000.0],
            speeds_ms=[2.0, 8.0, 14.0],
        )

    def test_rejects_invalid_member_count(self):
        with self.assertRaises(ValueError):
            run_ensemble(
                profile=object(),
                wind_profile=self.wind,
                n_members=0,
                D0_mm=5.0,
                z0=5000.0,
                T0_K=263.15,
            )

    @patch(
        "hailcast2d.ensemble.simulate_trajectory_2d",
        side_effect=fake_trajectory,
    )
    def test_zero_spread_preserves_initial_conditions(self, mocked):
        result = run_ensemble(
            profile=object(),
            wind_profile=self.wind,
            n_members=3,
            D0_mm=6.0,
            z0=5000.0,
            T0_K=263.15,
            x0=100.0,
            vx0=2.0,
            seed=42,
        )

        self.assertEqual(result["n_simulated"], 3)
        self.assertEqual(result["n_impacted"], 3)
        self.assertEqual(result["impact_fraction"], 1.0)

        for conditions in result["initial_conditions"]:
            self.assertEqual(conditions["D0_mm"], 6.0)
            self.assertEqual(conditions["x0_m"], 100.0)
            self.assertEqual(conditions["vx0_ms"], 2.0)
            self.assertEqual(conditions["wind_offset_ms"], 0.0)

        self.assertEqual(mocked.call_count, 3)

    @patch(
        "hailcast2d.ensemble.simulate_trajectory_2d",
        side_effect=fake_trajectory,
    )
    def test_seed_makes_sampling_reproducible(self, mocked):
        parameters = {
            "profile": object(),
            "wind_profile": self.wind,
            "n_members": 5,
            "D0_mm": 7.0,
            "z0": 5000.0,
            "T0_K": 263.15,
            "diameter_std_mm": 1.0,
            "x0_std_m": 100.0,
            "vx0_std_ms": 0.5,
            "wind_std_ms": 1.5,
            "seed": 123,
        }

        first = run_ensemble(**parameters)
        second = run_ensemble(**parameters)

        self.assertEqual(
            first["initial_conditions"],
            second["initial_conditions"],
        )

    @patch(
        "hailcast2d.ensemble.simulate_trajectory_2d",
        side_effect=fake_trajectory,
    )
    def test_wind_offset_preserves_vertical_shear(self, mocked):
        run_ensemble(
            profile=object(),
            wind_profile=self.wind,
            n_members=1,
            D0_mm=6.0,
            z0=5000.0,
            T0_K=263.15,
            wind_std_ms=2.0,
            seed=7,
        )

        sampled_wind = mocked.call_args.kwargs["wind_profile"]

        np.testing.assert_allclose(
            np.diff(sampled_wind.speeds_ms),
            np.diff(self.wind.speeds_ms),
        )


if __name__ == "__main__":
    unittest.main()
