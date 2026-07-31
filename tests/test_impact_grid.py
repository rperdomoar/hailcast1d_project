import unittest

import numpy as np

from hailcast2d.impact_grid import build_impact_grid


class TestImpactGrid(unittest.TestCase):

    def setUp(self):
        self.impacts = [
            {
                "x_m": 25.0,
                "diameter_mm": 10.0,
                "kinetic_energy_J": 2.0,
            },
            {
                "x_m": 75.0,
                "diameter_mm": 20.0,
                "kinetic_energy_J": 6.0,
            },
            {
                "x_m": 125.0,
                "diameter_mm": 30.0,
                "kinetic_energy_J": 12.0,
            },
        ]

    def test_aggregates_impacts_by_cell(self):
        grid = build_impact_grid(
            impacts=self.impacts,
            cell_size_m=100.0,
            x_min_m=0.0,
            x_max_m=200.0,
            n_simulated=5,
        )

        np.testing.assert_array_equal(
            grid["impact_count"],
            [2, 1],
        )
        np.testing.assert_allclose(
            grid["impact_fraction"],
            [0.4, 0.2],
        )
        np.testing.assert_allclose(
            grid["diameter_mean_mm"],
            [15.0, 30.0],
        )
        np.testing.assert_allclose(
            grid["diameter_max_mm"],
            [20.0, 30.0],
        )
        np.testing.assert_allclose(
            grid["kinetic_energy_sum_J"],
            [8.0, 12.0],
        )

    def test_impact_on_final_edge_is_included(self):
        impacts = [
            {
                "x_m": 200.0,
                "diameter_mm": 15.0,
                "kinetic_energy_J": 4.0,
            }
        ]

        grid = build_impact_grid(
            impacts=impacts,
            cell_size_m=100.0,
            x_min_m=0.0,
            x_max_m=200.0,
        )

        np.testing.assert_array_equal(
            grid["impact_count"],
            [0, 1],
        )

    def test_counts_impacts_outside_domain(self):
        grid = build_impact_grid(
            impacts=self.impacts,
            cell_size_m=100.0,
            x_min_m=50.0,
            x_max_m=150.0,
            n_simulated=3,
        )

        self.assertEqual(grid["n_impacts_inside"], 2)
        self.assertEqual(grid["n_impacts_excluded"], 1)
        np.testing.assert_array_equal(
            grid["impact_count"],
            [2],
        )

    def test_empty_impacts_require_explicit_bounds(self):
        with self.assertRaises(ValueError):
            build_impact_grid(
                impacts=[],
                cell_size_m=100.0,
                n_simulated=10,
            )

    def test_empty_domain_returns_zero_arrays(self):
        grid = build_impact_grid(
            impacts=[],
            cell_size_m=100.0,
            x_min_m=0.0,
            x_max_m=300.0,
            n_simulated=10,
        )

        np.testing.assert_array_equal(
            grid["impact_count"],
            [0, 0, 0],
        )
        np.testing.assert_allclose(
            grid["impact_fraction"],
            [0.0, 0.0, 0.0],
        )

    def test_rejects_missing_impact_fields(self):
        with self.assertRaises(ValueError):
            build_impact_grid(
                impacts=[{"x_m": 10.0}],
                cell_size_m=100.0,
            )


if __name__ == "__main__":
    unittest.main()
