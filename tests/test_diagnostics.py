import unittest

from hailcast2d.diagnostics import (
    descriptive_statistics,
    summarize_ensemble,
    summarize_grid,
)


class TestDiagnostics(unittest.TestCase):

    def setUp(self):
        self.impacts = [
            {
                "member_id": 0,
                "time_s": 100.0,
                "x_m": 1000.0,
                "diameter_mm": 10.0,
                "speed_ms": 20.0,
                "angle_from_horizontal_deg": 70.0,
                "kinetic_energy_J": 2.0,
            },
            {
                "member_id": 1,
                "time_s": 200.0,
                "x_m": 3000.0,
                "diameter_mm": 20.0,
                "speed_ms": 30.0,
                "angle_from_horizontal_deg": 80.0,
                "kinetic_energy_J": 8.0,
            },
        ]

        self.ensemble = {
            "trajectories": [{}, {}, {}],
            "impacts": self.impacts,
            "n_simulated": 3,
            "n_impacted": 2,
            "impact_fraction": 2.0 / 3.0,
        }

    def test_descriptive_statistics(self):
        result = descriptive_statistics([1.0, 2.0, 3.0])

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 3.0)

    def test_empty_statistics(self):
        result = descriptive_statistics([])

        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["max"])

    def test_summarizes_ensemble_and_extremes(self):
        result = summarize_ensemble(self.ensemble)

        self.assertEqual(result["n_simulated"], 3)
        self.assertEqual(result["n_impacted"], 2)
        self.assertEqual(result["n_not_impacted"], 1)

        self.assertEqual(
            result["statistics"]["impact_diameter_mm"]["mean"],
            15.0,
        )
        self.assertEqual(
            result["extreme_members"]["largest_displacement"][
                "member_id"
            ],
            1,
        )
        self.assertEqual(
            result["extreme_members"]["largest_energy"]["value"],
            8.0,
        )

    def test_rejects_inconsistent_ensemble_count(self):
        invalid = dict(self.ensemble)
        invalid["n_impacted"] = 1

        with self.assertRaises(ValueError):
            summarize_ensemble(invalid)

    def test_summarizes_grid_and_checks_conservation(self):
        grid = {
            "impact_count": [2, 0, 1],
            "diameter_max_mm": [20.0, 0.0, 15.0],
            "kinetic_energy_sum_J": [10.0, 0.0, 3.0],
            "n_impacts_inside": 3,
            "n_impacts_excluded": 1,
            "cell_size_m": 1000.0,
        }

        result = summarize_grid(
            grid,
            expected_n_impacts=4,
        )

        self.assertEqual(result["n_cells"], 3)
        self.assertEqual(result["n_occupied_cells"], 2)
        self.assertEqual(result["maximum_cell_count"], 2)
        self.assertEqual(result["maximum_diameter_mm"], 20.0)
        self.assertEqual(result["total_kinetic_energy_J"], 13.0)
        self.assertTrue(result["impact_conservation_ok"])

    def test_rejects_grid_without_conservation(self):
        grid = {
            "impact_count": [2],
            "diameter_max_mm": [20.0],
            "kinetic_energy_sum_J": [10.0],
            "n_impacts_inside": 2,
            "n_impacts_excluded": 0,
            "cell_size_m": 1000.0,
        }

        with self.assertRaises(ValueError):
            summarize_grid(grid, expected_n_impacts=3)


if __name__ == "__main__":
    unittest.main()