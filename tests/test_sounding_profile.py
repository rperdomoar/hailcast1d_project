import unittest

import numpy as np

from hailcast2d.sounding_profile import SoundingProfile


def sample_sounding():
    return SoundingProfile(
        pressure_hpa=[
            1000, 925, 850, 700, 500, 400, 300, 250, 200
        ],
        height_m=[
            100, 750, 1500, 3000, 5600, 7200, 9200, 10300, 11800
        ],
        temperature_c=[
            30, 24, 18, 4, -12, -24, -40, -49, -56
        ],
        dewpoint_c=[
            22, 18, 14, -2, -20, -32, -48, -58, -65
        ],
        u_wind_ms=[
            2, 5, 8, 12, 18, 22, 28, 32, 36
        ],
        v_wind_ms=[
            1, 2, 3, 5, 8, 10, 12, 14, 16
        ],
    )


class TestSoundingProfile(unittest.TestCase):

    def test_calculates_finite_diagnostics(self):
        sounding = sample_sounding()
        summary = sounding.summary()

        self.assertGreaterEqual(summary["cape_jkg"], 0.0)
        self.assertGreaterEqual(summary["ideal_wmax_ms"], 0.0)
        self.assertGreater(summary["bulk_shear_0_6km_ms"], 0.0)
        self.assertIsNotNone(summary["freezing_level_m"])

    def test_ideal_wmax_matches_cape_relation(self):
        sounding = sample_sounding()

        expected = np.sqrt(2.0 * sounding.cape_jkg)

        self.assertAlmostEqual(sounding.ideal_wmax_ms, expected)

    def test_interpolates_pressure_height(self):
        sounding = sample_sounding()

        height = sounding.height_at_pressure(775.0)

        self.assertGreater(height, 1500.0)
        self.assertLess(height, 3000.0)

    def test_builds_wind_component_profile(self):
        sounding = sample_sounding()
        wind = sounding.wind_profile(component="u")

        self.assertAlmostEqual(wind.velocity(100.0), 2.0)
        self.assertAlmostEqual(wind.velocity(5600.0), 18.0)

    def test_rejects_unsorted_pressure(self):
        with self.assertRaises(ValueError):
            SoundingProfile(
                pressure_hpa=[1000, 900, 950, 700, 500, 300],
                height_m=[0, 1000, 2000, 4000, 7000, 10000],
                temperature_c=[25, 20, 15, 5, -15, -40],
                dewpoint_c=[20, 15, 10, 0, -20, -45],
                u_wind_ms=[0, 2, 4, 8, 12, 16],
                v_wind_ms=[0, 1, 2, 3, 4, 5],
            )

    def test_rejects_shallow_sounding(self):
        with self.assertRaises(ValueError):
            SoundingProfile(
                pressure_hpa=[1000, 900, 800, 700, 600, 500],
                height_m=[0, 500, 1000, 1500, 2000, 2500],
                temperature_c=[25, 20, 15, 10, 5, 0],
                dewpoint_c=[20, 15, 10, 5, 0, -5],
                u_wind_ms=[0, 1, 2, 3, 4, 5],
                v_wind_ms=[0, 1, 2, 3, 4, 5],
            )


if __name__ == "__main__":
    unittest.main()