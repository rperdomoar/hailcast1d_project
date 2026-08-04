import unittest

import numpy as np

from hailcast2d.sounding_cloud import SoundingCloudProfile
from tests.test_sounding_profile import sample_sounding


class TestSoundingCloudProfile(unittest.TestCase):

    def test_builds_finite_profiles(self):
        cloud = SoundingCloudProfile(sample_sounding())

        self.assertTrue(np.all(np.isfinite(cloud.updraft_ms)))
        self.assertTrue(np.all(np.isfinite(cloud.lwc_kgm3)))
        self.assertTrue(np.all(cloud.updraft_ms >= 0.0))
        self.assertTrue(np.all(cloud.lwc_kgm3 >= 0.0))

    def test_effective_updraft_respects_efficiency(self):
        cloud = SoundingCloudProfile(
            sample_sounding(),
            updraft_efficiency=0.25,
        )

        self.assertAlmostEqual(
            cloud.w_peak,
            0.25 * np.max(cloud.ideal_updraft_ms),
        )

    def test_surface_is_zero_agl(self):
        cloud = SoundingCloudProfile(sample_sounding())

        self.assertEqual(cloud.z_min, 0.0)
        self.assertAlmostEqual(cloud.temperature(0.0), 303.15)
        self.assertAlmostEqual(cloud.pressure(0.0), 1000.0)

    def test_cloud_top_is_equilibrium_level_agl(self):
        sounding = sample_sounding()
        cloud = SoundingCloudProfile(sounding)

        expected_cloud_top = (
            sounding.el_height_m - sounding.height_m[0]
        )

        self.assertAlmostEqual(cloud.z_max, expected_cloud_top)
        self.assertAlmostEqual(
            cloud.profile_top_m,
            sounding.height_m[-1] - sounding.height_m[0],
        )
        self.assertLess(cloud.z_max, cloud.profile_top_m)

    def test_updraft_and_lwc_are_zero_at_and_above_cloud_top(self):
        cloud = SoundingCloudProfile(sample_sounding())

        heights = np.asarray(
            [cloud.z_max, cloud.z_max + 100.0]
        )

        np.testing.assert_allclose(cloud.w(heights), 0.0)
        np.testing.assert_allclose(cloud.lwc(heights), 0.0)

    def test_air_density_is_positive(self):
        cloud = SoundingCloudProfile(sample_sounding())

        self.assertGreater(cloud.air_density(0.0), 0.0)
        self.assertGreater(cloud.air_density(6000.0), 0.0)
        self.assertLess(
            cloud.air_density(6000.0),
            cloud.air_density(0.0),
        )

    def test_wind_profile_uses_agl_height(self):
        cloud = SoundingCloudProfile(sample_sounding())
        wind = cloud.wind_profile(component="u")

        self.assertAlmostEqual(wind.velocity(0.0), 2.0)
        self.assertAlmostEqual(wind.velocity(5500.0), 18.0)

    def test_rejects_invalid_efficiency(self):
        with self.assertRaises(ValueError):
            SoundingCloudProfile(
                sample_sounding(),
                updraft_efficiency=1.5,
            )
    def test_rejects_sounding_without_equilibrium_level(self):
        sounding = sample_sounding()
        sounding.el_height_m = None
        sounding.el_pressure_hpa = None

        with self.assertRaises(ValueError):
            SoundingCloudProfile(sounding)


if __name__ == "__main__":
    unittest.main()