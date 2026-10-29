"""Publication audit: invariants, not agreement with empirical observations.

Run from repo root: python -m unittest discover -s audit -v
Source snapshot: 24a1183503bb99223305e98021cc3602c5bffd3a.
Failures intentionally expose unresolved source-model inconsistencies.
"""
import unittest
import numpy as np
from cloud_model import CloudProfile
from hailcast2d.trajectory import simulate_trajectory_2d
from hailcast2d.wind_profile import HorizontalWindProfile

class PhysicalConsistencyAudit(unittest.TestCase):
    def trajectory(self, coupled):
        options = {'updraft_radius_m': 3000.0} if coupled else {}
        return simulate_trajectory_2d(
            profile=CloudProfile(updraft_type='strong'),
            wind_profile=HorizontalWindProfile.constant(10.0),
            D0_mm=5.0, z0=5000.0, T0_K=263.15,
            rho_init=500.0, t_max=0.1, dt=0.1, **options)

    def test_requested_initial_diameter_one_way(self):
        r=self.trajectory(False)
        self.assertAlmostEqual(2e3*r['radius'][0],5.0,places=10)

    def test_requested_initial_diameter_coupled(self):
        r=self.trajectory(True)
        self.assertAlmostEqual(2e3*r['radius'][0],5.0,places=10)

    def test_mass_volume_density_one_way(self):
        r=self.trajectory(False)
        np.testing.assert_allclose(r['mass'],r['density']*(4*np.pi/3)*r['radius']**3,rtol=1e-8,atol=0)

    def test_mass_volume_density_coupled(self):
        r=self.trajectory(True)
        np.testing.assert_allclose(r['mass'],r['density']*(4*np.pi/3)*r['radius']**3,rtol=1e-8,atol=0)

    def test_same_initial_geometry_in_both_modes(self):
        a,b=self.trajectory(False),self.trajectory(True)
        self.assertAlmostEqual(a['radius'][0],b['radius'][0],places=12)

if __name__=='__main__':unittest.main()
