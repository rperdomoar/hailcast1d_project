import unittest
import numpy as np
from hailcast2d.trajectory import simulate_trajectory_2d
from hailcast2d.wind_profile import HorizontalWindProfile
class Uniform:
    z_min=0.
    def w(self,z):return 0.
    def air_density(self,z):return 1.2
class TestMechanicsMode(unittest.TestCase):
    def run_case(self,**kw):
        params=dict(profile=Uniform(),wind_profile=HorizontalWindProfile.constant(0),
            D0_mm=10,z0=5000,T0_K=263.15,rho_init=900,dt=.2,t_max=10,
            max_horizontal_substep_s=.05,cd=.44,physics_mode='mechanics_reference')
        params.update(kw);return simulate_trajectory_2d(**params)
    def test_matches_independent_analytic_velocity_and_height(self):
        r=self.run_case();g=9.81*(1-1.2/900);vt=np.sqrt(4*.01*9.81*(900-1.2)/(3*.44*1.2))
        t=r['time'];np.testing.assert_allclose(r['velocity'],-vt*np.tanh(g*t/vt),atol=1e-7)
        np.testing.assert_allclose(r['height'],5000-vt**2/g*np.log(np.cosh(g*t/vt)),atol=1e-7)
    def test_mass_density_and_geometry_are_constant(self):
        r=self.run_case()
        for k in ['mass','density','radius']:np.testing.assert_array_equal(r[k],np.full(len(r[k]),r[k][0]))
    def test_ground_event_recomputed_and_consistent(self):
        r=self.run_case(z0=1,t_max=3,dt=.5,max_horizontal_substep_s=.1)
        self.assertTrue(r['impact']['impacted_ground']);self.assertTrue(r['event_refined'])
        self.assertEqual(r['height'][-1],0)
        self.assertAlmostEqual(r['impact']['time_s'],r['time'][-1],places=10)
    def test_output_interval_independence_with_same_actual_steps(self):
        a=self.run_case(dt=.2);b=self.run_case(dt=.4)
        np.testing.assert_allclose(a['height'][-1],b['height'][-1],atol=1e-8)
    def test_horizontal_galilean_invariance_with_moving_core(self):
        a=self.run_case(vx0=0,updraft_radius_m=100)
        b=self.run_case(vx0=7,wind_profile=HorizontalWindProfile.constant(7),storm_motion_ms=7,updraft_radius_m=100)
        np.testing.assert_allclose(a['height'],b['height'],atol=1e-8)
        np.testing.assert_allclose(b['x'],a['x']+7*a['time'],atol=1e-8)
    def test_invalid_mode_and_growth_arguments_rejected(self):
        with self.assertRaises(ValueError):self.run_case(physics_mode='bad')
        with self.assertRaises(TypeError):self.run_case(k_melt=0)
        with self.assertRaises(ValueError):self.run_case(dt=float('nan'))
