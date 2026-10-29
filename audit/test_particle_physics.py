import unittest
import numpy as np
from hailcast2d.particle_physics import (drag_acceleration, particle_acceleration,
    collection_rate, dry_bulk_density, dry_void_fraction)

class TestParticlePhysics(unittest.TestCase):
    def drag(self,v,u):
        return drag_acceleration(v,u,1e-4,.005,1.2)
    def test_zero_relative_velocity_has_no_drag(self):
        np.testing.assert_array_equal(self.drag([3,5],[3,5]),[0,0])
    def test_relative_drag_is_dissipative(self):
        for v in ([3,4],[-3,4],[3,-4],[-3,-4]):
            self.assertLess(np.dot(self.drag(v,[0,0]),v),0)
    def test_galilean_invariance(self):
        v,u=np.array([3.,-4]),np.array([10.,6]);boost=np.array([-5.,20])
        np.testing.assert_allclose(self.drag(v,u),self.drag(v+boost,u+boost))
    def test_rotation_covariance(self):
        rot=np.array([[0.,-1],[1,0]]);v,u=np.array([3.,-4]),np.array([10.,6])
        np.testing.assert_allclose(self.drag(rot@v,rot@u),rot@self.drag(v,u))
    def test_co_moving_sphere_only_has_effective_gravity(self):
        d,rho=.005,500.;m=np.pi*d**3/6*rho
        a=particle_acceleration([3,5],[3,5],m,d,1.2)
        np.testing.assert_allclose(a,[0,-9.81*(1-1.2/rho)])
    def test_fixed_cd_terminal_force_balance(self):
        d,rho,air,cd=.01,900.,1.2,.44;m=np.pi*d**3/6*rho
        vt=np.sqrt(4*d*9.81*(rho-air)/(3*cd*air))
        np.testing.assert_allclose(particle_acceleration([0,-vt],[0,0],m,d,air,cd=cd),[0,0],atol=1e-12)
    def test_horizontal_slip_contributes_to_collection(self):
        rate=collection_rate(.001,.5,.005,[3,4],[0,0])
        self.assertAlmostEqual(rate,.001*.5*np.pi*.005**2/4*5)
    def test_zero_lwc_or_zero_slip_yields_no_collection(self):
        self.assertEqual(collection_rate(0,.5,.005,[3,4],[0,0]),0)
        self.assertEqual(collection_rate(.001,.5,.005,[3,4],[3,4]),0)
    def test_dry_porosity_round_trip_and_compaction_sign(self):
        for rho in [200,500,900]:self.assertAlmostEqual(dry_bulk_density(dry_void_fraction(rho)),rho)
        self.assertGreater(dry_bulk_density(.2),dry_bulk_density(.5))
    def test_invalid_physical_inputs_rejected(self):
        for rho in [0,-1,1000,float('nan')]:
            with self.assertRaises(ValueError):dry_void_fraction(rho)
        with self.assertRaises(ValueError):self.drag([np.nan,0],[0,0])
        with self.assertRaises(ValueError):drag_acceleration([1,0],[0,0],0,.005,1.2)
        with self.assertRaises(ValueError):collection_rate(.001,1.2,.005,[0,0],[1,0])
