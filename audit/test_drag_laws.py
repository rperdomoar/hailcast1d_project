import unittest
import numpy as np
from hailcast2d.particle_physics import sphere_drag_coefficient,drag_acceleration
class TestDragLaws(unittest.TestCase):
    def test_stokes_limit(self):
        re=1e-8
        self.assertAlmostEqual(sphere_drag_coefficient(re,'clift_gauvin')*re/24,1,places=5)
    def test_candidate_is_continuous_at_legacy_switches(self):
        for re in [1,1000]:
            a=sphere_drag_coefficient(re*(1-1e-8),'clift_gauvin')
            b=sphere_drag_coefficient(re*(1+1e-8),'clift_gauvin')
            self.assertLess(abs(a-b)/a,1e-6)
    def test_default_preserves_legacy_coefficient(self):
        self.assertEqual(sphere_drag_coefficient(1000),.44)
    def test_candidate_drag_opposes_slip(self):
        for v in ([1,2],[-1,2],[1,-2]):
            a=drag_acceleration(v,[0,0],1e-4,.005,1.2,drag_law='clift_gauvin')
            self.assertLess(np.dot(v,a),0)
    def test_invalid_selection_and_out_of_range_rejected(self):
        with self.assertRaises(ValueError):sphere_drag_coefficient(3e5,'clift_gauvin')
        with self.assertRaises(ValueError):drag_acceleration([0,0],[0,0],1,.01,1,drag_law='bad')
