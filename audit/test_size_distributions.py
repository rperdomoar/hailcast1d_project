import unittest
from unittest.mock import patch
import numpy as np
from hailcast2d.size_distributions import sample_diameters
from hailcast2d.ensemble import run_ensemble
from hailcast2d.wind_profile import HorizontalWindProfile
class TestSizes(unittest.TestCase):
    def test_reproducible_and_bounded(self):
        s=dict(kind='truncated_normal',location_mm=1,scale_mm=2,low_mm=.2,high_mm=4)
        a=sample_diameters(1000,s,42)
        np.testing.assert_array_equal(a,sample_diameters(1000,s,42))
        self.assertTrue(np.all((a>.2)&(a<4)))
        self.assertEqual(len(np.unique(a)),1000)
    def test_lognormal_parameter_conversion(self):
        a=sample_diameters(100000,dict(kind='lognormal',mean_mm=3,std_mm=1),42)
        self.assertAlmostEqual(a.mean(),3,delta=.02);self.assertAlmostEqual(a.std(),1,delta=.02)
    def test_uniform_and_fixed(self):
        np.testing.assert_array_equal(sample_diameters(3,dict(kind='fixed',value_mm=2)),[2,2,2])
        a=sample_diameters(20,dict(kind='uniform',low_mm=1,high_mm=3),1)
        self.assertTrue(np.all((a>=1)&(a<3)))
    def test_invalid_parameters(self):
        for s in [dict(kind='fixed',value_mm=0),dict(kind='uniform',low_mm=3,high_mm=2),dict(kind='unknown'),dict(kind='lognormal',mean_mm=1,std_mm=-1)]:
            with self.assertRaises(ValueError):sample_diameters(3,s)
    def test_explicit_samples_reach_solver_unchanged(self):
        with patch('hailcast2d.ensemble.simulate_trajectory_2d',return_value={'impact':{'impacted_ground':False}}) as solver:
            r=run_ensemble(object(),HorizontalWindProfile.constant(0),3,1,100,263,diameters_mm=[1,2,3])
        self.assertEqual([c.kwargs['D0_mm'] for c in solver.call_args_list],[1,2,3])
        self.assertEqual(r['diameter_sampling']['method'],'explicit_samples')
