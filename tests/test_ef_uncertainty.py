import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.metric import calculate_ef_interval

class TestEFUncertainty(unittest.TestCase):
    
    def test_ef_interval_formula_standard(self):
        """
        Test standard case correctness.
        Vd = 100, Vs = 40 => EF = 60%
        Let dVd = 5, dVs = 5
        
        dEF/dVs = -1/100 = -0.01
        dEF/dVd = 40/10000 = 0.004
        
        Term1 = -0.01 * 5 = -0.05
        Term2 = 0.004 * 5 = 0.02
        
        Delta = sqrt(0.0025 + 0.0004) = sqrt(0.0029) approx 0.05385
        Percentage = 5.385%
        """
        ed_vol = 100.0
        es_vol = 40.0
        delta_v = 5.0
        
        interval = calculate_ef_interval(ed_vol, es_vol, delta_v, delta_v)
        
        expected = np.sqrt((-0.01 * 5)**2 + (0.004 * 5)**2) * 100
        self.assertAlmostEqual(interval, expected, places=4)
        
    def test_ef_interval_weird_heart(self):
        """
        Test 'weird' heart: Small Vd (e.g., pediatric or restrictive), Normal Vs.
        This maximizes penalty because of 1/Vd terms.
        
        Vd = 50, Vs = 30 -> EF = 40%
        dV = 5
        
        dEF/dVs = -1/50 = -0.02
        dEF/dVd = 30/2500 = 0.012
        
        Term1 = -0.02 * 5 = -0.1
        Term2 = 0.012 * 5 = 0.06
        
        Delta = sqrt(0.01 + 0.0036) = sqrt(0.0136) approx 0.1166
        Percentage = 11.66%
        
        Compared to normal heart (Vd=100), uncertainty effectively doubled/tripled.
        """
        ed_vol = 50.0
        es_vol = 30.0
        delta_v = 5.0
        
        interval = calculate_ef_interval(ed_vol, es_vol, delta_v, delta_v)
        
        expected = np.sqrt((-0.02 * 5)**2 + (0.012 * 5)**2) * 100
        self.assertAlmostEqual(interval, expected, places=4)
        print(f"\nWeird Heart Interval: {interval:.2f}% (Expected ~11.66%)")

    def test_division_by_zero_guard(self):
        """
        Test that Vd ~ 0 returns 0.0 instead of crash.
        """
        interval = calculate_ef_interval(0.0, 0.0, 1.0, 1.0)
        self.assertEqual(interval, 0.0)
        
        interval = calculate_ef_interval(1e-8, 10.0, 1.0, 1.0)
        self.assertEqual(interval, 0.0)

if __name__ == '__main__':
    unittest.main()
