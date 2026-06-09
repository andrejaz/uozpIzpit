import unittest

import numpy as np

from r2 import r2_score, baseline_predict


class UnitTests(unittest.TestCase):

    def test_r2_perfect(self):
        y = np.array([2.0, 4.0, 6.0])
        self.assertAlmostEqual(r2_score(y, y), 1.0)

    def test_r2_mean_zero(self):
        y = np.array([2.0, 4.0, 6.0])
        self.assertAlmostEqual(r2_score(y, np.full(3, y.mean())), 0.0)

    def test_baseline_predict(self):
        pred = baseline_predict(np.array([2.0, 4.0, 6.0]), 3)
        self.assertTrue(np.allclose(pred, [4.0, 4.0, 4.0]))
        self.assertEqual(len(baseline_predict(np.array([1.0, 3.0]), 5)), 5)


if __name__ == "__main__":
    unittest.main()
