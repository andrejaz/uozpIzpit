import unittest

import numpy as np

from mds import stress, mds


class UnitTests(unittest.TestCase):

    def test_stress_zero_for_exact(self):
        pos = {"A": np.array([0.0, 0.0]),
               "B": np.array([3.0, 0.0]),
               "C": np.array([0.0, 4.0])}
        d = {("A", "B"): 3.0, ("A", "C"): 4.0, ("B", "C"): 5.0}
        self.assertAlmostEqual(stress(pos, d), 0.0, places=6)

    def test_stress_value(self):
        pos = {"A": np.array([0.0, 0.0]), "B": np.array([3.0, 0.0])}
        d = {("A", "B"): 4.0}
        self.assertAlmostEqual(stress(pos, d), 1.0, places=6)

    def test_mds_reaches_low_stress(self):
        d = {("A", "B"): 3.0, ("A", "C"): 4.0, ("B", "C"): 5.0}
        pos = mds(d, epochs=4000, lr=0.01, seed=0)
        self.assertLess(stress(pos, d), 0.5)


if __name__ == "__main__":
    unittest.main()
