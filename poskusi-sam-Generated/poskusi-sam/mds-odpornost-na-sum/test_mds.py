import unittest

import numpy as np

from mds import perturb_distance, total_movement


class UnitTests(unittest.TestCase):

    def test_perturb_changes_pair(self):
        d = {("A", "B"): 10.0, ("A", "C"): 5.0}
        d2 = perturb_distance(d, ("A", "B"), 100.0)
        self.assertAlmostEqual(d2[("A", "B")], 110.0)
        self.assertAlmostEqual(d2[("A", "C")], 5.0)

    def test_original_unchanged(self):
        d = {("A", "B"): 10.0}
        _ = perturb_distance(d, ("A", "B"), 5.0)
        self.assertAlmostEqual(d[("A", "B")], 10.0)

    def test_total_movement(self):
        a = {"X": np.array([0.0, 0.0]), "Y": np.array([0.0, 0.0])}
        b = {"X": np.array([3.0, 4.0]), "Y": np.array([0.0, 0.0])}
        self.assertAlmostEqual(total_movement(a, b), 5.0, places=6)
        self.assertAlmostEqual(total_movement(a, a), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
