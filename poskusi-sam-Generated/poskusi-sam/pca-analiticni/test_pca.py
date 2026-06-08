import unittest

import numpy as np

from pca import analytic_pca, pca_gradient


class UnitTests(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, (400, 3))
        self.X = base @ np.array([[3.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.5]])

    def test_eigenvalues_descending(self):
        comps, vals, ratios = analytic_pca(self.X)
        self.assertTrue(np.all(np.diff(vals) <= 1e-9))

    def test_ratios_sum_to_one(self):
        comps, vals, ratios = analytic_pca(self.X)
        self.assertAlmostEqual(float(np.sum(ratios)), 1.0, places=6)

    def test_first_component_unit(self):
        comps, vals, ratios = analytic_pca(self.X)
        self.assertAlmostEqual(np.linalg.norm(comps[0]), 1.0, places=6)

    def test_matches_gradient(self):
        comps, vals, ratios = analytic_pca(self.X)
        u = pca_gradient(self.X)
        self.assertGreater(abs(float(comps[0] @ u)), 0.99)


if __name__ == "__main__":
    unittest.main()
