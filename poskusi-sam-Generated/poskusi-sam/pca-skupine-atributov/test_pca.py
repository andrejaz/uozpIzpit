import unittest

import numpy as np

from pca import make_grouped_data, top_components, explained_variance_ratios


class UnitTests(unittest.TestCase):

    def setUp(self):
        self.X = make_grouped_data(n=400, noise=0.1, seed=0)

    def test_components_shape_unit(self):
        C = top_components(self.X, 3)
        self.assertEqual(np.asarray(C).shape, (3, 9))
        for row in C:
            self.assertAlmostEqual(np.linalg.norm(row), 1.0, places=6)

    def test_ratios_sum_to_one(self):
        r = explained_variance_ratios(self.X)
        self.assertAlmostEqual(float(np.sum(r)), 1.0, places=6)
        # padajoče
        self.assertTrue(np.all(np.diff(r) <= 1e-9))

    def test_top3_explain_most(self):
        r = explained_variance_ratios(self.X)
        self.assertGreater(float(np.sum(r[:3])), 0.8)


if __name__ == "__main__":
    unittest.main()
