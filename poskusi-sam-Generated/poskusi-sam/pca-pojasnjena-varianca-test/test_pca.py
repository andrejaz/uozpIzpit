import unittest

import numpy as np

from pca import (make_data, standardize_fit, standardize_apply,
                 pca_fit, explained_on_test)


class UnitTests(unittest.TestCase):

    def setUp(self):
        X = make_data(n=400, seed=0)
        mean, std = standardize_fit(X[:200])
        self.train = standardize_apply(X[:200], mean, std)
        self.test = standardize_apply(X[200:], mean, std)
        self.comps = pca_fit(self.train)

    def test_full_k_explains_all(self):
        d = self.test.shape[1]
        self.assertAlmostEqual(explained_on_test(self.test, self.comps, d),
                               1.0, places=5)

    def test_cumulative_increasing_and_bounded(self):
        vals = [explained_on_test(self.test, self.comps, k) for k in range(1, 6)]
        for v in vals:
            self.assertGreaterEqual(v, -1e-9)
            self.assertLessEqual(v, 1.0 + 1e-9)
        self.assertTrue(all(vals[i] <= vals[i + 1] + 1e-9 for i in range(4)))


if __name__ == "__main__":
    unittest.main()
