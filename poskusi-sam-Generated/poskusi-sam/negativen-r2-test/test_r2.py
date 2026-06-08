import unittest

import numpy as np

from r2 import r2_score, polynomial_features, train_test_split


class UnitTests(unittest.TestCase):

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(r2_score(y, y), 1.0)

    def test_r2_mean_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.full(3, y.mean())
        self.assertAlmostEqual(r2_score(y, pred), 0.0)

    def test_r2_negative(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([3.0, 2.0, 1.0])  # slabše od povprečja
        self.assertLess(r2_score(y, pred), 0.0)

    def test_polynomial_features(self):
        P = polynomial_features(np.array([2.0, 3.0]), 3)
        self.assertEqual(np.asarray(P).shape, (2, 3))
        self.assertTrue(np.allclose(P, [[2, 4, 8], [3, 9, 27]]))

    def test_split_sizes_and_disjoint(self):
        X = np.arange(10).reshape(10, 1).astype(float)
        y = np.arange(10).astype(float)
        Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, seed=0)
        self.assertEqual(len(Xtr) + len(Xte), 10)
        self.assertEqual(len(yte), 3)
        all_vals = set(Xtr.ravel().tolist()) | set(Xte.ravel().tolist())
        self.assertEqual(all_vals, set(range(10)))


if __name__ == "__main__":
    unittest.main()
