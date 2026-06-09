import unittest

import numpy as np

from uhajanje import standardize_fit, standardize_apply, r2_score


class UnitTests(unittest.TestCase):

    def test_standardize_fit_uses_train_stats(self):
        X = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
        mean, std = standardize_fit(X)
        self.assertTrue(np.allclose(mean, [2.0, 20.0]))
        self.assertTrue(np.allclose(std, X.std(axis=0)))

    def test_standardize_apply(self):
        X = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
        mean, std = standardize_fit(X)
        Z = standardize_apply(X, mean, std)
        self.assertTrue(np.allclose(Z.mean(axis=0), 0.0, atol=1e-9))
        self.assertTrue(np.allclose(Z.std(axis=0), 1.0, atol=1e-9))

    def test_apply_uses_given_stats(self):
        mean = np.array([1.0, 1.0])
        std = np.array([2.0, 2.0])
        Z = standardize_apply(np.array([[3.0, 5.0]]), mean, std)
        self.assertTrue(np.allclose(Z, [[1.0, 2.0]]))

    def test_r2(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(r2_score(y, y), 1.0)
        self.assertAlmostEqual(r2_score(y, np.full(3, y.mean())), 0.0)


if __name__ == "__main__":
    unittest.main()
