import unittest

import numpy as np

from silhueta import silhouette_score, best_k


class UnitTests(unittest.TestCase):

    def test_silhouette_known(self):
        X = np.array([[0.0], [1.0], [10.0], [11.0]])
        s = silhouette_score(X, np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(s, 0.8997, places=3)

    def test_best_k_two_clusters(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (30, 2)),
                       rng.normal(10, 0.3, (30, 2))])
        self.assertEqual(best_k(X, [2, 3, 4], seed=0), 2)

    def test_best_k_returns_from_ks(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (100, 2))
        self.assertIn(best_k(X, [2, 3, 4], seed=0), [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
