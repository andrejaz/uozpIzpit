import unittest

import numpy as np

from tsne import gaussian_affinities, make_clusters


def pairwise(X):
    s = np.sum(X ** 2, axis=1)
    return np.sqrt(np.maximum(s[:, None] + s[None, :] - 2 * X @ X.T, 0.0))


class UnitTests(unittest.TestCase):

    def setUp(self):
        self.D = pairwise(np.array([[0.0], [1.0], [5.0], [6.0]]))

    def test_rows_sum_to_one(self):
        P = gaussian_affinities(self.D, sigma=2.0)
        self.assertTrue(np.allclose(P.sum(axis=1), 1.0))

    def test_zero_diagonal(self):
        P = gaussian_affinities(self.D, sigma=2.0)
        self.assertTrue(np.allclose(np.diag(P), 0.0))

    def test_larger_sigma_more_uniform(self):
        P_small = gaussian_affinities(self.D, sigma=0.5)
        P_large = gaussian_affinities(self.D, sigma=20.0)
        # pri velikem sigma je najvecja verjetnost v vrstici manjsa (bolj enakomerno)
        self.assertGreater(P_small[0].max(), P_large[0].max())

    def test_make_clusters_shapes(self):
        X, labels = make_clusters(15, 8.0, seed=0)
        self.assertEqual(X.shape, (45, 2))
        self.assertEqual(len(set(np.asarray(labels).tolist())), 3)


if __name__ == "__main__":
    unittest.main()
