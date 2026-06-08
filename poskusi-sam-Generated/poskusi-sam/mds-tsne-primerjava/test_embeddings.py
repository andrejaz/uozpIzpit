import unittest

import numpy as np

from embeddings import make_groups_highdim, silhouette


class UnitTests(unittest.TestCase):

    def test_shapes(self):
        X, labels = make_groups_highdim(n_per=20, dim=10, sep=6.0, seed=0)
        self.assertEqual(X.shape, (60, 10))
        self.assertEqual(len(labels), 60)
        self.assertEqual(len(set(np.asarray(labels).tolist())), 3)

    def test_silhouette_separated(self):
        X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
        labels = np.array([0, 0, 1, 1])
        self.assertGreater(silhouette(X, labels), 0.5)

    def test_silhouette_groups(self):
        X, labels = make_groups_highdim(n_per=30, dim=10, sep=8.0, seed=0)
        self.assertGreater(silhouette(X, labels), 0.5)


if __name__ == "__main__":
    unittest.main()
