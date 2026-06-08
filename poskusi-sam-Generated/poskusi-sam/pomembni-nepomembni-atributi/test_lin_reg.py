import unittest

import numpy as np

from lin_reg import make_data, train_multi, rank_features


class UnitTests(unittest.TestCase):

    def test_rank_simple(self):
        order = list(rank_features(np.array([0.1, -5.0, 2.0])))
        self.assertEqual(order, [1, 2, 0])

    def test_top3_are_informative(self):
        X, y = make_data(seed=0, n=2000, noise=0.0)
        w, b = train_multi(X, y, lr=0.05, n_epochs=3000)
        order = rank_features(w)
        self.assertEqual(set(order[:3]), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
