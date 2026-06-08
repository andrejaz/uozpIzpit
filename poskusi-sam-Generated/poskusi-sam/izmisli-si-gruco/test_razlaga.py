import unittest

import numpy as np

from razlaga import t_statistic, rank_features_ttest


class UnitTests(unittest.TestCase):

    def test_t_statistic_separated(self):
        t = t_statistic([10.0, 11.0, 12.0], [0.0, 1.0, 2.0])
        self.assertGreater(t, 5.0)

    def test_t_statistic_symmetric_sign(self):
        t = t_statistic([0.0, 1.0, 2.0], [10.0, 11.0, 12.0])
        self.assertLess(t, -5.0)

    def test_rank_picks_separating_feature(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (40, 4))
        mask = np.zeros(40, dtype=bool)
        mask[:10] = True
        X[:10, 2] += 5.0  # atribut 2 mocno loci gruco
        order = rank_features_ttest(X, mask)
        self.assertEqual(order[0], 2)


if __name__ == "__main__":
    unittest.main()
