import unittest

import numpy as np

from pca import explained_variance_first, mean_explained_first


class UnitTests(unittest.TestCase):

    def test_explained_known(self):
        X = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        # lastni vrednosti 4.5 in 0.5 -> 0.9
        self.assertAlmostEqual(explained_variance_first(X), 0.9, places=6)

    def test_random_high_d_near_one_over_d(self):
        m = mean_explained_first(n=3000, d=10, reps=8, seed=0)
        self.assertAlmostEqual(m, 0.1, delta=0.05)

    def test_random_d2(self):
        m = mean_explained_first(n=3000, d=2, reps=8, seed=1)
        self.assertGreater(m, 0.5)


if __name__ == "__main__":
    unittest.main()
