import unittest

import numpy as np

from regularizacija import make_data, regularization_path


class UnitTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_data(seed=0)
        self.lambdas = [0.0, 0.1, 1.0, 5.0]

    def test_path_shape(self):
        P = regularization_path(self.X, self.y, "l2", self.lambdas)
        self.assertEqual(np.asarray(P).shape, (4, self.X.shape[1]))

    def test_l2_shrinks(self):
        P = regularization_path(self.X, self.y, "l2", self.lambdas)
        self.assertLess(np.linalg.norm(P[-1]), np.linalg.norm(P[0]))

    def test_l1_zeros_increase(self):
        P = regularization_path(self.X, self.y, "l1", self.lambdas)
        zeros_small = int(np.sum(np.abs(P[1]) < 1e-9))
        zeros_large = int(np.sum(np.abs(P[-1]) < 1e-9))
        self.assertGreaterEqual(zeros_large, zeros_small)


if __name__ == "__main__":
    unittest.main()
