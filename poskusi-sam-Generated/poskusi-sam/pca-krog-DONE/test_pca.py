import unittest

import numpy as np

from pca import first_principal_component, make_circle, explained_variance_ratio


class UnitTests(unittest.TestCase):

    def test_circle_about_half(self):
        X = make_circle(1000, seed=0)
        v = first_principal_component(X)
        self.assertAlmostEqual(explained_variance_ratio(X, v), 0.5, places=1)

    def test_axis_spread_high(self):
        X = np.array([[-3.0, 0.0], [3.0, 0.0], [0.0, 0.5], [0.0, -0.5]])
        ratio = explained_variance_ratio(X, np.array([1.0, 0.0]))
        self.assertGreater(ratio, 0.8)


if __name__ == "__main__":
    unittest.main()
