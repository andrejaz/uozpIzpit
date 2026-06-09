import unittest

import numpy as np

from pca import first_principal_component, make_two_groups


class UnitTests(unittest.TestCase):

    def test_unit_length(self):
        v = first_principal_component(np.array([[1.0, 0.0], [-1.0, 0.0],
                                               [0.0, 0.5], [0.0, -0.5]]))
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=6)

    def test_horizontal_spread(self):
        X = np.array([[-3.0, 0.0], [3.0, 0.0], [-1.0, 0.1], [1.0, -0.1]])
        v = first_principal_component(X)
        self.assertGreater(abs(v[0]), abs(v[1]))

    def test_low_noise_is_horizontal(self):
        v = first_principal_component(make_two_groups(noise=0.2, seed=0))
        self.assertGreater(abs(v[0]), abs(v[1]))


if __name__ == "__main__":
    unittest.main()
