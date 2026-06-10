import unittest

import numpy as np

from voditelji import assign, train_leaders


class UnitTests(unittest.TestCase):

    def test_assign(self):
        X = np.array([[0.0, 0.0], [9.0, 9.0]])
        leaders = np.array([[0.0, 0.0], [10.0, 10.0]])
        a = assign(X, leaders)
        self.assertEqual(list(a), [0, 1])

    def test_leaders_near_centers(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal([0, 0], 0.4, (60, 2)),
                       rng.normal([10, 10], 0.4, (60, 2))])
        leaders = train_leaders(X, 2, lr=0.5, steps=400, seed=0)
        for center in [np.array([0.0, 0.0]), np.array([10.0, 10.0])]:
            dmin = min(np.linalg.norm(center - lead) for lead in leaders)
            self.assertLess(dmin, 1.5)


if __name__ == "__main__":
    unittest.main()
