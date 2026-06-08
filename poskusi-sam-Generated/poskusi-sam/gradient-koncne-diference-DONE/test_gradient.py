import unittest

from gradient import L, numerical_gradient, analytic_gradient


class UnitTests(unittest.TestCase):

    def test_analytic_gradient(self):
        self.assertEqual(tuple(analytic_gradient(2, 3, 10, 2)), (6, 4, 2, 16))

    def test_numerical_matches_analytic(self):
        num = numerical_gradient(L, (2, 3, 10, 2))
        ana = analytic_gradient(2, 3, 10, 2)
        self.assertEqual(len(num), 4)
        for x, y in zip(num, ana):
            self.assertAlmostEqual(x, y, places=3)

    def test_numerical_other_point(self):
        num = numerical_gradient(L, (1.0, 1.0, 1.0, 1.0))
        ana = analytic_gradient(1.0, 1.0, 1.0, 1.0)  # (1, 1, 1, 2)
        for x, y in zip(num, ana):
            self.assertAlmostEqual(x, y, places=3)


if __name__ == "__main__":
    unittest.main()
