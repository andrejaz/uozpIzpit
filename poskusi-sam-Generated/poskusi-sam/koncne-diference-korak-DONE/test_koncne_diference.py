import unittest

from koncne_diference import f, forward_difference, analytic_derivative


class UnitTests(unittest.TestCase):

    def test_analytic_derivative(self):
        self.assertAlmostEqual(analytic_derivative(6), 2.0)
        self.assertAlmostEqual(analytic_derivative(5), 0.0)

    def test_forward_difference_close(self):
        approx = forward_difference(f, 6.0, 1e-4)
        self.assertAlmostEqual(approx, 2.0, places=3)

    def test_forward_difference_linear_exact(self):
        # za linearno funkcijo je končna diferenca točna pri vsakem h
        g = lambda x: 3 * x + 1
        self.assertAlmostEqual(forward_difference(g, 2.0, 0.1), 3.0, places=9)


if __name__ == "__main__":
    unittest.main()
