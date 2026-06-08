import unittest

from gradientni_spust import f, numerical_derivative, gradient_descent


class UnitTests(unittest.TestCase):

    def test_numerical_derivative_zero_at_min(self):
        # f'(2.25) = 0
        self.assertAlmostEqual(numerical_derivative(f, 2.25), 0.0, places=3)

    def test_numerical_derivative_parabola(self):
        g = lambda x: (x - 1) ** 2
        self.assertAlmostEqual(numerical_derivative(g, 3.0), 4.0, places=3)

    def test_descent_finds_local_min(self):
        x = gradient_descent(f, 2.0, lr=0.01, steps=20000)
        self.assertAlmostEqual(x, 2.25, places=2)

    def test_descent_on_parabola(self):
        g = lambda x: (x - 1.0) ** 2
        x = gradient_descent(g, 5.0, lr=0.1, steps=2000)
        self.assertAlmostEqual(x, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
