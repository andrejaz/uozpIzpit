import math
import unittest

from quadratic_gradient_descent import derivative, find_minimum


class TestQuadraticGradientDescent(unittest.TestCase):
    def test_derivative_basic_case(self):
        self.assertEqual(derivative(6), 2)
        self.assertEqual(derivative(5), 0)
        self.assertEqual(derivative(3), -4)

    def test_find_minimum_converges_near_five(self):
        result = find_minimum(6, 0.1, 20)
        self.assertAlmostEqual(result, 5.0, delta=0.02)

    def test_zero_steps_returns_start_value(self):
        self.assertEqual(find_minimum(12, 0.1, 0), 12)

    def test_result_type(self):
        result = find_minimum(6, 0.1, 3)
        self.assertIsInstance(result, (int, float))

    def test_update_uses_negative_gradient_direction(self):
        result = find_minimum(6, 0.1, 1)
        self.assertAlmostEqual(result, 5.8)


if __name__ == "__main__":
    unittest.main()
