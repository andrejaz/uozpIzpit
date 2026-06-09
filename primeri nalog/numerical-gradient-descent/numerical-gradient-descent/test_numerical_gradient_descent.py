import math
import unittest

from numerical_gradient_descent import find_minimum, numerical_derivative


class TestNumericalGradientDescent(unittest.TestCase):
    def test_numerical_derivative_basic_cases(self):
        self.assertAlmostEqual(numerical_derivative(0.0), 2.0363, delta=0.001)
        self.assertAlmostEqual(numerical_derivative(-1.0), -0.7364, delta=0.001)
        self.assertAlmostEqual(numerical_derivative(2.0), -1.2100, delta=0.001)

    def test_numerical_derivative_uses_h_argument(self):
        self.assertAlmostEqual(numerical_derivative(1.0, 0.1), 0.3932, delta=0.001)
        self.assertNotAlmostEqual(
            numerical_derivative(1.0, 0.1), numerical_derivative(1.0, 0.5), delta=0.001
        )

    def test_zero_steps_returns_start_value(self):
        self.assertEqual(find_minimum(1.25, 0.05, 0), 1.25)

    def test_result_type(self):
        result = find_minimum(0.0, 0.05, 10)
        self.assertIsInstance(result, (int, float))

    def test_update_uses_negative_gradient_direction(self):
        self.assertAlmostEqual(find_minimum(0.0, 0.05, 1), -0.1018, delta=0.001)

    def test_find_minimum_reaches_local_solution(self):
        result = find_minimum(0.0, 0.05, 80)
        self.assertAlmostEqual(result, -0.899, delta=0.01)


if __name__ == "__main__":
    unittest.main()
