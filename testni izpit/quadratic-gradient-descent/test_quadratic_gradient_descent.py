import math
import unittest

from quadratic_gradient_descent import derivative, find_minimum


class UnitTests(unittest.TestCase):

    def test_derivative_basic_case(self):
        assert derivative(6) == 2
        assert derivative(5) == 0
        assert derivative(3) == -4

    def test_find_minimum_converges_near_five(self):
        result = find_minimum(6, 0.1, 20)
        assert math.isclose(result, 5, abs_tol=0.02)

    def test_zero_steps_returns_start_value(self):
        assert find_minimum(12, 0.1, 0) == 12

    def test_result_type(self):
        result = find_minimum(6, 0.1, 3)
        assert isinstance(result, (int, float))

    def test_update_uses_negative_gradient_direction(self):
        result = find_minimum(6, 0.1, 1)
        assert math.isclose(result, 5.8)
