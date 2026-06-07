import math
import unittest

from numerical_gradient_descent import (
    find_minimum,
    numerical_derivative,
)


# nekateri importi, ki jih bomo potrebova pri naslednjih naslogah
import numpy
import torch
import pandas
import matplotlib


class UnitTests(unittest.TestCase):

    def test_numerical_derivative_basic_cases(self):
        assert math.isclose(numerical_derivative(0.0), 2.0363, abs_tol=0.001)
        assert math.isclose(numerical_derivative(-1.0), -0.7364, abs_tol=0.001)
        assert math.isclose(numerical_derivative(2.0), -1.2100, abs_tol=0.001)

    def test_zero_steps_returns_start_value(self):
        assert find_minimum(1.25, 0.05, 0) == 1.25

    def test_result_type(self):
        result = find_minimum(0.0, 0.05, 10)
        assert isinstance(result, (int, float))

    def test_update_uses_negative_gradient_direction(self):
        start = 0.0
        learning_rate = 0.05
        expected = start - learning_rate * numerical_derivative(start)
        assert math.isclose(find_minimum(start, learning_rate, 1), expected)

    def test_find_minimum_reaches_local_solution(self):
        result = find_minimum(0.0, 0.05, 80)
        assert math.isclose(result, -0.899, abs_tol=0.01)

