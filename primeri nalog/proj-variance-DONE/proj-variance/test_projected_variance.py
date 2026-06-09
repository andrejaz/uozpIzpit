import unittest

from projected_variance import projected_variance


class TestProjectedVariance(unittest.TestCase):
    X = [(2.0, 0.0), (-2.0, 0.0), (0.0, 1.0), (0.0, -1.0)]

    def test_basic_case(self):
        self.assertAlmostEqual(
            projected_variance(self.X, (-1.0, 1.0)),
            1.25,
            places=6,
        )

    def test_second_direction(self):
        self.assertAlmostEqual(
            projected_variance(self.X, (-1.0, 3.0)),
            0.65,
            places=6,
        )

    def test_unit_direction(self):
        self.assertAlmostEqual(
            projected_variance(self.X, (1.0, 0.0)),
            2.0,
            places=6,
        )

    def test_unnormalized_direction_matches_unit(self):
        self.assertAlmostEqual(
            projected_variance(self.X, (2.0, 0.0)),
            2.0,
            places=6,
        )

    def test_result_type(self):
        self.assertIsInstance(
            projected_variance([(1.0, 0.0), (-1.0, 0.0)], (1.0, 0.0)),
            float,
        )

    def test_common_wrong_sample_variance(self):
        # Napačna rešitev z deljenjem z n - 1 da 8/3, ne 2.0.
        self.assertAlmostEqual(
            projected_variance(self.X, (1.0, 0.0)),
            2.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
