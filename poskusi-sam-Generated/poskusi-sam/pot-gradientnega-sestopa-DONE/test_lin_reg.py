import unittest

import numpy as np

from lin_reg import train_with_history, mse_loss


class UnitTests(unittest.TestCase):

    def setUp(self):
        self.xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.ys = 2 * self.xs - 1

    def test_history_length(self):
        path = train_with_history(self.xs, self.ys, 0.05, n_epochs=100)
        self.assertEqual(len(path), 101)

    def test_starts_at_zero(self):
        path = train_with_history(self.xs, self.ys, 0.05, n_epochs=10)
        self.assertAlmostEqual(path[0][0], 0.0)
        self.assertAlmostEqual(path[0][1], 0.0)

    def test_loss_decreases_along_path(self):
        path = train_with_history(self.xs, self.ys, 0.05, n_epochs=2000)
        w0, b0 = path[0]
        w1, b1 = path[-1]
        self.assertLess(mse_loss(self.xs, self.ys, w1, b1),
                        mse_loss(self.xs, self.ys, w0, b0))


if __name__ == "__main__":
    unittest.main()
