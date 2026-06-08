import unittest

import numpy as np

from lin_reg import train, mse_loss, make_data


class UnitTests(unittest.TestCase):

    def test_recovers_line(self):
        xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        ys = 2 * xs - 1
        w, b = train(xs, ys, lr=0.05, n_epochs=5000)
        self.assertAlmostEqual(w, 2.0, places=2)
        self.assertAlmostEqual(b, -1.0, places=2)

    def test_loss_decreases(self):
        xs, ys = make_data(seed=1, n=30)
        before = mse_loss(xs, ys, 0.0, 0.0)
        w, b = train(xs, ys, lr=0.01, n_epochs=500)
        self.assertLess(mse_loss(xs, ys, w, b), before)


if __name__ == "__main__":
    unittest.main()
