import unittest

import numpy as np

from lin_reg import train_multi, make_data, mse_loss


class UnitTests(unittest.TestCase):

    def test_recovers_weights_no_noise(self):
        X, y = make_data(seed=0, n=2000, noise=0.0)
        w, b = train_multi(X, y, lr=0.05, n_epochs=4000)
        self.assertTrue(np.allclose(w, [2.0, 3.0, -1.0], atol=0.05))
        self.assertAlmostEqual(b, 1.0, places=1)

    def test_loss_decreases(self):
        X, y = make_data(seed=1, n=500, noise=5.0)
        w, b = train_multi(X, y, lr=0.05, n_epochs=1000)
        self.assertLess(mse_loss(X, y, w, b),
                        mse_loss(X, y, np.zeros(3), 0.0))


if __name__ == "__main__":
    unittest.main()
