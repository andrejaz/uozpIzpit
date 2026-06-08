import unittest

import numpy as np

from lin_reg import train_batches, mse_loss, make_data


class UnitTests(unittest.TestCase):

    def test_full_batch_recovers(self):
        xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        ys = 2 * xs + 1
        w, b = train_batches(xs, ys, lr=0.05, n_epochs=3000, batch_size=5, seed=0)
        self.assertAlmostEqual(w, 2.0, places=2)
        self.assertAlmostEqual(b, 1.0, places=2)

    def test_minibatch_reduces_loss(self):
        xs, ys = make_data(n=300)
        before = mse_loss(xs, ys, 0.0, 0.0)
        w, b = train_batches(xs, ys, lr=0.001, n_epochs=300, batch_size=10, seed=1)
        self.assertLess(mse_loss(xs, ys, w, b), before)


if __name__ == "__main__":
    unittest.main()
