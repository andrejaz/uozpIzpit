import unittest

import numpy as np

from lin_reg import train_batches_with_history, mse_loss, make_data


class UnitTests(unittest.TestCase):

    def test_length_and_start(self):
        xs, ys = make_data(n=200)
        path = train_batches_with_history(xs, ys, 0.001, 50, batch_size=10, seed=0)
        self.assertEqual(len(path), 51)
        self.assertEqual(path[0], (0.0, 0.0))

    def test_loss_decreases(self):
        xs, ys = make_data(n=300)
        path = train_batches_with_history(xs, ys, 0.001, 300, batch_size=20, seed=2)
        w1, b1 = path[-1]
        self.assertLess(mse_loss(xs, ys, w1, b1), mse_loss(xs, ys, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
