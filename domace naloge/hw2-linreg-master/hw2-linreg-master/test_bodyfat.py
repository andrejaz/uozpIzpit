import unittest

from bodyfat import (LinReg, Value,
                     model_all, model_wh, model_wh_squared,
                     model_wh_squared_remove_row)


class TestBodyfat(unittest.TestCase):

    def test_loss_mean(self):
        lr = LinReg(n_inputs=2)
        lr.weights = [Value(2, label=f'w{i}')
                      for i in range(2)]
        lr.b = Value(0, label='b')
        X = [[1,1], [2,2], [1,1]]
        ys = [6, 10, 6]
        loss = lr.loss(X, ys).data  # (2**2 + 2**2 + 2**2) / 3
        self.assertEqual(loss, 4)

    def test_all(self):
        lr, X, ys = model_all(batch_size=252)
        self.assertIsInstance(lr, LinReg)
        self.assertEqual(len(X), 252)
        self.assertEqual(len(X[0]), 14)
        loss = lr.loss(X, ys).data
        self.assertTrue(13 < loss < 15.5)
        self.assertAlmostEqual(lr.b.data, 18.938, places=1)

    def test_wh(self):
        lr, X, ys = model_wh(batch_size=252)
        self.assertIsInstance(lr, LinReg)
        self.assertEqual(len(X), 252)
        self.assertEqual(len(X[0]), 2)
        loss = lr.loss(X, ys).data
        self.assertTrue(31 < loss < 33)
        self.assertAlmostEqual(lr.b.data, 18.938, places=1)
        weights = sorted([w.data for w in lr.weights])
        self.assertAlmostEqual(weights[0], -2.4, delta=0.5)
        self.assertAlmostEqual(weights[1], 5.5, delta=0.5)

    def test_wh_squared(self):
        lr, X, ys = model_wh_squared(batch_size=252)
        self.assertIsInstance(lr, LinReg)
        self.assertEqual(len(X), 252)
        self.assertEqual(len(X[0]), 4)
        loss = lr.loss(X, ys).data
        self.assertTrue(24 < loss < 26)
        self.assertAlmostEqual(lr.b.data, 18.938, places=1)
        weights = sorted([w.data for w in lr.weights])
        self.assertAlmostEqual(weights[0], -14, delta=2)
        self.assertAlmostEqual(weights[-1], 17, delta=2)
        self.assertGreater(min(map(abs, weights)), 6)

    def test_wh_squared_remove_row(self):
        lr, X, ys = model_wh_squared_remove_row(batch_size=251, row=41)
        self.assertIsInstance(lr, LinReg)
        self.assertEqual(len(X), 251)
        self.assertEqual(len(X[0]), 4)
        loss = lr.loss(X, ys).data
        self.assertTrue(24 < loss < 26)
        self.assertAlmostEqual(lr.b.data, 18.888, places=1)
        weights = sorted([w.data for w in lr.weights])
        self.assertAlmostEqual(weights[0], -11, delta=2)
        self.assertAlmostEqual(weights[-1], 17, delta=2)
        self.assertLess(min(map(abs, weights)), 2.5)


if __name__ == '__main__':
    unittest.main()