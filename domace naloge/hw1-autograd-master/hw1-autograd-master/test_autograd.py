import unittest

from autograd import Value, fn1, fn2


class TestAutograd(unittest.TestCase):

    def test_fn1_data(self):
        x1 = Value(2, label='x1')
        x2 = Value(1, label='x2')
        l = fn1(x1, x2)
        self.assertAlmostEqual(l.data, -0.833333, places=5)

        x1 = Value(0.2, label='x1')
        x2 = Value(0.3, label='x2')
        l = fn1(x1, x2)
        self.assertAlmostEqual(l.data, 2.500000, places=5)

    def test_fn1_grad(self):
        x1 = Value(2, label='x1')
        x2 = Value(1, label='x2')
        l = fn1(x1, x2)
        l.backward()
        self.assertEqual(l.grad, 1)
        self.assertAlmostEqual(x1.grad, 0.347222, places=5)
        self.assertAlmostEqual(x2.grad, 0.694444, places=5)

        x1 = Value(0.2, label='x1')
        x2 = Value(0.3, label='x2')
        l = fn1(x1, x2)
        l.backward()
        self.assertEqual(l.grad, 1)
        self.assertAlmostEqual(x1.grad, 3.125000, places=5)
        self.assertAlmostEqual(x2.grad, 6.250000, places=5)

    def test_fn2_data(self):
        x1 = Value(2, label='x1')
        x2 = Value(1, label='x2')
        l = fn2(x1, x2)
        self.assertAlmostEqual(l.data, -0.105083, places=5)

        x1 = Value(0.2, label='x1')
        x2 = Value(0.3, label='x2')
        l = fn2(x1, x2)
        self.assertAlmostEqual(l.data, -1.040054, places=5)

    def test_fn2_grad(self):
        x1 = Value(2, label='x1')
        x2 = Value(1, label='x2')
        l = fn2(x1, x2)
        l.backward()
        self.assertEqual(l.grad, 1)
        self.assertAlmostEqual(x1.grad, 0.049875, places=5)
        self.assertAlmostEqual(x2.grad, 0.199501, places=5)

        x1 = Value(0.2, label='x1')
        x2 = Value(0.3, label='x2')
        l = fn2(x1, x2)
        l.backward()
        self.assertEqual(l.grad, 1)
        self.assertAlmostEqual(x1.grad, 0.323282, places=5)
        self.assertAlmostEqual(x2.grad, 2.801780, places=5)


if __name__ == '__main__':
    unittest.main()