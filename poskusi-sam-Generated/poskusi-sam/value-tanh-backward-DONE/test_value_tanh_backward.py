import math
import unittest

from value_tanh_backward import Value


def fn(a, b, c):
    return math.tanh(a + b) * c


class UnitTests(unittest.TestCase):

    def test_known_gradients(self):
        a, b, c = Value(1.0), Value(0.5), Value(2.0)
        out = (a + b).tanh() * c
        out.backward()
        t = math.tanh(1.5)
        self.assertAlmostEqual(c.grad, t, places=5)              # ~0.9051
        self.assertAlmostEqual(a.grad, 2.0 * (1 - t ** 2), places=5)  # ~0.3614
        self.assertAlmostEqual(b.grad, 2.0 * (1 - t ** 2), places=5)

    def test_matches_finite_differences(self):
        a, b, c = Value(1.0), Value(0.5), Value(2.0)
        out = (a + b).tanh() * c
        out.backward()
        h = 1e-6
        da = (fn(1.0 + h, 0.5, 2.0) - fn(1.0 - h, 0.5, 2.0)) / (2 * h)
        db = (fn(1.0, 0.5 + h, 2.0) - fn(1.0, 0.5 - h, 2.0)) / (2 * h)
        dc = (fn(1.0, 0.5, 2.0 + h) - fn(1.0, 0.5, 2.0 - h)) / (2 * h)
        self.assertAlmostEqual(a.grad, da, places=4)
        self.assertAlmostEqual(b.grad, db, places=4)
        self.assertAlmostEqual(c.grad, dc, places=4)


if __name__ == "__main__":
    unittest.main()
