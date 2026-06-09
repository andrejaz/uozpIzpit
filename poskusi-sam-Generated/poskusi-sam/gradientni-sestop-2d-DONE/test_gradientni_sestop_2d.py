import unittest

from gradientni_sestop_2d import Value, f_value, train_2d, analytic_min


class UnitTests(unittest.TestCase):

    def test_f_value(self):
        v = f_value(Value(1.0), Value(2.0))
        # 1 + 4 + 2 - 6 - 8 = -7
        self.assertAlmostEqual(v.data, -7.0, places=6)

    def test_analytic_min(self):
        a, b, fv = analytic_min()
        self.assertAlmostEqual(a, 8.0 / 3.0, places=6)
        self.assertAlmostEqual(b, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(fv, -28.0 / 3.0, places=6)

    def test_train_converges(self):
        a, b, fv = train_2d(lr=0.1, steps=300)
        self.assertAlmostEqual(a, 8.0 / 3.0, places=2)
        self.assertAlmostEqual(b, 2.0 / 3.0, places=2)
        self.assertAlmostEqual(fv, -28.0 / 3.0, places=2)


if __name__ == "__main__":
    unittest.main()
