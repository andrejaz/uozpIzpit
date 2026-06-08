import math
import unittest

from value_tanh import Value


class UnitTests(unittest.TestCase):

    def setUp(self):
        self.a = Value(1.0, label="a")
        self.b = Value(0.5, label="b")
        self.c = Value(2.0, label="c")
        self.n = (self.a + self.b).tanh()
        self.out = self.n * self.c

    def test_value(self):
        self.assertAlmostEqual(self.out.data, math.tanh(1.5) * 2.0, places=6)

    def test_op_label(self):
        self.assertEqual(self.n._op, "tanh")

    def test_predecessors(self):
        self.assertIn(self.c, self.out._prev)
        self.assertIn(self.n, self.out._prev)
        # tanh ima natanko enega predhodnika (vsoto a + b)
        self.assertEqual(len(self.n._prev), 1)


if __name__ == "__main__":
    unittest.main()
