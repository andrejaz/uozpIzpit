import unittest

from lin_reg import diverged, find_critical_lr, make_data


class UnitTests(unittest.TestCase):

    def setUp(self):
        self.xs, self.ys = make_data(seed=42, n=5)

    def test_diverged_true_for_huge_lr(self):
        self.assertTrue(diverged(self.xs, self.ys, 1e9, 200))

    def test_diverged_false_for_tiny_lr(self):
        self.assertFalse(diverged(self.xs, self.ys, 1e-4, 200))

    def test_critical_lr_in_range(self):
        crit = find_critical_lr(self.xs, self.ys, n_epochs=200)
        self.assertGreater(crit, 0.0)
        self.assertFalse(diverged(self.xs, self.ys, crit / 50.0, 200))
        self.assertTrue(diverged(self.xs, self.ys, crit * 50.0, 200))


if __name__ == "__main__":
    unittest.main()
