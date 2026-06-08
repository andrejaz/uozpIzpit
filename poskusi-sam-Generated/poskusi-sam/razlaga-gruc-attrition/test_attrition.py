import unittest

from attrition import mann_whitney_auc


class UnitTests(unittest.TestCase):

    def test_book_example(self):
        auc = mann_whitney_auc([1, 3, 4], [2, 5, 6, 7, 8])
        self.assertAlmostEqual(auc, 13 / 15, places=4)

    def test_symmetric_half(self):
        auc = mann_whitney_auc([1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(auc, 0.5, places=6)

    def test_all_smaller(self):
        auc = mann_whitney_auc([1, 2], [10, 20])
        self.assertAlmostEqual(auc, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
