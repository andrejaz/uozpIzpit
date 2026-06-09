import unittest

from obogatenost import enrichment, count_significant


class UnitTests(unittest.TestCase):

    def setUp(self):
        self.all_items = set(range(12))
        self.selected = {0, 1, 2, 3, 11}

    def test_enrichment_book_example(self):
        e = enrichment(self.all_items, self.selected, set(range(5)))
        self.assertEqual(e["k"], 4)
        self.assertEqual(e["K"], 5)
        self.assertAlmostEqual(e["expected"], 5 * 5 / 12, places=4)
        self.assertAlmostEqual(e["p_value"], 36 / 792, places=4)

    def test_count_significant(self):
        groups = [set(range(5)), {5, 6, 7, 8, 9, 10, 11}]
        c = count_significant(self.all_items, self.selected, groups, alpha=0.05)
        self.assertEqual(c, 1)


if __name__ == "__main__":
    unittest.main()
