import unittest

from explain_tsne import (read_xy, read_groups,
                          mean_distance_group_allpairs,
                          unusual_group_pval)


class TestUnusual(unittest.TestCase):

    def test_mean_distance_group(self):

        locs = {
            "a": [1, 1],
            "b": [2, 2],
            "c": [1, 1]
        }

        d = mean_distance_group_allpairs(locs, ["a", "c"])
        self.assertAlmostEqual(d, 0)

        d = mean_distance_group_allpairs(locs, ["a", "b"])
        self.assertAlmostEqual(d, 1.4142135623730951)

        d = mean_distance_group_allpairs(locs, ["a", "b", "c"])
        self.assertAlmostEqual(d, 0.9428090415820635)

    def test_unusual_group_pval(self):

        tsne_xy = read_xy('climate-europe.tab')
        groups = read_groups('groups.yaml')

        northern = unusual_group_pval(tsne_xy, groups["Northern_Capitals"])
        river = unusual_group_pval(tsne_xy, groups["River_Cities"])
        scenic = unusual_group_pval(tsne_xy, groups["Scenic_Smaller_Capitals"])
        imperial = unusual_group_pval(tsne_xy, groups["Former_Imperial_Powerhouses"])

        self.assertLess(northern, 0.05)
        self.assertLess(river, 0.15)

        self.assertGreater(scenic, 0.75)
        self.assertGreater(imperial, 0.75)
        self.assertLess(scenic, 0.9)
        self.assertLess(imperial, 0.9)