import unittest

import numpy as np

from korelacije import fit, make_correlated_data, selected_features


class UnitTests(unittest.TestCase):

    def test_selected_features_basic(self):
        sel = selected_features(np.array([0.0, 1.0, 0.0, 2.0, 0.0]))
        self.assertEqual(set(sel), {1, 3})

    def test_stronger_reg_selects_fewer(self):
        X, y = make_correlated_data(seed=0)
        w_weak, _ = fit(X, y, reg="l1", reg_strength=0.02, seed=0)
        w_strong, _ = fit(X, y, reg="l1", reg_strength=0.6, seed=0)
        self.assertLessEqual(len(selected_features(w_strong)),
                             len(selected_features(w_weak)))


if __name__ == "__main__":
    unittest.main()
