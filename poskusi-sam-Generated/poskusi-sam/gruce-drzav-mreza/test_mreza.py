import unittest

import numpy as np

from mreza import cosine_distance, build_graph, label_propagation


class UnitTests(unittest.TestCase):

    def test_cosine_distance(self):
        self.assertAlmostEqual(cosine_distance([1.0, 0.0], [1.0, 0.0]), 0.0, places=6)
        self.assertAlmostEqual(cosine_distance([1.0, 0.0], [0.0, 1.0]), 1.0, places=6)

    def test_build_graph_groups(self):
        X = np.array([[1.0, 0.0], [1.0, 0.05], [0.0, 1.0], [0.05, 1.0]])
        adj = build_graph(X, threshold=0.1)
        self.assertIn(1, adj[0])
        self.assertNotIn(2, adj[0])

    def test_label_propagation_two_cliques(self):
        adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1},
               3: {4, 5}, 4: {3, 5}, 5: {3, 4}}
        labels = label_propagation(adj, seed=0)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[4], labels[5])
        self.assertNotEqual(labels[0], labels[3])


if __name__ == "__main__":
    unittest.main()
