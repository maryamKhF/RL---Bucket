"""The bounded observation search must retain the original channel ranking."""

import random
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from RL.state import State


class StateSelectionTests(unittest.TestCase):
    def assert_same_observation(self, graph, source, destination, k):
        state = State(graph, source, destination, k=k, graph_is_local=True)
        expected = state._observable_channels()[:k]
        actual = state._observable_channels(limit=k)
        self.assertEqual([item[:3] for item in actual], [item[:3] for item in expected])
        for old, new in zip(expected, actual):
            self.assertIs(old[3], new[3])
        fast = state.vector()
        with patch.object(state, "_nearest_local_channels", return_value=expected):
            np.testing.assert_array_equal(fast, state.vector())

    def test_matches_full_ranking_across_layers_parallel_edges_and_disconnected_nodes(self):
        for graph_type in (nx.DiGraph, nx.MultiDiGraph):
            for seed in range(12):
                rng = random.Random(seed)
                graph = graph_type()
                graph.add_nodes_from(range(35))
                for _ in range(100):
                    graph.add_edge(rng.randrange(30), rng.randrange(30),
                                   capacity=rng.choice([0, 10, 20, 100]))
                local = graph.subgraph(set(range(34)))
                for source, destination in ((0, 1), (0, 33), (32, 33), (99, 1)):
                    for k in (1, 15, 200):
                        with self.subTest(kind=graph_type.__name__, seed=seed,
                                          terminals=(source, destination), k=k):
                            self.assert_same_observation(local, source, destination, k)

    def test_live_attributes_and_capacity_ties_are_preserved(self):
        graph = nx.MultiDiGraph()
        for node in range(1, 25):
            graph.add_edge(0, node, key="a", capacity=10)
            graph.add_edge(node, 0, key="b", capacity=10)
        self.assert_same_observation(graph, 0, 1, 15)
        graph[0][24]["a"]["capacity"] = 1000
        self.assert_same_observation(graph, 0, 1, 15)
        chosen = State(graph, 0, 1, graph_is_local=True)._observable_channels(limit=15)
        self.assertEqual(chosen[0][:3], (0, 24, "a"))


if __name__ == "__main__":
    unittest.main()
