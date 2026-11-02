
# test_main.py

import unittest
from pathlib import Path

from main import (
    geo_to_json,
    load_cfg,
    set_seed
)

from Network.graph_builder import LNGraphBuilder
from Simulation.transaction_generator import generate_transactions


class TestMainPipeline(unittest.TestCase):

    # ==================================================
    # Setup
    # ==================================================

    @classmethod
    def setUpClass(cls):

        # ----------------------------------------------
        # Repository root
        # ----------------------------------------------

        cls.geo_file = Path(
            "20190501.gml.geo"
        )

        # ----------------------------------------------
        # Check GML.GEO file
        # ----------------------------------------------

        if not cls.geo_file.exists():

            raise FileNotFoundError(
                f"GML.GEO file not found: "
                f"{cls.geo_file.resolve()}"
            )

        # ----------------------------------------------
        # Load configuration
        # ----------------------------------------------

        cls.cfg = load_cfg()

        cls.seed = cls.cfg["seed"]

        set_seed(
            cls.seed
        )


    # ==================================================
    # Test 1
    # GML.GEO file exists
    # ==================================================

    def test_geo_file_exists(self):

        self.assertTrue(
            self.geo_file.exists(),
            "20190501.gml.geo does not exist."
        )


    # ==================================================
    # Test 2
    # GML.GEO -> JSON-like structure
    # ==================================================

    def test_geo_to_json(self):

        data = geo_to_json(
            self.geo_file
        )

        # ----------------------------------------------
        # Basic structure
        # ----------------------------------------------

        self.assertIsInstance(
            data,
            dict
        )

        self.assertIn(
            "directed",
            data
        )

        self.assertIn(
            "multigraph",
            data
        )

        self.assertIn(
            "nodes",
            data
        )

        self.assertIn(
            "edges",
            data
        )

        # ----------------------------------------------
        # Network should not be empty
        # ----------------------------------------------

        self.assertGreater(
            len(data["nodes"]),
            0,
            "No nodes were found in GML.GEO."
        )

        self.assertGreater(
            len(data["edges"]),
            0,
            "No edges were found in GML.GEO."
        )


    # ==================================================
    # Test 3
    # Check node structure
    # ==================================================

    def test_node_structure(self):

        data = geo_to_json(
            self.geo_file
        )

        node = data["nodes"][0]

        self.assertIn(
            "id",
            node
        )


    # ==================================================
    # Test 4
    # Check Lightning channel structure
    # ==================================================

    def test_edge_structure(self):

        data = geo_to_json(
            self.geo_file
        )

        edge = data["edges"][0]

        # ----------------------------------------------
        # Required topology fields
        # ----------------------------------------------

        self.assertIn(
            "source",
            edge
        )

        self.assertIn(
            "target",
            edge
        )

        # ----------------------------------------------
        # Lightning-specific fields
        # ----------------------------------------------

        lightning_fields = [

            "scid",

            "fee_base_msat",

            "fee_proportional_millionths"

        ]

        found_fields = [

            field

            for field in lightning_fields

            if field in edge

        ]

        self.assertGreater(
            len(found_fields),
            0,
            "No Lightning-specific channel fields found."
        )


    # ==================================================
    # Test 5
    # Build NetworkX graph
    # ==================================================

    def test_graph_builder(self):

        data = geo_to_json(
            self.geo_file
        )

        builder = LNGraphBuilder()

        G = builder.from_data(
            data
        )

        # ----------------------------------------------
        # Graph must exist
        # ----------------------------------------------

        self.assertIsNotNone(
            G
        )

        # ----------------------------------------------
        # Graph must contain nodes
        # ----------------------------------------------

        self.assertGreater(
            G.number_of_nodes(),
            0,
            "Graph contains no nodes."
        )

        # ----------------------------------------------
        # Graph must contain channels
        # ----------------------------------------------

        self.assertGreater(
            G.number_of_edges(),
            0,
            "Graph contains no channels."
        )


    # ==================================================
    # Test 6
    # Check important graph attributes
    # ==================================================

    def test_graph_channel_attributes(self):

        data = geo_to_json(
            self.geo_file
        )

        builder = LNGraphBuilder()

        G = builder.from_data(
            data
        )

        # ----------------------------------------------
        # Get first edge
        # ----------------------------------------------

        source, target, attributes = next(
            iter(
                G.edges(
                    data=True
                )
            )
        )

        # ----------------------------------------------
        # Basic attributes
        # ----------------------------------------------

        self.assertIn(
            "capacity",
            attributes
        )

        self.assertIn(
            "fee_base",
            attributes
        )

        self.assertIn(
            "fee_rate",
            attributes
        )

        self.assertIn(
            "available",
            attributes
        )


    # ==================================================
    # Test 7
    # Generate transactions
    # ==================================================

    def test_transaction_generation(self):

        data = geo_to_json(
            self.geo_file
        )

        builder = LNGraphBuilder()

        G = builder.from_data(
            data
        )

        # ----------------------------------------------
        # Generate a small number of transactions
        # ----------------------------------------------

        transactions = generate_transactions(
            G,
            10,
            self.seed,
            self.cfg["simulation"]["min_amount"],
            self.cfg["simulation"]["max_amount"]
        )

        # ----------------------------------------------
        # Check result
        # ----------------------------------------------

        self.assertIsNotNone(
            transactions
        )

        self.assertEqual(
            len(transactions),
            10
        )


# ==================================================
# Run Tests
# ==================================================

if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )

