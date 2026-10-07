"""Tests for spatial carbon matching and explicit coarse fallbacks."""

import json
import tempfile
import unittest
from pathlib import Path

from Network.carbon_intensity import CarbonIntensityDataset
from Network.graph_builder import LNGraphBuilder


class CarbonIntensityLocationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.country_path = self.root / "countries.json"
        self.country_path.write_text(
            json.dumps({
                "GBR": {
                    "iso_code": "GBR",
                    "carbon_intensity": 250,
                    "year": 2021,
                    "continent": "EU",
                },
                "world_average": 400,
                "continent_average": {"EU": 300},
            }),
            encoding="utf-8",
        )
        self.spatial_path = self.root / "grid.geojson"
        self.spatial_path.write_text(
            json.dumps({
                "type": "FeatureCollection",
                "year": 2024,
                "spatial_resolution": "10km grid",
                "features": [{
                    "type": "Feature",
                    "properties": {"carbon_intensity": 75},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [0, 0], [4, 0], [4, 4], [0, 4], [0, 0]
                        ]],
                    },
                }],
            }),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_coordinates_match_spatial_value_before_country_average(self):
        dataset = CarbonIntensityDataset(self.country_path, self.spatial_path)

        result = dataset.lookup_location(
            latitude=2,
            longitude=2,
            country_code="GB",
        )

        self.assertEqual(result, (75.0, "spatial", "GBR", "10km grid", 2024))

    def test_outside_spatial_coverage_is_labeled_country_resolution(self):
        dataset = CarbonIntensityDataset(self.country_path, self.spatial_path)

        result = dataset.lookup_location(
            latitude=10,
            longitude=10,
            country_code="GB",
        )

        self.assertEqual(result, (250.0, "country", "GBR", "country", 2021))

    def test_missing_country_uses_labeled_continent_or_world_fallback(self):
        dataset = CarbonIntensityDataset(self.country_path, self.spatial_path)

        continent = dataset.lookup_location(country_code="LI")
        world = dataset.lookup_location()

        self.assertEqual(continent, (300.0, "continent", "LIE", "continent", None))
        self.assertEqual(world, (400.0, "world", None, "world", None))

    def test_graph_builder_records_spatial_provenance_for_nodes(self):
        dataset = CarbonIntensityDataset(self.country_path, self.spatial_path)
        graph = LNGraphBuilder(carbon_dataset=dataset).from_data({
            "nodes": [{
                "id": "n1",
                "latitude": 2,
                "longitude": 2,
                "country_code": "GB",
            }],
            "channels": [],
        })

        attrs = graph.nodes["n1"]
        self.assertEqual(attrs["carbon_intensity"], 75.0)
        self.assertEqual(attrs["carbon_intensity_source"], "spatial")
        self.assertEqual(attrs["carbon_intensity_resolution"], "10km grid")
        self.assertEqual(attrs["carbon_intensity_year"], 2024)


if __name__ == "__main__":
    unittest.main()
