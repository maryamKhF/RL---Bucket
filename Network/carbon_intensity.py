"""Location-aware carbon-intensity lookup for Lightning Network nodes.

An optional GeoJSON polygon layer provides location-specific values. The
project's country energy-mix file remains a coarser fallback, so each result
also carries its spatial resolution and reference year.
"""

import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "global_energy_mix.json"
DEFAULT_SPATIAL_DATASET_PATH = PROJECT_ROOT / "carbon_intensity.geojson"

# ISO-2 codes present in the project's geolocated Lightning snapshots.
ISO2_TO_ISO3 = {
    "AE": "ARE", "AG": "ATG", "AR": "ARG", "AT": "AUT",
    "AU": "AUS", "BA": "BIH", "BE": "BEL", "BG": "BGR",
    "BH": "BHR", "BR": "BRA", "BS": "BHS", "CA": "CAN",
    "CH": "CHE", "CL": "CHL", "CN": "CHN", "CO": "COL",
    "CW": "CUW", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
    "EE": "EST", "ES": "ESP", "FI": "FIN", "FR": "FRA",
    "GB": "GBR", "GH": "GHA", "GR": "GRC", "HK": "HKG",
    "HR": "HRV", "HU": "HUN", "ID": "IDN", "IE": "IRL",
    "IL": "ISR", "IM": "IMN", "IN": "IND", "IT": "ITA",
    "JP": "JPN", "KE": "KEN", "KR": "KOR", "LI": "LIE",
    "LK": "LKA", "LT": "LTU", "LU": "LUX", "LV": "LVA",
    "MD": "MDA", "MO": "MAC", "MT": "MLT", "MX": "MEX",
    "MY": "MYS", "NL": "NLD", "NO": "NOR", "NZ": "NZL",
    "PE": "PER", "PH": "PHL", "PL": "POL", "PR": "PRI",
    "PT": "PRT", "PY": "PRY", "RE": "REU", "RO": "ROU",
    "RU": "RUS", "SE": "SWE", "SG": "SGP", "SI": "SVN",
    "SK": "SVK", "TH": "THA", "TR": "TUR", "TW": "TWN",
    "UA": "UKR", "US": "USA", "UY": "URY", "VA": "VAT",
    "VE": "VEN", "VG": "VGB", "VN": "VNM", "ZA": "ZAF",
}

# Snapshot territories without a separate energy-mix record use
# their reference continent average, matching the paper code's fallback.
ISO2_CONTINENT_FALLBACK = {
    "CW": "NA",  # Curaçao
    "IM": "EU",  # Isle of Man
    "LI": "EU",  # Liechtenstein
    "RE": "AF",  # Réunion
    "VA": "EU",  # Vatican City
}


class CarbonIntensityDataset:
    """Resolve a node's intensity by country, continent, then world."""

    def __init__(self, path=DEFAULT_DATASET_PATH, spatial_path=DEFAULT_SPATIAL_DATASET_PATH):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Carbon-intensity dataset not found: {self.path}"
            )

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read carbon-intensity dataset: {self.path}"
            ) from exc

        if not isinstance(raw, dict):
            raise TypeError("Carbon-intensity dataset must be a JSON object.")

        self.country_intensities = {}
        self.country_continents = {}
        self.country_years = {}
        for code, record in raw.items():
            if code in {"world_average", "continent_average"}:
                continue
            if not isinstance(record, dict):
                continue
            normalized = str(record.get("iso_code", code)).strip().upper()
            intensity = self._finite_nonnegative(
                record.get("carbon_intensity")
            )
            if len(normalized) == 3 and intensity is not None:
                self.country_intensities[normalized] = intensity
                self.country_years[normalized] = self._valid_year(
                    record.get("year")
                )
            continent = record.get("continent")
            if len(normalized) == 3 and isinstance(continent, str) and continent:
                self.country_continents[normalized] = continent.strip().upper()

        supplied_continents = raw.get("continent_average", {})
        self.continent_averages = {}
        if isinstance(supplied_continents, dict):
            for code, value in supplied_continents.items():
                intensity = self._finite_nonnegative(value)
                if intensity is not None:
                    self.continent_averages[str(code).strip().upper()] = intensity

        # Derive averages when the JSON does not include precomputed values.
        grouped = {}
        for country, intensity in self.country_intensities.items():
            continent = self.country_continents.get(country)
            if continent:
                grouped.setdefault(continent, []).append(intensity)
        for continent, values in grouped.items():
            self.continent_averages.setdefault(
                continent,
                sum(values) / len(values),
            )

        self.world_average = self._finite_nonnegative(raw.get("world_average"))
        if self.world_average is None:
            values = list(self.country_intensities.values())
            if not values:
                raise ValueError(
                    "Dataset has no valid country carbon-intensity values."
                )
            self.world_average = sum(values) / len(values)

        self.spatial_path = Path(spatial_path) if spatial_path else None
        self.spatial_features = self._load_spatial_features(self.spatial_path)

    @staticmethod
    def _valid_year(value):
        try:
            year = int(value)
        except (TypeError, ValueError):
            return None
        return year if 1800 <= year <= 2200 else None

    @classmethod
    def _load_spatial_features(cls, path):
        if path is None or not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read spatial carbon dataset: {path}") from exc
        if not isinstance(raw, dict) or raw.get("type") != "FeatureCollection":
            raise TypeError("Spatial carbon dataset must be a GeoJSON FeatureCollection.")

        features = []
        for feature in raw.get("features", []):
            if not isinstance(feature, dict):
                continue
            properties = feature.get("properties") or {}
            intensity = cls._finite_nonnegative(
                properties.get("carbon_intensity", properties.get("intensity"))
            )
            geometry = feature.get("geometry") or {}
            geometry_type = geometry.get("type")
            coordinates = geometry.get("coordinates")
            if intensity is None or geometry_type not in {"Polygon", "MultiPolygon"}:
                continue
            features.append({
                "geometry_type": geometry_type,
                "coordinates": coordinates,
                "intensity": intensity,
                "year": cls._valid_year(properties.get("year", raw.get("year"))),
                "resolution": str(
                    properties.get("spatial_resolution", raw.get("spatial_resolution", "polygon"))
                ),
            })
        return features

    @staticmethod
    def _finite_nonnegative(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number < 0.0:
            return None
        return number

    @staticmethod
    def normalize_country_code(code):
        if code is None:
            return None
        normalized = str(code).strip().upper()
        if len(normalized) == 2:
            return ISO2_TO_ISO3.get(normalized)
        if len(normalized) == 3:
            return normalized
        return None

    def continent_for(self, country_code):
        country = self.normalize_country_code(country_code)
        if country in self.country_continents:
            return self.country_continents[country]
        if country_code is not None:
            return ISO2_CONTINENT_FALLBACK.get(
                str(country_code).strip().upper()
            )
        return None

    def lookup(self, country_code=None, continent_code=None):
        """Return ``(gCO2/kWh, source, normalized_country_code)``."""
        value, source, country, _resolution, _year = self.lookup_location(
            country_code=country_code,
            continent_code=continent_code,
        )
        return value, source, country

    @staticmethod
    def _point_in_ring(longitude, latitude, ring):
        """Return whether a lon/lat point lies inside a GeoJSON linear ring."""
        if not isinstance(ring, list) or len(ring) < 4:
            return False
        inside = False
        previous = ring[-1]
        for current in ring:
            try:
                x1, y1 = float(previous[0]), float(previous[1])
                x2, y2 = float(current[0]), float(current[1])
            except (TypeError, ValueError, IndexError):
                previous = current
                continue
            if (y1 > latitude) != (y2 > latitude):
                crossing = (x2 - x1) * (latitude - y1) / (y2 - y1) + x1
                if longitude < crossing:
                    inside = not inside
            previous = current
        return inside

    @classmethod
    def _point_in_polygon(cls, longitude, latitude, polygon):
        if not isinstance(polygon, list) or not polygon:
            return False
        if not cls._point_in_ring(longitude, latitude, polygon[0]):
            return False
        return not any(
            cls._point_in_ring(longitude, latitude, hole)
            for hole in polygon[1:]
        )

    @classmethod
    def _contains(cls, feature, longitude, latitude):
        coordinates = feature["coordinates"]
        if feature["geometry_type"] == "Polygon":
            return cls._point_in_polygon(longitude, latitude, coordinates)
        return any(
            cls._point_in_polygon(longitude, latitude, polygon)
            for polygon in coordinates or []
        )

    def lookup_location(
        self,
        latitude=None,
        longitude=None,
        country_code=None,
        continent_code=None,
    ):
        """Return intensity, source, country, resolution and reference year.

        Location polygons take precedence when valid coordinates are available.
        Country, continent and world estimates are explicitly marked as such.
        """
        try:
            lat = float(latitude)
            lon = float(longitude)
        except (TypeError, ValueError):
            lat = lon = None
        if (
            lat is not None
            and lon is not None
            and math.isfinite(lat)
            and math.isfinite(lon)
            and -90.0 <= lat <= 90.0
            and -180.0 <= lon <= 180.0
        ):
            for feature in self.spatial_features:
                if self._contains(feature, lon, lat):
                    return (
                        feature["intensity"],
                        "spatial",
                        self.normalize_country_code(country_code),
                        feature["resolution"],
                        feature["year"],
                    )

        country = self.normalize_country_code(country_code)
        if country in self.country_intensities:
            value = self.country_intensities[country]
            # Match the reference implementation's fallback for zero/missing data.
            if value > 0.0:
                return value, "country", country, "country", self.country_years.get(country)

        continent = (
            str(continent_code).strip().upper()
            if continent_code is not None
            else None
        )
        if continent is None:
            continent = self.continent_for(country_code)
        if continent in self.continent_averages:
            return (
                self.continent_averages[continent],
                "continent",
                country,
                "continent",
                None,
            )

        return self.world_average, "world", country, "world", None
