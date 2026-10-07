"""Country and continent carbon-intensity lookup for LN nodes."""

import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "global_energy_mix.json"

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

    def __init__(self, path=DEFAULT_DATASET_PATH):
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
        country = self.normalize_country_code(country_code)
        if country in self.country_intensities:
            value = self.country_intensities[country]
            # Match the reference implementation's fallback for zero/missing data.
            if value > 0.0:
                return value, "country", country

        continent = (
            str(continent_code).strip().upper()
            if continent_code is not None
            else None
        )
        if continent is None:
            continent = self.continent_for(country_code)
        if continent in self.continent_averages:
            return self.continent_averages[continent], "continent", country

        return self.world_average, "world", country
