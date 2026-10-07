"""
Simulation/failure_model.py

Lightning Network failure simulation model.

Responsibilities
----------------
1. Assign static failure probabilities to channels.
2. Evaluate node/channel/liquidity failures during one payment.
3. Return the exact failed element and failure position.
4. Provide failure information required by Partial Backtracking.
5. Reset temporary runtime failure state between episodes.

Architecture
------------
The public NetworkX graph contains topology and public channel state. Hidden
directional liquidity is read from the private simulator ledger supplied by
PaymentSimulator; it is never exposed to the routing policy.

FailureModel is intentionally separated from:

    - routing
    - Bucket
    - Partial Backtracking
    - PPO
    - Onion routing
    - settlement

Important semantic rules
------------------------
- Channel capacity is NOT directional liquidity.
- Snapshot capacity is never used as a directional balance.
- The simulator's private balance causes deterministic failure when an
  attempted amount exceeds the hidden balance.
- Sufficient hidden liquidity may still experience stochastic forwarding
  failure.
- Exact MultiDiGraph channel keys are mandatory when the graph
  contains parallel channels.
- FailureModel evaluates exactly ONE payment attempt.
- FailureModel never selects an alternative route.
- FailureModel never modifies Bucket state.
- FailureModel never performs backtracking.
- FailureModel never performs settlement.
- Explicitly malformed probability attributes are rejected.
- No arbitrary parallel channel is selected silently.
- A node's stochastic failure state is evaluated at most once
  during a single payment attempt.
- Geographic intercontinental classification uses country
  information first and Haversine distance only as fallback.
"""

from __future__ import annotations

from datetime import datetime
import math
import random
from typing import Any

import numpy as np


# ============================================================
# Static Failure Probability Assignment
# ============================================================

def assign_failure_probabilities(
    G,
    average_rate,
    seed=42,
):
    """
    Assign deterministic channel-specific failure probabilities.

    Static channel attributes
    -------------------------
    simulator_failure_probability

    Runtime channel attributes
    --------------------------
    available
    failure_count
    last_failure

    Parameters
    ----------
    G : networkx.Graph
        Network graph.

    average_rate : float
        Mean channel failure probability in [0, 1].

    seed : int
        Deterministic NumPy RNG seed.

    Returns
    -------
    networkx.Graph
        The same graph instance.
    """

    if G is None:
        raise ValueError(
            "G cannot be None."
        )

    average_rate = _validate_probability(
        average_rate,
        "average_rate",
    )

    seed = _validate_seed(
        seed
    )

    rng = np.random.default_rng(
        seed
    )

    if G.is_multigraph():

        for u, v, key, data in G.edges(
            keys=True,
            data=True,
        ):

            if u == v:
                continue

            probability = _calculate_failure_probability(
                G=G,
                u=u,
                v=v,
                average_rate=average_rate,
                rng=rng,
            )

            data.pop("failure_probability", None)
            data["simulator_failure_probability"] = (
                probability
            )

            data["available"] = True
            data["failure_count"] = 0
            data["last_failure"] = None

    else:

        for u, v, data in G.edges(
            data=True,
        ):

            if u == v:
                continue

            probability = _calculate_failure_probability(
                G=G,
                u=u,
                v=v,
                average_rate=average_rate,
                rng=rng,
            )

            data.pop("failure_probability", None)
            data["simulator_failure_probability"] = (
                probability
            )

            data["available"] = True
            data["failure_count"] = 0
            data["last_failure"] = None

    return G


# ============================================================
# Probability Validation
# ============================================================

def _validate_probability(
    value,
    name,
):
    """
    Validate probability in [0, 1].

    bool is rejected explicitly.
    """

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be numeric, not bool."
        )

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            f"{name} must be numeric."
        ) from exc

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be in [0, 1]."
        )

    return value


# ============================================================
# Seed Validation
# ============================================================

def _validate_seed(
    seed,
):
    """
    Validate integer RNG seed.

    bool is rejected.
    """

    if isinstance(
        seed,
        bool,
    ):
        raise TypeError(
            "seed must be an integer."
        )

    if not isinstance(
        seed,
        (int, np.integer),
    ):
        raise TypeError(
            "seed must be an integer."
        )

    return int(seed)


# ============================================================
# Amount Validation
# ============================================================

def _validate_amount(
    amount,
):
    """
    Validate positive finite payment amount.

    Returns
    -------
    float or None
        Valid amount, otherwise None.
    """

    if isinstance(
        amount,
        bool,
    ):
        return None

    try:
        amount = float(amount)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        amount
    ):
        return None

    if amount <= 0.0:
        return None

    return amount


# ============================================================
# Failure Probability Calculation
# ============================================================

def _calculate_failure_probability(
    G,
    u,
    v,
    average_rate,
    rng,
):
    """
    Calculate channel failure probability.

    Factors
    -------
    Same country:
        0.7

    Intercontinental:
        2.0

    Other:
        1.0

    Random multiplier:
        uniform(0.8, 1.2)

    Final value is clipped to [0, 0.95].
    """

    country_u = G.nodes[u].get(
        "country"
    )

    country_v = G.nodes[v].get(
        "country"
    )

    normalized_country_u = (
        country_u.strip().upper()
        if isinstance(country_u, str)
        else None
    )

    normalized_country_v = (
        country_v.strip().upper()
        if isinstance(country_v, str)
        else None
    )

    if (
        normalized_country_u
        and normalized_country_v
        and normalized_country_u
        == normalized_country_v
    ):

        factor = 0.7

    elif _intercontinental(
        G,
        u,
        v,
    ):

        factor = 2.0

    else:

        factor = 1.0

    probability = (
        float(average_rate)
        * factor
        * float(
            rng.uniform(
                0.8,
                1.2,
            )
        )
    )

    return float(
        min(
            0.95,
            max(
                0.0,
                probability,
            ),
        )
    )


# ============================================================
# Country -> Continent
# ============================================================

def _country_to_continent(
    country,
):
    """
    Convert ISO-like country code to a continent.

    Unknown countries return None.

    This mapping is deliberately explicit. It does not infer
    a continent from an unknown country code.
    """

    if not isinstance(
        country,
        str,
    ):
        return None

    country = country.strip().upper()

    if not country:
        return None

    country_to_continent = {

        # ----------------------------------------------------
        # North America
        # ----------------------------------------------------

        "US": "North America",
        "CA": "North America",
        "MX": "North America",
        "GT": "North America",
        "BZ": "North America",
        "SV": "North America",
        "HN": "North America",
        "NI": "North America",
        "CR": "North America",
        "PA": "North America",
        "CU": "North America",
        "JM": "North America",
        "HT": "North America",
        "DO": "North America",

        # ----------------------------------------------------
        # South America
        # ----------------------------------------------------

        "BR": "South America",
        "AR": "South America",
        "CL": "South America",
        "CO": "South America",
        "PE": "South America",
        "VE": "South America",
        "EC": "South America",
        "BO": "South America",
        "PY": "South America",
        "UY": "South America",
        "GY": "South America",
        "SR": "South America",

        # ----------------------------------------------------
        # Europe
        # ----------------------------------------------------

        "FR": "Europe",
        "DE": "Europe",
        "GB": "Europe",
        "IT": "Europe",
        "ES": "Europe",
        "PT": "Europe",
        "NL": "Europe",
        "BE": "Europe",
        "CH": "Europe",
        "AT": "Europe",
        "SE": "Europe",
        "NO": "Europe",
        "DK": "Europe",
        "FI": "Europe",
        "PL": "Europe",
        "CZ": "Europe",
        "SK": "Europe",
        "HU": "Europe",
        "RO": "Europe",
        "BG": "Europe",
        "GR": "Europe",
        "IE": "Europe",
        "IS": "Europe",
        "LU": "Europe",
        "SI": "Europe",
        "HR": "Europe",
        "RS": "Europe",
        "BA": "Europe",
        "ME": "Europe",
        "AL": "Europe",
        "MK": "Europe",
        "EE": "Europe",
        "LV": "Europe",
        "LT": "Europe",
        "MT": "Europe",
        "CY": "Europe",
        "UA": "Europe",
        "MD": "Europe",

        # ----------------------------------------------------
        # Asia
        # ----------------------------------------------------

        "CN": "Asia",
        "JP": "Asia",
        "KR": "Asia",
        "IN": "Asia",
        "IR": "Asia",
        "TR": "Asia",
        "AE": "Asia",
        "SA": "Asia",
        "IL": "Asia",
        "IQ": "Asia",
        "JO": "Asia",
        "LB": "Asia",
        "SY": "Asia",
        "YE": "Asia",
        "OM": "Asia",
        "QA": "Asia",
        "KW": "Asia",
        "BH": "Asia",
        "PK": "Asia",
        "BD": "Asia",
        "LK": "Asia",
        "NP": "Asia",
        "TH": "Asia",
        "VN": "Asia",
        "MY": "Asia",
        "SG": "Asia",
        "ID": "Asia",
        "PH": "Asia",
        "TW": "Asia",
        "HK": "Asia",
        "KZ": "Asia",
        "UZ": "Asia",
        "TM": "Asia",
        "KG": "Asia",
        "TJ": "Asia",
        "MN": "Asia",

        # ----------------------------------------------------
        # Africa
        # ----------------------------------------------------

        "EG": "Africa",
        "ZA": "Africa",
        "NG": "Africa",
        "KE": "Africa",
        "MA": "Africa",
        "DZ": "Africa",
        "TN": "Africa",
        "LY": "Africa",
        "ET": "Africa",
        "GH": "Africa",
        "TZ": "Africa",
        "UG": "Africa",
        "SN": "Africa",
        "CI": "Africa",
        "CM": "Africa",
        "SD": "Africa",

        # ----------------------------------------------------
        # Oceania
        # ----------------------------------------------------

        "AU": "Oceania",
        "NZ": "Oceania",
        "FJ": "Oceania",
        "PG": "Oceania",
        "WS": "Oceania",
        "TO": "Oceania",
    }

    return country_to_continent.get(
        country
    )


# ============================================================
# Geographic Helper
# ============================================================

def _intercontinental(
    G,
    u,
    v,
):
    """
    Estimate whether u -> v is intercontinental.

    Priority
    --------
    1. Country-based continent classification.
    2. Geographic-distance fallback.

    Country-based classification
    ----------------------------
    If both country codes are recognized, their mapped
    continents are compared directly.

    Geographic fallback
    --------------------
    If one or both countries are unknown, a Haversine
    great-circle distance is used.

    A distance >= 3000 km is treated as a geographic
    intercontinental proxy.

    Important
    ---------
    This is a simulation proxy, not a geographic database.
    """

    if G is None:
        return False

    try:

        if u not in G.nodes:
            return False

        if v not in G.nodes:
            return False

        node_u = G.nodes[u]
        node_v = G.nodes[v]

    except Exception:
        return False

    # --------------------------------------------------------
    # Country-based classification
    # --------------------------------------------------------

    country_u = node_u.get(
        "country"
    )

    country_v = node_v.get(
        "country"
    )

    continent_u = _country_to_continent(
        country_u
    )

    continent_v = _country_to_continent(
        country_v
    )

    if (
        continent_u is not None
        and continent_v is not None
    ):

        return (
            continent_u
            != continent_v
        )

    # --------------------------------------------------------
    # Geographic fallback
    # --------------------------------------------------------

    latitude_u = _finite_coordinate(
        node_u.get(
            "latitude"
        )
    )

    longitude_u = _finite_coordinate(
        node_u.get(
            "longitude"
        )
    )

    latitude_v = _finite_coordinate(
        node_v.get(
            "latitude"
        )
    )

    longitude_v = _finite_coordinate(
        node_v.get(
            "longitude"
        )
    )

    if (
        latitude_u is None
        or longitude_u is None
        or latitude_v is None
        or longitude_v is None
    ):
        return False

    # --------------------------------------------------------
    # Geographic range validation
    # --------------------------------------------------------

    if not (
        -90.0
        <= latitude_u
        <= 90.0
    ):
        raise ValueError(
            "latitude_u must be in [-90, 90]."
        )

    if not (
        -90.0
        <= latitude_v
        <= 90.0
    ):
        raise ValueError(
            "latitude_v must be in [-90, 90]."
        )

    if not (
        -180.0
        <= longitude_u
        <= 180.0
    ):
        raise ValueError(
            "longitude_u must be in [-180, 180]."
        )

    if not (
        -180.0
        <= longitude_v
        <= 180.0
    ):
        raise ValueError(
            "longitude_v must be in [-180, 180]."
        )

    # --------------------------------------------------------
    # Haversine great-circle distance
    # --------------------------------------------------------

    earth_radius_km = 6371.0

    latitude_u_rad = math.radians(
        latitude_u
    )

    latitude_v_rad = math.radians(
        latitude_v
    )

    delta_lat_rad = math.radians(
        latitude_v
        - latitude_u
    )

    delta_lon_rad = math.radians(
        longitude_v
        - longitude_u
    )

    haversine_a = (
        math.sin(
            delta_lat_rad / 2.0
        ) ** 2
        +
        math.cos(
            latitude_u_rad
        )
        *
        math.cos(
            latitude_v_rad
        )
        *
        math.sin(
            delta_lon_rad / 2.0
        ) ** 2
    )

    # Numerical protection for floating-point roundoff.
    haversine_a = min(
        1.0,
        max(
            0.0,
            haversine_a,
        ),
    )

    angular_distance = (
        2.0
        *
        math.atan2(
            math.sqrt(
                haversine_a
            ),
            math.sqrt(
                1.0
                -
                haversine_a
            ),
        )
    )

    distance_km = (
        earth_radius_km
        *
        angular_distance
    )

    return (
        distance_km >= 3000.0
    )


# ============================================================
# Coordinate Validation
# ============================================================

def _finite_coordinate(
    value,
):
    """
    Return a finite coordinate or None.

    bool is rejected.
    """

    if isinstance(
        value,
        bool,
    ):
        return None

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        value
    ):
        return None

    return value


# ============================================================
# Failure Model
# ============================================================

class FailureModel:
    """
    Runtime failure evaluator for one Lightning payment attempt.

    Failure categories
    ------------------
    node_failure
    channel_failure
    liquidity_failure
    missing_channel
    invalid_route
    invalid_amount
    invalid_edge
    invalid_route_edges
    edge_path_mismatch
    missing_channel_key
    unexpected_channel_key

    This class does not:

        - select routes
        - generate routes
        - manage Bucket
        - perform backtracking
        - choose PPO actions
        - perform settlement
    """

    def __init__(
        self,
        node_failure_probability=0.01,
        liquidity_failure_probability=0.05,
        seed=42,
    ):

        self.node_failure_probability = (
            _validate_probability(
                node_failure_probability,
                "node_failure_probability",
            )
        )

        self.liquidity_failure_probability = (
            _validate_probability(
                liquidity_failure_probability,
                "liquidity_failure_probability",
            )
        )

        self.seed = _validate_seed(
            seed
        )

        self.rng = random.Random(
            self.seed
        )

        # Set by PaymentSimulator to the simulator-only ledger. The agent and
        # public routing graph never receive this mapping.
        self.hidden_liquidity = None

        # Per-payment node evaluation cache.
        self._node_attempt_cache = None

    # ========================================================
    # Reset Runtime State
    # ========================================================

    def reset_runtime_state(
        self,
        G,
        reset_counters=False,
        reset_rng=False,
    ):
        """
        Reset temporary runtime graph state.

        Static simulator_failure_probability attributes are preserved.
        """

        if G is None:
            raise ValueError(
                "G cannot be None."
            )

        if not isinstance(
            reset_counters,
            bool,
        ):
            raise TypeError(
                "reset_counters must be bool."
            )

        if not isinstance(
            reset_rng,
            bool,
        ):
            raise TypeError(
                "reset_rng must be bool."
            )

        # ----------------------------------------------------
        # Channel state
        # ----------------------------------------------------

        if G.is_multigraph():

            for _, _, _, data in G.edges(
                keys=True,
                data=True,
            ):

                data["available"] = True

                if reset_counters:

                    data["failure_count"] = 0
                    data["last_failure"] = None

        else:

            for _, _, data in G.edges(
                data=True,
            ):

                data["available"] = True

                if reset_counters:

                    data["failure_count"] = 0
                    data["last_failure"] = None

        # ----------------------------------------------------
        # Node state
        # ----------------------------------------------------

        for _, data in G.nodes(
            data=True,
        ):

            if "available" in data:
                data["available"] = True

            if "is_online" in data:
                data["is_online"] = True

            if "online" in data:
                data["online"] = True

        # ----------------------------------------------------
        # Runtime attempt cache
        # ----------------------------------------------------

        self._node_attempt_cache = None

        # ----------------------------------------------------
        # RNG
        # ----------------------------------------------------

        if reset_rng:

            self.rng.seed(
                self.seed
            )

    # ========================================================
    # Node Failure
    # ========================================================

    def check_node_failure(
        self,
        G,
        node,
    ):
        """
        Evaluate node availability/failure.

        Within one payment attempt, a node is evaluated at
        most once.

        Returns
        -------
        bool
            True means node failure.
        """

        if G is None:
            return True

        if node not in G.nodes:
            return True

        # ----------------------------------------------------
        # Per-payment cache
        # ----------------------------------------------------

        if self._node_attempt_cache is not None:

            if node in self._node_attempt_cache:

                return (
                    self._node_attempt_cache[node]
                )

        data = G.nodes[node]

        # ----------------------------------------------------
        # Explicit runtime state
        # ----------------------------------------------------

        if data.get(
            "available",
            True,
        ) is False:

            result = True

        elif data.get(
            "is_online",
            True,
        ) is False:

            result = True

        elif data.get(
            "online",
            True,
        ) is False:

            result = True

        else:

            result = (
                self.rng.random()
                <
                self.node_failure_probability
            )

        if self._node_attempt_cache is not None:

            self._node_attempt_cache[node] = result

        return result

    # ========================================================
    # Channel Failure
    # ========================================================

    def check_channel_failure(
        self,
        G,
        u,
        v,
        key=None,
    ):
        """
        Evaluate exact channel failure.

        MultiGraph / MultiDiGraph requires key.
        """

        edge = self._get_edge(
            G,
            u,
            v,
            key,
        )

        if edge is None:
            return True

        if edge.get(
            "available",
            True,
        ) is False:

            return True

        probability_field = (
            "simulator_failure_probability"
            if "simulator_failure_probability" in edge
            else "failure_probability"
        )

        if probability_field not in edge:

            raise ValueError(
                "Channel is missing required "
                "simulator failure-probability attribute."
            )

        probability = _validate_probability(
            edge[probability_field],
            probability_field,
        )

        failed = (
            self.rng.random()
            <
            probability
        )

        if failed:

            failure_count = edge.get(
                "failure_count",
                0,
            )

            if isinstance(
                failure_count,
                bool,
            ):
                raise TypeError(
                    "failure_count must be an integer."
                )

            if not isinstance(
                failure_count,
                int,
            ):
                raise TypeError(
                    "failure_count must be an integer."
                )

            if failure_count < 0:
                raise ValueError(
                    "failure_count cannot be negative."
                )

            edge["failure_count"] = (
                failure_count + 1
            )

            edge["last_failure"] = (
                datetime.now()
            )

            edge["available"] = False

        return failed

    # ========================================================
    # Directional Liquidity
    # ========================================================

    def _get_directional_liquidity(
        self,
        edge,
    ):
        """
        Return known directional liquidity.

        Priority
        --------
        1. balance_uv
        2. liquidity_uv
        3. liquidity
        4. estimated_liquidity

        capacity is intentionally excluded.

        Returns
        -------
        float or None
            None means unknown.
        """

        if not isinstance(
            edge,
            dict,
        ):
            raise TypeError(
                "edge must be a dictionary."
            )

        for field in (
            "balance_uv",
            "liquidity_uv",
            "liquidity",
            "estimated_liquidity",
        ):

            if field not in edge:
                continue

            value = edge[field]

            if value is None:
                continue

            if isinstance(
                value,
                bool,
            ):
                raise TypeError(
                    f"{field} must be numeric, not bool."
                )

            try:
                value = float(value)

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"{field} must be numeric."
                ) from exc

            if not math.isfinite(
                value
            ):
                raise ValueError(
                    f"{field} must be finite."
                )

            if value < 0:
                raise ValueError(
                    f"{field} cannot be negative."
                )

            return value

        return None

    # ========================================================
    # Liquidity Failure
    # ========================================================

    def check_liquidity_failure(
        self,
        G,
        u,
        v,
        amount,
        key=None,
    ):
        """
        Evaluate directional liquidity and stochastic
        forwarding failure.

        Rules
        -----
        capacity:
            ignored

        hidden simulator balance < amount:
            deterministic failure

        sufficient hidden balance:
            stochastic forwarding failure may still occur

        With a simulator ledger, a missing entry means unknown and only the
        configured stochastic failure applies. Direct legacy calls without a
        ledger may still use explicit balance fields on the graph.
        """

        edge = self._get_edge(
            G,
            u,
            v,
            key,
        )

        if edge is None:
            return True

        valid_amount = _validate_amount(
            amount
        )

        if valid_amount is None:
            return True

        hidden_ledger = self.hidden_liquidity
        edge_identity = (u, v, key)
        if hidden_ledger is not None:
            directional_liquidity = hidden_ledger.get(edge_identity)
        else:
            directional_liquidity = self._get_directional_liquidity(edge)

        if directional_liquidity is not None:

            if valid_amount > directional_liquidity:
                return True

        return (
            self.rng.random()
            <
            self.liquidity_failure_probability
        )

    # ========================================================
    # Single Edge Evaluation
    # ========================================================

    def evaluate_edge(
        self,
        G,
        u,
        v,
        amount,
        key=None,
    ):
        """
        Evaluate exactly one directed channel traversal.
        """

        valid_amount = _validate_amount(
            amount
        )

        if valid_amount is None:

            return {
                "success": False,
                "reason": "invalid_amount",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key,
                ),
            }

        edge = self._get_edge(
            G,
            u,
            v,
            key,
        )

        if edge is None:

            return {
                "success": False,
                "reason": "missing_channel",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key,
                ),
            }

        if edge.get(
            "available",
            True,
        ) is False:

            return {
                "success": False,
                "reason": "channel_failure",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key,
                ),
            }

        # ----------------------------------------------------
        # Source node
        # ----------------------------------------------------

        if self.check_node_failure(
            G,
            u,
        ):

            return {
                "success": False,
                "reason": "node_failure",
                "failed_node": u,
                "failed_edge": None,
            }

        # ----------------------------------------------------
        # Destination node
        # ----------------------------------------------------

        if self.check_node_failure(
            G,
            v,
        ):

            return {
                "success": False,
                "reason": "node_failure",
                "failed_node": v,
                "failed_edge": None,
            }

        # ----------------------------------------------------
        # Exact channel
        # ----------------------------------------------------

        if self.check_channel_failure(
            G,
            u,
            v,
            key,
        ):

            return {
                "success": False,
                "reason": "channel_failure",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key,
                ),
            }

        # ----------------------------------------------------
        # Directional liquidity / forwarding
        # ----------------------------------------------------

        if self.check_liquidity_failure(
            G,
            u,
            v,
            valid_amount,
            key,
        ):

            return {
                "success": False,
                "reason": "liquidity_failure",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key,
                ),
            }

        # ----------------------------------------------------
        # Successful edge evaluation
        #
        # IMPORTANT:
        # PaymentSimulator requires reason to always be a
        # string. Therefore successful evaluations explicitly
        # return "success" rather than None.
        # ----------------------------------------------------

        return {
            "success": True,
            "reason": "success",
            "failed_node": None,
            "failed_edge": None,
        }

    # ========================================================
    # Full Payment Evaluation
    # ========================================================

    def evaluate_payment_failure(
        self,
        route,
        amount,
        network,
        route_edges=None,
    ):
        """
        Evaluate exactly one payment attempt.

        Exact route-edge information is required for a
        MultiGraph / MultiDiGraph.

        The node cache is active only during this method and is
        cleared in the finally block.

        route_edges may contain:

            (u, v)

            (u, v, key)

        or dictionary edge records such as:

            {
                "source": u,
                "target": v,
                "channel_key": key,
                "scid": ...,
                "data": ...
            }

        The original route-edge representation is preserved in
        visited_edges.
        """

        if network is None:

            raise ValueError(
                "network cannot be None."
            )

        if not isinstance(
            route,
            (list, tuple),
        ):

            return self._failure_result(
                reason="invalid_route"
            )

        route = list(route)

        if len(route) < 2:

            return self._failure_result(
                reason="invalid_route"
            )

        valid_amount = _validate_amount(
            amount
        )

        if valid_amount is None:

            return self._failure_result(
                reason="invalid_amount"
            )

        # ----------------------------------------------------
        # Start one-payment node cache.
        # ----------------------------------------------------

        previous_cache = (
            self._node_attempt_cache
        )

        self._node_attempt_cache = {}

        try:

            visited_edges = []

            # ------------------------------------------------
            # Resolve route edges.
            # ------------------------------------------------

            if route_edges is None:

                route_edges = (
                    self._resolve_route_edges(
                        network,
                        route,
                    )
                )

            if route_edges is None:

                return self._failure_result(
                    reason="invalid_route_edges",
                    failure_index=0,
                    visited_edges=[],
                )

            try:

                route_edges = list(
                    route_edges
                )

            except TypeError:

                return self._failure_result(
                    reason="invalid_route_edges",
                    failure_index=0,
                    visited_edges=[],
                )

            if len(route_edges) != (
                len(route) - 1
            ):

                return self._failure_result(
                    reason="invalid_route_edges",
                    failure_index=0,
                    visited_edges=[],
                )

            # ------------------------------------------------
            # Sequential hop evaluation
            # ------------------------------------------------

            for index, edge_info in enumerate(
                route_edges
            ):

                parsed = self._parse_edge(
                    edge_info
                )

                if parsed is None:

                    return self._failure_result(
                        reason="invalid_edge",
                        failed_edge=edge_info,
                        failure_index=index,
                        visited_edges=visited_edges,
                    )

                u, v, key = parsed

                expected_u = route[index]
                expected_v = route[index + 1]

                if (
                    u != expected_u
                    or
                    v != expected_v
                ):

                    return self._failure_result(
                        reason="edge_path_mismatch",
                        failed_edge=(
                            u,
                            v,
                            key,
                        ),
                        failure_index=index,
                        visited_edges=visited_edges,
                    )

                # --------------------------------------------
                # MultiGraph exact-key requirement
                # --------------------------------------------

                if network.is_multigraph():

                    if key is None:

                        return self._failure_result(
                            reason="missing_channel_key",
                            failed_edge=(
                                u,
                                v,
                                key,
                            ),
                            failure_index=index,
                            visited_edges=visited_edges,
                        )

                else:

                    if key is not None:

                        return self._failure_result(
                            reason="unexpected_channel_key",
                            failed_edge=(
                                u,
                                v,
                                key,
                            ),
                            failure_index=index,
                            visited_edges=visited_edges,
                        )

                result = self.evaluate_edge(
                    network,
                    u,
                    v,
                    valid_amount,
                    key,
                )

                if not isinstance(
                    result,
                    dict,
                ):

                    raise TypeError(
                        "evaluate_edge() must return dict."
                    )

                if "success" not in result:

                    raise ValueError(
                        "evaluate_edge() result must contain "
                        "'success'."
                    )

                if not isinstance(
                    result["success"],
                    bool,
                ):

                    raise TypeError(
                        "evaluate_edge() success must be bool."
                    )

                # ------------------------------------------------
                # Validate result reason contract.
                #
                # PaymentSimulator also requires reason to be
                # a string, including on success.
                # ------------------------------------------------

                if "reason" not in result:

                    raise ValueError(
                        "evaluate_edge() result must contain "
                        "'reason'."
                    )

                if not isinstance(
                    result["reason"],
                    str,
                ):

                    raise TypeError(
                        "evaluate_edge() reason must be a string."
                    )

                if not result["success"]:

                    return self._failure_result(
                        reason=result["reason"],
                        failed_node=result.get(
                            "failed_node"
                        ),
                        failed_edge=result.get(
                            "failed_edge"
                        ),
                        failure_index=index,
                        visited_edges=visited_edges,
                    )

                # ------------------------------------------------
                # Preserve the original edge representation.
                #
                # This is important for Top-K / Bucket edges
                # represented as dictionaries containing:
                #
                # source
                # target
                # channel_key
                # scid
                # data
                #
                # Do not collapse these to (u, v, key).
                # ------------------------------------------------

                visited_edges.append(
                    edge_info
                )

            # ------------------------------------------------
            # Successful payment evaluation.
            #
            # IMPORTANT:
            # reason must be a string because PaymentSimulator
            # validates the FailureModel result contract.
            # ------------------------------------------------

            return {
                "success": True,
                "reason": "success",
                "failed_node": None,
                "failed_edge": None,
                "failure_index": None,
                "visited_edges": list(
                    visited_edges
                ),
            }

        finally:

            self._node_attempt_cache = (
                previous_cache
            )

    # ========================================================
    # Failure Result
    # ========================================================

    @staticmethod
    def _failure_result(
        reason,
        failed_node=None,
        failed_edge=None,
        failure_index=None,
        visited_edges=None,
    ):
        """
        Build normalized failure result.
        """

        if not isinstance(
            reason,
            str,
        ):
            raise TypeError(
                "reason must be a string."
            )

        if visited_edges is None:
            visited_edges = []

        if not isinstance(
            visited_edges,
            (list, tuple),
        ):
            raise TypeError(
                "visited_edges must be a list or tuple."
            )

        return {
            "success": False,
            "reason": reason,
            "failed_node": failed_node,
            "failed_edge": failed_edge,
            "failure_index": failure_index,
            "visited_edges": list(
                visited_edges
            ),
        }

    # ========================================================
    # Route Edge Resolution
    # ========================================================

    def _resolve_route_edges(
        self,
        G,
        route,
    ):
        """
        Resolve route edges only when this is unambiguous.

        Simple Graph / DiGraph
        ----------------------
        A node path uniquely identifies an edge.

        MultiGraph / MultiDiGraph
        -------------------------
        A node path does NOT uniquely identify a channel.

        Therefore this method returns None for a MultiGraph
        when exact route_edges were not supplied.

        This prevents silent parallel-channel selection.
        """

        if G is None:
            return None

        if not isinstance(
            route,
            (list, tuple),
        ):
            return None

        if len(route) < 2:
            return None

        # ----------------------------------------------------
        # Parallel channels require exact identity.
        # ----------------------------------------------------

        if G.is_multigraph():
            return None

        edges = []

        for i in range(
            len(route) - 1
        ):

            u = route[i]
            v = route[i + 1]

            try:

                if not G.has_edge(
                    u,
                    v,
                ):
                    return None

                data = G.get_edge_data(
                    u,
                    v,
                )

            except Exception:

                return None

            if data is None:
                return None

            if data.get(
                "available",
                True,
            ) is False:

                return None

            edges.append(
                (
                    u,
                    v,
                    None,
                )
            )

        return edges

    # ========================================================
    # Edge Parser
    # ========================================================

    @staticmethod
    def _parse_edge(
        edge_info,
    ):
        """
        Normalize an edge representation.

        Accepted representations
        -------------------------
        Dictionary:
            {
                "source": u,
                "target": v,
                "channel_key": key,
                ...
            }

        Compatible dictionary aliases:
            {
                "u": u,
                "v": v,
                "key": key,
                ...
            }

        Tuple:
            (u, v)

        Keyed tuple:
            (u, v, key)

        Returns
        -------
        tuple or None
            Canonical form:

                (u, v, key)
        """

        if edge_info is None:
            return None

        # ----------------------------------------------------
        # Dictionary edge representation
        # ----------------------------------------------------

        if isinstance(
            edge_info,
            dict,
        ):

            if "source" in edge_info:

                u = edge_info.get(
                    "source"
                )

            elif "u" in edge_info:

                u = edge_info.get(
                    "u"
                )

            else:

                return None

            if "target" in edge_info:

                v = edge_info.get(
                    "target"
                )

            elif "v" in edge_info:

                v = edge_info.get(
                    "v"
                )

            else:

                return None

            if "channel_key" in edge_info:

                key = edge_info.get(
                    "channel_key"
                )

            elif "key" in edge_info:

                key = edge_info.get(
                    "key"
                )

            else:

                key = None

            if u is None or v is None:
                return None

            return (
                u,
                v,
                key,
            )

        # ----------------------------------------------------
        # Tuple / list edge representation
        # ----------------------------------------------------

        if isinstance(
            edge_info,
            (list, tuple),
        ):

            if len(edge_info) == 2:

                return (
                    edge_info[0],
                    edge_info[1],
                    None,
                )

            if len(edge_info) == 3:

                return (
                    edge_info[0],
                    edge_info[1],
                    edge_info[2],
                )

            return None

        return None

    # ========================================================
    # Exact Graph Edge Access
    # ========================================================

    @staticmethod
    def _get_edge(
        G,
        u,
        v,
        key=None,
    ):
        """
        Return exact edge data.

        MultiGraph / MultiDiGraph
        -------------------------
        key is mandatory.

        Graph / DiGraph
        ---------------
        key must be None.

        No arbitrary parallel channel is selected.
        """

        if G is None:
            return None

        try:

            if not G.has_edge(
                u,
                v,
            ):
                return None

        except Exception:

            return None

        # ----------------------------------------------------
        # MultiGraph / MultiDiGraph
        # ----------------------------------------------------

        if G.is_multigraph():

            if key is None:
                return None

            try:

                return G.get_edge_data(
                    u,
                    v,
                    key,
                )

            except Exception:

                return None

        # ----------------------------------------------------
        # Graph / DiGraph
        # ----------------------------------------------------

        if key is not None:
            return None

        try:

            return G.get_edge_data(
                u,
                v,
            )

        except Exception:

            return None


# ============================================================
# Deterministic Standalone Test
# ============================================================

def _run_standalone_test():
    """
    Deterministic smoke test.
    """

    import networkx as nx

    print()
    print("=" * 72)
    print("FAILURE MODEL STANDALONE TEST")
    print("=" * 72)

    G = nx.MultiDiGraph()

    G.add_node(
        "A",
        available=True,
        country="US",
        latitude=40,
        longitude=-74,
    )

    G.add_node(
        "B",
        available=True,
        country="US",
        latitude=41,
        longitude=-73,
    )

    G.add_node(
        "C",
        available=True,
        country="FR",
        latitude=48,
        longitude=2,
    )

    G.add_edge(
        "A",
        "B",
        key=0,
        capacity=10000,
        balance_uv=500,
        available=True,
        failure_probability=0.0,
        failure_count=0,
        last_failure=None,
    )

    G.add_edge(
        "A",
        "B",
        key=1,
        capacity=10000,
        balance_uv=10000,
        available=True,
        failure_probability=0.0,
        failure_count=0,
        last_failure=None,
    )

    G.add_edge(
        "B",
        "C",
        key=0,
        capacity=10000,
        available=True,
        failure_probability=0.0,
        failure_count=0,
        last_failure=None,
    )

    model = FailureModel(
        node_failure_probability=0.0,
        liquidity_failure_probability=0.0,
        seed=42,
    )

    # --------------------------------------------------------
    # 1. Capacity semantics
    # --------------------------------------------------------

    result = model.evaluate_payment_failure(
        route=["B", "C"],
        amount=5000,
        network=G,
        route_edges=[
            ("B", "C", 0),
        ],
    )

    assert result["success"] is True

    print(
        "[01] capacity is not liquidity                 PASS"
    )

    # --------------------------------------------------------
    # 2. Known insufficient liquidity
    # --------------------------------------------------------

    result = model.evaluate_payment_failure(
        route=["A", "B"],
        amount=600,
        network=G,
        route_edges=[
            ("A", "B", 0),
        ],
    )

    assert result["success"] is False
    assert result["reason"] == "liquidity_failure"
    assert result["failed_edge"] == (
        "A",
        "B",
        0,
    )
    assert result["failure_index"] == 0

    print(
        "[02] insufficient directional liquidity        PASS"
    )

    # --------------------------------------------------------
    # 3. Exact parallel channel
    # --------------------------------------------------------

    result = model.evaluate_payment_failure(
        route=["A", "B"],
        amount=600,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    assert result["success"] is True

    assert result["visited_edges"] == [
        ("A", "B", 1)
    ]

    print(
        "[03] exact parallel channel                    PASS"
    )

    # --------------------------------------------------------
    # 4. Missing key
    # --------------------------------------------------------

    result = model.evaluate_payment_failure(
        route=["A", "B"],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B"),
        ],
    )

    assert result["success"] is False
    assert result["reason"] == "missing_channel_key"

    print(
        "[04] missing MultiDiGraph key rejected          PASS"
    )

    # --------------------------------------------------------
    # 5. No automatic parallel selection
    # --------------------------------------------------------

    result = model.evaluate_payment_failure(
        route=["A", "B"],
        amount=100,
        network=G,
        route_edges=None,
    )

    assert result["success"] is False
    assert result["reason"] == "invalid_route_edges"

    print(
        "[05] no automatic parallel-channel selection   PASS"
    )

    # --------------------------------------------------------
    # 6. Failure index
    # --------------------------------------------------------

    G["B"]["C"][0]["failure_probability"] = 1.0

    result = model.evaluate_payment_failure(
        route=["A", "B", "C"],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
            ("B", "C", 0),
        ],
    )

    assert result["success"] is False

    assert result["reason"] == (
        "channel_failure"
    )

    assert result["failed_edge"] == (
        "B",
        "C",
        0,
    )

    assert result["failure_index"] == 1

    assert result["visited_edges"] == [
        ("A", "B", 1)
    ]

    print(
        "[06] failure index and visited edges            PASS"
    )

    # --------------------------------------------------------
    # 7. Runtime state
    # --------------------------------------------------------

    assert G["B"]["C"][0]["available"] is False

    assert G["B"]["C"][0]["failure_count"] == 1

    print(
        "[07] runtime channel failure state              PASS"
    )

    # --------------------------------------------------------
    # 8. Reset
    # --------------------------------------------------------

    model.reset_runtime_state(
        G,
        reset_counters=True,
        reset_rng=True,
    )

    assert G["B"]["C"][0]["available"] is True

    assert G["B"]["C"][0]["failure_count"] == 0

    assert G["B"]["C"][0]["last_failure"] is None

    print(
        "[08] runtime state reset                        PASS"
    )

    # --------------------------------------------------------
    # 9. Country-based classification
    # --------------------------------------------------------

    assert _intercontinental(
        G,
        "A",
        "B",
    ) is False

    assert _intercontinental(
        G,
        "B",
        "C",
    ) is True

    print(
        "[09] country-based continent classification    PASS"
    )

    # --------------------------------------------------------
    # 10. Invalid probability
    # --------------------------------------------------------

    G["A"]["B"][0]["failure_probability"] = "INVALID"

    raised = False

    try:

        model.check_channel_failure(
            G,
            "A",
            "B",
            0,
        )

    except ValueError:

        raised = True

    assert raised is True

    print(
        "[10] invalid probability rejected               PASS"
    )

    # --------------------------------------------------------
    # 11. Dictionary edge representation
    # --------------------------------------------------------

    model.reset_runtime_state(
        G,
        reset_counters=True,
        reset_rng=True,
    )

    dictionary_edge = {
        "source": "A",
        "target": "B",
        "channel_key": 1,
        "scid": "A-B-1",
        "data": {
            "capacity": 10000,
        },
    }

    result = model.evaluate_payment_failure(
        route=["A", "B"],
        amount=600,
        network=G,
        route_edges=[
            dictionary_edge
        ],
    )

    assert result["success"] is True

    assert result["visited_edges"] == [
        dictionary_edge
    ]

    print(
        "[11] dictionary edge + exact key preserved    PASS"
    )

    # --------------------------------------------------------
    # 12. Dictionary edge aliases
    # --------------------------------------------------------

    alias_edge = {
        "u": "A",
        "v": "B",
        "key": 1,
        "scid": "A-B-1",
    }

    result = model.evaluate_payment_failure(
        route=["A", "B"],
        amount=600,
        network=G,
        route_edges=[
            alias_edge
        ],
    )

    assert result["success"] is True

    assert result["visited_edges"] == [
        alias_edge
    ]

    print(
        "[12] dictionary aliases supported                PASS"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FAILURE MODEL STANDALONE TEST: PASS")
    print("=" * 72)


# ============================================================
# Standalone Execution
# ============================================================

if __name__ == "__main__":
    _run_standalone_test()
