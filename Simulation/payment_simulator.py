"""
Simulation/payment_simulator.py

Execute exactly ONE selected payment route over a simulated
Lightning Network.

Responsibilities
----------------
1. Validate the selected route.
2. Preserve exact channel identity.
3. Calculate deterministic route metrics.
4. Delegate forwarding/failure evaluation to FailureModel.
5. Commit successful settlement through NetworkDynamics.
6. Return a structured PaymentResult.
7. Expose exact failure information required by:
       - Bucket.backtrack()
       - Simulation.backtrack.PartialBacktracker

This module does NOT:
    - select candidates
    - manage Bucket state
    - perform candidate-level backtracking
    - perform partial route recovery
    - perform full rerouting
    - make PPO/RL decisions
    - retry a failed payment

Execution flow
--------------

    Bucket
       |
       v
selected candidate
       |
       v
Bucket.record_attempt()
       |
       v
PaymentSimulator
       |
       v
FailureModel
       |
       +---- failure ------> PaymentResult(False)
       |
       +---- forwarding OK
                    |
                    v
             NetworkDynamics
                    |
                    +---- rejected --> PaymentResult(False)
                    |
                    +---- accepted --> PaymentResult(True)

Channel identity
----------------
For MultiGraph / MultiDiGraph:

    (u, v, key)

is mandatory.

Top-K / Bucket may represent a physical edge as:

    {
        "source": u,
        "target": v,
        "channel_key": key,
        "scid": ...,
        "data": ...
    }

PaymentSimulator accepts both representations and internally
normalizes them to:

    (u, v, key)

The original edge representation is preserved when passing
route_edges to FailureModel and when storing it in PaymentResult.

Capacity
--------
Capacity is NOT interpreted as directional liquidity.

Strictness
----------
This module intentionally avoids silent numerical fallbacks.

A malformed explicitly supplied attribute is an error.

A missing optional metric is handled according to the
explicit contract of that metric rather than being silently
converted into a physical zero.

FailureModel and NetworkDynamics failures are not converted
into ordinary payment failures. Internal component errors
must remain visible to tests and callers.

Geographic / continent metric
-----------------------------
When country information is available and recognized, the
continent metric uses country -> continent classification.

When country information is unavailable or the country is
unknown, the simulator uses the deterministic geographic
proxy:

    abs(latitude_u  - latitude_v) > 10
    AND
    abs(longitude_u - longitude_v) > 45

The geographic rule is a fallback proxy only. It is not a
complete geographic database.
"""

import math
import time


# ============================================================
# Geographic / Country Helper
# ============================================================

def estimate_inter_continent(G, u, v):
    """
    Estimate whether a hop crosses continents.

    Priority
    --------
    1. Country-based continent classification when both
       countries are available and recognized.
    2. Geographic coordinate proxy when country information
       is unavailable or unknown.

    Country-based classification prevents false continent
    transitions caused by simple latitude/longitude
    thresholds.

    Geographic fallback
    --------------------
    The fallback proxy requires:

        abs(latitude_u  - latitude_v) > 10
        AND
        abs(longitude_u - longitude_v) > 45

    Missing or invalid coordinates return 0.
    """

    if G is None:
        return 0

    try:
        if u not in G.nodes or v not in G.nodes:
            return 0
    except Exception:
        return 0

    source = G.nodes[u]
    target = G.nodes[v]

    # --------------------------------------------------------
    # Country-based classification
    # --------------------------------------------------------

    source_country = source.get("country")
    target_country = target.get("country")

    if (
        isinstance(source_country, str)
        and isinstance(target_country, str)
        and source_country.strip()
        and target_country.strip()
    ):
        source_country = source_country.strip().upper()
        target_country = target_country.strip().upper()

        source_continent = (
            PaymentSimulator._country_to_continent(
                source_country
            )
        )

        target_continent = (
            PaymentSimulator._country_to_continent(
                target_country
            )
        )

        if (
            source_continent is not None
            and target_continent is not None
        ):
            return int(
                source_continent != target_continent
            )

    # --------------------------------------------------------
    # Geographic fallback
    # --------------------------------------------------------

    source_lat = PaymentSimulator._finite_float(
        source.get("latitude")
    )

    source_lon = PaymentSimulator._finite_float(
        source.get("longitude")
    )

    target_lat = PaymentSimulator._finite_float(
        target.get("latitude")
    )

    target_lon = PaymentSimulator._finite_float(
        target.get("longitude")
    )

    if (
        source_lat is None
        or source_lon is None
        or target_lat is None
        or target_lon is None
    ):
        return 0

    return int(
        abs(source_lat - target_lat) > 10.0
        and
        abs(source_lon - target_lon) > 45.0
    )


# ============================================================
# Payment Result
# ============================================================

class PaymentResult:
    """
    Structured result of exactly one payment attempt.

    The object is intentionally passive:

        - no retry
        - no rerouting
        - no backtracking
        - no graph modification
    """

    def __init__(
        self,
        success,
        path,
        edges,
        fee=0.0,
        delay=0.0,
        carbon=0.0,
        inter_country_hops=0,
        inter_continent_hops=0,
        reason=None,
        elapsed=0.0,
        failed_node=None,
        failed_edge=None,
        failure_index=None,
        visited_edges=None
    ):

        if not isinstance(success, bool):
            raise TypeError(
                "success must be a bool."
            )

        if not isinstance(path, (list, tuple)):
            raise TypeError(
                "path must be a list or tuple."
            )

        if not isinstance(edges, (list, tuple)):
            raise TypeError(
                "edges must be a list or tuple."
            )

        self.success = success

        self.path = list(path)
        self.edges = list(edges)

        self.fee = PaymentSimulator._non_negative_finite(
            fee,
            "fee"
        )

        self.delay = PaymentSimulator._non_negative_finite(
            delay,
            "delay"
        )

        self.carbon = PaymentSimulator._non_negative_finite(
            carbon,
            "carbon"
        )

        self.inter_country_hops = (
            PaymentSimulator._non_negative_int(
                inter_country_hops,
                "inter_country_hops"
            )
        )

        self.inter_continent_hops = (
            PaymentSimulator._non_negative_int(
                inter_continent_hops,
                "inter_continent_hops"
            )
        )

        if (
            reason is not None
            and not isinstance(reason, str)
        ):
            raise TypeError(
                "reason must be a string or None."
            )

        self.reason = reason

        self.elapsed = PaymentSimulator._non_negative_finite(
            elapsed,
            "elapsed"
        )

        self.failed_node = failed_node
        self.failed_edge = failed_edge

        if failure_index is not None:

            if (
                isinstance(failure_index, bool)
                or not isinstance(failure_index, int)
                or failure_index < 0
            ):
                raise ValueError(
                    "failure_index must be a non-negative "
                    "integer or None."
                )

        self.failure_index = failure_index

        if visited_edges is not None:

            if not isinstance(
                visited_edges,
                (list, tuple)
            ):
                raise TypeError(
                    "visited_edges must be a list, tuple, or None."
                )

            self.visited_edges = list(
                visited_edges
            )

        else:

            self.visited_edges = []

    # ========================================================
    # Dictionary Representation
    # ========================================================

    def to_dict(self):
        """
        Return a serializable dictionary representation.
        """

        return {
            "success": self.success,
            "path": list(self.path),
            "edges": list(self.edges),
            "fee": self.fee,
            "delay": self.delay,
            "carbon": self.carbon,
            "inter_country_hops":
                self.inter_country_hops,
            "inter_continent_hops":
                self.inter_continent_hops,
            "reason": self.reason,
            "elapsed": self.elapsed,
            "failed_node": self.failed_node,
            "failed_edge": self.failed_edge,
            "failure_index": self.failure_index,
            "visited_edges":
                list(self.visited_edges)
        }

    # ========================================================
    # Representation
    # ========================================================

    def __repr__(self):

        return (
            "PaymentResult("
            f"success={self.success}, "
            f"reason={self.reason!r}, "
            f"fee={self.fee:.4f}, "
            f"delay={self.delay:.4f}, "
            f"carbon={self.carbon:.4f}, "
            f"failed_edge={self.failed_edge!r}, "
            f"failure_index={self.failure_index!r}"
            ")"
        )


# ============================================================
# Payment Simulator
# ============================================================

class PaymentSimulator:
    """
    Execute exactly one selected payment route.

    PaymentSimulator is intentionally unaware of:

        Bucket
        PartialBacktracker
        PPO
        RL environment
        Top-K candidate selection

    It executes exactly one route supplied by its caller.

    Edge compatibility
    ------------------
    The simulator accepts both:

        (u, v)
        (u, v, key)

    and Top-K / Bucket dictionaries:

        {
            "source": u,
            "target": v,
            "channel_key": key,
            ...
        }

    Dictionary edges are preserved when passed to external
    components and stored in PaymentResult.

    Internally, all edge comparisons and graph lookups use
    the canonical identity:

        (u, v, key)
    """

    def __init__(
        self,
        G,
        failure_model,
        network_dynamics
    ):

        if G is None:
            raise ValueError(
                "G cannot be None."
            )

        if failure_model is None:
            raise ValueError(
                "failure_model cannot be None."
            )

        if network_dynamics is None:
            raise ValueError(
                "network_dynamics cannot be None."
            )

        self.G = G
        self.failure_model = failure_model
        self.network_dynamics = network_dynamics

    # ========================================================
    # Execute Exactly One Payment
    # ========================================================

    def simulate_payment(
        self,
        path,
        edges,
        amount,
        tx_id=None
    ):
        """
        Execute exactly ONE payment attempt.

        There is:

            - no retry
            - no route selection
            - no backtracking
            - no rerouting

        Component exceptions are intentionally not swallowed.
        """

        start_time = time.perf_counter()

        amount_value = self._positive_amount(
            amount
        )

        if amount_value is None:

            result = self._result(
                success=False,
                path=(
                    path
                    if isinstance(path, (list, tuple))
                    else []
                ),
                edges=(
                    edges
                    if isinstance(edges, (list, tuple))
                    else []
                ),
                reason="invalid_amount",
                elapsed=self._elapsed(start_time)
            )

            return self._finalize_result(
                result,
                tx_id
            )

        if not isinstance(
            path,
            (list, tuple)
        ):

            result = self._result(
                success=False,
                path=[],
                edges=(
                    edges
                    if isinstance(edges, (list, tuple))
                    else []
                ),
                reason="invalid_path",
                elapsed=self._elapsed(start_time)
            )

            return self._finalize_result(
                result,
                tx_id
            )

        if not isinstance(
            edges,
            (list, tuple)
        ):

            result = self._result(
                success=False,
                path=list(path),
                edges=[],
                reason="invalid_edges",
                elapsed=self._elapsed(start_time)
            )

            return self._finalize_result(
                result,
                tx_id
            )

        if len(path) < 2:

            result = self._result(
                success=False,
                path=path,
                edges=edges,
                reason="no_path",
                elapsed=self._elapsed(start_time)
            )

            return self._finalize_result(
                result,
                tx_id
            )

        validation = self._validate_route(
            path,
            edges
        )

        if not validation["valid"]:

            result = self._result(
                success=False,
                path=path,
                edges=edges,
                reason=validation["reason"],
                failed_edge=validation.get(
                    "failed_edge"
                ),
                failure_index=validation.get(
                    "failure_index"
                ),
                elapsed=self._elapsed(start_time)
            )

            return self._finalize_result(
                result,
                tx_id
            )

        metrics = self._calculate_metrics(
            edges,
            amount_value
        )

        # ----------------------------------------------------
        # FailureModel
        #
        # IMPORTANT:
        # Preserve the ORIGINAL edge representation here.
        #
        # Top-K/Bucket dictionary edges must reach the
        # FailureModel unchanged so that a failure model can
        # inspect:
        #
        #     source
        #     target
        #     channel_key
        #     scid
        #     data
        # ----------------------------------------------------

        failure = (
            self.failure_model
            .evaluate_payment_failure(
                route=list(path),
                amount=amount_value,
                network=self.G,
                route_edges=list(edges)
            )
        )

        self._validate_failure_model_result(
            failure
        )

        # ----------------------------------------------------
        # Forwarding failed
        # ----------------------------------------------------

        if failure["success"] is False:

            visited_edges = (
                self._validate_visited_edges(
                    failure.get(
                        "visited_edges",
                        []
                    ),
                    edges
                )
            )

            result = self._result(
                success=False,
                path=path,
                edges=edges,
                fee=metrics["fee"],
                delay=metrics["delay"],
                carbon=metrics["carbon"],
                inter_country_hops=(
                    metrics[
                        "inter_country_hops"
                    ]
                ),
                inter_continent_hops=(
                    metrics[
                        "inter_continent_hops"
                    ]
                ),
                reason=failure["reason"],
                failed_node=failure.get(
                    "failed_node"
                ),
                failed_edge=failure.get(
                    "failed_edge"
                ),
                failure_index=failure.get(
                    "failure_index"
                ),
                visited_edges=visited_edges,
                elapsed=self._elapsed(start_time)
            )

            return self._finalize_result(
                result,
                tx_id
            )

        # ----------------------------------------------------
        # Forwarding succeeded.
        #
        # Settlement is attempted exactly once.
        #
        # The original edge representation is preserved.
        # NetworkDynamics therefore receives the same physical
        # channel identity supplied by Top-K/Bucket.
        # ----------------------------------------------------

        settlement_success = (
            self.network_dynamics
            .settle_route(
                route_edges=list(edges),
                amount=amount_value
            )
        )

        if not isinstance(
            settlement_success,
            bool
        ):
            raise TypeError(
                "NetworkDynamics.settle_route() "
                "must return bool."
            )

        if settlement_success is False:

            result = self._result(
                success=False,
                path=path,
                edges=edges,
                fee=metrics["fee"],
                delay=metrics["delay"],
                carbon=metrics["carbon"],
                inter_country_hops=(
                    metrics[
                        "inter_country_hops"
                    ]
                ),
                inter_continent_hops=(
                    metrics[
                        "inter_continent_hops"
                    ]
                ),
                reason="settlement_failed",
                elapsed=self._elapsed(start_time)
            )

            return self._finalize_result(
                result,
                tx_id
            )

        # ----------------------------------------------------
        # Successful payment
        # ----------------------------------------------------

        result = self._result(
            success=True,
            path=path,
            edges=edges,
            fee=metrics["fee"],
            delay=metrics["delay"],
            carbon=metrics["carbon"],
            inter_country_hops=(
                metrics[
                    "inter_country_hops"
                ]
            ),
            inter_continent_hops=(
                metrics[
                    "inter_continent_hops"
                ]
            ),
            reason="success",
            visited_edges=list(edges),
            elapsed=self._elapsed(start_time)
        )

        return self._finalize_result(
            result,
            tx_id
        )

    # ========================================================
    # Finalize / Record
    # ========================================================

    def _finalize_result(
        self,
        result,
        tx_id
    ):
        """
        Record the completed payment attempt.

        Recording errors are intentionally NOT swallowed.
        """

        if tx_id is not None:

            self.network_dynamics.record_payment(
                tx_id,
                result.to_dict()
            )

        return result

    # ========================================================
    # Route Validation
    # ========================================================

    def _validate_route(
        self,
        path,
        edges
    ):
        """
        Validate node sequence and exact edge identity.

        Edge representation may be either:

            tuple/list:
                (u, v)
                (u, v, key)

        or:

            dictionary:
                {
                    "source": u,
                    "target": v,
                    "channel_key": key,
                    ...
                }
        """

        if not isinstance(
            path,
            (list, tuple)
        ):
            return self._invalid_route(
                "invalid_path"
            )

        if not isinstance(
            edges,
            (list, tuple)
        ):
            return self._invalid_route(
                "invalid_edges"
            )

        if len(path) < 2:

            return self._invalid_route(
                "path_too_short"
            )

        if len(path) != len(edges) + 1:

            return self._invalid_route(
                "route_edge_mismatch"
            )

        try:

            if len(path) != len(set(path)):

                return self._invalid_route(
                    "route_contains_loop"
                )

        except TypeError:

            return self._invalid_route(
                "unhashable_route_node"
            )

        for index, edge in enumerate(edges):

            parsed = self._parse_edge(
                edge
            )

            if parsed is None:

                return self._invalid_route(
                    "invalid_edge",
                    edge,
                    index
                )

            u, v, key = parsed

            if u != path[index]:

                return self._invalid_route(
                    "invalid_route_source",
                    edge,
                    index
                )

            if v != path[index + 1]:

                return self._invalid_route(
                    "invalid_route_destination",
                    edge,
                    index
                )

            try:

                if not self.G.has_edge(u, v):

                    return self._invalid_route(
                        "missing_channel",
                        edge,
                        index
                    )

            except Exception:

                return self._invalid_route(
                    "channel_lookup_error",
                    edge,
                    index
                )

            if self.G.is_multigraph():

                if key is None:

                    return self._invalid_route(
                        "missing_channel_key",
                        edge,
                        index
                    )

                try:

                    if not self.G.has_edge(
                        u,
                        v,
                        key
                    ):

                        return self._invalid_route(
                            "missing_channel",
                            edge,
                            index
                        )

                except Exception:

                    return self._invalid_route(
                        "channel_lookup_error",
                        edge,
                        index
                    )

            else:

                if key is not None:

                    return self._invalid_route(
                        "unexpected_channel_key",
                        edge,
                        index
                    )

        return {
            "valid": True,
            "reason": None,
            "failed_edge": None,
            "failure_index": None
        }

    # ========================================================
    # Route Metrics
    # ========================================================

    def _calculate_metrics(
        self,
        edges,
        amount
    ):
        """
        Calculate deterministic route-level metrics.

        Explicitly malformed metric attributes raise an error.
        """

        fee = 0.0
        delay = 0.0
        carbon = 0.0
        inter_country = 0
        inter_continent = 0

        for edge in edges:

            parsed = self._parse_edge(
                edge
            )

            if parsed is None:

                raise ValueError(
                    "Invalid edge during metric calculation."
                )

            u, v, key = parsed

            data = self._get_edge(
                u,
                v,
                key
            )

            if data is None:

                raise ValueError(
                    f"Cannot access exact channel "
                    f"({u!r}, {v!r}, {key!r})."
                )

            fee_base, fee_rate = (
                self._get_fee_components(
                    data
                )
            )

            fee += (
                fee_base
                +
                fee_rate
                *
                float(amount)
                /
                1_000_000.0
            )

            delay_value = self._get_optional_numeric(
                data,
                primary="delay",
                secondary="cltv_expiry_delta",
                default=0.0,
                field_name="delay"
            )

            delay += delay_value

            if u not in self.G.nodes:

                raise ValueError(
                    f"Source node {u!r} is missing."
                )

            if v not in self.G.nodes:

                raise ValueError(
                    f"Destination node {v!r} is missing."
                )

            source = self.G.nodes[u]
            target = self.G.nodes[v]

            source_carbon = self._get_node_metric(
                source,
                "carbon_intensity",
                default=0.0
            )

            target_carbon = self._get_node_metric(
                target,
                "carbon_intensity",
                default=0.0
            )

            carbon += (
                source_carbon
                +
                target_carbon
            ) / 2.0

            source_country = source.get(
                "country"
            )

            target_country = target.get(
                "country"
            )

            if (
                source_country is not None
                and
                target_country is not None
                and
                source_country != ""
                and
                target_country != ""
                and
                source_country != target_country
            ):
                inter_country += 1

            inter_continent += (
                estimate_inter_continent(
                    self.G,
                    u,
                    v
                )
            )

        if len(edges) == 0:

            average_carbon = 0.0

        else:

            average_carbon = (
                carbon / len(edges)
            )

        return {
            "fee": fee,
            "delay": delay,
            "carbon": average_carbon,
            "inter_country_hops":
                inter_country,
            "inter_continent_hops":
                inter_continent
        }

    # ========================================================
    # Country -> Continent Mapping
    # ========================================================

    @staticmethod
    def _country_to_continent(country):
        """
        Return the continent associated with a country code.

        Unknown countries return None.
        """

        country_to_continent = {

            # ------------------------------------------------
            # North America
            # ------------------------------------------------

            "US": "North America",
            "CA": "North America",
            "MX": "North America",

            # ------------------------------------------------
            # Central America / Caribbean
            # ------------------------------------------------

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

            # ------------------------------------------------
            # South America
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Europe
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Asia
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Africa
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Oceania
            # ------------------------------------------------

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

    # ========================================================
    # Fee Components
    # ========================================================

    @staticmethod
    def _get_fee_components(data):
        """
        Extract Lightning fee components.
        """

        if not isinstance(data, dict):

            raise TypeError(
                "Channel data must be a dictionary."
            )

        if "fee_base" in data:

            base = (
                PaymentSimulator
                ._required_non_negative_float(
                    data["fee_base"],
                    "fee_base"
                )
            )

        elif "fee_base_msat" in data:

            base = (
                PaymentSimulator
                ._required_non_negative_float(
                    data["fee_base_msat"],
                    "fee_base_msat"
                )
            )

        else:

            base = 0.0

        if "fee_rate" in data:

            rate = (
                PaymentSimulator
                ._required_non_negative_float(
                    data["fee_rate"],
                    "fee_rate"
                )
            )

        elif "fee_proportional_millionths" in data:

            rate = (
                PaymentSimulator
                ._required_non_negative_float(
                    data[
                        "fee_proportional_millionths"
                    ],
                    "fee_proportional_millionths"
                )
            )

        else:

            rate = 0.0

        return base, rate

    # ========================================================
    # Optional Numeric Attribute
    # ========================================================

    @staticmethod
    def _get_optional_numeric(
        data,
        primary,
        secondary,
        default,
        field_name
    ):
        """
        Read one of two optional numeric attributes.
        """

        if primary in data:

            return (
                PaymentSimulator
                ._required_non_negative_float(
                    data[primary],
                    field_name
                )
            )

        if secondary in data:

            return (
                PaymentSimulator
                ._required_non_negative_float(
                    data[secondary],
                    secondary
                )
            )

        return float(default)

    # ========================================================
    # Node Metric
    # ========================================================

    @staticmethod
    def _get_node_metric(
        node_data,
        field_name,
        default=0.0
    ):
        """
        Read a node metric.
        """

        if field_name not in node_data:
            return float(default)

        return (
            PaymentSimulator
            ._required_non_negative_float(
                node_data[field_name],
                field_name
            )
        )

    # ========================================================
    # FailureModel Validation
    # ========================================================

    def _validate_failure_model_result(
        self,
        failure
    ):
        """
        Validate the FailureModel contract.

        failed_edge may be either:

            (u, v)
            (u, v, key)

        or a Top-K/Bucket dictionary.
        """

        if not isinstance(
            failure,
            dict
        ):

            raise TypeError(
                "FailureModel must return a dictionary."
            )

        if "success" not in failure:

            raise ValueError(
                "FailureModel result must contain 'success'."
            )

        if not isinstance(
            failure["success"],
            bool
        ):

            raise TypeError(
                "FailureModel 'success' must be bool."
            )

        if "reason" not in failure:

            raise ValueError(
                "FailureModel result must contain 'reason'."
            )

        if not isinstance(
            failure["reason"],
            str
        ):

            raise TypeError(
                "FailureModel 'reason' must be a string."
            )

        failure_index = failure.get(
            "failure_index"
        )

        if failure_index is not None:

            if (
                isinstance(
                    failure_index,
                    bool
                )
                or
                not isinstance(
                    failure_index,
                    int
                )
                or
                failure_index < 0
            ):

                raise ValueError(
                    "FailureModel failure_index must be "
                    "a non-negative integer or None."
                )

        failed_edge = failure.get(
            "failed_edge"
        )

        if failed_edge is not None:

            if self._parse_edge(
                failed_edge
            ) is None:

                raise ValueError(
                    "FailureModel failed_edge has invalid format."
                )

        if "visited_edges" in failure:

            visited = failure[
                "visited_edges"
            ]

            if not isinstance(
                visited,
                (list, tuple)
            ):

                raise TypeError(
                    "FailureModel visited_edges must be "
                    "a list or tuple."
                )

    # ========================================================
    # Visited Edge Validation
    # ========================================================

    def _validate_visited_edges(
        self,
        visited_edges,
        route_edges
    ):
        """
        Validate FailureModel visited edges.

        Both tuple/list and dictionary edge representations
        are accepted.

        The original representation is preserved in the
        returned list.
        """

        if visited_edges is None:
            return []

        if not isinstance(
            visited_edges,
            (list, tuple)
        ):

            raise TypeError(
                "visited_edges must be a list or tuple."
            )

        route_parsed = []

        for edge in route_edges:

            parsed = self._parse_edge(
                edge
            )

            if parsed is None:

                raise ValueError(
                    "Route contains an invalid edge."
                )

            route_parsed.append(
                parsed
            )

        result = []

        for index, edge in enumerate(
            visited_edges
        ):

            parsed = self._parse_edge(
                edge
            )

            if parsed is None:

                raise ValueError(
                    "visited_edges contains an invalid edge."
                )

            if parsed not in route_parsed:

                raise ValueError(
                    "visited_edges contains an edge "
                    "outside the selected route."
                )

            if index >= len(route_parsed):

                raise ValueError(
                    "visited_edges contains too many edges."
                )

            if parsed != route_parsed[index]:

                raise ValueError(
                    "visited_edges must preserve route order."
                )

            result.append(
                edge
            )

        return result

    # ========================================================
    # Result Factory
    # ========================================================

    @staticmethod
    def _result(
        success,
        path,
        edges,
        fee=0.0,
        delay=0.0,
        carbon=0.0,
        inter_country_hops=0,
        inter_continent_hops=0,
        reason=None,
        elapsed=0.0,
        failed_node=None,
        failed_edge=None,
        failure_index=None,
        visited_edges=None
    ):

        return PaymentResult(
            success=success,
            path=path,
            edges=edges,
            fee=fee,
            delay=delay,
            carbon=carbon,
            inter_country_hops=(
                inter_country_hops
            ),
            inter_continent_hops=(
                inter_continent_hops
            ),
            reason=reason,
            elapsed=elapsed,
            failed_node=failed_node,
            failed_edge=failed_edge,
            failure_index=failure_index,
            visited_edges=visited_edges
        )

    # ========================================================
    # Edge Parser
    # ========================================================

    @staticmethod
    def _parse_edge(edge):
        """
        Normalize a physical channel edge.

        Accepted representations
        -------------------------

        Top-K / Bucket dictionary:

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

        Important
        ---------

        This method does NOT replace the original edge.

        It is used only for:

            - route validation
            - exact graph lookup
            - edge identity comparison
            - metric calculation
            - failure information validation

        The original object is preserved by callers.
        """

        if edge is None:
            return None

        # ----------------------------------------------------
        # Dictionary representation
        # ----------------------------------------------------

        if isinstance(edge, dict):

            # ------------------------------------------------
            # Source
            # ------------------------------------------------

            if "source" in edge:

                u = edge.get(
                    "source"
                )

            elif "u" in edge:

                u = edge.get(
                    "u"
                )

            else:

                return None

            # ------------------------------------------------
            # Target
            # ------------------------------------------------

            if "target" in edge:

                v = edge.get(
                    "target"
                )

            elif "v" in edge:

                v = edge.get(
                    "v"
                )

            else:

                return None

            # ------------------------------------------------
            # Exact channel key
            # ------------------------------------------------

            if "channel_key" in edge:

                key = edge.get(
                    "channel_key"
                )

            elif "key" in edge:

                key = edge.get(
                    "key"
                )

            else:

                key = None

            if u is None or v is None:

                return None

            return (
                u,
                v,
                key
            )

        # ----------------------------------------------------
        # Tuple / list representation
        # ----------------------------------------------------

        if isinstance(
            edge,
            (tuple, list)
        ):

            if len(edge) == 2:

                return (
                    edge[0],
                    edge[1],
                    None
                )

            if len(edge) == 3:

                return (
                    edge[0],
                    edge[1],
                    edge[2]
                )

            return None

        return None

    # ========================================================
    # Canonical Edge Identity
    # ========================================================

    @staticmethod
    def _edge_identity(edge):
        """
        Return canonical physical-channel identity.

        The identity is:

            (source, target, channel_key)

        or None when the edge representation is invalid.
        """

        return PaymentSimulator._parse_edge(
            edge
        )

    # ========================================================
    # Graph Edge Access
    # ========================================================

    def _get_edge(
        self,
        u,
        v,
        key=None
    ):
        """
        Return exact channel data.

        MultiGraph / MultiDiGraph:
            key is mandatory.

        Graph / DiGraph:
            key must be None.
        """

        try:

            if not self.G.has_edge(
                u,
                v
            ):

                return None

        except Exception:

            return None

        if self.G.is_multigraph():

            if key is None:
                return None

            try:

                data = self.G.get_edge_data(
                    u,
                    v,
                    key
                )

            except Exception:

                return None

            return data

        if key is not None:
            return None

        try:

            return self.G.get_edge_data(
                u,
                v
            )

        except Exception:

            return None

    # ========================================================
    # Utility: Invalid Route
    # ========================================================

    @staticmethod
    def _invalid_route(
        reason,
        failed_edge=None,
        failure_index=None
    ):

        return {
            "valid": False,
            "reason": reason,
            "failed_edge": failed_edge,
            "failure_index": failure_index
        }

    # ========================================================
    # Utility: Elapsed
    # ========================================================

    @staticmethod
    def _elapsed(start_time):

        return max(
            0.0,
            time.perf_counter() - start_time
        )

    # ========================================================
    # Utility: Positive Amount
    # ========================================================

    @staticmethod
    def _positive_amount(amount):
        """
        Return finite positive float or None.
        """

        if isinstance(
            amount,
            bool
        ):
            return None

        try:

            value = float(amount)

        except (
            TypeError,
            ValueError
        ):

            return None

        if not math.isfinite(value):
            return None

        if value <= 0:
            return None

        return value

    # ========================================================
    # Utility: Finite Float
    # ========================================================

    @staticmethod
    def _finite_float(value):
        """
        Return finite float or None.
        """

        if isinstance(
            value,
            bool
        ):
            return None

        try:

            value = float(value)

        except (
            TypeError,
            ValueError
        ):

            return None

        if not math.isfinite(value):
            return None

        return value

    # ========================================================
    # Utility: Required Non-Negative Float
    # ========================================================

    @staticmethod
    def _required_non_negative_float(
        value,
        field_name
    ):
        """
        Strict finite non-negative numeric validation.
        """

        if isinstance(
            value,
            bool
        ):

            raise TypeError(
                f"{field_name} must be numeric, not bool."
            )

        try:

            number = float(value)

        except (
            TypeError,
            ValueError
        ) as exc:

            raise ValueError(
                f"{field_name} must be numeric."
            ) from exc

        if not math.isfinite(number):

            raise ValueError(
                f"{field_name} must be finite."
            )

        if number < 0:

            raise ValueError(
                f"{field_name} cannot be negative."
            )

        return number

    # ========================================================
    # Utility: Non-Negative Finite
    # ========================================================

    @staticmethod
    def _non_negative_finite(
        value,
        field_name
    ):

        return (
            PaymentSimulator
            ._required_non_negative_float(
                value,
                field_name
            )
        )

    # ========================================================
    # Utility: Non-Negative Integer
    # ========================================================

    @staticmethod
    def _non_negative_int(
        value,
        field_name
    ):

        if (
            isinstance(value, bool)
            or
            not isinstance(value, int)
        ):

            raise TypeError(
                f"{field_name} must be an integer."
            )

        if value < 0:

            raise ValueError(
                f"{field_name} cannot be negative."
            )

        return value


# ============================================================
# Backward-Compatible Functional API
# ============================================================

def simulate_payment(
    G,
    path,
    edges,
    amount,
    failure_model=None,
    network_dynamics=None,
    tx_id=None
):
    """
    Functional wrapper.

    Executes exactly ONE payment attempt.
    """

    if failure_model is None:

        raise ValueError(
            "failure_model is required."
        )

    if network_dynamics is None:

        raise ValueError(
            "network_dynamics is required."
        )

    simulator = PaymentSimulator(
        G=G,
        failure_model=failure_model,
        network_dynamics=network_dynamics
    )

    return simulator.simulate_payment(
        path=path,
        edges=edges,
        amount=amount,
        tx_id=tx_id
    )


# ============================================================
# Standalone Diagnostic Test
# ============================================================

if __name__ == "__main__":

    import networkx as nx

    print()
    print("=" * 72)
    print("PAYMENT SIMULATOR DIAGNOSTIC TEST")
    print("=" * 72)

    G = nx.MultiDiGraph()

    G.add_edge(
        "A",
        "B",
        key=10,
        fee_base_msat=1000,
        fee_proportional_millionths=100,
        cltv_expiry_delta=40,
        balance_uv=10000,
        capacity=20000,
        available=True
    )

    G.add_edge(
        "B",
        "C",
        key=20,
        fee_base_msat=1000,
        fee_proportional_millionths=200,
        cltv_expiry_delta=50,
        balance_uv=10000,
        capacity=20000,
        available=True
    )

    for node in G.nodes:

        G.nodes[node]["available"] = True
        G.nodes[node]["is_online"] = True
        G.nodes[node]["carbon_intensity"] = 100.0

    class TestFailureModel:

        def evaluate_payment_failure(
            self,
            route,
            amount,
            network,
            route_edges=None
        ):

            return {
                "success": True,
                "reason": "success",
                "visited_edges": list(
                    route_edges
                )
            }

    class TestNetworkDynamics:

        def __init__(self):

            self.records = []
            self.settlement_calls = 0

        def settle_route(
            self,
            route_edges,
            amount
        ):

            self.settlement_calls += 1

            return True

        def record_payment(
            self,
            tx_id,
            result
        ):

            self.records.append(
                (
                    tx_id,
                    result
                )
            )

    failure_model = TestFailureModel()

    network_dynamics = TestNetworkDynamics()

    simulator = PaymentSimulator(
        G=G,
        failure_model=failure_model,
        network_dynamics=network_dynamics
    )

    path = [
        "A",
        "B",
        "C"
    ]

    # --------------------------------------------------------
    # Tuple representation
    # --------------------------------------------------------

    edges = [
        ("A", "B", 10),
        ("B", "C", 20)
    ]

    result = simulator.simulate_payment(
        path=path,
        edges=edges,
        amount=1000,
        tx_id="TX-001"
    )

    print()
    print("Tuple edge test")
    print("----------------")
    print("Success          :", result.success)
    print("Reason           :", result.reason)
    print("Path             :", result.path)
    print("Edges            :", result.edges)
    print("Visited edges    :", result.visited_edges)
    print("Fee              :", result.fee)
    print("Delay            :", result.delay)
    print("Carbon           :", result.carbon)
    print(
        "Settlement calls :",
        network_dynamics.settlement_calls
    )
    print(
        "Records          :",
        len(network_dynamics.records)
    )

    assert result.success is True
    assert result.reason == "success"

    assert result.path == [
        "A",
        "B",
        "C"
    ]

    assert result.edges == [
        ("A", "B", 10),
        ("B", "C", 20)
    ]

    assert result.visited_edges == [
        ("A", "B", 10),
        ("B", "C", 20)
    ]

    assert network_dynamics.settlement_calls == 1
    assert len(network_dynamics.records) == 1

    # --------------------------------------------------------
    # Exact channel identity
    # --------------------------------------------------------

    exact_data = simulator._get_edge(
        "A",
        "B",
        10
    )

    assert exact_data is not None
    assert exact_data["fee_base_msat"] == 1000

    G.add_edge(
        "A",
        "B",
        key=99,
        fee_base_msat=999999,
        balance_uv=10000,
        available=True
    )

    assert simulator._get_edge(
        "A",
        "B",
        777
    ) is None

    assert simulator._get_edge(
        "A",
        "B"
    ) is None

    # --------------------------------------------------------
    # Dictionary edge compatibility diagnostic
    # --------------------------------------------------------

    dictionary_edges = [
        {
            "source": "A",
            "target": "B",
            "channel_key": 10,
            "scid": "A-B-10"
        },
        {
            "source": "B",
            "target": "C",
            "channel_key": 20,
            "scid": "B-C-20"
        }
    ]

    parsed_first = simulator._parse_edge(
        dictionary_edges[0]
    )

    parsed_second = simulator._parse_edge(
        dictionary_edges[1]
    )

    assert parsed_first == (
        "A",
        "B",
        10
    )

    assert parsed_second == (
        "B",
        "C",
        20
    )

    dictionary_validation = simulator._validate_route(
        path,
        dictionary_edges
    )

    assert dictionary_validation["valid"] is True

    dictionary_metrics = simulator._calculate_metrics(
        dictionary_edges,
        1000
    )

    assert dictionary_metrics["fee"] >= 0.0

    dictionary_result = simulator.simulate_payment(
        path=path,
        edges=dictionary_edges,
        amount=1000,
        tx_id="TX-DICT-001"
    )

    assert dictionary_result.success is True
    assert dictionary_result.edges == dictionary_edges
    assert dictionary_result.visited_edges == dictionary_edges

    # --------------------------------------------------------
    # FailureModel receives original dictionary edges
    # --------------------------------------------------------

    class DictionaryAwareFailureModel:

        def __init__(self):

            self.received_edges = None

        def evaluate_payment_failure(
            self,
            route,
            amount,
            network,
            route_edges=None
        ):

            self.received_edges = list(
                route_edges
            )

            assert isinstance(
                self.received_edges[0],
                dict
            )

            assert (
                self.received_edges[0][
                    "channel_key"
                ] == 10
            )

            return {
                "success": True,
                "reason": "success",
                "visited_edges": list(
                    route_edges
                )
            }

    dictionary_failure_model = (
        DictionaryAwareFailureModel()
    )

    dictionary_dynamics = TestNetworkDynamics()

    dictionary_simulator = PaymentSimulator(
        G=G,
        failure_model=dictionary_failure_model,
        network_dynamics=dictionary_dynamics
    )

    dictionary_e2e_result = (
        dictionary_simulator.simulate_payment(
            path=path,
            edges=dictionary_edges,
            amount=1000,
            tx_id="TX-DICT-002"
        )
    )

    assert dictionary_e2e_result.success is True

    assert (
        dictionary_failure_model.received_edges
        == dictionary_edges
    )

    # --------------------------------------------------------
    # Failure propagation
    # --------------------------------------------------------

    class FailedPaymentModel:

        def evaluate_payment_failure(
            self,
            route,
            amount,
            network,
            route_edges=None
        ):

            return {
                "success": False,
                "reason": "channel_failure",
                "failed_node": "B",
                "failed_edge": (
                    "B",
                    "C",
                    20
                ),
                "failure_index": 1,
                "visited_edges": [
                    ("A", "B", 10)
                ]
            }

    failed_dynamics = TestNetworkDynamics()

    failed_simulator = PaymentSimulator(
        G=G,
        failure_model=FailedPaymentModel(),
        network_dynamics=failed_dynamics
    )

    failed_result = (
        failed_simulator.simulate_payment(
            path=path,
            edges=edges,
            amount=1000,
            tx_id="TX-002"
        )
    )

    assert failed_result.success is False
    assert failed_result.reason == "channel_failure"

    assert failed_result.failed_node == "B"

    assert failed_result.failed_edge == (
        "B",
        "C",
        20
    )

    assert failed_result.failure_index == 1

    assert failed_result.visited_edges == [
        ("A", "B", 10)
    ]

    # FailureModel failure must NOT settle the route.
    assert failed_dynamics.settlement_calls == 0

    # --------------------------------------------------------
    # Dictionary failed_edge compatibility
    # --------------------------------------------------------

    class DictionaryFailureModel:

        def evaluate_payment_failure(
            self,
            route,
            amount,
            network,
            route_edges=None
        ):

            return {
                "success": False,
                "reason": "channel_failure",
                "failed_node": "B",
                "failed_edge": {
                    "source": "B",
                    "target": "C",
                    "channel_key": 20,
                    "scid": "B-C-20"
                },
                "failure_index": 1,
                "visited_edges": [
                    {
                        "source": "A",
                        "target": "B",
                        "channel_key": 10,
                        "scid": "A-B-10"
                    }
                ]
            }

    dictionary_failed_dynamics = TestNetworkDynamics()

    dictionary_failed_simulator = PaymentSimulator(
        G=G,
        failure_model=DictionaryFailureModel(),
        network_dynamics=dictionary_failed_dynamics
    )

    dictionary_failed_result = (
        dictionary_failed_simulator.simulate_payment(
            path=path,
            edges=dictionary_edges,
            amount=1000,
            tx_id="TX-DICT-FAIL"
        )
    )

    assert dictionary_failed_result.success is False
    assert (
        dictionary_failed_result.reason
        == "channel_failure"
    )

    assert (
        dictionary_failed_result.failed_edge[
            "source"
        ]
        == "B"
    )

    assert (
        dictionary_failed_result.failed_edge[
            "target"
        ]
        == "C"
    )

    assert (
        dictionary_failed_result.failed_edge[
            "channel_key"
        ]
        == 20
    )

    assert dictionary_failed_result.failure_index == 1

    assert (
        dictionary_failed_result.visited_edges
        == [
            dictionary_edges[0]
        ]
    )

    assert (
        dictionary_failed_dynamics.settlement_calls
        == 0
    )

    # --------------------------------------------------------
    # Final diagnostics
    # --------------------------------------------------------

    print()
    print("Exact channel identity : PASS")
    print("Successful settlement  : PASS")
    print("Tuple edge format      : PASS")
    print("Dictionary edge format : PASS")
    print("Failure propagation    : PASS")
    print("Dictionary failure     : PASS")
    print("No retry               : PASS")
    print("PAYMENT SIMULATOR      : SUCCESS")