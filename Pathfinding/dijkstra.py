"""
Pathfinding/dijkstra.py

Dijkstra routing engine for the adaptive routing model.

Unified routing objective
-------------------------

    C_LND(e)
        |
        +----------------------+
        |                      |
        v                      v
    native cost          adaptive heuristic
                               |
                               v
                       normalized penalty
                               |
                               v
                  C'(e) = C_LND(e)
                           * (1 + lambda_h * penalty)

The same adaptive edge-cost definition is used by
Pathfinding.top_k_paths.

Important design principles
----------------------------

1. No silent fallback for malformed routing data.
2. Liquidity is a hard feasibility constraint.
3. Channel identity is preserved for MultiGraph/MultiDiGraph.
4. Geographic metrics are obtained from Network.topology.
5. RGB carbon proxy is defined centrally in Network.topology.
6. Adaptive cost is calculated exactly once per edge.
7. PPO controls eta; k is controlled outside this class.
8. Optional Boolean state flags must contain real Boolean values.
9. max_hops must be a genuine integer.
"""

import heapq
import itertools
import math

import networkx as nx

from .heuristics import (
    lnd_cost,
    adaptive_edge_cost,
    validate_eta,
    validate_lambda_h,
    channel_fee,
    channel_delay,
)

from Network.topology import geo_features


class Dijkstra:
    """
    Dijkstra-based routing engine for a Lightning-style
    payment network.

    Pipeline:

        Network State
             |
             v
            PPO
             |
             v
            eta
             |
             v
      Adaptive Heuristic
             |
             v
      Normalized Penalty
             |
             v
       Unified Edge Cost
             |
             v
          Dijkstra
             |
             v
        Candidate Route
    """

    # ======================================================
    # Initialization
    # ======================================================

    def __init__(self, graph):
        if graph is None:
            raise ValueError("graph must not be None")

        if not isinstance(
            graph,
            (
                nx.Graph,
                nx.DiGraph,
                nx.MultiGraph,
                nx.MultiDiGraph,
            ),
        ):
            raise TypeError(
                "graph must be a NetworkX Graph, DiGraph, "
                "MultiGraph, or MultiDiGraph"
            )

        self.G = graph

    # ======================================================
    # Main Routing
    # ======================================================

    def shortest_path(
        self,
        source,
        target,
        amount,
        heuristic_fn=None,
        eta=0.0,
        max_hops=12,
        lambda_h=1.0,
    ):
        """
        Find a minimum-cost feasible path.

        The adaptive objective is always calculated through
        adaptive_edge_cost().

        Parameters
        ----------
        source : node
            Source node.

        target : node
            Destination node.

        amount : float
            Payment amount in the same unit used by the
            graph's fee and liquidity fields.

        heuristic_fn : callable or None
            Native edge-cost function.

            If None:
                lnd_cost is used.

            If a custom function is supplied, it is used
            directly. Exceptions are intentionally propagated.

        eta : float
            Adaptive routing parameter in [0, 1].

        max_hops : int
            Maximum number of edges in the route.

            A genuine integer is required. Values such as
            3.8 are rejected rather than silently converted
            to 3.

        lambda_h : float
            Adaptive penalty scaling parameter.

        Returns
        -------
        dict
            Structured success or failure result.
        """

        # --------------------------------------------------
        # Validate nodes
        # --------------------------------------------------

        if source not in self.G:
            raise ValueError(
                f"Unknown source node: {source}"
            )

        if target not in self.G:
            raise ValueError(
                f"Unknown target node: {target}"
            )

        if source == target:
            return self._failure(
                "source_equals_target"
            )

        # --------------------------------------------------
        # Validate amount
        # --------------------------------------------------

        try:
            amount = float(amount)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "amount must be numeric"
            ) from exc

        if not math.isfinite(amount) or amount <= 0.0:
            raise ValueError(
                "amount must be finite and greater than zero"
            )

        # --------------------------------------------------
        # Validate hop limit
        # --------------------------------------------------

        max_hops = self._validate_max_hops(max_hops)

        # --------------------------------------------------
        # Validate adaptive parameters
        # --------------------------------------------------

        eta = validate_eta(eta)
        lambda_h = validate_lambda_h(lambda_h)

        # --------------------------------------------------
        # Validate endpoint availability
        # --------------------------------------------------

        if not self._node_online(source):
            return self._failure(
                "source_offline"
            )

        if not self._node_online(target):
            return self._failure(
                "target_offline"
            )

        # --------------------------------------------------
        # Native routing objective
        # --------------------------------------------------

        if heuristic_fn is None:
            heuristic_fn = lnd_cost

        if not callable(heuristic_fn):
            raise TypeError(
                "heuristic_fn must be callable"
            )

        # --------------------------------------------------
        # Priority queue
        #
        # Entry:
        #
        # (
        #   cost,
        #   hops,
        #   counter,
        #   node,
        #   path,
        #   edges,
        #   total_fee,
        #   total_delay,
        #   path_reliability,
        #   total_distance,
        #   total_carbon,
        #   inter_country_hops,
        #   inter_continent_hops,
        # )
        # --------------------------------------------------

        counter = itertools.count()
        pq = []

        heapq.heappush(
            pq,
            (
                0.0,
                0,
                next(counter),
                source,
                (source,),
                (),
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0,
                0,
            ),
        )

        # --------------------------------------------------
        # Best cost for (node, hop_count)
        # --------------------------------------------------

        best = {}

        # ==================================================
        # Dijkstra loop
        # ==================================================

        while pq:

            (
                cost,
                hops,
                _counter,
                u,
                path,
                edges,
                total_fee,
                total_delay,
                path_reliability,
                total_distance,
                total_carbon,
                inter_country_hops,
                inter_continent_hops,
            ) = heapq.heappop(pq)

            # ----------------------------------------------
            # Target reached
            # ----------------------------------------------

            if u == target:
                return self._success(
                    path=path,
                    edges=edges,
                    cost=cost,
                    hop_count=hops,
                    total_fee=total_fee,
                    total_delay=total_delay,
                    path_reliability=path_reliability,
                    total_distance=total_distance,
                    total_carbon=total_carbon,
                    inter_country_hops=inter_country_hops,
                    inter_continent_hops=inter_continent_hops,
                    eta=eta,
                    lambda_h=lambda_h,
                )

            # ----------------------------------------------
            # Hop constraint
            # ----------------------------------------------

            if hops >= max_hops:
                continue

            state = (
                u,
                hops,
            )

            previous_best = best.get(
                state,
                math.inf,
            )

            if cost >= previous_best:
                continue

            best[state] = cost

            # ==================================================
            # Explore neighbors
            # ==================================================

            for v, channel_dict in self.G[u].items():

                # ------------------------------------------
                # Prevent cycles
                # ------------------------------------------

                if v in path:
                    continue

                # ------------------------------------------
                # Skip offline destination/intermediate node
                # ------------------------------------------

                if not self._node_online(v):
                    continue

                # ------------------------------------------
                # Resolve channel(s)
                # ------------------------------------------

                if self._is_multigraph():

                    channel_items = channel_dict.items()

                else:

                    channel_items = [
                        (
                            channel_dict.get(
                                "channel_key"
                            ),
                            channel_dict,
                        )
                    ]

                # ==================================================
                # Channel loop
                # ==================================================

                for key, raw_data in channel_items:

                    # ----------------------------------------------
                    # Validate raw edge data
                    # ----------------------------------------------

                    if not isinstance(raw_data, dict):
                        raise TypeError(
                            f"Invalid edge data for "
                            f"{u}->{v}, key={key!r}"
                        )

                    data = dict(raw_data)

                    # ----------------------------------------------
                    # Normalize schema aliases only
                    # ----------------------------------------------

                    self._normalize_edge_data(data)

                    # ----------------------------------------------
                    # Channel availability
                    # ----------------------------------------------

                    if not self._channel_available(data):
                        continue

                    # ----------------------------------------------
                    # Liquidity
                    #
                    # Liquidity is a hard feasibility constraint.
                    # ----------------------------------------------

                    liquidity = self._estimated_liquidity(
                        data
                    )

                    if liquidity is None:
                        raise ValueError(
                            f"Missing valid liquidity for edge "
                            f"{u}->{v}, key={key!r}"
                        )

                    if liquidity < amount:
                        continue

                    # ----------------------------------------------
                    # Adaptive edge cost
                    #
                    # This is the ONLY place where the unified
                    # adaptive cost is calculated.
                    # ----------------------------------------------

                    edge_info = adaptive_edge_cost(
                        G=self.G,
                        u=u,
                        v=v,
                        data=data,
                        amount=amount,
                        eta=eta,
                        heuristic_fn=heuristic_fn,
                        lambda_h=lambda_h,
                    )

                    if not isinstance(edge_info, dict):
                        raise TypeError(
                            "adaptive_edge_cost() must return a dict"
                        )

                    required_fields = (
                        "native_cost",
                        "raw_heuristic",
                        "adaptive_penalty",
                        "cost",
                    )

                    missing = [
                        field
                        for field in required_fields
                        if field not in edge_info
                    ]

                    if missing:
                        raise KeyError(
                            "adaptive_edge_cost() result is missing "
                            f"fields: {missing}"
                        )

                    native_cost = edge_info[
                        "native_cost"
                    ]

                    adaptive_h = edge_info[
                        "raw_heuristic"
                    ]

                    adaptive_penalty_value = edge_info[
                        "adaptive_penalty"
                    ]

                    edge_cost = edge_info[
                        "cost"
                    ]

                    # ----------------------------------------------
                    # Validate adaptive-cost outputs
                    # ----------------------------------------------

                    if not self._valid_cost(
                        native_cost
                    ):
                        raise ValueError(
                            f"Invalid native cost on edge "
                            f"{u}->{v}, key={key!r}: "
                            f"{native_cost!r}"
                        )

                    if not self._valid_number(
                        adaptive_h
                    ):
                        raise ValueError(
                            f"Invalid adaptive heuristic on edge "
                            f"{u}->{v}, key={key!r}: "
                            f"{adaptive_h!r}"
                        )

                    if not self._valid_metric(
                        adaptive_penalty_value
                    ):
                        raise ValueError(
                            f"Invalid adaptive penalty on edge "
                            f"{u}->{v}, key={key!r}: "
                            f"{adaptive_penalty_value!r}"
                        )

                    if not self._valid_cost(
                        edge_cost
                    ):
                        raise ValueError(
                            f"Invalid adaptive edge cost on edge "
                            f"{u}->{v}, key={key!r}: "
                            f"{edge_cost!r}"
                        )

                    # ----------------------------------------------
                    # Fee
                    # ----------------------------------------------

                    fee = channel_fee(
                        data,
                        amount,
                    )

                    if not self._valid_metric(fee):
                        raise ValueError(
                            f"Invalid fee on edge "
                            f"{u}->{v}, key={key!r}: "
                            f"{fee!r}"
                        )

                    # ----------------------------------------------
                    # Delay
                    # ----------------------------------------------

                    delay = channel_delay(
                        data
                    )

                    if not self._valid_metric(delay):
                        raise ValueError(
                            f"Invalid delay on edge "
                            f"{u}->{v}, key={key!r}: "
                            f"{delay!r}"
                        )

                    # ----------------------------------------------
                    # Failure probability
                    # ----------------------------------------------

                    failure_probability = (
                        self._failure_probability(
                            data
                        )
                    )

                    channel_reliability = (
                        1.0 - failure_probability
                    )

                    new_reliability = (
                        path_reliability
                        * channel_reliability
                    )

                    # ----------------------------------------------
                    # Geographic metrics
                    # ----------------------------------------------

                    (
                        distance_km,
                        carbon_intensity,
                        inter_country,
                        inter_continent,
                    ) = self._geographic_metrics(
                        u,
                        v,
                    )

                    # ----------------------------------------------
                    # Accumulate path metrics
                    # ----------------------------------------------

                    new_cost = (
                        cost
                        + edge_cost
                    )

                    new_hops = (
                        hops
                        + 1
                    )

                    if new_hops > max_hops:
                        continue

                    new_fee = (
                        total_fee
                        + fee
                    )

                    new_delay = (
                        total_delay
                        + delay
                    )

                    new_distance = (
                        total_distance
                        + distance_km
                    )

                    new_carbon = (
                        total_carbon
                        + carbon_intensity
                    )

                    new_inter_country = (
                        inter_country_hops
                        + int(inter_country)
                    )

                    new_inter_continent = (
                        inter_continent_hops
                        + int(inter_continent)
                    )

                    # ----------------------------------------------
                    # Preserve exact edge identity
                    # ----------------------------------------------

                    new_edges = (
                        edges
                        + (
                            (
                                u,
                                v,
                                key,
                            ),
                        )
                    )

                    # ----------------------------------------------
                    # Push candidate
                    # ----------------------------------------------

                    heapq.heappush(
                        pq,
                        (
                            new_cost,
                            new_hops,
                            next(counter),
                            v,
                            path + (v,),
                            new_edges,
                            new_fee,
                            new_delay,
                            new_reliability,
                            new_distance,
                            new_carbon,
                            new_inter_country,
                            new_inter_continent,
                        ),
                    )

        # --------------------------------------------------
        # No feasible route
        # --------------------------------------------------

        return self._failure(
            "no_feasible_path"
        )

    # ======================================================
    # Graph Type
    # ======================================================

    def _is_multigraph(self):
        return isinstance(
            self.G,
            (
                nx.MultiGraph,
                nx.MultiDiGraph,
            ),
        )

    # ======================================================
    # max_hops Validation
    # ======================================================

    @staticmethod
    def _validate_max_hops(value):
        """
        Validate max_hops without silently truncating values.

        Accepted:
            int
            integer-valued finite float, e.g. 5.0

        Rejected:
            5.5
            "5"
            True
            NaN
            inf
            zero
            negative values
        """

        if isinstance(value, bool):
            raise ValueError(
                "max_hops must be an integer"
            )

        if isinstance(value, int):
            max_hops = value

        elif isinstance(value, float):

            if not math.isfinite(value):
                raise ValueError(
                    "max_hops must be finite"
                )

            if not value.is_integer():
                raise ValueError(
                    "max_hops must be an integer"
                )

            max_hops = int(value)

        else:
            raise ValueError(
                "max_hops must be an integer"
            )

        if max_hops <= 0:
            raise ValueError(
                "max_hops must be greater than zero"
            )

        return max_hops

    # ======================================================
    # Edge Data Normalization
    # ======================================================

    @staticmethod
    def _normalize_edge_data(data):
        """
        Normalize schema aliases only.

        This function does not manufacture numerical routing
        values such as zero fee, zero delay, zero liquidity,
        or default failure probability.

        The 'available' field is an optional Boolean state flag.
        If absent, the channel is considered available because
        the underlying GML schema does not necessarily contain
        this field.
        """

        if "fee_base" not in data:

            if "fee_base_msat" in data:
                data["fee_base"] = data[
                    "fee_base_msat"
                ]
            else:
                raise KeyError(
                    "Missing fee_base / fee_base_msat"
                )

        if "fee_rate" not in data:

            if "fee_proportional_millionths" in data:
                data["fee_rate"] = data[
                    "fee_proportional_millionths"
                ]
            else:
                raise KeyError(
                    "Missing fee_rate / "
                    "fee_proportional_millionths"
                )

        if "delay" not in data:

            if "cltv_expiry_delta" in data:
                data["delay"] = data[
                    "cltv_expiry_delta"
                ]
            else:
                raise KeyError(
                    "Missing delay / cltv_expiry_delta"
                )

        # --------------------------------------------------
        # Optional availability state
        # --------------------------------------------------

        if "available" in data:
            Dijkstra._validate_boolean_field(
                data["available"],
                "available",
            )

        # --------------------------------------------------
        # Validate optional online state if supplied
        # --------------------------------------------------

        for field in (
            "is_online",
            "online",
        ):
            if field in data:
                Dijkstra._validate_boolean_field(
                    data[field],
                    field,
                )

    # ======================================================
    # Boolean Validation
    # ======================================================

    @staticmethod
    def _validate_boolean_field(
        value,
        field_name,
    ):
        """
        Require a genuine Boolean value.

        Strings such as:
            "false"
            "true"
            "0"
            "1"

        are intentionally rejected because Python's bool()
        would otherwise silently interpret non-empty strings
        as True.
        """

        if not isinstance(value, bool):
            raise ValueError(
                f"{field_name} must be a Boolean value, "
                f"got {value!r}"
            )

        return value

    # ======================================================
    # Directional Liquidity
    # ======================================================

    @staticmethod
    def _estimated_liquidity(data):
        """
        Return directional liquidity.

        Priority:
            estimated_liquidity
            liquidity_uv
            balance_uv

        Invalid values are never silently converted to zero.
        """

        found = False

        for field in (
            "estimated_liquidity",
            "liquidity_uv",
            "balance_uv",
        ):

            if field not in data:
                continue

            found = True

            value = data[field]

            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid liquidity value in field "
                    f"{field!r}: {value!r}"
                ) from exc

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite liquidity value in field "
                    f"{field!r}: {value!r}"
                )

            if value < 0.0:
                raise ValueError(
                    f"Negative liquidity value in field "
                    f"{field!r}: {value!r}"
                )

            return value

        if found:
            raise ValueError(
                "Liquidity fields exist but contain no valid value"
            )

        return None

    # ======================================================
    # Node Availability
    # ======================================================

    def _node_online(self, node):
        """
        Check optional node availability flags.

        Missing flags mean that no offline state has been
        explicitly declared.

        If a flag is present, it must be Boolean.
        """

        if node not in self.G:
            return False

        data = self.G.nodes[node]

        for field in (
            "available",
            "is_online",
            "online",
        ):

            if field not in data:
                continue

            value = data[field]

            self._validate_boolean_field(
                value,
                field,
            )

            if not value:
                return False

        return True

    # ======================================================
    # Channel Availability
    # ======================================================

    @staticmethod
    def _channel_available(data):
        """
        Check channel availability.

        Missing 'available' means no explicit unavailable state
        has been declared.

        If present, the value must be Boolean.
        """

        if "available" not in data:
            return True

        Dijkstra._validate_boolean_field(
            data["available"],
            "available",
        )

        return data["available"]

    # ======================================================
    # Failure Probability
    # ======================================================

    @staticmethod
    def _failure_probability(data):
        """
        Calculate empirical channel failure probability.

        If success_count and/or failure_count exist:

            p = failure / (failure + success)

        If both counts are zero, an explicit
        failure_probability must be available.

        Otherwise, an explicit failure_probability is required.

        No arbitrary default probability is introduced.
        """

        has_success = (
            "success_count" in data
        )

        has_failure = (
            "failure_count" in data
        )

        if has_success or has_failure:

            success = (
                Dijkstra._strict_nonnegative_float(
                    data.get(
                        "success_count",
                        0.0,
                    ),
                    "success_count",
                )
            )

            failure = (
                Dijkstra._strict_nonnegative_float(
                    data.get(
                        "failure_count",
                        0.0,
                    ),
                    "failure_count",
                )
            )

            total = (
                success
                + failure
            )

            if total > 0.0:

                probability = (
                    failure
                    /
                    total
                )

                return min(
                    max(
                        probability,
                        0.0,
                    ),
                    1.0,
                )

        if "failure_probability" not in data:
            raise ValueError(
                "Missing failure_probability and "
                "no valid success/failure history"
            )

        probability = (
            Dijkstra._strict_probability(
                data["failure_probability"],
                "failure_probability",
            )
        )

        return probability

    # ======================================================
    # Geographic Metrics
    # ======================================================

    def _geographic_metrics(
        self,
        u,
        v,
    ):
        """
        Obtain geographic metrics from Network.topology.

        No zero-valued fallback is used.
        """

        features = geo_features(
            self.G,
            u,
            v,
        )

        if not isinstance(features, dict):
            raise TypeError(
                "geo_features() must return a dict"
            )

        required = (
            "distance_km",
            "carbon_intensity",
            "inter_country",
            "inter_continent",
        )

        missing = [
            key
            for key in required
            if key not in features
        ]

        if missing:
            raise KeyError(
                "geo_features() is missing required "
                f"fields: {missing}"
            )

        distance = (
            self._strict_nonnegative_float(
                features["distance_km"],
                "distance_km",
            )
        )

        carbon = (
            self._strict_nonnegative_float(
                features["carbon_intensity"],
                "carbon_intensity",
            )
        )

        inter_country = int(
            bool(
                features["inter_country"]
            )
        )

        inter_continent = int(
            bool(
                features["inter_continent"]
            )
        )

        return (
            distance,
            carbon,
            inter_country,
            inter_continent,
        )

    # ======================================================
    # Minimum Liquidity
    # ======================================================

    def _minimum_liquidity(
        self,
        edges,
    ):
        """
        Return minimum directional liquidity across the
        exact edges used by the selected route.
        """

        if not edges:
            return None

        values = []

        for u, v, key in edges:

            data = self._get_exact_edge_data(
                u,
                v,
                key,
            )

            liquidity = (
                self._estimated_liquidity(
                    data
                )
            )

            if liquidity is None:
                raise ValueError(
                    f"Missing liquidity for selected edge "
                    f"{u}->{v}, key={key!r}"
                )

            values.append(
                liquidity
            )

        if not values:
            return None

        return min(values)

    # ======================================================
    # Exact Edge Retrieval
    # ======================================================

    def _get_exact_edge_data(
        self,
        u,
        v,
        key,
    ):
        """
        Retrieve the exact channel represented by
        (u, v, key).

        For MultiGraph/MultiDiGraph, silently selecting
        the first channel is forbidden.
        """

        if self._is_multigraph():

            if key is None:
                raise ValueError(
                    f"Missing channel key for multigraph edge "
                    f"{u}->{v}"
                )

            try:
                return self.G[u][v][key]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown channel key {key!r} "
                    f"for edge {u}->{v}"
                ) from exc

        # --------------------------------------------------
        # Simple graph
        # --------------------------------------------------

        try:
            return self.G[u][v]
        except KeyError as exc:
            raise KeyError(
                f"Unknown edge {u}->{v}"
            ) from exc

    # ======================================================
    # Success Result
    # ======================================================

    def _success(
        self,
        path,
        edges,
        cost,
        hop_count,
        total_fee,
        total_delay,
        path_reliability,
        total_distance,
        total_carbon,
        inter_country_hops,
        inter_continent_hops,
        eta=0.0,
        lambda_h=1.0,
    ):

        min_liquidity = (
            self._minimum_liquidity(
                edges
            )
        )

        return {
            "success": True,

            "path": list(path),

            "edges": list(edges),

            "cost": float(cost),

            "hop_count": int(
                hop_count
            ),

            "total_fee": float(
                total_fee
            ),

            "total_delay": float(
                total_delay
            ),

            "min_liquidity": (
                None
                if min_liquidity is None
                else float(
                    min_liquidity
                )
            ),

            "reliability": float(
                path_reliability
            ),

            "failure_probability": float(
                1.0 - path_reliability
            ),

            "total_distance_km": float(
                total_distance
            ),

            "total_carbon": float(
                total_carbon
            ),

            "inter_country_hops": int(
                inter_country_hops
            ),

            "inter_continent_hops": int(
                inter_continent_hops
            ),

            "eta": float(eta),

            "lambda_h": float(
                lambda_h
            ),

            "candidate": True,

            "reason": None,
        }

    # ======================================================
    # Failure Result
    # ======================================================

    @staticmethod
    def _failure(
        reason="no_feasible_path"
    ):
        return {
            "success": False,

            "path": None,

            "edges": None,

            "cost": math.inf,

            "hop_count": 0,

            "total_fee": 0.0,

            "total_delay": 0.0,

            "min_liquidity": None,

            "reliability": 0.0,

            "failure_probability": 1.0,

            "total_distance_km": 0.0,

            "total_carbon": 0.0,

            "inter_country_hops": 0,

            "inter_continent_hops": 0,

            "eta": None,

            "lambda_h": None,

            "candidate": False,

            "reason": reason,
        }

    # ======================================================
    # Strict Numeric Helpers
    # ======================================================

    @staticmethod
    def _strict_nonnegative_float(
        value,
        field_name,
    ):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be numeric: {value!r}"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"{field_name} must be finite: {value!r}"
            )

        if value < 0.0:
            raise ValueError(
                f"{field_name} must be non-negative: {value!r}"
            )

        return value

    # ======================================================
    # Probability Validation
    # ======================================================

    @staticmethod
    def _strict_probability(
        value,
        field_name,
    ):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be numeric: {value!r}"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"{field_name} must be finite: {value!r}"
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{field_name} must be in [0, 1]: {value!r}"
            )

        return value

    # ======================================================
    # Generic Number Validation
    # ======================================================

    @staticmethod
    def _valid_number(value):

        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        return math.isfinite(value)

    # ======================================================
    # Nonnegative Metric Validation
    # ======================================================

    @staticmethod
    def _valid_metric(value):

        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        return (
            math.isfinite(value)
            and value >= 0.0
        )

    # ======================================================
    # Cost Validation
    # ======================================================

    @staticmethod
    def _valid_cost(value):

        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        return (
            math.isfinite(value)
            and value >= 0.0
        )