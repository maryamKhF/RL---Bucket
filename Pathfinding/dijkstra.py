"""
Dijkstra routing engine for the adaptive routing model.

Unified routing objective:

    native_cost
        +
        adaptive signal
        |
        v
    normalized penalty
        |
        v
    C'(e) =
        C_LND(e) * (1 + lambda_h * penalty)

The exact same adaptive-cost definition is used by
Pathfinding.top_k_paths.
"""

import heapq
import itertools
import math

import networkx as nx

from .heuristics import (
    lnd_cost,
    modified_cost,
    adaptive_heuristic,
    adaptive_penalty,
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

    def __init__(self, graph):
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

        heuristic_fn is expected to represent the native
        routing cost only, normally lnd_cost.

        The adaptive component is added exactly once by
        adaptive_edge_cost().
        """

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

        try:
            amount = float(amount)
        except (TypeError, ValueError):
            raise ValueError(
                "amount must be numeric"
            )

        if (
            not math.isfinite(amount)
            or amount <= 0.0
        ):
            raise ValueError(
                "amount must be finite and greater than zero"
            )

        try:
            max_hops = int(max_hops)
        except (TypeError, ValueError):
            raise ValueError(
                "max_hops must be an integer"
            )

        if max_hops <= 0:
            raise ValueError(
                "max_hops must be greater than zero"
            )

        eta = validate_eta(
            eta
        )

        lambda_h = validate_lambda_h(
            lambda_h
        )

        if not self._node_online(source):
            return self._failure(
                "source_offline"
            )

        if not self._node_online(target):
            return self._failure(
                "target_offline"
            )

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
            )
        )

        best = {}

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

            for v, channel_dict in self.G[u].items():

                if v in path:
                    continue

                if not self._node_online(v):
                    continue

                if self._is_multigraph():

                    channel_items = (
                        channel_dict.items()
                    )

                else:

                    channel_items = [
                        (
                            channel_dict.get(
                                "channel_key",
                                None,
                            ),
                            channel_dict,
                        )
                    ]

                for key, raw_data in channel_items:

                    data = dict(
                        raw_data
                    )

                    self._normalize_edge_data(
                        data
                    )

                    if not self._channel_available(
                        data
                    ):
                        continue

                    liquidity = (
                        self._estimated_liquidity(
                            data
                        )
                    )

                    if (
                        liquidity is not None
                        and liquidity < amount
                    ):
                        continue

                    # --------------------------------------
                    # Native cost + adaptive cost
                    # --------------------------------------

                    try:

                        edge_info = adaptive_edge_cost(
                            G=self.G,
                            u=u,
                            v=v,
                            data=data,
                            amount=amount,
                            eta=eta,
                            heuristic_fn=(
                                heuristic_fn
                                if heuristic_fn is not None
                                else lnd_cost
                            ),
                            lambda_h=lambda_h,
                        )

                    except (
                        TypeError,
                        ValueError,
                        KeyError,
                        AttributeError,
                    ):

                        continue

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

                    if not self._valid_cost(
                        edge_cost
                    ):
                        continue

                    fee = channel_fee(
                        data,
                        amount,
                    )

                    if not self._valid_metric(
                        fee
                    ):
                        continue

                    delay = channel_delay(
                        data
                    )

                    if not self._valid_metric(
                        delay
                    ):
                        continue

                    failure_probability = (
                        self._failure_probability(
                            data
                        )
                    )

                    channel_reliability = (
                        1.0
                        -
                        failure_probability
                    )

                    new_reliability = (
                        path_reliability
                        *
                        channel_reliability
                    )

                    (
                        distance_km,
                        carbon_intensity,
                        inter_country,
                        inter_continent,
                    ) = self._geographic_metrics(
                        u,
                        v,
                        data,
                    )

                    new_cost = (
                        cost
                        +
                        edge_cost
                    )

                    new_hops = (
                        hops
                        +
                        1
                    )

                    if new_hops > max_hops:
                        continue

                    new_fee = (
                        total_fee
                        +
                        fee
                    )

                    new_delay = (
                        total_delay
                        +
                        delay
                    )

                    new_distance = (
                        total_distance
                        +
                        distance_km
                    )

                    new_carbon = (
                        total_carbon
                        +
                        carbon_intensity
                    )

                    new_inter_country = (
                        inter_country_hops
                        +
                        int(inter_country)
                    )

                    new_inter_continent = (
                        inter_continent_hops
                        +
                        int(inter_continent)
                    )

                    heapq.heappush(
                        pq,
                        (
                            new_cost,
                            new_hops,
                            next(counter),
                            v,
                            path + (v,),
                            edges + (
                                (
                                    u,
                                    v,
                                    key,
                                ),
                            ),
                            new_fee,
                            new_delay,
                            new_reliability,
                            new_distance,
                            new_carbon,
                            new_inter_country,
                            new_inter_continent,
                        )
                    )

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
            )
        )

    # ======================================================
    # Edge Normalization
    # ======================================================

    @staticmethod
    def _normalize_edge_data(data):

        if "fee_base" not in data:

            data["fee_base"] = (
                Dijkstra._safe_float(
                    data.get(
                        "fee_base_msat",
                        0.0,
                    )
                )
            )

        if "fee_rate" not in data:

            data["fee_rate"] = (
                Dijkstra._safe_float(
                    data.get(
                        "fee_proportional_millionths",
                        0.0,
                    )
                )
            )

        if "delay" not in data:

            data["delay"] = (
                Dijkstra._safe_float(
                    data.get(
                        "cltv_expiry_delta",
                        0.0,
                    )
                )
            )

        if "available" not in data:
            data["available"] = True

        if "failure_count" not in data:
            data["failure_count"] = 0

        if "success_count" not in data:
            data["success_count"] = 0

    # ======================================================
    # Directional Liquidity
    # ======================================================

    @staticmethod
    def _estimated_liquidity(data):

        for field in (
            "estimated_liquidity",
            "liquidity_uv",
            "balance_uv",
        ):

            if field not in data:
                continue

            value = data.get(
                field
            )

            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if (
                math.isfinite(value)
                and value >= 0.0
            ):
                return value

        return None

    # ======================================================
    # Node Availability
    # ======================================================

    def _node_online(self, node):

        if node not in self.G:
            return False

        data = self.G.nodes[node]

        if not bool(
            data.get(
                "available",
                True,
            )
        ):
            return False

        if not bool(
            data.get(
                "is_online",
                True,
            )
        ):
            return False

        if not bool(
            data.get(
                "online",
                True,
            )
        ):
            return False

        return True

    # ======================================================
    # Channel Availability
    # ======================================================

    @staticmethod
    def _channel_available(data):

        return bool(
            data.get(
                "available",
                True,
            )
        )

    # ======================================================
    # Failure Probability
    # ======================================================

    @staticmethod
    def _failure_probability(data):

        success = Dijkstra._safe_float(
            data.get(
                "success_count",
                0,
            )
        )

        failure = Dijkstra._safe_float(
            data.get(
                "failure_count",
                0,
            )
        )

        total = (
            success
            +
            failure
        )

        if total > 0.0:

            probability = (
                failure
                /
                total
            )

        elif "failure_probability" in data:

            probability = Dijkstra._safe_float(
                data.get(
                    "failure_probability"
                ),
                default=0.01,
            )

        else:

            probability = 0.01

        return min(
            max(
                probability,
                0.0,
            ),
            1.0,
        )

    # ======================================================
    # Geographic Metrics
    # ======================================================

    def _geographic_metrics(
        self,
        u,
        v,
        data,
    ):

        try:

            features = geo_features(
                self.G,
                u,
                v,
            )

            distance = self._safe_float(
                features.get(
                    "distance_km",
                    0.0,
                )
            )

            carbon = self._safe_float(
                features.get(
                    "carbon_intensity",
                    0.0,
                )
            )

            inter_country = int(
                bool(
                    features.get(
                        "inter_country",
                        False,
                    )
                )
            )

            inter_continent = int(
                bool(
                    features.get(
                        "inter_continent",
                        False,
                    )
                )
            )

            return (
                distance,
                carbon,
                inter_country,
                inter_continent,
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
        ):

            return (
                0.0,
                0.0,
                0,
                0,
            )

    # ======================================================
    # Minimum Liquidity
    # ======================================================

    def _minimum_liquidity(
        self,
        edges,
    ):

        if not edges:
            return None

        values = []

        for u, v, key in edges:

            try:

                if key is None:

                    data = self.G[u][v]

                else:

                    data = self.G[u][v][key]

            except (
                KeyError,
                TypeError,
            ):
                continue

            liquidity = (
                self._estimated_liquidity(
                    data
                )
            )

            if liquidity is not None:
                values.append(
                    liquidity
                )

        if not values:
            return None

        return min(values)

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
                1.0
                -
                path_reliability
            ),

            "total_distance_km": float(
                total_distance
            ),

            "total_carbon": float(
                total_carbon
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

            "eta": None,

            "lambda_h": None,

            "candidate": False,

            "reason": reason,
        }

    # ======================================================
    # Numeric Helpers
    # ======================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0,
    ):

        try:

            value = float(value)

            if not math.isfinite(value):
                return float(default)

            return value

        except (
            TypeError,
            ValueError,
        ):

            return float(default)

    @staticmethod
    def _valid_number(value):

        try:

            value = float(value)

            return math.isfinite(value)

        except (
            TypeError,
            ValueError,
        ):

            return False

    @staticmethod
    def _valid_metric(value):

        try:

            value = float(value)

            return (
                math.isfinite(value)
                and value >= 0.0
            )

        except (
            TypeError,
            ValueError,
        ):

            return False

    @staticmethod
    def _valid_cost(value):

        try:

            value = float(value)

            return (
                math.isfinite(value)
                and value >= 0.0
            )

        except (
            TypeError,
            ValueError,
        ):

            return False