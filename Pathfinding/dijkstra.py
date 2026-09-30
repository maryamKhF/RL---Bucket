# Pathfinding/dijkstra.py

import heapq
import itertools
import math

from .heuristics import (
    modified_cost,
    geographic_penalty,
    reliability_penalty
)

from Network.topology import geo_features


class Dijkstra:
    """
    Dijkstra-based routing engine for the Lightning-style
    payment network.

    Compatible with:
        - raw 20190501.gml.geo snapshot
        - normalized repository MultiDiGraph
        - learned channel-state information

    Important distinction
    ---------------------
    The raw snapshot provides topology and channel attributes,
    but it does NOT provide the actual directional channel
    balance.

    Therefore:

        capacity != balance
        capacity != liquidity
        capacity != available payment amount

    The Dijkstra engine NEVER uses the snapshot capacity as
    channel liquidity.

    If a learned/observed liquidity estimate is available,
    it may be used as a feasibility constraint.

    Otherwise, routing proceeds without a liquidity constraint
    and the actual payment feasibility is determined by the
    payment simulator.

    Routing pipeline:

        State
          |
          v
        PPO
          |
          v
        eta
          |
          v
        Modified Cost
          |
          v
        Dijkstra
          |
          v
      Candidate Path
          |
          v
    Payment Simulation
          |
          v
       Feedback
          |
          v
    Learned Channel State
    """

    def __init__(self, graph):
        """
        Parameters
        ----------
        graph : networkx.MultiDiGraph
            Lightning-style directed multigraph.

        The graph may be loaded directly from:

            20190501.gml.geo

        or generated through the repository graph builder.
        """

        self.G = graph

    # ========================================================
    # Main Routing
    # ========================================================

    def shortest_path(
        self,
        source,
        target,
        amount,
        heuristic_fn,
        eta=0.0,
        max_hops=12
    ):
        """
        Find the minimum-cost feasible routing path.

        Parameters
        ----------
        source : node id
            Payment source.

        target : node id
            Payment destination.

        amount : float
            Payment amount.

        heuristic_fn : callable
            Native routing cost function.

            Expected signature:

                heuristic_fn(
                    G,
                    u,
                    v,
                    data,
                    amount
                )

        eta : float
            PPO-generated routing correction factor.

            Valid range:

                -1 <= eta <= 1

        max_hops : int
            Maximum number of routing hops.

        Returns
        -------
        dict
            Standardized routing result.

        Important
        ---------
        This method does not infer channel balance from the
        snapshot capacity.

        Actual payment feasibility is evaluated separately by
        the payment simulation layer.
        """

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

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

        except (
            TypeError,
            ValueError
        ):
            raise ValueError(
                "amount must be numeric"
            )

        if not math.isfinite(amount):
            raise ValueError(
                "amount must be finite"
            )

        if amount <= 0:
            raise ValueError(
                "amount must be greater than zero"
            )

        if not isinstance(
            max_hops,
            int
        ):
            raise ValueError(
                "max_hops must be an integer"
            )

        if max_hops <= 0:
            raise ValueError(
                "max_hops must be greater than zero"
            )

        try:
            eta = float(eta)

        except (
            TypeError,
            ValueError
        ):
            raise ValueError(
                "eta must be numeric"
            )

        if not math.isfinite(eta):
            raise ValueError(
                "eta must be finite"
            )

        if eta < -1.0 or eta > 1.0:
            raise ValueError(
                "eta must satisfy -1 <= eta <= 1"
            )

        # ----------------------------------------------------
        # Source / target availability
        # ----------------------------------------------------

        if not self._node_online(source):
            return self._failure(
                "source_offline"
            )

        if not self._node_online(target):
            return self._failure(
                "target_offline"
            )

        # ----------------------------------------------------
        # Priority queue
        # ----------------------------------------------------

        pq = []

        counter = itertools.count()

        heapq.heappush(
            pq,
            (
                0.0,              # accumulated cost
                0,                # hop count
                next(counter),    # tie breaker
                source,           # current node
                (source,),        # node path
                (),               # edge path
                0.0,              # total fee
                0.0,              # total delay
                1.0,              # path reliability
                0.0,              # total distance
                0.0,              # total carbon
                0,                # inter-country hops
                0                 # inter-continent hops
            )
        )

        # ----------------------------------------------------
        # Best known state
        # ----------------------------------------------------

        best = {}

        # ====================================================
        # Dijkstra Search
        # ====================================================

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
                inter_continent_hops
            ) = heapq.heappop(
                pq
            )

            # ------------------------------------------------
            # Destination reached
            # ------------------------------------------------

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
                    inter_continent_hops=inter_continent_hops
                )

            # ------------------------------------------------
            # Maximum hop constraint
            # ------------------------------------------------

            if hops >= max_hops:
                continue

            # ------------------------------------------------
            # Dominance check
            # ------------------------------------------------

            state = (
                u,
                hops
            )

            previous_best = best.get(
                state,
                math.inf
            )

            if cost >= previous_best:
                continue

            best[state] = cost

            # =================================================
            # Explore outgoing channels
            # =================================================

            for v, channel_dict in self.G[u].items():

                # ------------------------------------------------
                # Loop prevention
                # ------------------------------------------------

                if v in path:
                    continue

                # ------------------------------------------------
                # Destination node availability
                # ------------------------------------------------

                if not self._node_online(v):
                    continue

                # ------------------------------------------------
                # MultiDiGraph channels
                # ------------------------------------------------

                for key, data in channel_dict.items():

                    # --------------------------------------------
                    # Normalize ONLY routing attributes.
                    #
                    # No balance is created from capacity.
                    # --------------------------------------------

                    self._normalize_edge_data(
                        data
                    )

                    # --------------------------------------------
                    # Channel availability
                    # --------------------------------------------

                    if not self._channel_available(
                        data
                    ):
                        continue

                    # --------------------------------------------
                    # Learned liquidity feasibility
                    #
                    # IMPORTANT:
                    #
                    # None means:
                    #     liquidity is unknown.
                    #
                    # It does NOT mean zero.
                    #
                    # capacity is NEVER used here.
                    # --------------------------------------------

                    estimated_liquidity = (
                        self._estimated_liquidity(
                            data
                        )
                    )

                    if (
                        estimated_liquidity is not None
                        and
                        estimated_liquidity < amount
                    ):
                        continue

                    # --------------------------------------------
                    # Native routing cost
                    # --------------------------------------------

                    try:

                        native_cost = heuristic_fn(
                            self.G,
                            u,
                            v,
                            data,
                            amount
                        )

                    except Exception:

                        continue

                    if not self._valid_cost(
                        native_cost
                    ):
                        continue

                    native_cost = float(
                        native_cost
                    )

                    # --------------------------------------------
                    # Geographic penalty
                    # --------------------------------------------

                    geo = data.get(
                        "geo_penalty"
                    )

                    if geo is None:

                        try:

                            geo = geographic_penalty(
                                self.G,
                                u,
                                v,
                                data
                            )

                        except Exception:

                            geo = 0.0

                    if not self._valid_cost(
                        geo
                    ):
                        continue

                    geo = float(
                        geo
                    )

                    # --------------------------------------------
                    # PPO-modified routing cost
                    # --------------------------------------------

                    edge_cost = modified_cost(
                        native_cost,
                        geo,
                        eta
                    )

                    if not self._valid_cost(
                        edge_cost
                    ):
                        continue

                    edge_cost = float(
                        edge_cost
                    )

                    # --------------------------------------------
                    # Forwarding fee
                    #
                    # Raw snapshot:
                    #
                    # fee_base_msat
                    # fee_proportional_millionths
                    #
                    # Normalized:
                    #
                    # fee_base
                    # fee_rate
                    # --------------------------------------------

                    fee = self._channel_fee(
                        data,
                        amount
                    )

                    if not self._valid_metric(
                        fee
                    ):
                        continue

                    # --------------------------------------------
                    # Delay
                    # --------------------------------------------

                    delay = self._channel_delay(
                        data
                    )

                    # --------------------------------------------
                    # Reliability estimate
                    #
                    # This is NOT channel balance.
                    #
                    # It represents estimated probability of
                    # forwarding failure based on available
                    # history/statistics.
                    # --------------------------------------------

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

                    # --------------------------------------------
                    # Geographic metrics
                    # --------------------------------------------

                    (
                        distance_km,
                        carbon_intensity,
                        inter_country,
                        inter_continent
                    ) = self._geographic_metrics(
                        u,
                        v,
                        data
                    )

                    # --------------------------------------------
                    # Accumulated values
                    # --------------------------------------------

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

                    # --------------------------------------------
                    # Add new state to priority queue
                    # --------------------------------------------

                    heapq.heappush(
                        pq,
                        (
                            new_cost,
                            new_hops,
                            next(counter),

                            v,

                            path + (
                                v,
                            ),

                            edges + (
                                (
                                    u,
                                    v,
                                    key
                                ),
                            ),

                            new_fee,
                            new_delay,
                            new_reliability,
                            new_distance,
                            new_carbon,
                            new_inter_country,
                            new_inter_continent
                        )
                    )

        # ----------------------------------------------------
        # No feasible topological route
        # ----------------------------------------------------

        return self._failure(
            "no_feasible_path"
        )

    # ========================================================
    # Snapshot Attribute Normalization
    # ========================================================

    @staticmethod
    def _normalize_edge_data(
        data
    ):
        """
        Normalize only attributes required by the routing
        implementation.

        IMPORTANT:

        The snapshot's capacity is preserved as capacity.

        It is NOT converted into:

            balance_uv
            liquidity_uv
            estimated_liquidity

        because the snapshot does not reveal directional
        channel balance.
        """

        # ----------------------------------------------------
        # Base fee
        # ----------------------------------------------------

        if "fee_base" not in data:

            if "fee_base_msat" in data:

                data["fee_base"] = (
                    Dijkstra._safe_float(
                        data[
                            "fee_base_msat"
                        ]
                    )
                )

            else:

                data["fee_base"] = 0.0

        # ----------------------------------------------------
        # Proportional fee rate
        # ----------------------------------------------------

        if "fee_rate" not in data:

            if (
                "fee_proportional_millionths"
                in data
            ):

                data["fee_rate"] = (
                    Dijkstra._safe_float(
                        data[
                            "fee_proportional_millionths"
                        ]
                    )
                )

            else:

                data["fee_rate"] = 0.0

        # ----------------------------------------------------
        # Delay
        # ----------------------------------------------------

        if "delay" not in data:

            if "cltv_expiry_delta" in data:

                data["delay"] = (
                    Dijkstra._safe_float(
                        data[
                            "cltv_expiry_delta"
                        ]
                    )
                )

            else:

                data["delay"] = 0.0

        # ----------------------------------------------------
        # Availability
        #
        # Raw snapshot does not necessarily contain an
        # explicit availability field.
        # ----------------------------------------------------

        if "available" not in data:

            data["available"] = True

        # ----------------------------------------------------
        # Failure statistics
        #
        # These are optional learned/simulation attributes.
        # They are NOT claimed to originate from the GML.
        # ----------------------------------------------------

        if "failure_count" not in data:

            data["failure_count"] = 0

        if "success_count" not in data:

            data["success_count"] = 0

    # ========================================================
    # Learned / Observed Liquidity
    # ========================================================

    @staticmethod
    def _estimated_liquidity(
        data
    ):
        """
        Return learned or observed directional liquidity.

        Priority:

            1. estimated_liquidity
            2. liquidity_uv
            3. balance_uv

        If none exists:

            return None

        This is intentional.

        The raw 20190501.gml.geo snapshot does NOT provide
        directional channel balance.

        Therefore:

            capacity is NEVER used here.

        None means that the routing engine currently has no
        liquidity estimate for this channel.
        """

        if "estimated_liquidity" in data:

            value = Dijkstra._safe_float(
                data[
                    "estimated_liquidity"
                ],
                default=-1.0
            )

            if value >= 0.0:

                return value

        if "liquidity_uv" in data:

            value = Dijkstra._safe_float(
                data[
                    "liquidity_uv"
                ],
                default=-1.0
            )

            if value >= 0.0:

                return value

        if "balance_uv" in data:

            value = Dijkstra._safe_float(
                data[
                    "balance_uv"
                ],
                default=-1.0
            )

            if value >= 0.0:

                return value

        return None

    # ========================================================
    # Node Validation
    # ========================================================

    def _node_online(
        self,
        node
    ):
        """
        Check node availability.

        Raw snapshot:
            if 'online' is absent, node is considered usable.

        Learned/processed graph:
            explicit 'online' value is respected.
        """

        return bool(
            self.G.nodes[node].get(
                "online",
                True
            )
        )

    # ========================================================
    # Channel Availability
    # ========================================================

    @staticmethod
    def _channel_available(
        data
    ):
        """
        Check explicit channel availability.

        Absence of the attribute means that the channel is
        not known to be unavailable.
        """

        return bool(
            data.get(
                "available",
                True
            )
        )

    # ========================================================
    # Channel Fee
    # ========================================================

    @staticmethod
    def _channel_fee(
        data,
        amount
    ):
        """
        Calculate forwarding fee.

        Supported snapshot attributes:

            fee_base_msat
            fee_proportional_millionths

        Supported normalized attributes:

            fee_base
            fee_rate

        Formula:

            fee =
                base_fee
                +
                amount * fee_rate / 1,000,000

        fee_rate is expressed in PPM.
        """

        base_fee = Dijkstra._safe_float(
            data.get(
                "fee_base",
                data.get(
                    "fee_base_msat",
                    0.0
                )
            )
        )

        fee_rate = Dijkstra._safe_float(
            data.get(
                "fee_rate",
                data.get(
                    "fee_proportional_millionths",
                    0.0
                )
            )
        )

        return (
            base_fee
            +
            (
                amount
                *
                fee_rate
                /
                1_000_000.0
            )
        )

    # ========================================================
    # Channel Delay
    # ========================================================

    @staticmethod
    def _channel_delay(
        data
    ):
        """
        Obtain channel delay.

        Supported attributes:

            delay
            cltv_expiry_delta
        """

        return Dijkstra._safe_float(
            data.get(
                "delay",
                data.get(
                    "cltv_expiry_delta",
                    0.0
                )
            )
        )

    # ========================================================
    # Failure Probability
    # ========================================================

    @staticmethod
    def _failure_probability(
        data
    ):
        """
        Estimate channel forwarding failure probability.

        If experience statistics exist:

            p_failure =
                failures / (successes + failures)

        Otherwise, use an explicitly supplied estimate.

        This quantity is separate from channel liquidity.
        """

        success = Dijkstra._safe_float(
            data.get(
                "success_count",
                0
            )
        )

        failure = Dijkstra._safe_float(
            data.get(
                "failure_count",
                0
            )
        )

        total = (
            success
            +
            failure
        )

        if total > 0:

            probability = (
                failure
                /
                total
            )

        elif "failure_probability" in data:

            probability = (
                Dijkstra._safe_float(
                    data[
                        "failure_probability"
                    ]
                )
            )

        else:

            # No experience yet.
            #
            # This is only a neutral initial estimate.
            # It is NOT obtained from the GML snapshot.

            probability = 0.01

        return min(
            max(
                probability,
                0.0
            ),
            1.0
        )

    # ========================================================
    # Geographic Metrics
    # ========================================================

    def _geographic_metrics(
        self,
        u,
        v,
        data
    ):
        """
        Obtain geographic features from node metadata.

        Expected topology features include:

            latitude
            longitude
            country
            continent
            carbon_intensity

        geo_features() handles the actual calculation.
        """

        try:

            features = geo_features(
                self.G,
                u,
                v
            )

            distance = self._safe_float(
                features.get(
                    "distance_km",
                    0.0
                )
            )

            carbon = self._safe_float(
                features.get(
                    "carbon_intensity",
                    0.0
                )
            )

            inter_country = int(
                bool(
                    features.get(
                        "inter_country",
                        False
                    )
                )
            )

            inter_continent = int(
                bool(
                    features.get(
                        "inter_continent",
                        False
                    )
                )
            )

            return (
                distance,
                carbon,
                inter_country,
                inter_continent
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            return (
                0.0,
                0.0,
                0,
                0
            )

    # ========================================================
    # Minimum Known Liquidity
    # ========================================================

    def _minimum_liquidity(
        self,
        edges
    ):
        """
        Return the minimum KNOWN directional liquidity.

        Important:

        Unknown liquidity is represented by None.

        Unknown liquidity must NOT be interpreted as zero.

        If no channel in the route has a learned/observed
        liquidity estimate, the result is None.
        """

        if not edges:
            return None

        values = []

        for u, v, key in edges:

            try:

                data = self.G[u][v][key]

            except (
                KeyError,
                TypeError
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

        return min(
            values
        )

    # ========================================================
    # Success Result
    # ========================================================

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
        inter_continent_hops
    ):
        """
        Build standardized successful routing result.
        """

        min_liquidity = (
            self._minimum_liquidity(
                edges
            )
        )

        return {

            "success": True,

            "path": list(
                path
            ),

            "edges": list(
                edges
            ),

            "cost": float(
                cost
            ),

            "hop_count": int(
                hop_count
            ),

            "total_fee": float(
                total_fee
            ),

            "total_delay": float(
                total_delay
            ),

            # None means:
            # no learned/observed liquidity estimate.
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

            "inter_country_hops": int(
                inter_country_hops
            ),

            "inter_continent_hops": int(
                inter_continent_hops
            )
        }

    # ========================================================
    # Failure Result
    # ========================================================

    @staticmethod
    def _failure(
        reason="no_feasible_path"
    ):
        """
        Standardized routing failure.

        This means that no route satisfying the currently
        available routing constraints was found.

        It does NOT represent stochastic payment failure.
        """

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

            "reason": reason
        }

    # ========================================================
    # Numeric Helpers
    # ========================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0
    ):
        """
        Safely convert value to finite float.
        """

        try:

            value = float(
                value
            )

            if not math.isfinite(
                value
            ):

                return float(
                    default
                )

            return value

        except (
            TypeError,
            ValueError
        ):

            return float(
                default
            )

    @staticmethod
    def _valid_metric(
        value
    ):
        """
        Validate a non-negative finite metric.
        """

        try:

            value = float(
                value
            )

            return (
                math.isfinite(
                    value
                )
                and
                value >= 0.0
            )

        except (
            TypeError,
            ValueError
        ):

            return False

    @staticmethod
    def _valid_cost(
        value
    ):
        """
        Dijkstra requires non-negative edge costs.
        """

        try:

            value = float(
                value
            )

            return (
                math.isfinite(
                    value
                )
                and
                value >= 0.0
            )

        except (
            TypeError,
            ValueError
        ):

            return False