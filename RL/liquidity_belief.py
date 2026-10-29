"""Online interval estimates for hidden directed channel liquidity."""

from __future__ import annotations

import math


class LiquidityBelief:
    """Track conservative per-direction bounds from observed outcomes.

    The initial interval is ``[0, capacity]``. Capacity is only a public
    physical upper bound; it is never treated as available liquidity.
    Successful transfers shift the interval with the observed settlement,
    while a liquidity failure narrows the upper bound below the attempted
    amount.
    """

    def __init__(self, graph, default_capacity=None):
        if graph is None:
            raise ValueError("graph is required")
        fallback_capacity = self._number(default_capacity)
        if fallback_capacity is not None and fallback_capacity < 0:
            raise ValueError("default_capacity must be nonnegative")
        self.default_capacity = fallback_capacity
        self.bounds = {}
        self.capacities = {}
        self._reverse_edges = {}
        self._initialize(graph)

    def _initialize(self, graph):
        groups = {}
        edges = (
            graph.edges(keys=True, data=True)
            if graph.is_multigraph()
            else ((u, v, None, data) for u, v, data in graph.edges(data=True))
        )
        for u, v, key, data in edges:
            edge = (u, v, key)
            capacity = self._number(data.get("capacity"))
            if capacity is None or capacity < 0:
                capacity = (
                    self.default_capacity
                    if self.default_capacity is not None
                    else 0.0
                )
            self.capacities[edge] = capacity
            self.bounds[edge] = [0.0, capacity]

            channel_id = str(data.get("channel_id", key))
            base_id = channel_id[:-4] if channel_id.endswith("-rev") else channel_id
            group_key = (base_id, frozenset((u, v)))
            groups.setdefault(group_key, []).append((edge, channel_id.endswith("-rev")))

        for members in groups.values():
            if len(members) == 2:
                (first, _), (second, _) = members
                if first[:2] == second[:2][::-1]:
                    self._reverse_edges[first] = second
                    self._reverse_edges[second] = first

    @staticmethod
    def _number(value):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _edge(edge):
        if isinstance(edge, dict):
            return edge.get("source"), edge.get("target"), edge.get("channel_key")
        if isinstance(edge, (tuple, list)) and len(edge) == 3:
            return tuple(edge)
        return None

    def interval(self, edge, capacity=None):
        identity = self._edge(edge)
        if identity is None:
            return 0.0, max(0.0, self._number(capacity) or 0.0)
        if identity in self.bounds:
            return tuple(self.bounds[identity])
        maximum = self._number(capacity)
        return 0.0, max(0.0, maximum or 0.0)

    def observe_failure(self, edge, amount):
        identity = self._edge(edge)
        amount = self._number(amount)
        if identity not in self.bounds or amount is None or amount <= 0:
            return
        lower, upper = self.bounds[identity]
        strict_upper = math.nextafter(amount, -math.inf)
        if strict_upper < lower:
            lower = 0.0
        self.bounds[identity] = [lower, max(lower, min(upper, strict_upper))]

    def observe_success(self, edges, amount):
        amount = self._number(amount)
        if amount is None or amount <= 0:
            return
        identities = [self._edge(edge) for edge in edges]
        for identity in identities:
            if identity not in self.bounds:
                continue
            lower, upper = self.bounds[identity]
            self.bounds[identity] = [max(0.0, lower - amount), max(0.0, upper - amount)]
            reverse = self._reverse_edges.get(identity)
            if reverse in self.bounds:
                reverse_lower, reverse_upper = self.bounds[reverse]
                capacity = self.capacities[reverse]
                self.bounds[reverse] = [
                    min(capacity, reverse_lower + amount),
                    min(capacity, reverse_upper + amount),
                ]


__all__ = ["LiquidityBelief"]
