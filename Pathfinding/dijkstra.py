# Pathfinding/dijkstra.py

import heapq
import math

from .heuristics import modified_cost, geographic_penalty


class Dijkstra:
    """
    Generic Dijkstra router for Lightning Network.

    Compatible with:
         Network
         Simulation
         Bucket
         RL
         Evaluation
    """

    def __init__(self, graph):
        self.G = graph

    # --------------------------------------------------------
    # Main Routing
    # --------------------------------------------------------

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
        Parameters
        ----------
        source : node id

        target : node id

        amount : payment amount

        heuristic_fn :
            Cost function from heuristics.py

        eta :
            Geographic penalty coefficient.

        max_hops :
            Maximum allowed hops.

        Returns
        -------
        dict
        """

        if source not in self.G:
            raise ValueError(f"Unknown source node: {source}")

        if target not in self.G:
            raise ValueError(f"Unknown target node: {target}")

        if not self.G.nodes[source].get("online", True):
            return self._failure()

        if not self.G.nodes[target].get("online", True):
            return self._failure()

        pq = []

        heapq.heappush(
            pq,
            (
                0.0,                # accumulated cost
                0,                  # hops
                source,
                (source,),
                (),
                0.0,                # total fee
                0.0                 # total delay
            )
        )

        best = {}

        while pq:

            (
                cost,
                hops,
                u,
                path,
                edges,
                total_fee,
                total_delay

            ) = heapq.heappop(pq)

            if u == target:

                return {

                    "success": True,

                    "path": list(path),

                    "edges": list(edges),

                    "cost": cost,

                    "hop_count": hops,

                    "total_fee": total_fee,

                    "total_delay": total_delay,

                    "min_liquidity": self._minimum_liquidity(edges)

                }

            if hops >= max_hops:
                continue

            state = (u, hops)

            if cost >= best.get(state, math.inf):
                continue

            best[state] = cost

            # ------------------------------------------
            # Explore outgoing channels
            # ------------------------------------------

            for v, keys in self.G[u].items():

                if v in path:
                    continue

                if not self.G.nodes[v].get("online", True):
                    continue

                for key, data in keys.items():

                    if not data.get("available", True):
                        continue

                    # IMPORTANT:
                    # Lightning routes depend on liquidity,
                    # not channel capacity.

                    if data.get("balance_uv", 0) < amount:
                        continue

                    native_cost = heuristic_fn(
                        self.G,
                        u,
                        v,
                        data,
                        amount
                    )

                    geo = data.get("geo_penalty")

                    if geo is None:

                        geo = geographic_penalty(
                            self.G,
                            u,
                            v,
                            data
                        )

                    edge_cost = modified_cost(
                        native_cost,
                        geo,
                        eta
                    )

                    heapq.heappush(

                        pq,

                        (

                            cost + edge_cost,

                            hops + 1,

                            v,

                            path + (v,),

                            edges + ((u, v, key),),

                            total_fee + data.get("fee_base", 0)
                            + amount * data.get("fee_rate", 0),

                            total_delay + data.get("delay", 0)

                        )

                    )

        return self._failure()

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _minimum_liquidity(self, edges):

        if not edges:
            return 0

        values = []

        for u, v, key in edges:

            values.append(

                self.G[u][v][key]["balance_uv"]

            )

        return min(values)

    @staticmethod
    def _failure():

        return {

            "success": False,

            "path": None,

            "edges": None,

            "cost": math.inf,

            "hop_count": 0,

            "total_fee": 0,

            "total_delay": 0,

            "min_liquidity": 0

        }