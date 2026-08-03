"""
environment.py

Lightning Network Simulation Environment.

Responsibilities:
- Execute payments
- Update channel states
- Track failures/successes
- Provide feedback for RL and Bucket routing

Routing algorithms are NOT implemented here.
"""


import random


class LightningEnvironment:


    def __init__(
            self,
            graph,
            seed=42
    ):
        """
        Parameters
        ----------
        graph:
            NetworkX MultiDiGraph
        """

        self.G = graph

        self.rng = random.Random(seed)


        # Simulation statistics

        self.total_payments = 0

        self.successful_payments = 0

        self.failed_payments = 0




    # Network Observation



    def get_neighbors(
            self,
            node_id
    ):
        """
        Return available next-hop nodes.
        """

        neighbors = []


        for n in self.G.successors(node_id):

            node_data = self.G.nodes[n]

            if node_data.get(
                    "online",
                    True
            ):
                neighbors.append(n)


        return neighbors



    def get_channel(
            self,
            u,
            v,
            key=None
    ):
        """
        Return channel attributes.
        """

        if not self.G.has_edge(u, v):

            return None


        edges = self.G[u][v]


        if key:

            return edges.get(key)


        # return first available channel

        return list(
            edges.values()
        )[0]




    # Payment Execution



    def execute_payment(
            self,
            path,
            amount
    ):
        """
        Execute payment through given path.

        Example:

        A -> B -> C -> D

        """

        self.total_payments += 1


        used_edges = []


        for i in range(
            len(path)-1
        ):


            u = path[i]

            v = path[i+1]


            if not self.G.has_edge(
                    u,
                    v
            ):

                return self._failure(
                    reason="missing_channel",
                    edge=(u,v)
                )


            # Select first available channel

            channel_key, channel = self._select_channel(
                u,
                v
            )


            if channel is None:

                return self._failure(
                    reason="no_available_channel",
                    edge=(u,v)
                )



            # Check liquidity

            if (
                channel["balance_uv"]
                <
                amount
            ):

                return self._failure(
                    reason="insufficient_liquidity",
                    edge=(u,v)
                )



            # Failure probability simulation

            if self.rng.random() < channel["failure_probability"]:

                channel["failure_count"] += 1


                return self._failure(
                    reason="probabilistic_failure",
                    edge=(u,v)
                )



            # Update balances

            channel["balance_uv"] -= amount

            channel["balance_vu"] += amount


            channel["success_count"] += 1


            used_edges.append(
                (u,v)
            )



        self.successful_payments += 1


        return True, {

            "status":
                "success",

            "used_edges":
                used_edges,

            "amount":
                amount,

            "reward":
                self.calculate_reward(
                    True
                )
        }




    # Channel Selection



    def _select_channel(
            self,
            u,
            v
    ):
        """
        Select a channel between two nodes.

        MultiDiGraph can contain multiple channels.
        """


        channels = self.G[u][v]


        for key,data in channels.items():

            if data.get(
                    "available",
                    True
            ):

                return key,data


        return None,None




    # Failure Handling



    def _failure(
            self,
            reason,
            edge
    ):

        self.failed_payments += 1


        return False, {

            "status":
                "failed",

            "reason":
                reason,

            "failed_edge":
                edge,

            "reward":
                self.calculate_reward(
                    False
                )
        }




    # RL Support



    def get_state(
            self,
            node_id
    ):
        """
        Basic state representation.

        Later connected to state_encoder.py.
        """


        neighbors = self.get_neighbors(
            node_id
        )


        state = {

            "node":
                node_id,

            "available_neighbors":
                len(neighbors),


            "degree":
                self.G.degree(node_id),


            "online":
                int(
                    self.G.nodes[node_id]
                    .get("online",True)
                )

        }


        return state



    def calculate_reward(
            self,
            success
    ):
        """
        Reward function.

        Later can be replaced
        by multi-objective reward.
        """


        if success:

            return 1.0


        return -1.0



    # Reset and Statistics



    def reset_statistics(self):

        self.total_payments = 0

        self.successful_payments = 0

        self.failed_payments = 0



    def statistics(self):

        total = self.total_payments


        return {

            "total_payments":
                total,

            "successful":
                self.successful_payments,


            "failed":
                self.failed_payments,


            "success_rate":

                (
                    self.successful_payments /
                    total
                    if total > 0
                    else 0
                )
        }