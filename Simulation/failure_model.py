"""
failure_model.py

Lightning Network failure simulation model.

This module provides:

1. Static failure probability assignment
   based on network topology and geography.

2. Dynamic payment failure evaluation
   during routing simulation.

Compatible with:
    - Network.graph_builder
    - Network.channel
    - Simulation.payment_simulator
"""


import numpy as np
import random
from datetime import datetime




# Static Failure Probability Assignment



def assign_failure_probabilities(
        G,
        average_rate,
        seed=42
):
    """
    Assign failure probabilities to all channels.

    This should be executed after graph construction.

    Parameters
    ----------
    G :
        NetworkX MultiGraph

    average_rate :
        Base failure rate

    seed :
        Random seed
    """

    rng = np.random.default_rng(seed)


    for u, v, k, d in G.edges(
            keys=True,
            data=True
    ):

        if u == v:
            continue


        country_u = G.nodes[u].get(
            "country",
            ""
        )

        country_v = G.nodes[v].get(
            "country",
            ""
        )


        # Geographic failure abstraction
        if country_u != country_v:

            factor = (
                2.0
                if _intercontinental(
                    G,
                    u,
                    v
                )
                else 1.0
            )

        else:

            factor = 0.7



        probability = (
            average_rate
            *
            factor
            *
            rng.uniform(
                0.8,
                1.2
            )
        )


        d["failure_probability"] = float(
            min(
                0.95,
                probability
            )
        )


        # Channel state

        d["available"] = True

        d["failure_count"] = 0

        d["last_failure"] = None



def _intercontinental(
        G,
        u,
        v
):
    """
    Detect long-distance links.
    """

    a = G.nodes[u]
    b = G.nodes[v]


    return (
        abs(
            a.get(
                "longitude",
                0
            )
            -
            b.get(
                "longitude",
                0
            )
        )
        >
        45

        and

        abs(
            a.get(
                "latitude",
                0
            )
            -
            b.get(
                "latitude",
                0
            )
        )
        >
        10
    )




# Dynamic Failure Model



class FailureModel:

    """
    Runtime payment failure evaluator.

    Uses probabilities assigned during
    network initialization.
    """


    def __init__(
            self,
            node_failure_probability=0.01,
            liquidity_failure_probability=0.05,
            random_failure_probability=0.01
    ):


        self.node_failure_probability = (
            node_failure_probability
        )


        self.liquidity_failure_probability = (
            liquidity_failure_probability
        )


        self.random_failure_probability = (
            random_failure_probability
        )



    # --------------------------------------------------------
    # Node failure
    # --------------------------------------------------------

    def check_node_failure(
            self,
            node
    ):


        if hasattr(
            node,
            "is_online"
        ):

            if not node.is_online:
                return True



        return (
            random.random()
            <
            self.node_failure_probability
        )



    # --------------------------------------------------------
    # Channel failure
    # --------------------------------------------------------

    def check_channel_failure(
            self,
            channel
    ):
        """
        Uses channel-specific probability
        assigned in preprocessing.
        """


        # Already unavailable

        if hasattr(
            channel,
            "available"
        ):

            if not channel.available:
                return True



        probability = getattr(
            channel,
            "failure_probability",
            0.01
        )


        failed = (
            random.random()
            <
            probability
        )


        if failed:

            channel.failure_count += 1

            channel.last_failure = (
                datetime.now()
            )


            channel.available = False



        return failed



    # --------------------------------------------------------
    # Liquidity failure
    # --------------------------------------------------------

    def check_liquidity_failure(
            self,
            channel,
            amount,
            sender=None
    ):


        # If channel implements liquidity check

        if hasattr(
            channel,
            "has_liquidity"
        ):

            return not channel.has_liquidity(
                amount,
                sender
            )


        # Generic capacity model

        capacity = getattr(
            channel,
            "capacity",
            float("inf")
        )


        if amount > capacity:

            return True



        # stochastic liquidity failure

        return (
            random.random()
            <
            self.liquidity_failure_probability
        )



    # --------------------------------------------------------
    # Full Payment Evaluation
    # --------------------------------------------------------

    def evaluate_payment_failure(
            self,
            route,
            amount,
            network
    ):


        for node_id in route:


            node = self._get_node(
                network,
                node_id
            )


            if node:

                if self.check_node_failure(
                    node
                ):

                    return {

                        "success":False,

                        "reason":
                            "node_failure",

                        "failed_element":
                            node_id
                    }



        for i in range(
            len(route)-1
        ):


            src = route[i]

            dst = route[i+1]


            channel = self._get_channel(
                network,
                src,
                dst
            )


            if channel is None:

                return {

                    "success":False,

                    "reason":
                        "missing_channel",

                    "failed_element":
                        f"{src}-{dst}"
                }



            if self.check_channel_failure(
                channel
            ):

                return {

                    "success":False,

                    "reason":
                        "channel_failure",

                    "failed_element":
                        f"{src}-{dst}"
                }



            if self.check_liquidity_failure(
                channel,
                amount,
                src
            ):

                return {

                    "success":False,

                    "reason":
                        "liquidity_failure",

                    "failed_element":
                        f"{src}-{dst}"
                }



        if (
            random.random()
            <
            self.random_failure_probability
        ):

            return {

                "success":False,

                "reason":
                    "random_failure",

                "failed_element":
                    None
            }



        return {

            "success":True,

            "reason":None,

            "failed_element":None
        }



    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _get_node(
            self,
            network,
            node_id
    ):


        if hasattr(
            network,
            "nodes"
        ):

            return network.nodes.get(
                node_id
            )


        return None



    def _get_channel(
            self,
            network,
            src,
            dst
    ):


        if hasattr(
            network,
            "channels"
        ):

            for channel in network.channels:


                if (
                    channel.node1 == src
                    and
                    channel.node2 == dst
                ):

                    return channel



                if (
                    channel.node1 == dst
                    and
                    channel.node2 == src
                ):

                    return channel


        return None