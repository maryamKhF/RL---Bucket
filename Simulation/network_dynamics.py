"""
network_dynamics.py

Responsible for dynamic behavior of Lightning Network during simulation.

Functions:
- Check channel forwarding capability
- Update balances after successful payment
- Record payment history
- Simulate channel/node availability changes
- Maintain network state evolution
"""


from collections import defaultdict
import random


class NetworkDynamics:

    def __init__(
        self,
        G,
        channel_failure_rate=0.01,
        node_failure_rate=0.005,
        recovery_rate=0.05
    ):
        """
        Parameters
        ----------
        G:
            NetworkX Lightning graph

        channel_failure_rate:
            Probability of channel becoming unavailable

        node_failure_rate:
            Probability of node going offline

        recovery_rate:
            Probability of recovery
        """

        self.G = G

        self.history = defaultdict(list)

        self.channel_failure_rate = (
            channel_failure_rate
        )

        self.node_failure_rate = (
            node_failure_rate
        )

        self.recovery_rate = (
            recovery_rate
        )


    # ==================================================
    # Check forwarding possibility
    # ==================================================

    def can_forward(
        self,
        u,
        v,
        key,
        amount
    ):

        if not self.G.has_edge(
            u,
            v,
            key
        ):
            return False


        d = self.G.edges[
            u,
            v,
            key
        ]


        return (
            d.get(
                "available",
                True
            )
            and
            d.get(
                "capacity",
                0
            )
            >= amount
            and
            d.get(
                "balance_uv",
                0
            )
            >= amount
        )



    # ==================================================
    # Record transaction result
    # ==================================================

    def record(
        self,
        tx_id,
        edge,
        success
    ):

        u,v,k = edge

        d = self.G.edges[
            u,
            v,
            k
        ]


        d.setdefault(
            "success_count",
            0
        )

        d.setdefault(
            "failure_count",
            0
        )


        d["success_count"] += int(success)

        d["failure_count"] += int(
            not success
        )


        self.history[
            tx_id
        ].append(
            (
                edge,
                success
            )
        )



    # ==================================================
    # Update channel balance after payment
    # ==================================================

    def settle(
        self,
        edge,
        amount
    ):

        u,v,k = edge

        d = self.G.edges[
            u,
            v,
            k
        ]


        d["balance_uv"] = max(
            0,
            d.get(
                "balance_uv",
                0
            )
            -
            amount
        )


        d["balance_vu"] = min(
            d.get(
                "capacity",
                0
            ),
            d.get(
                "balance_vu",
                0
            )
            +
            amount
        )



    # ==================================================
    # Network evolution step
    # ==================================================

    def update(self):
        """
        Simulate network changes
        after each transaction step.
        """

        self.update_nodes()

        self.update_channels()



    # ==================================================
    # Node availability
    # ==================================================

    def update_nodes(self):

        for node,data in self.G.nodes(
            data=True
        ):

            active = data.get(
                "available",
                True
            )


            if active:

                if random.random() < self.node_failure_rate:

                    data["available"] = False


            else:

                if random.random() < self.recovery_rate:

                    data["available"] = True



    # ==================================================
    # Channel availability
    # ==================================================

    def update_channels(self):

        for u,v,k,data in self.G.edges(
            keys=True,
            data=True
        ):

            active = data.get(
                "available",
                True
            )


            if active:

                if random.random() < self.channel_failure_rate:

                    data["available"] = False


            else:

                if random.random() < self.recovery_rate:

                    data["available"] = True



    # ==================================================
    # Get current usable network
    # ==================================================

    def get_state(self):

        return self.G



    # ==================================================
    # Reset simulation
    # ==================================================

    def reset(self):

        self.history.clear()


        for node,data in self.G.nodes(
            data=True
        ):

            data["available"] = True


        for u,v,k,data in self.G.edges(
            keys=True,
            data=True
        ):

            data["available"] = True
            data["success_count"] = 0
            data["failure_count"] = 0