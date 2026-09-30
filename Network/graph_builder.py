import json
from pathlib import Path

import networkx as nx
import numpy as np

from .node import Node
from .channel import Channel


class LNGraphBuilder:
    """
    Builds a NetworkX MultiDiGraph from:

        1. Real Lightning Network JSON snapshot
        2. In-memory JSON-like data
        3. Synthetic Lightning topology

    Important modeling rules
    ------------------------
    - rgb_color is used as the numeric carbon-intensity value.
    - Larger rgb_color means higher carbon intensity.
    - Channel capacity is NOT treated as directional liquidity.
    - If directional balance/liquidity is not provided by the input,
      it remains unknown (None).
    - Learned/observed liquidity can be added later by the routing
      and learning modules.
    """

    # ==========================================================
    # Load Real LN Snapshot from JSON File
    # ==========================================================

    def from_json(self, path):

        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        return self.from_data(data)

    # ==========================================================
    # Load Real LN Snapshot from In-Memory Data
    # ==========================================================

    def from_data(self, data):

        G = nx.MultiDiGraph()

        nodes = data.get(
            "nodes",
            data.get(
                "vertices",
                []
            )
        )

        channels = data.get(
            "channels",
            data.get(
                "edges",
                []
            )
        )

        # ======================================================
        # Add Nodes
        # ======================================================

        for n in nodes:

            nid = n.get(
                "node_id",
                n.get(
                    "id"
                )
            )

            if nid is None:
                continue

            nid = str(nid)

            # --------------------------------------------------
            # Geographic information
            # --------------------------------------------------

            latitude = n.get(
                "latitude",
                n.get(
                    "lat",
                    0.0
                )
            )

            longitude = n.get(
                "longitude",
                n.get(
                    "lon",
                    0.0
                )
            )

            country = n.get(
                "country",
                "US"
            )

            # --------------------------------------------------
            # Carbon intensity
            #
            # rgb_color is treated as a numeric carbon value.
            #
            # Larger rgb_color
            #       ->
            # higher carbon intensity
            # --------------------------------------------------

            rgb_color = n.get(
                "rgb_color",
                None
            )

            if rgb_color is not None:

                try:
                    rgb_color = float(
                        rgb_color
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    rgb_color = None

            # Backward compatibility:
            # if rgb_color is unavailable, use explicitly
            # provided carbon_intensity.
            #
            # No artificial default such as 300 is introduced
            # for real snapshot data.

            if rgb_color is not None:

                carbon_intensity = rgb_color

            else:

                carbon_value = n.get(
                    "carbon_intensity",
                    None
                )

                if carbon_value is not None:

                    try:
                        carbon_intensity = float(
                            carbon_value
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        carbon_intensity = 0.0

                else:

                    carbon_intensity = 0.0

            # --------------------------------------------------
            # Online status
            # --------------------------------------------------

            online = n.get(
                "online",
                True
            )

            # --------------------------------------------------
            # Create Node
            # --------------------------------------------------

            node = Node(

                node_id=nid,

                country=str(
                    country
                ),

                latitude=float(
                    latitude
                ),

                longitude=float(
                    longitude
                ),

                carbon_intensity=float(
                    carbon_intensity
                ),

                online=bool(
                    online
                )
            )

            # Keep rgb_color explicitly as a node attribute.
            #
            # This allows the graph to retain the original
            # carbon-related field from the dataset.

            node_attributes = {
                **node.__dict__,
                "rgb_color": rgb_color
            }

            G.add_node(
                nid,
                **node_attributes
            )

        # ======================================================
        # Add Channels
        # ======================================================

        for i, e in enumerate(channels):

            source = e.get(
                "source"
            )

            target = e.get(
                "target"
            )

            if source is None or target is None:
                continue

            u = str(
                source
            )

            v = str(
                target
            )

            if (
                u not in G
                or
                v not in G
            ):
                continue

            # --------------------------------------------------
            # Channel Capacity
            #
            # Capacity is retained as an advertised/static
            # channel attribute.
            #
            # IMPORTANT:
            # capacity is NOT directional liquidity.
            # --------------------------------------------------

            capacity = e.get(
                "capacity",
                e.get(
                    "capacity_sat",
                    None
                )
            )

            if capacity is not None:

                try:
                    capacity = float(
                        capacity
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    capacity = None

            # --------------------------------------------------
            # Channel ID
            # --------------------------------------------------

            channel_id = e.get(
                "channel_id",
                e.get(
                    "scid",
                    f"ch-{i}"
                )
            )

            channel_id = str(
                channel_id
            )

            # --------------------------------------------------
            # Base Fee
            #
            # fee_base_msat -> fee_base
            # --------------------------------------------------

            fee_base = e.get(
                "fee_base",
                e.get(
                    "fee_base_msat",
                    0
                )
            )

            fee_base = float(
                fee_base
            )

            # --------------------------------------------------
            # Fee Rate
            #
            # fee_proportional_millionths -> fee_rate
            # --------------------------------------------------

            fee_rate = e.get(
                "fee_rate",
                e.get(
                    "fee_proportional_millionths",
                    0
                )
            )

            fee_rate = float(
                fee_rate
            )

            # --------------------------------------------------
            # Delay
            #
            # cltv_expiry_delta -> delay
            # --------------------------------------------------

            delay = e.get(
                "delay",
                e.get(
                    "cltv_expiry_delta",
                    0.0
                )
            )

            delay = float(
                delay
            )

            # --------------------------------------------------
            # Failure Probability
            # --------------------------------------------------

            failure_probability = e.get(
                "failure_probability",
                0.01
            )

            failure_probability = float(
                failure_probability
            )

            # --------------------------------------------------
            # Availability
            # --------------------------------------------------

            available = e.get(
                "available",
                True
            )

            available = bool(
                available
            )

            # --------------------------------------------------
            # Directional Liquidity / Balance
            #
            # Do NOT use capacity / 2.
            #
            # The real snapshot may not contain directional
            # balances. In that case the value remains unknown.
            # --------------------------------------------------

            balance_uv = e.get(
                "balance_uv",
                None
            )

            balance_vu = e.get(
                "balance_vu",
                None
            )

            if balance_uv is not None:

                try:
                    balance_uv = float(
                        balance_uv
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    balance_uv = None

            if balance_vu is not None:

                try:
                    balance_vu = float(
                        balance_vu
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    balance_vu = None

            # --------------------------------------------------
            # Learned / Observed Liquidity
            #
            # This is optional and is NOT inferred from capacity.
            # It can later be populated from routing experience.
            # --------------------------------------------------

            estimated_liquidity_uv = e.get(
                "estimated_liquidity_uv",
                e.get(
                    "estimated_liquidity",
                    None
                )
            )

            if estimated_liquidity_uv is not None:

                try:
                    estimated_liquidity_uv = float(
                        estimated_liquidity_uv
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    estimated_liquidity_uv = None

            estimated_liquidity_vu = e.get(
                "estimated_liquidity_vu",
                None
            )

            if estimated_liquidity_vu is not None:

                try:
                    estimated_liquidity_vu = float(
                        estimated_liquidity_vu
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    estimated_liquidity_vu = None

            # --------------------------------------------------
            # Create Channel
            # --------------------------------------------------

            channel = Channel(

                channel_id=channel_id,

                capacity=capacity,

                fee_base=fee_base,

                fee_rate=fee_rate,

                delay=delay,

                failure_probability=failure_probability,

                available=available,

                balance_uv=balance_uv,

                balance_vu=balance_vu

            )

            # --------------------------------------------------
            # Forward Channel: u -> v
            # --------------------------------------------------

            forward_attributes = {
                **channel.__dict__
            }

            if estimated_liquidity_uv is not None:

                forward_attributes[
                    "estimated_liquidity"
                ] = estimated_liquidity_uv

            G.add_edge(

                u,

                v,

                key=channel.channel_id,

                **forward_attributes

            )

            # --------------------------------------------------
            # Reverse Channel: v -> u
            #
            # Only create the reverse representation because
            # the graph model uses directed routing.
            #
            # Directional balances are swapped only when they
            # are actually known.
            # --------------------------------------------------

            reverse_channel = {
                **channel.__dict__,

                "channel_id":
                    channel.channel_id + "-rev",

                "balance_uv":
                    channel.balance_vu,

                "balance_vu":
                    channel.balance_uv
            }

            if estimated_liquidity_vu is not None:

                reverse_channel[
                    "estimated_liquidity"
                ] = estimated_liquidity_vu

            elif estimated_liquidity_uv is not None:

                # If only the forward estimate exists, do not
                # incorrectly reuse it for the reverse direction.

                reverse_channel[
                    "estimated_liquidity"
                ] = None

            G.add_edge(

                v,

                u,

                key=channel.channel_id + "-rev",

                **reverse_channel

            )

        return G

    # ==========================================================
    # Synthetic LN Generator
    # ==========================================================

    def synthetic(
        self,
        n=250,
        extra_edges=500,
        seed=42
    ):

        rng = np.random.default_rng(
            seed
        )

        countries = [

            "US",
            "CA",
            "DE",
            "FR",
            "GB",
            "JP",
            "AU",
            "BR",
            "SG",
            "IN"

        ]

        # ------------------------------------------------------
        # Create Scale-Free Topology
        # ------------------------------------------------------

        base_graph = nx.barabasi_albert_graph(

            n,

            max(
                2,
                min(
                    5,
                    n - 1
                )
            ),

            seed=seed

        )

        G = nx.MultiDiGraph()

        # ------------------------------------------------------
        # Add Bidirectional Channels
        # ------------------------------------------------------

        for u, v in base_graph.edges():

            G.add_edge(
                int(u),
                int(v)
            )

            G.add_edge(
                int(v),
                int(u)
            )

        # ------------------------------------------------------
        # Add Extra Random Connections
        # ------------------------------------------------------

        for _ in range(
            extra_edges
        ):

            u, v = rng.choice(

                n,

                2,

                replace=False

            )

            G.add_edge(
                int(u),
                int(v)
            )

            G.add_edge(
                int(v),
                int(u)
            )

        # ------------------------------------------------------
        # Add Node Attributes
        # ------------------------------------------------------

        for node_id in G.nodes:

            rgb_color = float(
                rng.uniform(
                    100,
                    700
                )
            )

            node = Node(

                node_id=str(
                    node_id
                ),

                country=str(
                    rng.choice(
                        countries
                    )
                ),

                latitude=float(
                    rng.uniform(
                        -55,
                        60
                    )
                ),

                longitude=float(
                    rng.uniform(
                        -170,
                        170
                    )
                ),

                carbon_intensity=rgb_color,

                online=True

            )

            G.nodes[
                node_id
            ].update(
                node.__dict__
            )

            # Synthetic data also follows the same semantic
            # meaning as the real data.

            G.nodes[
                node_id
            ][
                "rgb_color"
            ] = rgb_color

        # ------------------------------------------------------
        # Add Channel Attributes
        # ------------------------------------------------------

        for idx, (u, v, k) in enumerate(
            G.edges(
                keys=True
            )
        ):

            capacity = float(
                rng.lognormal(
                    np.log(300000),
                    0.8
                )
            )

            channel = Channel(

                channel_id=f"syn-{idx}",

                capacity=capacity,

                fee_base=float(
                    rng.integers(
                        500,
                        3000
                    )
                ),

                fee_rate=float(
                    rng.integers(
                        1,
                        100
                    )
                ),

                delay=float(
                    rng.uniform(
                        0.1,
                        3.0
                    )
                ),

                failure_probability=0.01,

                available=True,

                balance_uv=capacity / 2,

                balance_vu=capacity / 2,

                failure_count=0,

                success_count=0

            )

            G.edges[
                u,
                v,
                k
            ].update(
                channel.__dict__
            )

        return G

    # ==========================================================
    # Utility Functions
    # ==========================================================

    def number_of_nodes(self, G):

        return G.number_of_nodes()

    def number_of_channels(self, G):

        return G.number_of_edges()

    def summary(self, G):

        print(
            "Lightning Network Graph"
        )

        print(
            f"Nodes: {G.number_of_nodes()}"
        )

        print(
            f"Channels: {G.number_of_edges()}"
        )