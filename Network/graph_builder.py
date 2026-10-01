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
    -------------------------
    - rgb_color is preserved in its original representation.
    - RGB values are converted to a numeric carbon-intensity proxy
      using:

          0.299R + 0.587G + 0.114B

    - If rgb_color is missing or invalid, carbon_intensity is set
      to 0.0.

    - Channel capacity is NOT treated as directional liquidity.
    - If directional balance/liquidity is not provided by the input,
      it remains unknown (None).
    - Learned/observed liquidity can be added later by the routing
      and learning modules.
    """

    # ==========================================================
    # RGB / Carbon Utility
    # ==========================================================

    @staticmethod
    def _rgb_to_carbon_intensity(rgb_color):
        """
        Convert a supported RGB representation into the numeric
        carbon-intensity proxy used by the routing heuristics.

        Supported representations:

            "3399ff"
            "#3399ff"

            [51, 153, 255]
            (51, 153, 255)

            {"r": 51, "g": 153, "b": 255}
            {"red": 51, "green": 153, "blue": 255}

        Returns
        -------
        float or None
            Numeric RGB luminance/carbon proxy.

            None is returned when the RGB representation is
            absent or invalid.

        Important
        ---------
        The original rgb_color value is NOT modified.

        The graph keeps the original value separately as
        "rgb_color".
        """

        # ------------------------------------------------------
        # Missing RGB
        # ------------------------------------------------------

        if rgb_color is None:
            return None

        r = None
        g = None
        b = None

        # ------------------------------------------------------
        # Dictionary representation
        # ------------------------------------------------------

        if isinstance(rgb_color, dict):

            if all(
                key in rgb_color
                for key in ("r", "g", "b")
            ):
                r = rgb_color["r"]
                g = rgb_color["g"]
                b = rgb_color["b"]

            elif all(
                key in rgb_color
                for key in ("red", "green", "blue")
            ):
                r = rgb_color["red"]
                g = rgb_color["green"]
                b = rgb_color["blue"]

            else:
                return None

        # ------------------------------------------------------
        # List / Tuple representation
        # ------------------------------------------------------

        elif isinstance(
            rgb_color,
            (list, tuple)
        ):

            if len(rgb_color) != 3:
                return None

            r, g, b = rgb_color

        # ------------------------------------------------------
        # Hexadecimal string representation
        # ------------------------------------------------------

        elif isinstance(
            rgb_color,
            str
        ):

            value = rgb_color.strip()

            if value.startswith("#"):
                value = value[1:]

            if len(value) != 6:
                return None

            try:
                r = int(
                    value[0:2],
                    16
                )

                g = int(
                    value[2:4],
                    16
                )

                b = int(
                    value[4:6],
                    16
                )

            except ValueError:
                return None

        # ------------------------------------------------------
        # Unsupported representation
        # ------------------------------------------------------

        else:
            return None

        # ------------------------------------------------------
        # Validate RGB components
        # ------------------------------------------------------

        try:
            r = float(r)
            g = float(g)
            b = float(b)

        except (
            TypeError,
            ValueError
        ):
            return None

        # ------------------------------------------------------
        # Validate finite values
        # ------------------------------------------------------

        if not all(
            np.isfinite(value)
            for value in (r, g, b)
        ):
            return None

        # ------------------------------------------------------
        # Validate RGB range
        # ------------------------------------------------------

        if not all(
            0.0 <= value <= 255.0
            for value in (r, g, b)
        ):
            return None

        # ------------------------------------------------------
        # RGB luminance / carbon-intensity proxy
        #
        # Formula:
        #
        #     0.299R + 0.587G + 0.114B
        #
        # ------------------------------------------------------

        carbon_intensity = (
            0.299 * r
            +
            0.587 * g
            +
            0.114 * b
        )

        return float(
            carbon_intensity
        )

    # ==========================================================
    # Load Real LN Snapshot from JSON File
    # ==========================================================

    def from_json(self, path):
        """
        Load a Lightning Network snapshot from a JSON file.

        Parameters
        ----------
        path:
            Path to the JSON snapshot.

        Returns
        -------
        networkx.MultiDiGraph
        """

        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        return self.from_data(
            data
        )

    # ==========================================================
    # Load Real LN Snapshot from In-Memory Data
    # ==========================================================

    def from_data(self, data):
        """
        Build a MultiDiGraph from JSON-like data.

        Important:
        - RGB is preserved.
        - Missing RGB produces carbon_intensity=0.0.
        - Capacity is never interpreted as directional liquidity.
        - Unknown directional liquidity remains None.
        """

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

            nid = str(
                nid
            )

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
            # RGB / Carbon information
            #
            # The original RGB representation is preserved.
            #
            # Example:
            #
            #     "3399ff"
            #
            # is NOT converted to float and is NOT discarded.
            #
            # Instead:
            #
            #     R = 51
            #     G = 153
            #     B = 255
            #
            # and:
            #
            #     carbon =
            #         0.299R
            #       + 0.587G
            #       + 0.114B
            #
            # If RGB is missing or invalid:
            #
            #     carbon_intensity = 0.0
            #
            # --------------------------------------------------

            rgb_color = n.get(
                "rgb_color",
                None
            )

            rgb_carbon_intensity = (
                self._rgb_to_carbon_intensity(
                    rgb_color
                )
            )

            # --------------------------------------------------
            # Determine carbon intensity
            # --------------------------------------------------
            #
            # Priority:
            #
            # 1. Valid RGB
            # 2. Explicit carbon_intensity
            # 3. Zero
            #
            # The third case is important for real snapshot
            # nodes that contain no RGB information.
            #
            # --------------------------------------------------

            if rgb_carbon_intensity is not None:

                carbon_intensity = (
                    rgb_carbon_intensity
                )

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

                        if not np.isfinite(
                            carbon_intensity
                        ):
                            carbon_intensity = 0.0

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

            # --------------------------------------------------
            # Preserve original RGB representation.
            #
            # Examples:
            #
            #     "3399ff"
            #
            # or:
            #
            #     None
            #
            # for nodes where RGB is unavailable.
            # --------------------------------------------------

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

        for i, e in enumerate(
            channels
        ):

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
            #
            # capacity is NOT directional liquidity.
            #
            # No capacity / 2 is used here.
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
            # IMPORTANT:
            #
            # Do NOT use capacity / 2.
            #
            # The real snapshot may not contain directional
            # balances.
            #
            # In that case:
            #
            #     balance_uv = None
            #     balance_vu = None
            #
            # Unknown liquidity is kept unknown.
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
            # This is optional.
            #
            # It is NOT inferred from capacity.
            #
            # It can later be populated by routing experience.
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
        """
        Generate a synthetic Lightning Network topology.

        Synthetic topology behavior is kept separate from the
        real snapshot behavior.

        In particular, synthetic channels have known directional
        balances initialized to capacity / 2.
        """

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

            # --------------------------------------------------
            # Preserve synthetic RGB representation.
            # --------------------------------------------------

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

    def number_of_nodes(
        self,
        G
    ):
        """
        Return the number of nodes in the graph.
        """

        return G.number_of_nodes()

    def number_of_channels(
        self,
        G
    ):
        """
        Return the number of directed channels in the graph.
        """

        return G.number_of_edges()

    def summary(
        self,
        G
    ):
        """
        Print a concise graph summary.
        """

        print(
            "Lightning Network Graph"
        )

        print(
            f"Nodes: {G.number_of_nodes()}"
        )

        print(
            f"Channels: {G.number_of_edges()}"
        )