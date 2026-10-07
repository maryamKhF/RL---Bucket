# RL/state.py

import numpy as np
import networkx as nx

from Network.topology import (
    geo_features,
    channel_features
)


# ==========================================================
# State Configuration
# ==========================================================

# K:
# Maximum number of observable channels.
#
# M:
# Ego-Graph observation radius.
#
# The current implementation uses:
#
#     k = K
#     radius = M
#
# Therefore M is NOT passed to State as "m".
# ==========================================================

K = 15
M = 5


# ==========================================================
# Channel Features
# ==========================================================
#
# Fixed feature order.
#
# 0   fee_base
# 1   fee_rate
# 2   capacity
# 3   delay
# 4   failure_probability
# 5   availability
# 6   liquidity
# 7   geographic_distance
# 8   inter_country
# 9   inter_continent
# 10  carbon_intensity
#
# Therefore:
#
#     D = 11
#
# State matrix:
#
#     S(u,v) ∈ R^(K × D)
#
# Complete observation:
#
#     [flatten(S), mask]
#
# Dimension:
#
#     K × D + K
#
#     15 × 11 + 15 = 180
# ==========================================================

FEATURE_NAMES = (
    "fee_base",
    "fee_rate",
    "capacity",
    "delay",
    "failure_probability",
    "availability",
    "liquidity",
    "geographic_distance",
    "inter_country",
    "inter_continent",
    "carbon_intensity"
)

FEATURE_DIM = len(
    FEATURE_NAMES
)


# ==========================================================
# Feature Groups
# ==========================================================

BINARY_FEATURES = {
    "availability",
    "inter_country",
    "inter_continent"
}


LOG_FEATURES = {
    "fee_base",
    "fee_rate",
    "capacity",
    "delay",
    "geographic_distance",
    "carbon_intensity"
}


# ==========================================================
# State Class
# ==========================================================

class State:

    def __init__(
        self,
        G,
        source,
        destination,
        transaction=None,
        candidate_paths=None,
        simulation_info=None,
        bucket_info=None,
        k=K,
        radius=M,
        normalization_stats=None
    ):

        self.G = G

        self.source = source

        self.destination = destination

        self.transaction = (
            transaction
            if transaction is not None
            else {}
        )

        self.candidate_paths = (
            candidate_paths
            if candidate_paths is not None
            else []
        )

        self.simulation_info = (
            simulation_info
            if simulation_info is not None
            else {}
        )

        self.bucket_info = (
            bucket_info
            if bucket_info is not None
            else {}
        )

        self.k = max(
            int(k),
            1
        )

        self.radius = max(
            int(radius),
            0
        )

        self.normalization_stats = (
            normalization_stats
            if normalization_stats is not None
            else {}
        )

    # ======================================================
    # Complete State Vector
    # ======================================================

    def vector(self):

        state_matrix, mask = (
            self.state_matrix()
        )

        vector = np.concatenate(
            [
                state_matrix.flatten(),
                mask
            ]
        )

        return np.asarray(
            vector,
            dtype=np.float32
        )

    # ======================================================
    # State Matrix
    # ======================================================
    #
    # Returns:
    #
    #     matrix: K × 11
    #     mask:   K
    #
    # Empty rows are zero-padded.
    # Mask = 1 -> valid channel
    # Mask = 0 -> padding
    # ======================================================

    def state_matrix(self):

        channels = (
            self._observable_channels()
        )

        matrix = np.zeros(
            (
                self.k,
                FEATURE_DIM
            ),
            dtype=np.float32
        )

        mask = np.zeros(
            self.k,
            dtype=np.float32
        )

        for i, channel in enumerate(
            channels[:self.k]
        ):

            matrix[i] = (
                self._channel_vector(
                    channel
                )
            )

            mask[i] = 1.0

        return matrix, mask

    # ======================================================
    # Observable Channels
    # ======================================================
    #
    # Observation is constructed from the local ego-graphs
    # around source and destination.
    #
    # The observation does NOT perform route selection.
    # PPO receives these features and learns eta.
    # ======================================================

    def _observable_channels(self):

        if self.G is None:

            return []

        source_exists = (
            self.source is not None
            and
            self.G.has_node(
                self.source
            )
        )

        destination_exists = (
            self.destination is not None
            and
            self.G.has_node(
                self.destination
            )
        )

        if not source_exists and not destination_exists:

            return []

        # --------------------------------------------------
        # Build local ego-graph node set
        # --------------------------------------------------

        nodes = set()

        if source_exists:

            source_nodes = (
                nx.single_source_shortest_path_length(
                    self.G,
                    self.source,
                    cutoff=self.radius
                )
            )

            nodes.update(
                source_nodes.keys()
            )

        if destination_exists:

            destination_nodes = (
                nx.single_source_shortest_path_length(
                    self.G,
                    self.destination,
                    cutoff=self.radius
                )
            )

            nodes.update(
                destination_nodes.keys()
            )

        # --------------------------------------------------
        # Precompute terminal distances.
        #
        # This is much faster than calling
        # shortest_path_length() for every channel.
        # --------------------------------------------------

        source_distances = {}

        destination_distances = {}

        if source_exists:

            source_distances = (
                nx.single_source_shortest_path_length(
                    self.G,
                    self.source
                )
            )

        if destination_exists:

            destination_distances = (
                nx.single_source_shortest_path_length(
                    self.G,
                    self.destination
                )
            )

        # --------------------------------------------------
        # Collect channels
        # --------------------------------------------------

        channels = []

        for node in nodes:

            if not self.G.has_node(
                node
            ):
                continue

            if self.G.is_multigraph():

                if self.G.is_directed():

                    edges = list(
                        self.G.in_edges(
                            node,
                            keys=True,
                            data=True
                        )
                    )

                    edges += list(
                        self.G.out_edges(
                            node,
                            keys=True,
                            data=True
                        )
                    )

                else:

                    edges = list(
                        self.G.edges(
                            node,
                            keys=True,
                            data=True
                        )
                    )

            else:

                if self.G.is_directed():

                    edges = []

                    edges.extend(
                        [
                            (
                                u,
                                v,
                                None,
                                data
                            )
                            for (
                                u,
                                v,
                                data
                            )
                            in self.G.in_edges(
                                node,
                                data=True
                            )
                        ]
                    )

                    edges.extend(
                        [
                            (
                                u,
                                v,
                                None,
                                data
                            )
                            for (
                                u,
                                v,
                                data
                            )
                            in self.G.out_edges(
                                node,
                                data=True
                            )
                        ]
                    )

                else:

                    edges = [
                        (
                            u,
                            v,
                            None,
                            data
                        )
                        for (
                            u,
                            v,
                            data
                        )
                        in self.G.edges(
                            node,
                            data=True
                        )
                    ]

            for edge in edges:

                u, v, key, data = edge

                channels.append(
                    (
                        u,
                        v,
                        key,
                        data
                    )
                )

        # --------------------------------------------------
        # Remove exact duplicate edge records
        # --------------------------------------------------

        unique = {}

        for (
            u,
            v,
            key,
            data
        ) in channels:

            identifier = (
                u,
                v,
                key
            )

            unique[
                identifier
            ] = (
                u,
                v,
                key,
                data
            )

        channels = list(
            unique.values()
        )

        # --------------------------------------------------
        # Rank observable channels
        # --------------------------------------------------
        #
        # Ranking is ONLY for deterministic observation
        # ordering.
        #
        # It does not select the payment route.
        # PPO still determines eta.
        # --------------------------------------------------

        ranked = []

        for channel in channels:

            u, v, key, data = channel

            distance = (
                self._distance_to_terminals(
                    u,
                    v,
                    source_distances,
                    destination_distances
                )
            )

            capacity = self._safe_float(
                data.get(
                    "capacity",
                    0.0
                )
            )

            ranked.append(
                (
                    distance,
                    -capacity,
                    str(u),
                    str(v),
                    str(key),
                    channel
                )
            )

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
                item[4]
            )
        )

        return [
            item[-1]
            for item in ranked
        ]

    # ======================================================
    # Distance to Source / Destination
    # ======================================================

    def _distance_to_terminals(
        self,
        u,
        v,
        source_distances=None,
        destination_distances=None
    ):

        if source_distances is None:

            source_distances = {}

            if (
                self.source is not None
                and
                self.G.has_node(
                    self.source
                )
            ):

                try:

                    source_distances = (
                        nx.single_source_shortest_path_length(
                            self.G,
                            self.source
                        )
                    )

                except Exception:

                    source_distances = {}

        if destination_distances is None:

            destination_distances = {}

            if (
                self.destination is not None
                and
                self.G.has_node(
                    self.destination
                )
            ):

                try:

                    destination_distances = (
                        nx.single_source_shortest_path_length(
                            self.G,
                            self.destination
                        )
                    )

                except Exception:

                    destination_distances = {}

        distances = []

        for node in (
            u,
            v
        ):

            d_source = (
                source_distances.get(
                    node,
                    999
                )
            )

            d_destination = (
                destination_distances.get(
                    node,
                    999
                )
            )

            distances.append(
                min(
                    d_source,
                    d_destination
                )
            )

        if not distances:

            return 999

        return min(
            distances
        )

    # ======================================================
    # Channel Feature Vector
    # ======================================================

    def _channel_vector(
        self,
        channel
    ):

        u, v, key, data = channel

        if data is None:

            data = {}

        # --------------------------------------------------
        # Channel features
        # --------------------------------------------------

        try:

            extracted = (
                channel_features(
                    self.G,
                    u,
                    v,
                    key
                )
            )

            if extracted is None:

                extracted = {}

        except Exception:

            extracted = {}

        # --------------------------------------------------
        # Basic channel features
        # --------------------------------------------------

        fee_base = self._safe_float(
            extracted.get(
                "fee_base",
                data.get(
                    "fee_base",
                    0.0
                )
            )
        )

        fee_rate = self._safe_float(
            extracted.get(
                "fee_rate",
                data.get(
                    "fee_rate",
                    data.get(
                        "fee_proportional_millionths",
                        0.0
                    )
                )
            )
        )

        capacity = self._safe_float(
            extracted.get(
                "capacity",
                data.get(
                    "capacity",
                    0.0
                )
            )
        )

        delay = self._safe_float(
            extracted.get(
                "delay",
                data.get(
                    "delay",
                    0.0
                )
            )
        )

        failure_probability = self._safe_float(
            extracted.get(
                "failure_probability",
                data.get(
                    "failure_probability",
                    0.0
                )
            )
        )

        availability = self._safe_float(
            extracted.get(
                "available",
                data.get(
                    "available",
                    data.get(
                        "active",
                        1.0
                    )
                )
            )
        )

        # --------------------------------------------------
        # Liquidity
        # --------------------------------------------------

        if (
            "liquidity_uv"
            in extracted
        ):

            liquidity = self._safe_float(
                extracted.get(
                    "liquidity_uv"
                )
            )

        elif (
            "liquidity"
            in data
        ):

            liquidity = self._safe_float(
                data.get(
                    "liquidity"
                )
            )

        elif (
            "balance_uv"
            in data
            and
            capacity > 0
        ):

            liquidity = (
                self._safe_float(
                    data.get(
                        "balance_uv"
                    )
                )
                /
                capacity
            )

        else:

            liquidity = 0.0

        # --------------------------------------------------
        # Geographic features
        # --------------------------------------------------

        distance = 0.0

        inter_country = 0.0

        inter_continent = 0.0

        carbon = 0.0

        try:

            geo = geo_features(
                self.G,
                u,
                v
            )

            if geo is None:

                geo = {}

            distance = self._safe_float(
                geo.get(
                    "distance_km",
                    0.0
                )
            )

            inter_country = self._safe_float(
                geo.get(
                    "inter_country",
                    0.0
                )
            )

            inter_continent = self._safe_float(
                geo.get(
                    "inter_continent",
                    0.0
                )
            )

            carbon = self._safe_float(
                geo.get(
                    "carbon_intensity",
                    0.0
                )
            )

        except Exception:

            distance = 0.0
            inter_country = 0.0
            inter_continent = 0.0
            carbon = 0.0

        # --------------------------------------------------
        # Raw feature dictionary
        # --------------------------------------------------

        raw_features = {

            "fee_base":
                fee_base,

            "fee_rate":
                fee_rate,

            "capacity":
                capacity,

            "delay":
                delay,

            "failure_probability":
                failure_probability,

            "availability":
                availability,

            "liquidity":
                liquidity,

            "geographic_distance":
                distance,

            "inter_country":
                inter_country,

            "inter_continent":
                inter_continent,

            "carbon_intensity":
                carbon
        }

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        normalized = []

        for name in FEATURE_NAMES:

            value = raw_features[
                name
            ]

            normalized_value = (
                self._normalize_feature(
                    name,
                    value
                )
            )

            normalized.append(
                normalized_value
            )

        return np.asarray(
            normalized,
            dtype=np.float32
        )

    # ======================================================
    # Feature Normalization
    # ======================================================

    def _normalize_feature(
        self,
        name,
        value
    ):

        value = self._safe_float(
            value
        )

        # --------------------------------------------------
        # Binary features
        # --------------------------------------------------

        if name in BINARY_FEATURES:

            return float(
                np.clip(
                    value,
                    0.0,
                    1.0
                )
            )

        # --------------------------------------------------
        # Log transformation
        # --------------------------------------------------

        if name in LOG_FEATURES:

            value = np.log1p(
                max(
                    value,
                    0.0
                )
            )

        # --------------------------------------------------
        # Standardization
        # --------------------------------------------------

        stats = self.normalization_stats.get(
            name
        )

        if stats is not None:

            mean = self._safe_float(
                stats.get(
                    "mean",
                    0.0
                )
            )

            std = self._safe_float(
                stats.get(
                    "std",
                    1.0
                )
            )

            if std > 1e-8:

                value = (
                    value - mean
                ) / std

        # --------------------------------------------------
        # Safety clipping
        # --------------------------------------------------

        value = np.clip(
            value,
            -10.0,
            10.0
        )

        return float(
            value
        )

    # ======================================================
    # Safe Numeric Conversion
    # ======================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0
    ):

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return float(
                default
            )

        if not np.isfinite(
            value
        ):

            return float(
                default
            )

        return value

    # ======================================================
    # Dimension
    # ======================================================

    @property
    def dimension(self):

        return (
            self.k * FEATURE_DIM
            +
            self.k
        )

    # ======================================================
    # Matrix Dimension
    # ======================================================

    @property
    def matrix_dimension(self):

        return (
            self.k,
            FEATURE_DIM
        )

    # ======================================================
    # Mask
    # ======================================================

    def mask(self):

        _, mask = (
            self.state_matrix()
        )

        return mask

    # ======================================================
    # Human-readable Feature Description
    # ======================================================

    @staticmethod
    def feature_names():

        return FEATURE_NAMES


# ==========================================================
# Normalization Statistics Helper
# ==========================================================

def fit_normalization_stats(
    graphs,
    max_samples=None
):
    """
    Estimate normalization parameters.

    Statistics should be calculated only from training data
    in the final experiment.

    Parameters
    ----------
    graphs:
        Iterable of NetworkX graphs.

    max_samples:
        Optional maximum number of channels.

    Returns
    -------
    dict
        {
            feature_name: {
                "mean": ...,
                "std": ...
            }
        }
    """

    values = {

        name: []

        for name in FEATURE_NAMES

        if name not in BINARY_FEATURES
    }

    sample_count = 0

    # ======================================================
    # Graph loop
    # ======================================================

    for G in graphs:

        if G is None:

            continue

        # --------------------------------------------------
        # Obtain edges
        # --------------------------------------------------

        if G.is_multigraph():

            edges = G.edges(
                keys=True,
                data=True
            )

        else:

            edges = (

                (
                    u,
                    v,
                    None,
                    data
                )

                for (
                    u,
                    v,
                    data
                )
                in G.edges(
                    data=True
                )
            )

        # --------------------------------------------------
        # Edge loop
        # --------------------------------------------------

        for (
            u,
            v,
            key,
            data
        ) in edges:

            if data is None:

                data = {}

            # --------------------------------------------------
            # Channel features
            # --------------------------------------------------

            try:

                cf = channel_features(
                    G,
                    u,
                    v,
                    key
                )

                if cf is None:

                    cf = {}

            except Exception:

                cf = {}

            # --------------------------------------------------
            # Geographic features
            # --------------------------------------------------

            try:

                geo = geo_features(
                    G,
                    u,
                    v
                )

                if geo is None:

                    geo = {}

            except Exception:

                geo = {}

            # --------------------------------------------------
            # Basic values
            # --------------------------------------------------

            fee_base = State._safe_float(
                cf.get(
                    "fee_base",
                    data.get(
                        "fee_base",
                        0.0
                    )
                )
            )

            fee_rate = State._safe_float(
                cf.get(
                    "fee_rate",
                    data.get(
                        "fee_rate",
                        data.get(
                            "fee_proportional_millionths",
                            0.0
                        )
                    )
                )
            )

            capacity = State._safe_float(
                cf.get(
                    "capacity",
                    data.get(
                        "capacity",
                        0.0
                    )
                )
            )

            delay = State._safe_float(
                cf.get(
                    "delay",
                    data.get(
                        "delay",
                        0.0
                    )
                )
            )

            failure_probability = State._safe_float(
                cf.get(
                    "failure_probability",
                    data.get(
                        "failure_probability",
                        0.0
                    )
                )
            )

            if "liquidity_uv" in cf:

                liquidity = State._safe_float(
                    cf.get(
                        "liquidity_uv"
                    )
                )

            elif "liquidity" in data:

                liquidity = State._safe_float(
                    data.get(
                        "liquidity"
                    )
                )

            elif (
                "balance_uv" in data
                and
                capacity > 0
            ):

                liquidity = (
                    State._safe_float(
                        data.get(
                            "balance_uv"
                        )
                    )
                    /
                    capacity
                )

            else:

                liquidity = 0.0

            distance = State._safe_float(
                geo.get(
                    "distance_km",
                    0.0
                )
            )

            carbon = State._safe_float(
                geo.get(
                    "carbon_intensity",
                    0.0
                )
            )

            # --------------------------------------------------
            # Raw normalization values
            # --------------------------------------------------

            raw = {

                "fee_base":
                    fee_base,

                "fee_rate":
                    fee_rate,

                "capacity":
                    capacity,

                "delay":
                    delay,

                "failure_probability":
                    failure_probability,

                "liquidity":
                    liquidity,

                "geographic_distance":
                    distance,

                "carbon_intensity":
                    carbon
            }

            # --------------------------------------------------
            # Collect values
            # --------------------------------------------------

            for name in values:

                value = raw[
                    name
                ]

                if name in LOG_FEATURES:

                    value = np.log1p(
                        max(
                            value,
                            0.0
                        )
                    )

                values[
                    name
                ].append(
                    value
                )

            sample_count += 1

            if (
                max_samples is not None
                and
                sample_count >= max_samples
            ):

                break

        if (
            max_samples is not None
            and
            sample_count >= max_samples
        ):

            break

    # ======================================================
    # Calculate statistics
    # ======================================================

    stats = {}

    for name, data in values.items():

        if not data:

            stats[name] = {

                "mean":
                    0.0,

                "std":
                    1.0
            }

            continue

        arr = np.asarray(
            data,
            dtype=np.float64
        )

        mean = float(
            np.mean(
                arr
            )
        )

        std = float(
            np.std(
                arr
            )
        )

        if (
            not np.isfinite(std)
            or
            std < 1e-8
        ):

            std = 1.0

        stats[name] = {

            "mean":
                mean,

            "std":
                std
        }

    return stats


# ==========================================================
# External Interface
# ==========================================================

def neighborhood_state(
    G,
    source,
    destination,
    k=K,
    radius=M,
    transaction=None,
    candidate_paths=None,
    simulation_info=None,
    bucket_info=None,
    normalization_stats=None
):
    """
    Public interface for generating the RL observation.

    Pipeline:

        Local Ego-Graph
              ↓
        K observable channels
              ↓
        K × D State Matrix
              ↓
        Padding + Mask
              ↓
        PPO

    Returns
    -------
    numpy.ndarray
        Flattened state matrix followed by mask.
    """

    state = State(

        G,

        source,

        destination,

        transaction=transaction,

        candidate_paths=candidate_paths,

        simulation_info=simulation_info,

        bucket_info=bucket_info,

        k=k,

        radius=radius,

        normalization_stats=normalization_stats
    )

    return state.vector()
