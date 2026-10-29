"""
onion.py

Simulated Onion Routing for the Lightning Network simulation.

Responsibilities
----------------
1. Build a simulated onion packet after route selection.
2. Create one encrypted layer for each hop.
3. Allow each node to peel only its own layer.
4. Preserve transaction / attempt / Bucket identifiers.
5. Validate the selected route.
6. Support rebuilding an onion for an alternative route/suffix.
7. Preserve original hop indices when an onion is rebuilt.

Separation of responsibilities
-------------------------------
Router:
    Executes the already-selected route.

Bucket:
    Stores alternative candidate routes.

PartialBacktracker:
    Selects an alternative suffix after failure.

FailureModel:
    Detects payment failures.

OnionRouter:
    Builds and processes the simulated onion packet.

Important
---------
This module is a simulation abstraction.

It is NOT an implementation of the cryptographic onion
construction used by production Lightning Network
implementations such as BOLT 4 / Sphinx.
"""

import base64
import hashlib
import json


class OnionRouter:
    """
    Simulated Onion Router.

    The implementation models the logical behavior of
    per-hop onion layers without claiming production-level
    Lightning cryptographic security.
    """

    # ========================================================
    # Key Generation
    # ========================================================

    def _generate_key(self, node):
        """
        Generate a deterministic simulation key for a node.

        This is only a simulation abstraction.
        """

        return hashlib.sha256(
            str(node).encode("utf-8")
        ).digest()

    # ========================================================
    # Simulated Encryption
    # ========================================================

    def _encrypt(self, data, key):
        """
        Simulated XOR + Base64 encoding.

        This is NOT secure cryptography and is used only
        to model the concept of per-hop encrypted layers.
        """

        raw = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        encrypted = bytes(
            byte ^ key[i % len(key)]
            for i, byte in enumerate(raw)
        )

        return base64.b64encode(
            encrypted
        ).decode("ascii")

    # ========================================================
    # Simulated Decryption
    # ========================================================

    def _decrypt(self, encrypted, key):
        """
        Decrypt a simulated onion layer.
        """

        try:
            raw = base64.b64decode(
                encrypted
            )

            decrypted = bytes(
                byte ^ key[i % len(key)]
                for i, byte in enumerate(raw)
            )

            return json.loads(
                decrypted.decode("utf-8")
            )

        except (
            ValueError,
            TypeError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:

            raise ValueError(
                "Invalid or corrupted onion layer."
            ) from exc

    # ========================================================
    # Build Onion
    # ========================================================

    def build_onion(
        self,
        path,
        bucket_id,
        tx_id,
        amount,
        metadata=None,
        attempt_id=0,
        route_id=None,
        hop_offset=0,
    ):
        """
        Build a simulated onion packet for a selected route.

        Parameters
        ----------
        path : list
            Ordered node path.

        bucket_id : int / str
            Identifier of the Bucket entry associated with
            the selected route.

        tx_id : int / str
            Payment/transaction identifier.

        amount : float
            Payment amount.

        metadata : dict, optional
            Additional simulation metadata.

        attempt_id : int
            Current payment attempt.

        route_id : int / str, optional
            Identifier for the selected route.

        hop_offset : int
            Original hop index of path[0].

            This is important when rebuilding an onion for
            an alternative suffix after Partial Backtracking.

            Example:

                Original route:
                    A -> B -> C -> D

                Failure:
                    B -> C

                Alternative suffix:
                    B -> E -> D

                Rebuilt onion:
                    B -> E -> D

            The hop indices remain associated with the
            original route position instead of restarting
            incorrectly from zero.
        """

        self._validate_build_arguments(
            path,
            tx_id,
            amount,
            hop_offset,
        )

        if metadata is None:
            metadata = {}

        metadata = dict(metadata)

        if route_id is None:
            route_id = self._make_route_id(path)

        layers = None

        # ----------------------------------------------------
        # Build from destination to source
        # ----------------------------------------------------

        for local_index in range(
            len(path) - 1,
            -1,
            -1,
        ):

            node = path[local_index]

            next_hop = (
                path[local_index + 1]
                if local_index < len(path) - 1
                else None
            )

            hop_index = (
                hop_offset + local_index
            )

            layer = {
                "hop_index": hop_index,
                "node": node,
                "next_hop": next_hop,
                "amount": amount,
                "bucket_id": bucket_id,
                "tx_id": tx_id,
                "attempt_id": attempt_id,
                "route_id": route_id,
                "metadata": metadata,
                "inner_layer": layers,
            }

            key = self._generate_key(node)

            layers = {
                "node": node,
                "payload": self._encrypt(
                    layer,
                    key,
                ),
            }

        return {
            "version": 1,
            "bucket_id": bucket_id,
            "tx_id": tx_id,
            "attempt_id": attempt_id,
            "route_id": route_id,
            "source": path[0],
            "destination": path[-1],
            "path_length": len(path),
            "hop_offset": hop_offset,
            "layers": layers,
        }

    # ========================================================
    # Rebuild Onion
    # ========================================================

    def rebuild_onion(
        self,
        path,
        bucket_id,
        tx_id,
        amount,
        metadata=None,
        attempt_id=0,
        route_id=None,
        hop_offset=0,
    ):
        """
        Build a new onion packet for a new route or
        alternative suffix.

        hop_offset preserves the original route position.

        For a completely new route:

            hop_offset = 0

        For an alternative suffix beginning at original
        hop index 2:

            hop_offset = 2
        """

        return self.build_onion(
            path=path,
            bucket_id=bucket_id,
            tx_id=tx_id,
            amount=amount,
            metadata=metadata,
            attempt_id=attempt_id,
            route_id=route_id,
            hop_offset=hop_offset,
        )

    # ========================================================
    # Peel Layer
    # ========================================================

    def peel_layer(self, packet, node):
        """
        Remove and decrypt the current node's onion layer.

        The node can process only its own layer.

        Returns
        -------
        dict
            Information required for the next forwarding step.
        """

        self._validate_packet(packet)

        current_layer = packet.get(
            "layers"
        )

        if current_layer is None:
            raise ValueError(
                "Onion packet contains no layers."
            )

        if current_layer.get("node") != node:
            raise ValueError(
                "Node cannot peel this onion layer."
            )

        key = self._generate_key(node)

        data = self._decrypt(
            current_layer["payload"],
            key,
        )

        # ----------------------------------------------------
        # Validate decrypted layer
        # ----------------------------------------------------

        if data.get("node") != node:
            raise ValueError(
                "Onion layer node mismatch."
            )

        if data.get("tx_id") != packet["tx_id"]:
            raise ValueError(
                "Onion layer transaction mismatch."
            )

        if data.get("bucket_id") != packet["bucket_id"]:
            raise ValueError(
                "Onion layer Bucket mismatch."
            )

        if data.get("attempt_id") != packet["attempt_id"]:
            raise ValueError(
                "Onion layer attempt mismatch."
            )

        if data.get("route_id") != packet["route_id"]:
            raise ValueError(
                "Onion layer route mismatch."
            )

        return {
            "hop_index": data.get(
                "hop_index"
            ),

            "node": data.get(
                "node"
            ),

            "next_hop": data.get(
                "next_hop"
            ),

            "amount": data.get(
                "amount"
            ),

            "bucket_id": data.get(
                "bucket_id"
            ),

            "tx_id": data.get(
                "tx_id"
            ),

            "attempt_id": data.get(
                "attempt_id"
            ),

            "route_id": data.get(
                "route_id"
            ),

            "metadata": data.get(
                "metadata",
                {},
            ),

            "next_layer": data.get(
                "inner_layer"
            ),
        }

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, packet, node):
        """
        Process the current onion layer.

        Returns a forwarding state containing a valid
        next_packet when another layer exists.

        The destination produces next_packet=None.
        """

        layer = self.peel_layer(
            packet,
            node,
        )

        next_layer = layer[
            "next_layer"
        ]

        next_packet = None

        # ----------------------------------------------------
        # Build the packet for the next hop
        # ----------------------------------------------------

        if next_layer is not None:

            next_packet = {
                "version": packet["version"],
                "bucket_id": packet["bucket_id"],
                "tx_id": packet["tx_id"],
                "attempt_id": packet["attempt_id"],
                "route_id": packet["route_id"],
                "source": packet["source"],
                "destination": packet["destination"],
                "path_length": packet["path_length"],
                "hop_offset": packet.get(
                    "hop_offset",
                    0,
                ),
                "layers": next_layer,
            }

        return {
            "current_node": node,

            "next_hop": layer[
                "next_hop"
            ],

            "hop_index": layer[
                "hop_index"
            ],

            "amount": layer[
                "amount"
            ],

            "bucket_id": layer[
                "bucket_id"
            ],

            "tx_id": layer[
                "tx_id"
            ],

            "attempt_id": layer[
                "attempt_id"
            ],

            "route_id": layer[
                "route_id"
            ],

            "metadata": layer[
                "metadata"
            ],

            "next_layer": next_layer,

            "next_packet": next_packet,

            "destination_reached": (
                layer["next_hop"] is None
            ),
        }

    # ========================================================
    # Layer Verification
    # ========================================================

    def verify_layer(
        self,
        layer,
        bucket_id,
        tx_id,
        attempt_id=None,
        route_id=None,
    ):
        """
        Verify that a decrypted onion layer belongs to the
        expected payment attempt and selected route.
        """

        if layer is None:
            return False

        if layer.get(
            "bucket_id"
        ) != bucket_id:
            return False

        if layer.get(
            "tx_id"
        ) != tx_id:
            return False

        if (
            attempt_id is not None
            and layer.get(
                "attempt_id"
            ) != attempt_id
        ):
            return False

        if (
            route_id is not None
            and layer.get(
                "route_id"
            ) != route_id
        ):
            return False

        return True

    # ========================================================
    # Route Validation
    # ========================================================

    def validate_path(self, path):
        """
        Validate a selected route.

        Conditions
        ----------
        1. Path must be a list/tuple.
        2. At least source and destination must exist.
        3. Nodes must be unique.
        4. Source and destination must be different.
        """

        if not isinstance(
            path,
            (list, tuple),
        ):
            return False

        if len(path) < 2:
            return False

        if path[0] == path[-1]:
            return False

        if len(set(path)) != len(path):
            return False

        return True

    # ========================================================
    # Packet Validation
    # ========================================================

    def validate_packet(self, packet):
        """
        Validate the basic structure of an onion packet.
        """

        try:
            self._validate_packet(
                packet
            )

            return True

        except ValueError:
            return False

    # ========================================================
    # Internal Packet Validation
    # ========================================================

    def _validate_packet(self, packet):
        """
        Internal strict packet validation.
        """

        if not isinstance(
            packet,
            dict,
        ):
            raise ValueError(
                "Onion packet must be a dictionary."
            )

        required_fields = (
            "version",
            "bucket_id",
            "tx_id",
            "attempt_id",
            "route_id",
            "source",
            "destination",
            "path_length",
            "layers",
        )

        for field in required_fields:

            if field not in packet:
                raise ValueError(
                    f"Missing onion field: {field}"
                )

        if packet["layers"] is None:
            raise ValueError(
                "Onion packet has no layers."
            )

        if packet["path_length"] < 2:
            raise ValueError(
                "Invalid onion path length."
            )

        if packet["source"] == packet["destination"]:
            raise ValueError(
                "Onion source and destination cannot be identical."
            )

        if not isinstance(
            packet["layers"],
            dict,
        ):
            raise ValueError(
                "Invalid onion layers structure."
            )

        if "node" not in packet["layers"]:
            raise ValueError(
                "Onion layer has no node."
            )

        if "payload" not in packet["layers"]:
            raise ValueError(
                "Onion layer has no payload."
            )

    # ========================================================
    # Build Argument Validation
    # ========================================================

    def _validate_build_arguments(
        self,
        path,
        tx_id,
        amount,
        hop_offset,
    ):
        """
        Validate arguments before constructing the packet.
        """

        if not self.validate_path(
            path
        ):
            raise ValueError(
                "Invalid routing path."
            )

        if tx_id is None:
            raise ValueError(
                "tx_id cannot be None."
            )

        if amount <= 0:
            raise ValueError(
                "amount must be positive."
            )

        if not isinstance(
            hop_offset,
            int,
        ):
            raise ValueError(
                "hop_offset must be an integer."
            )

        if hop_offset < 0:
            raise ValueError(
                "hop_offset cannot be negative."
            )

    # ========================================================
    # Route Identifier
    # ========================================================

    @staticmethod
    def _make_route_id(path):
        """
        Generate a deterministic route identifier.

        The identifier is not a security mechanism.
        """

        serialized = json.dumps(
            list(path),
            separators=(",", ":"),
            default=str,
        )

        return hashlib.sha256(
            serialized.encode(
                "utf-8"
            )
        ).hexdigest()[:16]