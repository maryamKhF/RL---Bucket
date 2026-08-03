import hashlib
import json
import base64


class OnionRouter:
    """
    Simulated Onion Routing for RL + Bucket Lightning Network.

    This module:
    - Creates onion packet from selected route
    - Encrypts each hop layer
    - Allows nodes to peel their own layer
    - Validates transaction integrity
    """


    # -------------------------------------------------
    # Key generation
    # -------------------------------------------------

    def _generate_key(self, node):

        return hashlib.sha256(
            str(node).encode()
        ).digest()



    # -------------------------------------------------
    # Simulated encryption
    # -------------------------------------------------

    def _encrypt(self, data, key):

        raw = json.dumps(
            data,
            sort_keys=True
        ).encode()


        encrypted = bytes(

            [
                byte ^ key[i % len(key)]
                for i, byte in enumerate(raw)
            ]

        )


        return base64.b64encode(
            encrypted
        ).decode()



    def _decrypt(self, encrypted, key):

        raw = base64.b64decode(
            encrypted
        )


        decrypted = bytes(

            [
                byte ^ key[i % len(key)]
                for i, byte in enumerate(raw)
            ]

        )


        return json.loads(
            decrypted.decode()
        )



    # -------------------------------------------------
    # Build Onion Packet
    # -------------------------------------------------

    def build_onion(
            self,
            path,
            bucket_id,
            tx_id,
            amount,
            metadata=None
    ):

        """
        Creates onion packet.

        Parameters
        ----------
        path:
            Routing path from source to destination

        bucket_id:
            Selected bucket identifier

        tx_id:
            Transaction identifier

        amount:
            Payment amount

        metadata:
            Extra transaction information
        """


        if metadata is None:
            metadata = {}


        layers = None



        # Build layers backward

        for hop in range(
            len(path)-1,
            -1,
            -1
        ):


            node = path[hop]


            next_hop = (

                path[hop+1]
                if hop < len(path)-1
                else None

            )


            layer = {

                "hop": hop,

                "node": node,

                "next_hop": next_hop,

                "amount": amount,

                "bucket_id": bucket_id,

                "tx_id": tx_id,

                "metadata": metadata,

                "inner_layer": layers

            }


            key = self._generate_key(
                node
            )


            layers = {

                "node": node,

                "payload":
                    self._encrypt(
                        layer,
                        key
                    )

            }


        return {


            "bucket_id": bucket_id,

            "tx_id": tx_id,

            "layers": layers

        }



    # -------------------------------------------------
    # Peel Onion Layer
    # -------------------------------------------------

    def peel_layer(
            self,
            packet,
            node
    ):

        """
        Node removes its own onion layer.
        """


        if packet["layers"]["node"] != node:

            raise Exception(
                "Invalid onion layer"
            )


        key = self._generate_key(
            node
        )


        data = self._decrypt(

            packet["layers"]["payload"],

            key

        )


        return {


            "hop": data["hop"],

            "next_hop":
                data["next_hop"],

            "amount":
                data["amount"],

            "bucket_id":
                data["bucket_id"],

            "tx_id":
                data["tx_id"],

            "metadata":
                data["metadata"],

            "next_layer":
                data["inner_layer"]

        }



    # -------------------------------------------------
    # Layer verification
    # -------------------------------------------------

    def verify_layer(
            self,
            layer,
            bucket_id,
            tx_id
    ):


        return (

            layer["bucket_id"]
            == bucket_id

            and

            layer["tx_id"]
            == tx_id

        )



    # -------------------------------------------------
    # Route validation
    # -------------------------------------------------

    def validate_path(
            self,
            path
    ):


        if len(path) < 2:

            return False


        if len(set(path)) != len(path):

            return False


        return True