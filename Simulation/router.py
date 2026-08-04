# Simulation/router.py

"""
Routing execution layer for RL + Bucket.

This module connects:
    Pathfinding
        |
        RL Agent
        |
        Bucket
        |
        Onion Routing
        |
        Payment Simulation
"""


from Simulation.onion import OnionRouter



class Router:
    """
    Executes selected payment route.

    Responsibilities:
    - Validate path
    - Build onion packet
    - Forward packet hop by hop
    - Handle onion peeling
    - Return routing result
    """



    def __init__(self):

        self.onion = OnionRouter()



    # -------------------------------------------------
    # Route payment
    # -------------------------------------------------

    def route(
            self,
            path,
            bucket_id,
            tx_id,
            amount,
            metadata=None
    ):


        # -------------------------
        # Validate path
        # -------------------------

        if not self.onion.validate_path(path):

            return {

                "success": False,

                "reason":
                    "invalid_path"

            }



        # -------------------------
        # Create onion packet
        # -------------------------

        packet = self.onion.build_onion(

            path,

            bucket_id,

            tx_id,

            amount,

            metadata

        )



        current_packet = packet



        visited = []



        # -------------------------
        # Forward through nodes
        # -------------------------

        for node in path:


            try:

                layer = self.onion.peel_layer(

                    current_packet,

                    node

                )


            except Exception as e:


                return {

                    "success": False,

                    "failed_node": node,

                    "error": str(e),

                    "visited": visited

                }



            # verify integrity

            valid = self.onion.verify_layer(

                layer,

                bucket_id,

                tx_id

            )


            if not valid:

                return {

                    "success": False,

                    "failed_node": node,

                    "reason":
                        "integrity_check_failed"

                }



            visited.append(node)



            # -------------------------
            # Destination reached
            # -------------------------

            if layer["next_hop"] is None:


                return {

                    "success": True,

                    "tx_id": tx_id,

                    "amount": amount,

                    "path": visited

                }



            # create next packet layer

            current_packet = {

                "bucket_id":
                    bucket_id,

                "tx_id":
                    tx_id,

                "layers":
                    layer["next_layer"]

            }



        return {

            "success": False,

            "reason":
                "route_finished_without_destination"

        }



    # -------------------------------------------------
    # Simple test
    # -------------------------------------------------

    def test_route(self):

        path = [

            "Alice",

            "Bob",

            "Carol"

        ]


        return self.route(

            path,

            bucket_id="B001",

            tx_id="TX001",

            amount=1000

        )