import json
from pathlib import Path

import networkx as nx
import numpy as np

from .node import Node
from .channel import Channel



class LNGraphBuilder:
    """
    Builds a NetworkX MultiDiGraph
    from:
        1. Real Lightning Network JSON snapshot
        2. Synthetic Lightning topology
    """


    # =====================================================
    # Load Real LN Snapshot
    # =====================================================

    def from_json(self, path):

        data = json.loads(
            Path(path)
            .read_text(
                encoding="utf-8"
            )
        )


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


        # -----------------------------
        # Add Nodes
        # -----------------------------

        for n in nodes:


            nid = str(
                n.get(
                    "node_id",
                    n.get("id")
                )
            )


            if not nid:
                continue



            node = Node(

                node_id=nid,

                country=str(
                    n.get(
                        "country",
                        "US"
                    )
                ),

                latitude=float(
                    n.get(
                        "latitude",
                        n.get("lat",0.0)
                    )
                ),

                longitude=float(
                    n.get(
                        "longitude",
                        n.get("lon",0.0)
                    )
                ),

                carbon_intensity=float(
                    n.get(
                        "carbon_intensity",
                        300.0
                    )
                ),

                online=bool(
                    n.get(
                        "online",
                        True
                    )
                )
            )


            G.add_node(
                nid,
                **node.__dict__
            )



        # -----------------------------
        # Add Channels
        # -----------------------------

        for i,e in enumerate(channels):


            u = str(
                e["source"]
            )

            v = str(
                e["target"]
            )


            if (
                u not in G
                or
                v not in G
            ):
                continue



            capacity = float(
                e.get(
                    "capacity",
                    1_000_000
                )
            )



            channel = Channel(

                channel_id=str(
                    e.get(
                        "channel_id",
                        f"ch-{i}"
                    )
                ),

                capacity=capacity,

                fee_base=float(
                    e.get(
                        "fee_base",
                        1000
                    )
                ),

                fee_rate=float(
                    e.get(
                        "fee_rate",
                        10
                    )
                ),

                delay=float(
                    e.get(
                        "delay",
                        1.0
                    )
                ),

                failure_probability=float(
                    e.get(
                        "failure_probability",
                        0.01
                    )
                ),

                available=bool(
                    e.get(
                        "available",
                        True
                    )
                ),

                balance_uv=float(
                    e.get(
                        "balance_uv",
                        capacity/2
                    )
                ),

                balance_vu=float(
                    e.get(
                        "balance_vu",
                        capacity/2
                    )
                )
            )


            # u -> v

            G.add_edge(

                u,

                v,

                key=channel.channel_id,

                **channel.__dict__

            )


            # v -> u

            reverse_channel = {

                **channel.__dict__,

                "channel_id":
                    channel.channel_id + "-rev",

                "balance_uv":
                    channel.balance_vu,

                "balance_vu":
                    channel.balance_uv

            }


            G.add_edge(

                v,

                u,

                key=channel.channel_id + "-rev",

                **reverse_channel

            )


        return G



    # =====================================================
    # Synthetic LN Generator
    # =====================================================


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



        # ---------------------------------
        # Create Scale-Free topology
        # ---------------------------------

        base_graph = nx.barabasi_albert_graph(

            n,

            max(
                2,
                min(
                    5,
                    n-1
                )
            ),

            seed=seed

        )


        G = nx.MultiDiGraph()



        # Add bidirectional channels

        for u,v in base_graph.edges():


            G.add_edge(
                int(u),
                int(v)
            )


            G.add_edge(
                int(v),
                int(u)
            )



        # Add extra random connections

        for _ in range(extra_edges):


            u,v = rng.choice(

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



        # ---------------------------------
        # Add node attributes
        # ---------------------------------

        for node_id in G.nodes:


            node = Node(

                node_id=str(node_id),

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

                carbon_intensity=float(
                    rng.uniform(
                        100,
                        700
                    )
                ),

                online=True

            )


            G.nodes[node_id].update(
                node.__dict__
            )



        # ---------------------------------
        # Add channel attributes
        # ---------------------------------

        for idx,(u,v,k) in enumerate(
                G.edges(
                    keys=True
                )
        ):


            capacity=float(
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

                balance_uv=capacity/2,

                balance_vu=capacity/2,

                failure_count=0,

                success_count=0

            )


            G.edges[u,v,k].update(
                channel.__dict__
            )



        return G



    # =====================================================
    # Utility Functions
    # =====================================================


    def number_of_nodes(self,G):

        return G.number_of_nodes()



    def number_of_channels(self,G):

        return G.number_of_edges()



    def summary(self,G):

        print(
            "Lightning Network Graph"
        )

        print(
            f"Nodes: {G.number_of_nodes()}"
        )

        print(
            f"Channels: {G.number_of_edges()}"
        )