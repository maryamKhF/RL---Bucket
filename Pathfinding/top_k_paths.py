# Pathfinding/top_k_paths.py

import networkx as nx

from .heuristics import (
    modified_cost,
    geographic_penalty
)


def top_k_paths(
        G,
        source,
        target,
        amount,
        heuristic_fn,
        eta=0.0,
        k=5,
        max_hops=12
):

    H = nx.DiGraph()


    # -------------------------------------
    # Build weighted routing graph
    # -------------------------------------

    for u,v,key,data in G.edges(
            keys=True,
            data=True
    ):

        if not data.get(
                "available",
                True
        ):
            continue


        # Lightning liquidity check

        if data.get(
                "balance_uv",
                0
        ) < amount:
            continue



        native = heuristic_fn(
            G,
            u,
            v,
            data,
            amount
        )


        geo = geographic_penalty(
            G,
            u,
            v,
            data
        )


        weight = modified_cost(
            native,
            geo,
            eta
        )



        # Keep cheapest channel

        if (
            not H.has_edge(u,v)
            or
            weight < H[u][v]["weight"]
        ):

            H.add_edge(

                u,
                v,

                weight=weight,

                channel_key=key,

                channel_data=data

            )


    results=[]


    try:

        paths = nx.shortest_simple_paths(
            H,
            source,
            target,
            weight="weight"
        )


        for path in paths:


            hops=len(path)-1


            if hops > max_hops:
                continue



            edges=[]

            total_cost=0



            for a,b in zip(
                    path[:-1],
                    path[1:]
            ):

                key = H[a][b]["channel_key"]

                edges.append(
                    (
                        a,
                        b,
                        key
                    )
                )

                total_cost += H[a][b]["weight"]



            results.append(

                {

                    "success": True,

                    "path": path,

                    "edges": edges,

                    "cost": total_cost,

                    "hop_count": hops

                }

            )


            if len(results)>=k:
                break



    except (
        nx.NetworkXNoPath,
        nx.NodeNotFound
    ):

        return []



    return results