"""
Evaluations/baselines.py

Baseline evaluation framework.

Supports:

- Native LND heuristic
- Native CLN heuristic
- Native ECL heuristic
- Static heuristic
- Improved routing with Bucket
- RL assisted routing

Produces raw transaction rows
compatible with metrics.py
"""


from __future__ import annotations


import numpy as np


from Pathfinding.dijkstra import Dijkstra

from Pathfinding.heuristics import (
    lnd_cost,
    cln_cost,
    ecl_cost
)

from Pathfinding.top_k_paths import top_k_paths


from Bucket.bucket import (
    Bucket,
    execute_bucket
)


from Simulation.payment_simulator import simulate_payment


from RL.state import neighborhood_state




# ============================================================
# Heuristic Configuration
# ============================================================


HEURISTICS = {

    "native_lnd": lnd_cost,

    "native_cln": cln_cost,

    "native_ecl": ecl_cost

}



STATIC_ETA = {

    "static_lnd": 0.27876,

    "static_cln": 0.25556,

    "static_ecl": 0.35784

}





# ============================================================
# Helper Functions
# ============================================================


def calculate_path_metrics(
    G,
    edges,
    amount
):

    fee = 0

    delay = 0

    carbon_values = []

    country_hops = 0



    for e in edges:


        channel = G.edges[e]


        fee += (

            channel.get(
                "fee_base",
                0
            )

            +

            channel.get(
                "fee_rate",
                0
            )
            *
            amount
            /
            1e6

        )


        delay += channel.get(
            "delay",
            0
        )



        n1, n2 = e[0], e[1]


        carbon_values.append(

            (

                G.nodes[n1].get(
                    "carbon_intensity",
                    0
                )

                +

                G.nodes[n2].get(
                    "carbon_intensity",
                    0
                )

            )
            /
            2

        )



        if (

            G.nodes[n1].get(
                "country"
            )

            !=

            G.nodes[n2].get(
                "country"
            )

        ):

            country_hops += 1



    return {


        "fee":
            float(fee),


        "delay":
            float(delay),


        "carbon":
            float(
                np.mean(carbon_values)
            )
            if carbon_values
            else 0,


        "inter_country_hops":
            country_hops,


        "inter_continent_hops":
            0

    }





# ============================================================
# Main Baseline Runner
# ============================================================


def run_method(
    G,
    transactions,
    name,
    cfg,
    model=None,
    seed=42
):


    rng = np.random.default_rng(seed)


    rows = []


    router = Dijkstra(G)



    # --------------------------------------------------------
    # Select heuristic
    # --------------------------------------------------------


    if name.startswith("native_"):


        heuristic = HEURISTICS[name]

        eta = 0



    elif name.startswith("static_"):


        heuristic = HEURISTICS[

            name.replace(
                "static_",
                "native_"
            )

        ]


        eta = STATIC_ETA[name]



    else:


        heuristic = HEURISTICS[

            name.replace(
                "improved_",
                "native_"
            )

        ]


        eta = None





    # ========================================================
    # Transaction Loop
    # ========================================================


    for tx in transactions:



        # ----------------------------------------------------
        # RL ETA prediction
        # ----------------------------------------------------


        if model is not None:


            obs = neighborhood_state(

                G,

                tx.source,

                tx.destination,

                cfg["graph"]["neighborhood_k"],

                cfg["graph"]["neighborhood_m"],

                cfg["graph"]["ego_radius"]

            )


            action, _ = model.predict(

                obs,

                deterministic=True

            )


            eta = float(
                action[0]
            )





        # ----------------------------------------------------
        # Improved Bucket Routing
        # ----------------------------------------------------


        if name.startswith("improved"):


            paths = top_k_paths(

                G,

                tx.source,

                tx.destination,

                tx.amount,

                heuristic,

                eta or 0,

                cfg["graph"]["top_k_paths"],

                cfg["graph"]["max_hops"]

            )



            if paths:


                bucket = Bucket(

                    tx.tx_id,

                    tx.tx_id,

                    paths

                )


                success, candidate, attempts = execute_bucket(

                    G,

                    bucket,

                    tx.amount,

                    rng

                )



                if success:


                    path, edges, _ = candidate


                    metrics = calculate_path_metrics(

                        G,

                        edges,

                        tx.amount

                    )


                    rows.append({

                        "success": True,

                        "path_length": len(edges),

                        **metrics,

                        "runtime": 0

                    })



                else:


                    rows.append({

                        "success": False,

                        "path_length": 0,

                        "fee": 0,

                        "delay": 0,

                        "carbon": 0,

                        "inter_country_hops": 0,

                        "inter_continent_hops": 0,

                        "runtime": 0

                    })



            else:


                rows.append({

                    "success": False,

                    "path_length": 0,

                    "fee": 0,

                    "delay": 0,

                    "carbon": 0,

                    "inter_country_hops": 0,

                    "inter_continent_hops": 0,

                    "runtime": 0

                })



            continue





        # ----------------------------------------------------
        # Normal Baseline Routing
        # ----------------------------------------------------


        route = router.shortest_path(

            tx.source,

            tx.destination,

            tx.amount,

            heuristic,

            eta or 0,

            cfg["graph"]["max_hops"]

        )



        if not route["success"]:


            rows.append({

                "success": False,

                "path_length": 0,

                "fee": 0,

                "delay": 0,

                "carbon": 0,

                "inter_country_hops": 0,

                "inter_continent_hops": 0,

                "runtime": 0

            })


            continue




        path = route["path"]

        edges = route["edges"]





        result = simulate_payment(

            G,

            path,

            edges,

            tx.amount,

            rng

        )





        rows.append({

            "success":

                bool(

                    result

                    and

                    result.success

                ),



            "path_length":

                len(edges),



            "fee":

                result.fee

                if result

                else 0,



            "delay":

                result.delay

                if result

                else 0,



            "carbon":

                result.carbon

                if result

                else 0,



            "inter_country_hops":

                result.inter_country_hops

                if result

                else 0,



            "inter_continent_hops":

                result.inter_continent_hops

                if result

                else 0,



            "runtime":

                result.elapsed

                if result

                else 0

        })



    return rows