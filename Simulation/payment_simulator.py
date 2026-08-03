"""
payment_simulator.py

Execute payment over simulated Lightning Network.

Responsibilities:

1. Validate selected route
2. Check channel availability
3. Check liquidity
4. Simulate stochastic failures
5. Calculate routing metrics
6. Update balances
7. Return PaymentResult
"""

import time
import numpy as np



# ==========================================================
# Geographic Helper
# ==========================================================

def estimate_inter_continent(G, u, v):
    """
    Estimate if a hop crosses continents.

    Uses geographic abstraction only.
    """

    source = G.nodes[u]
    target = G.nodes[v]


    latitude_difference = abs(
        source.get("latitude", 0)
        -
        target.get("latitude", 0)
    )


    longitude_difference = abs(
        source.get("longitude", 0)
        -
        target.get("longitude", 0)
    )


    return int(

        latitude_difference > 10

        and

        longitude_difference > 45

    )



# ==========================================================
# Payment Result
# ==========================================================

class PaymentResult:


    def __init__(
        self,
        success,
        path,
        edges,
        fee,
        delay,
        carbon,
        inter_country_hops,
        inter_continent_hops,
        reason,
        elapsed
    ):


        self.success = success

        self.path = path

        self.edges = edges


        self.fee = fee

        self.delay = delay

        self.carbon = carbon


        self.inter_country_hops = (
            inter_country_hops
        )

        self.inter_continent_hops = (
            inter_continent_hops
        )


        self.reason = reason

        self.elapsed = elapsed



    def to_dict(self):

        return {

            "success": self.success,

            "path": self.path,

            "edges": self.edges,

            "fee": self.fee,

            "delay": self.delay,

            "carbon": self.carbon,

            "inter_country_hops":
                self.inter_country_hops,

            "inter_continent_hops":
                self.inter_continent_hops,

            "reason": self.reason,

            "elapsed":
                self.elapsed
        }




# ==========================================================
# Payment Simulation
# ==========================================================


def simulate_payment(
        G,
        path,
        edges,
        amount,
        rng=None
):


    rng = rng or np.random.default_rng()


    start_time = time.perf_counter()



    # ------------------------------------------------------
    # No Route
    # ------------------------------------------------------

    if not path or not edges:


        return PaymentResult(

            False,

            path or [],

            edges or [],

            0,

            0,

            0,

            0,

            0,

            "no_path",

            time.perf_counter()
            -
            start_time
        )



    fee = 0.0

    delay = 0.0

    carbon = 0.0


    inter_country = 0

    inter_continent = 0




    # ------------------------------------------------------
    # Validation Phase
    # ------------------------------------------------------

    for u,v,k in edges:



        d = G.edges[u,v,k]



        # Channel availability

        if not d.get(
            "available",
            True
        ):


            return PaymentResult(

                False,
                path,
                edges,
                fee,
                delay,
                carbon,
                inter_country,
                inter_continent,
                "channel_unavailable",
                time.perf_counter()
                -
                start_time
            )




        # Liquidity check

        capacity = d.get(
            "capacity",
            0
        )


        balance = d.get(
            "balance_uv",
            0
        )


        if (

            capacity < amount

            or

            balance < amount

        ):


            return PaymentResult(

                False,
                path,
                edges,
                fee,
                delay,
                carbon,
                inter_country,
                inter_continent,
                "insufficient_liquidity",
                time.perf_counter()
                -
                start_time
            )




        # Random failure


        probability = d.get(

            "failure_probability",

            0.01

        )


        if rng.random() < probability:


            d["failure_count"] = (

                d.get(
                    "failure_count",
                    0
                )
                +
                1

            )


            return PaymentResult(

                False,
                path,
                edges,
                fee,
                delay,
                carbon,
                inter_country,
                inter_continent,
                "stochastic_failure",
                time.perf_counter()
                -
                start_time
            )




        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------


        fee += (

            d.get(
                "fee_base",
                0
            )

            +

            d.get(
                "fee_rate",
                0
            )
            *
            amount
            /
            1_000_000

        )



        delay += d.get(
            "delay",
            0
        )



        source = G.nodes[u]

        target = G.nodes[v]



        carbon += (

            source.get(
                "carbon_intensity",
                0
            )

            +

            target.get(
                "carbon_intensity",
                0
            )

        ) / 2



        inter_country += int(

            source.get(
                "country"
            )
            !=
            target.get(
                "country"
            )

        )



        inter_continent += estimate_inter_continent(

            G,

            u,

            v

        )





    # ------------------------------------------------------
    # Commit Payment
    # ------------------------------------------------------

    for u,v,k in edges:


        d = G.edges[u,v,k]



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
                amount
            ),

            d.get(
                "balance_vu",
                0
            )
            +
            amount

        )



        d["success_count"] = (

            d.get(
                "success_count",
                0
            )
            +
            1

        )




    average_carbon = (

        carbon

        /

        max(
            1,
            len(edges)
        )

    )




    return PaymentResult(

        True,

        path,

        edges,

        fee,

        delay,

        average_carbon,

        inter_country,

        inter_continent,

        "success",

        time.perf_counter()
        -
        start_time

    )