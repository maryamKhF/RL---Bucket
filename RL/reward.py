# RL/reward.py

import numpy as np



def calculate_reward(
    success,
    path_length,
    carbon_intensity,
    fee=0,
    delay=0,
    scale=1000.0
):


    carbon = max(
        float(carbon_intensity),
        1e-3
    )


    # ----------------------------------
    # Failed Payment
    # ----------------------------------

    if not success:

        return float(
            -1.0
            -
            0.05 * path_length
        )



    # ----------------------------------
    # Success Objective
    # ----------------------------------

    path_quality = (

        1.0

        /

        (
            max(path_length,1)
            *
            carbon
        )

    )



    # ----------------------------------
    # Additional Costs
    # ----------------------------------

    cost_penalty = (

        0.01 * fee

        +

        0.01 * delay

    )



    raw_reward = (

        scale
        *
        path_quality

        -

        cost_penalty

    )



    return float(

        np.tanh(
            raw_reward / 100.0
        )

    )