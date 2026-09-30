# RL/reward.py

import numpy as np


def calculate_reward(
    success,
    path_length,
    carbon_intensity,
    fee=0.0,
    delay=0.0,
    scale=100.0
):
    """
    Calculate routing reward for the PPO agent.

    Main objectives:
        1. Payment success
        2. Shorter routing path
        3. Lower fee
        4. Lower delay
        5. Lower carbon intensity

    Parameters
    ----------
    success : bool
        Whether the payment was successfully completed.

    path_length : int
        Number of edges in the selected route.

    carbon_intensity : float
        Carbon intensity associated with the selected route.

    fee : float
        Total routing/payment fee.

    delay : float
        Total routing delay.

    scale : float
        Reward scaling factor.

    Returns
    -------
    float
        Reward approximately in [-1, 1].
    """

    # =====================================================
    # Numerical safety
    # =====================================================

    path_length = max(
        int(path_length),
        0
    )

    fee = max(
        float(fee),
        0.0
    )

    delay = max(
        float(delay),
        0.0
    )

    carbon = max(
        float(carbon_intensity),
        0.0
    )

    scale = max(
        float(scale),
        1.0
    )

    # =====================================================
    # FAILED PAYMENT
    # =====================================================

    if not success:

        # Payment failure is the dominant negative signal.
        #
        # The penalty is bounded so that the reward remains
        # within approximately [-1, 1].

        failure_penalty = (
            1.0
            +
            0.05 * path_length
        )

        reward = -np.tanh(
            failure_penalty
        )

        return float(reward)

    # =====================================================
    # SUCCESSFUL PAYMENT
    # =====================================================

    # -----------------------------------------------------
    # Path efficiency
    # -----------------------------------------------------

    # Shorter paths receive a larger score.
    path_score = 1.0 / max(
        path_length,
        1
    )

    # -----------------------------------------------------
    # Normalize routing criteria
    # -----------------------------------------------------

    # The following terms are normalized relative to the
    # reward scale so that one metric does not dominate
    # the others.

    path_penalty = (
        1.0
        -
        path_score
    )

    fee_penalty = (
        fee / scale
    )

    delay_penalty = (
        delay / scale
    )

    carbon_penalty = (
        carbon / scale
    )

    # =====================================================
    # Combined successful-payment score
    # =====================================================

    # Successful payment receives a positive base reward.
    # The remaining terms distinguish better and worse
    # successful routes.

    success_bonus = 1.0

    raw_reward = (
        success_bonus
        +
        0.5 * path_score
        -
        0.20 * path_penalty
        -
        0.20 * fee_penalty
        -
        0.20 * delay_penalty
        -
        0.20 * carbon_penalty
    )

    # =====================================================
    # Normalize reward
    # =====================================================

    reward = np.tanh(
        raw_reward
    )

    return float(reward)