# RL/reward.py

import numpy as np


def calculate_reward(
    success,
    path_length,
    carbon_intensity,
    fee=0.0,
    delay=0.0,
    partial_backtrack_count=0,
    full_reroute_count=0,
    attempt_count=1,
    scale=100.0,
    fee_reference=1000.0,
    delay_reference=10.0,
    carbon_reference=100.0,
    path_reference=10.0,
):
    """
    Calculate routing reward for PPO.

    PPO controls only eta.

    Top-K is fixed to 5 and is NOT part of the reward.

    Reward components for a successful payment:

        - path efficiency
        - transaction fee
        - delay
        - carbon intensity
        - partial backtracking
        - full rerouting
        - additional attempts

    Reward range:

        approximately [-1, 1]

    Successful payments receive a positive reward.

    Failed payments receive a dominant negative reward.
    """

    # =====================================================
    # NUMERICAL SAFETY
    # =====================================================

    path_length = max(
        int(path_length),
        0,
    )

    fee = max(
        float(fee),
        0.0,
    )

    delay = max(
        float(delay),
        0.0,
    )

    carbon = max(
        float(carbon_intensity),
        0.0,
    )

    partial_backtrack_count = max(
        int(partial_backtrack_count),
        0,
    )

    full_reroute_count = max(
        int(full_reroute_count),
        0,
    )

    attempt_count = max(
        int(attempt_count),
        1,
    )

    fee_reference = max(
        float(fee_reference),
        1.0,
    )

    delay_reference = max(
        float(delay_reference),
        1.0,
    )

    carbon_reference = max(
        float(carbon_reference),
        1.0,
    )

    path_reference = max(
        float(path_reference),
        1.0,
    )

    # scale is retained for compatibility with the existing
    # configuration, but is NOT used to dilute recovery costs.
    scale = max(
        float(scale),
        1.0,
    )


    # =====================================================
    # FAILURE
    # =====================================================

    if not success:

        failure_penalty = (
            1.0
            +
            0.05 * path_length
            +
            0.10 * partial_backtrack_count
            +
            0.20 * full_reroute_count
            +
            0.05 * max(
                attempt_count - 1,
                0,
            )
        )

        reward = -np.tanh(
            failure_penalty
        )

        return float(
            reward
        )


    # =====================================================
    # SUCCESS
    # =====================================================

    # -----------------------------------------------------
    # Normalize routing metrics
    # -----------------------------------------------------

    path_ratio = (
        path_length
        /
        path_reference
    )

    fee_ratio = (
        fee
        /
        fee_reference
    )

    delay_ratio = (
        delay
        /
        delay_reference
    )

    carbon_ratio = (
        carbon
        /
        carbon_reference
    )


    # -----------------------------------------------------
    # Bounded quality scores
    # -----------------------------------------------------

    path_score = np.exp(
        -path_ratio
    )

    fee_score = np.exp(
        -fee_ratio
    )

    delay_score = np.exp(
        -delay_ratio
    )

    carbon_score = np.exp(
        -carbon_ratio
    )


    # =====================================================
    # ROUTING QUALITY
    # =====================================================

    quality = (
        0.30 * path_score
        +
        0.30 * fee_score
        +
        0.20 * delay_score
        +
        0.20 * carbon_score
    )

    quality = float(
        np.clip(
            quality,
            0.0,
            1.0,
        )
    )


    # =====================================================
    # RECOVERY COST
    # =====================================================

    # These penalties are deliberately NOT divided by
    # "scale". They must remain visible to PPO.

    backtrack_penalty = (
        0.05
        *
        partial_backtrack_count
    )

    reroute_penalty = (
        0.10
        *
        full_reroute_count
    )

    additional_attempt_penalty = (
        0.02
        *
        max(
            attempt_count - 1,
            0,
        )
    )

    recovery_penalty = (
        backtrack_penalty
        +
        reroute_penalty
        +
        additional_attempt_penalty
    )


    # =====================================================
    # FINAL SUCCESS REWARD
    # =====================================================

    # Base successful-payment reward:
    #
    #     0.60
    #
    # Route quality contribution:
    #
    #     0.00 ... 0.40
    #
    # Therefore an uncomplicated successful payment
    # normally falls in:
    #
    #     0.60 ... 1.00

    raw_reward = (
        0.60
        +
        0.40 * quality
        -
        recovery_penalty
    )


    reward = float(
        np.clip(
            raw_reward,
            -1.0,
            1.0,
        )
    )


    return reward