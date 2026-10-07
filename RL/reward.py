"""Reward used by the paper's carbon-aware routing agent."""

import math


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
    """Return success × 1000 / average route carbon intensity.

    ``carbon_intensity`` is the average intensity of the route's nodes,
    matching the paper's path-length-over-total-carbon reward. The other
    parameters remain accepted for call-site compatibility; they are not
    reward terms in the paper's RL objective.
    """

    if not success:
        return 0.0

    try:
        average_carbon = float(carbon_intensity)
    except (TypeError, ValueError) as exc:
        raise ValueError("carbon_intensity must be numeric.") from exc

    if not math.isfinite(average_carbon) or average_carbon <= 0.0:
        raise ValueError(
            "A successful route must have a finite, positive average "
            "carbon intensity."
        )

    # The reference implementation computes route node count × 1000 / sum
    # of node intensities, which simplifies to 1000 / their average.
    return float(1000.0 / average_carbon)
