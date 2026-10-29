"""Paper baseline and carbon-aware reward variants for routing experiments."""

import math


REWARD_FORMULAS = {
    "paper_base",
    "amount",
    "liquidity_margin",
    "persistence",
    "fee",
    "delay",
    "hops",
}


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
    formula="paper_base",
    payment_amount=0.0,
    liquidity_margin=1.0,
    persistence=1.0,
    lambda_fee=1.0,
    lambda_delay=1.0,
    lambda_hops=1.0,
    amount_reference=1_000_000.0,
):
    """Calculate the paper reward or one single-factor ablation.

    The article reward is ``S * D * 1e3 / C``. Here ``D`` is the number of
    route nodes (``path_length + 1``) and ``C`` is their summed carbon
    intensity. For numeric stability, the amount variant multiplies by the
    dimensionless ``payment_amount / amount_reference``; this differs from
    raw satoshis only by a fixed experiment-wide constant.

    Fee, delay, and hop penalties are converted to carbon-equivalent units:
    ``lambda * (metric / metric_reference) * carbon_reference``. The
    liquidity and persistence factors are expected to be finite and
    nonnegative; their observable-only operational definitions are supplied
    by RoutingEnv.
    """
    if not success:
        return 0.0
    if formula not in REWARD_FORMULAS:
        raise ValueError(
            f"Unknown reward formula {formula!r}; expected one of "
            f"{sorted(REWARD_FORMULAS)}."
        )

    average_carbon = _finite_number(carbon_intensity, "carbon_intensity")
    if average_carbon <= 0.0:
        raise ValueError("A successful route needs positive carbon intensity.")
    distance = max(int(path_length) + 1, 1)
    carbon_total = average_carbon * distance
    numerator = distance * 1000.0

    if formula == "amount":
        reference = _positive_number(amount_reference, "amount_reference")
        amount = _finite_number(payment_amount, "payment_amount")
        if amount < 0:
            raise ValueError("payment_amount must be nonnegative.")
        numerator *= amount / reference
    elif formula == "liquidity_margin":
        numerator *= _nonnegative_number(liquidity_margin, "liquidity_margin")
    elif formula == "persistence":
        numerator *= _nonnegative_number(persistence, "persistence")

    denominator = carbon_total
    if formula == "fee":
        denominator += _carbon_equivalent(
            fee, fee_reference, carbon_reference, lambda_fee, "fee"
        )
    elif formula == "delay":
        denominator += _carbon_equivalent(
            delay, delay_reference, carbon_reference, lambda_delay, "delay"
        )
    elif formula == "hops":
        denominator += _carbon_equivalent(
            path_length, path_reference, carbon_reference, lambda_hops, "path_length"
        )

    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("Reward denominator must be finite and positive.")
    reward = numerator / denominator
    if not math.isfinite(reward):
        raise ValueError("Calculated reward is not finite.")
    return float(reward)


def _finite_number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _positive_number(value, name):
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number


def _nonnegative_number(value, name):
    number = _finite_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return number


def _carbon_equivalent(value, reference, carbon_reference, weight, name):
    metric = _nonnegative_number(value, name)
    metric_reference = _positive_number(reference, f"{name}_reference")
    carbon_scale = _positive_number(carbon_reference, "carbon_reference")
    coefficient = _nonnegative_number(weight, f"lambda_{name}")
    return coefficient * (metric / metric_reference) * carbon_scale


__all__ = ["REWARD_FORMULAS", "calculate_reward"]
