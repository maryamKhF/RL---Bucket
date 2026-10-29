"""Deterministic exact 70/30 train/held-out transaction split."""


TRAIN_RATIO = 0.70


def normalize_evaluation_count(evaluation_count):
    """Round a requested held-out count to the nearest exact 30% count.

    An exact 70/30 split with integer rows requires the evaluation count to be
    divisible by three. Keep the requested count when possible; otherwise use
    the nearest positive multiple of three (ties round upward).
    """
    if isinstance(evaluation_count, bool):
        raise ValueError("evaluation_count must be a positive integer.")

    try:
        evaluation_count = int(evaluation_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "evaluation_count must be a positive integer."
        ) from exc

    if evaluation_count <= 0:
        raise ValueError("evaluation_count must be a positive integer.")

    lower = (evaluation_count // 3) * 3
    upper = lower + 3
    if lower < 3:
        return upper
    return lower if evaluation_count - lower < upper - evaluation_count else upper


def training_count_for_evaluation_count(evaluation_count):
    """Return the training count matching the normalized 30% test count."""
    exact_evaluation_count = normalize_evaluation_count(evaluation_count)
    return exact_evaluation_count * 7 // 3


def split_transaction_pool(transactions, evaluation_count):
    """Split one ordered transaction pool into disjoint train/test slices."""
    pool = list(transactions)
    exact_evaluation_count = normalize_evaluation_count(evaluation_count)
    training_count = training_count_for_evaluation_count(exact_evaluation_count)
    expected_count = training_count + exact_evaluation_count

    if len(pool) != expected_count:
        raise ValueError(
            f"Transaction pool has {len(pool)} rows; the 70/30 split for "
            f"{exact_evaluation_count} evaluation rows requires {expected_count}."
        )

    training = pool[:training_count]
    evaluation = pool[training_count:training_count + exact_evaluation_count]
    return training, evaluation
