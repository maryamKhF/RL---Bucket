# Pathfinding/test_eta.py
"""
Diagnostic eta sweep for the adaptive routing criterion.

Purpose
-------
This test answers one specific question before PPO training:

    Does changing eta actually change the routing result?

The test keeps k FIXED at 5 and sweeps eta over:
    0.0, 0.1, ..., 1.0

For one fixed source/destination/amount, it prints:
    - selected candidate (#1)
    - all five candidates
    - path
    - cost
    - raw heuristic (if available)
    - adaptive penalty (if available)
    - fee
    - delay
    - reliability

It also compares candidate rankings between eta values.

Run:
    py -m Pathfinding.test_eta
"""

import math
import numpy as np

from Pathfinding.top_k_paths import top_k_paths
from Pathfinding.heuristics import lnd_cost

from RL.test_rl import (
    CONFIG,
    GML_FILENAME,
    load_real_graph,
    prepare_graph,
    prepare_channel_attributes,
    build_real_transactions,
)


# ==========================================================
# TEST SETTINGS
# ==========================================================

ETA_VALUES = [
    round(x, 1)
    for x in np.arange(0.0, 1.01, 0.1)
]

FIXED_K = 5


# ==========================================================
# HELPERS
# ==========================================================

def _fmt(value, digits=6):
    if value is None:
        return "None"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(value):
        return "nan"

    return f"{value:.{digits}f}"


def _path_signature(candidate):
    path = candidate.get("path", [])
    return tuple(path)


def _candidate_signature(candidates):
    return tuple(
        _path_signature(candidate)
        for candidate in candidates
    )


def _print_candidate(rank, candidate):
    print(
        f"  #{rank} "
        f"path={candidate.get('path')} "
        f"hops={candidate.get('hop_count')} "
        f"cost={_fmt(candidate.get('cost'))} "
        f"fee={_fmt(candidate.get('total_fee'))} "
        f"delay={_fmt(candidate.get('total_delay'))} "
        f"reliability={_fmt(candidate.get('reliability'))}"
    )

    if "raw_heuristic" in candidate:
        print(
            f"      raw_heuristic="
            f"{_fmt(candidate.get('raw_heuristic'))} "
            f"adaptive_penalty="
            f"{_fmt(candidate.get('adaptive_penalty'))}"
        )

    print(
        f"      eta={_fmt(candidate.get('eta'))} "
        f"edges={candidate.get('edges')}"
    )


def _print_change_summary(results):
    print(
        "\n"
        "======================================================"
    )
    print(" ETA EFFECT SUMMARY")
    print(
        "======================================================"
    )

    previous_eta = None
    previous_signature = None

    changed_count = 0

    for eta, candidates in results:
        signature = _candidate_signature(candidates)

        if previous_signature is None:
            status = "BASELINE"
        elif signature != previous_signature:
            status = "CHANGED"
            changed_count += 1
        else:
            status = "UNCHANGED"

        selected = (
            candidates[0].get("path")
            if candidates
            else None
        )

        print(
            f"eta={eta:.1f}  "
            f"status={status:<9}  "
            f"selected={selected}"
        )

        if previous_eta is not None and signature != previous_signature:
            print(
                f"  ranking changed: "
                f"eta {previous_eta:.1f} -> {eta:.1f}"
            )

        previous_eta = eta
        previous_signature = signature

    print(
        f"\nNumber of eta transitions with "
        f"candidate-order change: {changed_count}"
    )

    unique_rankings = {
        _candidate_signature(candidates)
        for _, candidates in results
    }

    print(
        f"Unique candidate rankings observed: "
        f"{len(unique_rankings)}"
    )

    if len(unique_rankings) == 1:
        print(
            "\nDIAGNOSIS: eta has NO observable effect on "
            "the top-5 candidate ranking for this transaction."
        )
        print(
            "Do NOT retrain PPO yet. The routing objective/action "
            "interface needs further inspection."
        )
    else:
        print(
            "\nDIAGNOSIS: eta DOES change the candidate ranking "
            "for this transaction."
        )
        print(
            "The routing layer exposes an observable action effect; "
            "PPO can now be tested meaningfully."
        )


# ==========================================================
# MAIN TEST
# ==========================================================

def main():
    print(
        "\n"
        "======================================================"
    )
    print(" ETA SWEEP DIAGNOSTIC")
    print(
        f" Snapshot: {GML_FILENAME}"
    )
    print(
        f" Fixed K: {FIXED_K}"
    )
    print(
        "======================================================"
    )

    # ------------------------------------------------------
    # 1. Load real GML
    # ------------------------------------------------------

    G = load_real_graph()

    # ------------------------------------------------------
    # 2. Convert to routing graph
    # ------------------------------------------------------

    G = prepare_graph(G)
    G = prepare_channel_attributes(G)

    print(
        "\nPrepared graph:"
    )
    print(
        f"  type  = {type(G).__name__}"
    )
    print(
        f"  nodes = {G.number_of_nodes()}"
    )
    print(
        f"  edges = {G.number_of_edges()}"
    )

    # ------------------------------------------------------
    # 3. Build one deterministic transaction
    # ------------------------------------------------------

    transactions = build_real_transactions(
        G,
        count=1,
        amount=1000,
    )

    if not transactions:
        raise RuntimeError(
            "No transaction was generated."
        )

    tx = transactions[0]

    print(
        "\nFixed transaction:"
    )
    print(
        f"  source      = {tx.source}"
    )
    print(
        f"  destination = {tx.destination}"
    )
    print(
        f"  amount      = {tx.amount}"
    )

    # ------------------------------------------------------
    # 4. Sweep eta
    # ------------------------------------------------------

    results = []

    lambda_h = float(
        CONFIG["graph"].get(
            "lambda_h",
            1.0,
        )
    )

    max_hops = int(
        CONFIG["graph"].get(
            "max_hops",
            12,
        )
    )

    print(
        "\n"
        "======================================================"
    )
    print(" ETA SWEEP")
    print(
        "======================================================"
    )

    for eta in ETA_VALUES:
        print(
            f"\n------------------------------------------------------"
        )
        print(
            f"ETA = {eta:.1f}   K = {FIXED_K}"
        )
        print(
            "------------------------------------------------------"
        )

        candidates = top_k_paths(
            G=G,
            source=tx.source,
            target=tx.destination,
            amount=tx.amount,
            heuristic_fn=lnd_cost,
            eta=eta,
            k=FIXED_K,
            max_hops=max_hops,
            lambda_h=lambda_h,
        )

        results.append(
            (eta, candidates)
        )

        if not candidates:
            print("  NO CANDIDATE PATH FOUND")
            continue

        print(
            f"  Candidate count = {len(candidates)}"
        )

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            _print_candidate(
                rank,
                candidate,
            )

    # ------------------------------------------------------
    # 5. Compare eta values
    # ------------------------------------------------------

    _print_change_summary(results)

    # ------------------------------------------------------
    # 6. Check fixed K invariant
    # ------------------------------------------------------

    print(
        "\n"
        "======================================================"
    )
    print(" FIXED-K CHECK")
    print(
        "======================================================"
    )

    for eta, candidates in results:
        if len(candidates) > FIXED_K:
            raise AssertionError(
                f"eta={eta} produced more than K={FIXED_K} "
                f"candidates."
            )

        print(
            f"eta={eta:.1f}: "
            f"{len(candidates)} candidates"
        )

    print(
        "\nETA SWEEP COMPLETED"
    )


if __name__ == "__main__":
    main()
