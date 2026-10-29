
"""
Evaluations/baselines.py

Evaluation runners for LND-based baselines and the proposed
RL + Bucket routing method.

Baseline methods
----------------
- native_lnd
- native_cln
- native_ecl
- static_lnd
- static_cln
- static_ecl

Proposed method
---------------
- improved_* / RL-assisted routing

Architecture
------------
LND-based baselines:

    Transaction
        |
        v
    Dijkstra
        |
        v
    one selected route
        |
        v
    PaymentSimulator
        |
        +--> FailureModel
        |
        +--> NetworkDynamics
        |
        v
    PaymentResult

Proposed method:

    Transaction
        |
        v
    RoutingEnv
        |
        v
    PPO -> eta
        |
        v
    Top-K
        |
        v
    Bucket
        |
        v
    PaymentSimulator
        |
        v
    Partial Backtracking
        |
        v
    Payment result

Important
---------
The LND-based baselines are intentionally NOT routed through
RoutingEnv. They remain independent routing baselines.

The proposed method uses the existing RoutingEnv so that
evaluation does not duplicate the project's current Bucket,
PaymentSimulator, FailureModel, and PartialBacktracking logic.

MultiDiGraph channel identity is preserved throughout the
baseline execution.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from Pathfinding.dijkstra import Dijkstra

from Pathfinding.heuristics import (
    lnd_cost,
    cln_cost,
    ecl_cost,
)

from Simulation.failure_model import (
    FailureModel,
    assign_failure_probabilities,
)

from Simulation.network_dynamics import (
    NetworkDynamics,
)

from Simulation.payment_simulator import (
    PaymentSimulator,
)

from RL.environment import RoutingEnv


# ============================================================
# Heuristic Configuration
# ============================================================

HEURISTICS = {
    "native_lnd": lnd_cost,
    "native_cln": cln_cost,
    "native_ecl": ecl_cost,
}


STATIC_ETA = {
    "static_lnd": 0.27876,
    "static_cln": 0.25556,
    "static_ecl": 0.35784,
}


# ============================================================
# Method Classification
# ============================================================

BASELINE_METHODS = {
    "native_lnd",
    "native_cln",
    "native_ecl",
    "static_lnd",
    "static_cln",
    "static_ecl",
}


def is_baseline_method(name: str) -> bool:
    """
    Return True when name identifies an LND-based baseline.
    """

    return name in BASELINE_METHODS


def is_proposed_method(name: str) -> bool:
    """
    Return True for the RL/improved routing method.
    """

    return (
        name.startswith("improved")
        or name.startswith("rl")
    )


# ============================================================
# Configuration Helpers
# ============================================================

def _simulation_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the simulation configuration safely.
    """

    simulation = cfg.get(
        "simulation",
        {}
    )

    if not isinstance(
        simulation,
        dict
    ):
        return {}

    return simulation


def _graph_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the graph configuration safely.
    """

    graph = cfg.get(
        "graph",
        {}
    )

    if not isinstance(
        graph,
        dict
    ):
        return {}

    return graph


def _resolve_seed(
    cfg: Dict[str, Any],
    seed: int,
) -> int:
    """
    Resolve the deterministic seed.

    Explicit run_method seed has priority over config.
    """

    if seed is not None:
        return int(seed)

    return int(
        cfg.get(
            "seed",
            42
        )
    )


def _resolve_failure_probability(
    cfg: Dict[str, Any],
) -> float:
    """
    Resolve the channel failure probability used by
    LND-based baseline evaluation.

    Priority
    --------
    1. simulation.channel_failure_rate
    2. simulation.failure_rate
    3. top-level failure_rate
    4. default 0.01

    A list such as `failure_rates` is intentionally not
    silently converted into one value. Evaluation code should
    explicitly provide the desired failure-rate condition.
    """

    simulation = _simulation_config(
        cfg
    )

    candidates = [
        simulation.get(
            "channel_failure_rate"
        ),
        simulation.get(
            "failure_rate"
        ),
        cfg.get(
            "failure_rate"
        ),
    ]

    for value in candidates:

        if value is None:
            continue

        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Failure probability must be in [0, 1]."
            )

        return value

    return 0.01


def _resolve_node_failure_probability(
    cfg: Dict[str, Any],
) -> float:
    """
    Resolve node failure probability.
    """

    simulation = _simulation_config(
        cfg
    )

    value = simulation.get(
        "node_failure_probability",
        0.01
    )

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "node_failure_probability must be in [0, 1]."
        )

    return value


def _resolve_liquidity_failure_probability(
    cfg: Dict[str, Any],
) -> float:
    """
    Resolve liquidity failure probability.
    """

    simulation = _simulation_config(
        cfg
    )

    value = simulation.get(
        "liquidity_failure_probability",
        0.05
    )

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "liquidity_failure_probability must be in [0, 1]."
        )

    return value


def _resolve_node_failure_rate(
    cfg: Dict[str, Any],
) -> float:
    """
    Resolve NetworkDynamics node failure rate.
    """

    simulation = _simulation_config(
        cfg
    )

    value = simulation.get(
        "node_failure_rate",
        0.005
    )

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "node_failure_rate must be in [0, 1]."
        )

    return value


def _resolve_recovery_rate(
    cfg: Dict[str, Any],
) -> float:
    """
    Resolve NetworkDynamics recovery rate.
    """

    simulation = _simulation_config(
        cfg
    )

    value = simulation.get(
        "recovery_rate",
        0.05
    )

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "recovery_rate must be in [0, 1]."
        )

    return value


# ============================================================
# Failure / Simulation Components
# ============================================================

def _build_payment_components(
    G,
    cfg: Dict[str, Any],
    seed: int,
):
    """
    Build the exact simulation components required by
    PaymentSimulator.

    Returns
    -------
    failure_model
    network_dynamics
    payment_simulator
    """

    channel_failure_rate = (
        _resolve_failure_probability(
            cfg
        )
    )

    node_failure_probability = (
        _resolve_node_failure_probability(
            cfg
        )
    )

    liquidity_failure_probability = (
        _resolve_liquidity_failure_probability(
            cfg
        )
    )

    node_failure_rate = (
        _resolve_node_failure_rate(
            cfg
        )
    )

    recovery_rate = (
        _resolve_recovery_rate(
            cfg
        )
    )

    # --------------------------------------------------------
    # Assign deterministic channel failure probabilities.
    #
    # This writes the static failure_probability attribute
    # expected by FailureModel.
    # --------------------------------------------------------

    assign_failure_probabilities(
        G,
        average_rate=channel_failure_rate,
        seed=seed,
    )

    # --------------------------------------------------------
    # FailureModel
    # --------------------------------------------------------

    failure_model = FailureModel(
        node_failure_probability=(
            node_failure_probability
        ),
        liquidity_failure_probability=(
            liquidity_failure_probability
        ),
        seed=seed,
    )

    # --------------------------------------------------------
    # NetworkDynamics
    # --------------------------------------------------------

    network_dynamics = NetworkDynamics(
        G=G,
        channel_failure_rate=(
            channel_failure_rate
        ),
        node_failure_rate=(
            node_failure_rate
        ),
        recovery_rate=(
            recovery_rate
        ),
        seed=seed,
    )

    # Start from a clean runtime state.
    network_dynamics.reset(
        reset_balances=False
    )

    failure_model.reset_runtime_state()

    # --------------------------------------------------------
    # PaymentSimulator
    # --------------------------------------------------------

    payment_simulator = PaymentSimulator(
        G=G,
        failure_model=failure_model,
        network_dynamics=network_dynamics,
    )

    return (
        failure_model,
        network_dynamics,
        payment_simulator,
    )


# ============================================================
# Result Conversion
# ============================================================

def _zero_result_row(
    tx,
    reason: str,
    eta: Optional[float] = None,
    top_k: int = 1,
) -> Dict[str, Any]:
    """
    Create a schema-compatible failed evaluation row.
    """

    return {
        "transaction_id": getattr(
            tx,
            "tx_id",
            None,
        ),
        "tx_id": getattr(
            tx,
            "tx_id",
            None,
        ),
        "eta": eta,
        "top_k": top_k,

        "candidate_path_count": 0,
        "usable_candidate_count": 0,
        "bucket_size": 0,

        "success": False,
        "payment_success": False,

        "path": [],
        "path_length": 0,

        "fee": 0.0,
        "delay": 0.0,
        "carbon": 0.0,

        "failure_probability": 0.0,
        "average_delay": 0.0,

        "backtrack_count": 0,
        "partial_backtrack_count": 0,
        "partial_backtrack_success": 0,

        "full_reroute": False,
        "full_reroute_count": 0,

        "attempt_count": 0,

        "reason": reason,
        "episode_reward": 0.0,

        "inter_country_hops": 0,
        "inter_continent_hops": 0,

        "runtime": 0.0,

        "recovered": False,
        "amount": float(
            getattr(
                tx,
                "amount",
                0.0,
            )
        ),
    }


def _payment_result_to_row(
    tx,
    result,
    eta: Optional[float],
) -> Dict[str, Any]:
    """
    Convert PaymentResult into the common evaluation row schema.

    PaymentSimulator remains the single source of truth for:
        fee
        delay
        carbon
        geographic hop metrics
        success
        reason
        elapsed
    """

    success = bool(
        result
        and result.success
    )

    path = (
        list(result.path)
        if result is not None
        and result.path is not None
        else []
    )

    edges = (
        list(result.edges)
        if result is not None
        and result.edges is not None
        else []
    )

    return {
        "transaction_id": getattr(
            tx,
            "tx_id",
            None,
        ),
        "tx_id": getattr(
            tx,
            "tx_id",
            None,
        ),

        "eta": eta,
        "top_k": 1,

        "candidate_path_count": (
            1
            if result is not None
            else 0
        ),

        "usable_candidate_count": (
            1
            if result is not None
            else 0
        ),

        "bucket_size": 0,

        "success": success,
        "payment_success": success,

        "path": path,
        "path_length": len(edges),

        "fee": float(
            result.fee
            if result is not None
            else 0.0
        ),

        "delay": float(
            result.delay
            if result is not None
            else 0.0
        ),

        "carbon": float(
            result.carbon
            if result is not None
            else 0.0
        ),

        "failure_probability": float(
            0.0
        ),

        "average_delay": float(
            result.delay
            if result is not None
            else 0.0
        ),

        "backtrack_count": 0,
        "partial_backtrack_count": 0,
        "partial_backtrack_success": 0,

        "full_reroute": False,
        "full_reroute_count": 0,

        "attempt_count": 1
        if result is not None
        else 0,

        "reason": (
            result.reason
            if result is not None
            else "no_payment_result"
        ),

        "episode_reward": 0.0,

        "inter_country_hops": int(
            result.inter_country_hops
            if result is not None
            else 0
        ),

        "inter_continent_hops": int(
            result.inter_continent_hops
            if result is not None
            else 0
        ),

        "runtime": float(
            result.elapsed
            if result is not None
            else 0.0
        ),

        "recovered": False,

        "amount": float(
            getattr(
                tx,
                "amount",
                0.0,
            )
        ),
    }


# ============================================================
# Baseline Route
# ============================================================

def _run_lnd_based_baseline(
    G,
    transactions: Iterable,
    name: str,
    cfg: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Execute one of the six LND-based baseline methods.

    These methods use Dijkstra to select ONE route and then
    execute that route through PaymentSimulator.

    No Bucket is used here.

    No RoutingEnv is used here.

    No PPO is used here.
    """

    if name not in BASELINE_METHODS:
        raise ValueError(
            f"Unsupported baseline method: {name}"
        )

    # --------------------------------------------------------
    # Select heuristic
    # --------------------------------------------------------

    if name.startswith(
        "native_"
    ):

        heuristic = HEURISTICS[
            name
        ]

        eta = 0.0

    else:

        native_name = name.replace(
            "static_",
            "native_",
        )

        heuristic = HEURISTICS[
            native_name
        ]

        eta = float(
            STATIC_ETA[name]
        )

    graph_cfg = _graph_config(
        cfg
    )

    max_hops = int(
        graph_cfg.get(
            "max_hops",
            12,
        )
    )

    # --------------------------------------------------------
    # Build actual payment execution stack.
    # --------------------------------------------------------

    (
        failure_model,
        network_dynamics,
        payment_simulator,
    ) = _build_payment_components(
        G=G,
        cfg=cfg,
        seed=seed,
    )

    router = Dijkstra(
        G
    )

    rows: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Transaction loop
    # --------------------------------------------------------

    for tx in transactions:

        # Each baseline transaction represents one independent
        # payment attempt. Runtime failure state is reset before
        # selecting/executing the next payment.
        network_dynamics.reset(
            reset_balances=False
        )

        failure_model.reset_runtime_state()

        # ----------------------------------------------------
        # Route selection
        # ----------------------------------------------------

        route = router.shortest_path(
            tx.source,
            tx.destination,
            tx.amount,
            heuristic,
            eta,
            max_hops,
        )

        if not route.get(
            "success",
            False
        ):

            rows.append(
                _zero_result_row(
                    tx=tx,
                    reason="routing_failed",
                    eta=eta,
                    top_k=1,
                )
            )

            continue

        path = route.get(
            "path"
        )

        edges = route.get(
            "edges"
        )

        if not path or not edges:

            rows.append(
                _zero_result_row(
                    tx=tx,
                    reason="invalid_routing_result",
                    eta=eta,
                    top_k=1,
                )
            )

            continue

        # ----------------------------------------------------
        # Execute exactly ONE route.
        #
        # PaymentSimulator validates exact MultiDiGraph
        # channel identity and delegates failure handling.
        # ----------------------------------------------------

        result = payment_simulator.simulate_payment(
            path=path,
            edges=edges,
            amount=tx.amount,
            tx_id=getattr(
                tx,
                "tx_id",
                None,
            ),
        )

        row = _payment_result_to_row(
            tx=tx,
            result=result,
            eta=eta,
        )

        rows.append(
            row
        )

    return rows


# ============================================================
# Proposed RL + Bucket Method
# ============================================================

def _run_proposed_method(
    G,
    transactions: Iterable,
    name: str,
    cfg: Dict[str, Any],
    model=None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Execute the proposed RL-assisted routing method.

    The existing RoutingEnv is deliberately used as the
    orchestration layer so that evaluation follows the same
    implementation used by the validated end-to-end pipeline.

    Pipeline:

        PPO
          ->
        eta
          ->
        Top-K (5)
          ->
        Bucket
          ->
        PaymentSimulator
          ->
        Partial Backtracking
          ->
        result
    """

    if model is None:
        raise ValueError(
            "A trained model is required for the proposed "
            "RL + Bucket method."
        )

    # --------------------------------------------------------
    # RoutingEnv owns the proposed routing pipeline.
    # --------------------------------------------------------

    env = RoutingEnv(
        G=G,
        transactions=list(
            transactions
        ),
        config=cfg,
    )

    rows: List[Dict[str, Any]] = []

    try:

        # ----------------------------------------------------
        # One environment reset starts at the first transaction.
        # ----------------------------------------------------

        obs, reset_info = env.reset(
            seed=seed,
            options={"transaction_index": 0},
        )

        transaction_index = 0

        while transaction_index < len(
            env.transactions
        ):

            # ------------------------------------------------
            # PPO predicts only eta.
            # ------------------------------------------------

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action_array = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(-1)

            if action_array.size != 1:
                raise ValueError(
                    "RL model must produce exactly one "
                    "eta action."
                )

            eta = float(
                action_array[0]
            )

            # ------------------------------------------------
            # Let RoutingEnv perform the complete proposed
            # routing pipeline.
            # ------------------------------------------------

            obs, _, terminated, truncated, info = (
                env.step(
                    np.asarray(
                        [eta],
                        dtype=np.float32,
                    )
                )
            )

            info = dict(
                info or {}
            )

            # ------------------------------------------------
            # Normalize the information produced by env.
            # ------------------------------------------------

            tx = env.transactions[
                transaction_index
            ]

            row = {
                "transaction_id": info.get(
                    "transaction_id",
                    getattr(
                        tx,
                        "tx_id",
                        None,
                    ),
                ),

                "tx_id": info.get(
                    "transaction_id",
                    getattr(
                        tx,
                        "tx_id",
                        None,
                    ),
                ),

                "eta": float(
                    info.get(
                        "eta",
                        eta,
                    )
                ),

                "top_k": int(
                    info.get(
                        "top_k",
                        5,
                    )
                ),

                "candidate_path_count": int(
                    info.get(
                        "candidate_path_count",
                        0,
                    )
                ),

                "usable_candidate_count": int(
                    info.get(
                        "usable_candidate_count",
                        0,
                    )
                ),

                "bucket_size": int(
                    info.get(
                        "bucket_size",
                        0,
                    )
                ),

                "success": bool(
                    info.get(
                        "success",
                        info.get(
                            "payment_success",
                            False,
                        ),
                    )
                ),

                "payment_success": bool(
                    info.get(
                        "payment_success",
                        info.get(
                            "success",
                            False,
                        ),
                    )
                ),

                "path": list(
                    info.get(
                        "path",
                        [],
                    )
                    or []
                ),

                "path_length": int(
                    info.get(
                        "path_length",
                        0,
                    )
                ),

                "fee": float(
                    info.get(
                        "fee",
                        0.0,
                    )
                ),

                "delay": float(
                    info.get(
                        "delay",
                        0.0,
                    )
                ),

                "carbon": float(
                    info.get(
                        "carbon",
                        0.0,
                    )
                ),

                "failure_probability": float(
                    info.get(
                        "failure_probability",
                        0.0,
                    )
                ),

                "average_delay": float(
                    info.get(
                        "average_delay",
                        0.0,
                    )
                ),

                "backtrack_count": int(
                    info.get(
                        "backtrack_count",
                        0,
                    )
                ),

                "partial_backtrack_count": int(
                    info.get(
                        "partial_backtrack_count",
                        0,
                    )
                ),

                "partial_backtrack_success": int(
                    info.get(
                        "partial_backtrack_success",
                        0,
                    )
                ),

                "full_reroute": bool(
                    info.get(
                        "full_reroute",
                        False,
                    )
                ),

                "full_reroute_count": int(
                    info.get(
                        "full_reroute_count",
                        0,
                    )
                ),

                "attempt_count": int(
                    info.get(
                        "attempt_count",
                        0,
                    )
                ),

                "reason": info.get(
                    "reason",
                    "unknown",
                ),

                "episode_reward": float(
                    info.get(
                        "episode_reward",
                        0.0,
                    )
                ),

                "inter_country_hops": int(
                    info.get(
                        "inter_country_hops",
                        0,
                    )
                ),

                "inter_continent_hops": int(
                    info.get(
                        "inter_continent_hops",
                        0,
                    )
                ),

                "runtime": float(
                    info.get(
                        "runtime",
                        0.0,
                    )
                ),

                "recovered": bool(
                    info.get(
                        "partial_backtrack_success",
                        0,
                    ) > 0
                ),

                "amount": float(
                    getattr(
                        tx,
                        "amount",
                        0.0,
                    )
                ),
            }

            rows.append(
                row
            )

            transaction_index += 1

            if terminated or truncated:
                break

    finally:

        # RoutingEnv does not need explicit simulation
        # finalization, but closing the Gym environment here
        # keeps this runner safe for future environment changes.
        try:
            env.close()
        except Exception:
            pass

    return rows


# ============================================================
# Main Public Runner
# ============================================================

def run_method(
    G,
    transactions,
    name,
    cfg,
    model=None,
    seed=42,
):
    """
    Execute one evaluation method.

    Parameters
    ----------
    G :
        NetworkX graph.

    transactions :
        Iterable of transaction objects.

    name :
        One of:

            native_lnd
            native_cln
            native_ecl
            static_lnd
            static_cln
            static_ecl

        or an improved/RL method name.

    cfg :
        Project configuration dictionary.

    model :
        Trained PPO model for the proposed method.

    seed :
        Deterministic evaluation seed.

    Returns
    -------
    list of dict
        Raw transaction-level evaluation rows.
    """

    if G is None:
        raise ValueError(
            "G cannot be None."
        )

    if transactions is None:
        raise ValueError(
            "transactions cannot be None."
        )

    seed = _resolve_seed(
        cfg,
        seed,
    )

    # Materialize once because the proposed environment and
    # baseline loop both require deterministic transaction access.
    transactions = list(
        transactions
    )

    if is_baseline_method(
        name
    ):

        return _run_lnd_based_baseline(
            G=G,
            transactions=transactions,
            name=name,
            cfg=cfg,
            seed=seed,
        )

    if is_proposed_method(
        name
    ):

        return _run_proposed_method(
            G=G,
            transactions=transactions,
            name=name,
            cfg=cfg,
            model=model,
            seed=seed,
        )

    raise ValueError(
        f"Unknown evaluation method: {name}"
    )


# ============================================================
# Public API
# ============================================================

__all__ = [
    "HEURISTICS",
    "STATIC_ETA",
    "BASELINE_METHODS",
    "is_baseline_method",
    "is_proposed_method",
    "run_method",
]

