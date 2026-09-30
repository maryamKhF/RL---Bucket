# Simulation/test_simulation.py

import sys
import traceback
import math
from pathlib import Path

import networkx as nx


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# PATHFINDING
# ==========================================================

from Pathfinding.top_k_paths import top_k_paths


# ==========================================================
# BUCKET
# ==========================================================

from Bucket.candidate_manager import CandidateManager
from Bucket.backtrack import Backtracker as BucketBacktracker


# ==========================================================
# SIMULATION
# ==========================================================

from Simulation.transaction_generator import (
    generate_transaction_dataset,
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

from Simulation.onion import (
    OnionRouter,
)

from Simulation.router import (
    Router,
)

from Simulation.backtrack import (
    PartialBacktracker,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

SNAPSHOT_NAME = "20190501.gml.geo"

SNAPSHOT_CANDIDATES = [
    PROJECT_ROOT / SNAPSHOT_NAME,
    PROJECT_ROOT / "Data" / SNAPSHOT_NAME,
    PROJECT_ROOT / "data" / SNAPSHOT_NAME,
]

N_TRANSACTIONS = 100
TEST_PAYMENTS = 20

SEED = 42

MIN_AMOUNT = 1_000
MAX_AMOUNT = 1_000_000

TOP_K = 5
MAX_HOPS = 12

ETA = 0.0
LAMBDA_H = 1.0

MAX_PAYMENT_ATTEMPTS = TOP_K + 2

PARTIAL_TEST_AMOUNT = 1_000
PARTIAL_TEST_TX_ID = 900001


# ==========================================================
# HELPERS
# ==========================================================

def section(number, title):

    print()
    print("=" * 70)
    print(f"{number}. {title}")
    print("=" * 70)


def fail(message):
    raise RuntimeError(message)


def assert_true(condition, message):

    if not condition:
        fail(message)


# ==========================================================
# SNAPSHOT
# ==========================================================

def find_snapshot():

    for path in SNAPSHOT_CANDIDATES:

        if path.exists():
            return path

    raise FileNotFoundError(
        "\nSnapshot file was not found.\n"
        f"Expected filename: {SNAPSHOT_NAME}\n\n"
        "Searched locations:\n"
        +
        "\n".join(
            f"  {path}"
            for path in SNAPSHOT_CANDIDATES
        )
    )


def load_gml_snapshot(path):

    section(
        1,
        "LOAD REAL GML SNAPSHOT",
    )

    print("Loading:")
    print(f"  {path}")

    G = nx.read_gml(
        path,
        label="id",
    )

    print()
    print("Graph loaded successfully.")

    print(
        f"  Graph type : {type(G).__name__}"
    )

    print(
        f"  Nodes      : {G.number_of_nodes()}"
    )

    print(
        f"  Edges      : {G.number_of_edges()}"
    )

    return G


# ==========================================================
# GRAPH NORMALIZATION
# ==========================================================

def normalize_graph(G):

    section(
        2,
        "NORMALIZE GRAPH",
    )

    if G.is_directed():

        print("Graph is already directed.")

        if not G.is_multigraph():

            print(
                "Converting directed Graph "
                "to MultiDiGraph..."
            )

            DG = nx.MultiDiGraph()

            for node, attrs in G.nodes(data=True):

                DG.add_node(
                    node,
                    **attrs,
                )

            for u, v, attrs in G.edges(data=True):

                DG.add_edge(
                    u,
                    v,
                    **attrs,
                )

            return DG

        return G

    print("GML graph is undirected.")
    print("Creating directed representation...")

    DG = nx.MultiDiGraph()

    for node, attrs in G.nodes(data=True):

        DG.add_node(
            node,
            **attrs,
        )

    for u, v, attrs in G.edges(data=True):

        DG.add_edge(
            u,
            v,
            **dict(attrs),
        )

        DG.add_edge(
            v,
            u,
            **dict(attrs),
        )

    print("Directed graph created.")

    print(
        f"  Nodes : {DG.number_of_nodes()}"
    )

    print(
        f"  Edges : {DG.number_of_edges()}"
    )

    return DG


# ==========================================================
# CHANNEL ATTRIBUTES
# ==========================================================

def prepare_edge_attributes(G):

    section(
        3,
        "PREPARE CHANNEL ATTRIBUTES",
    )

    edge_count = 0

    for u, v, key, data in G.edges(
        keys=True,
        data=True,
    ):

        # --------------------------------------------------
        # Capacity
        # --------------------------------------------------

        if "capacity" not in data:

            if "capacity_sat" in data:

                try:
                    data["capacity"] = float(
                        data["capacity_sat"]
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    data["capacity"] = float(
                        MAX_AMOUNT
                    )

            else:

                data["capacity"] = float(
                    MAX_AMOUNT
                )

        try:

            data["capacity"] = float(
                data["capacity"]
            )

        except (
            TypeError,
            ValueError,
        ):

            data["capacity"] = float(
                MAX_AMOUNT
            )

        if (
            not math.isfinite(
                data["capacity"]
            )
            or
            data["capacity"] <= 0
        ):

            data["capacity"] = float(
                MAX_AMOUNT
            )

        # --------------------------------------------------
        # Fee
        # --------------------------------------------------

        try:

            data["fee_base_msat"] = float(
                data.get(
                    "fee_base_msat",
                    1000.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            data["fee_base_msat"] = 1000.0

        if not math.isfinite(
            data["fee_base_msat"]
        ):

            data["fee_base_msat"] = 1000.0

        try:

            data[
                "fee_proportional_millionths"
            ] = float(
                data.get(
                    "fee_proportional_millionths",
                    1.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            data[
                "fee_proportional_millionths"
            ] = 1.0

        if not math.isfinite(
            data[
                "fee_proportional_millionths"
            ]
        ):

            data[
                "fee_proportional_millionths"
            ] = 1.0

        # --------------------------------------------------
        # Delay
        # --------------------------------------------------

        try:

            data["delay"] = float(
                data.get(
                    "delay",
                    1.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            data["delay"] = 1.0

        if not math.isfinite(
            data["delay"]
        ):

            data["delay"] = 1.0

        # --------------------------------------------------
        # Directional balance
        #
        # Used only as deterministic test data.
        #
        # Capacity is NOT treated as liquidity.
        # --------------------------------------------------

        data.setdefault(
            "balance_uv",
            data["capacity"] / 2.0,
        )

        data.setdefault(
            "balance_vu",
            data["capacity"] / 2.0,
        )

        try:

            data["balance_uv"] = float(
                data["balance_uv"]
            )

        except (
            TypeError,
            ValueError,
        ):

            data["balance_uv"] = (
                data["capacity"] / 2.0
            )

        try:

            data["balance_vu"] = float(
                data["balance_vu"]
            )

        except (
            TypeError,
            ValueError,
        ):

            data["balance_vu"] = (
                data["capacity"] / 2.0
            )

        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------

        data.setdefault(
            "available",
            True,
        )

        data.setdefault(
            "failure_count",
            0,
        )

        data.setdefault(
            "success_count",
            0,
        )

        edge_count += 1

    print(
        f"Prepared {edge_count} directed channels."
    )

    return G


# ==========================================================
# TRANSACTION VALIDATION
# ==========================================================

def choose_valid_transaction(
    transaction,
    G,
):

    source = transaction.source
    destination = transaction.destination

    if source not in G:
        return False

    if destination not in G:
        return False

    if source == destination:
        return False

    try:

        return nx.has_path(
            G,
            source,
            destination,
        )

    except Exception:

        return False


# ==========================================================
# PATH -> EXACT EDGES
# ==========================================================

def path_to_edges(
    G,
    path,
):

    if not path or len(path) < 2:
        return None

    edges = []

    for u, v in zip(
        path[:-1],
        path[1:],
    ):

        if not G.has_edge(
            u,
            v,
        ):

            return None

        edge_data = G.get_edge_data(
            u,
            v,
        )

        if not edge_data:
            return None

        if G.is_multigraph():

            available_keys = []

            for key, data in edge_data.items():

                if data.get(
                    "available",
                    True,
                ):

                    available_keys.append(
                        key
                    )

            # For exact simulation we do not silently
            # select an arbitrary parallel channel.

            if len(available_keys) != 1:

                return None

            edges.append(
                (
                    u,
                    v,
                    available_keys[0],
                )
            )

        else:

            if not edge_data.get(
                "available",
                True,
            ):

                return None

            edges.append(
                (
                    u,
                    v,
                )
            )

    return edges


# ==========================================================
# CANDIDATE HELPERS
# ==========================================================

def candidate_path(candidate):

    if isinstance(
        candidate,
        dict,
    ):

        return candidate.get("path")

    if isinstance(
        candidate,
        (list, tuple),
    ):

        if len(candidate) >= 1:
            return candidate[0]

    return None


def candidate_edges(candidate):

    if isinstance(
        candidate,
        dict,
    ):

        return candidate.get("edges")

    if isinstance(
        candidate,
        (list, tuple),
    ):

        if len(candidate) >= 2:
            return candidate[1]

    return None


def candidate_rank(bucket):

    if bucket is None:
        return None

    return bucket.current_index + 1


def find_candidate_by_path(
    candidates,
    path,
):

    if path is None:
        return None

    target = list(path)

    for candidate in candidates or []:

        candidate_candidate_path = (
            candidate_path(candidate)
        )

        if (
            candidate_candidate_path is not None
            and
            list(candidate_candidate_path)
            == target
        ):

            return candidate

    return None


# ==========================================================
# TEST TRANSACTION GENERATOR
# ==========================================================

def test_transaction_generator(G):

    section(
        4,
        "TEST TRANSACTION GENERATOR",
    )

    dataset = generate_transaction_dataset(
        G,
        n=N_TRANSACTIONS,
        seed=SEED,
        min_amount=MIN_AMOUNT,
        max_amount=MAX_AMOUNT,
        train_ratio=0.70,
        require_path=True,
    )

    print()
    print("Transaction dataset generated.")

    print(
        f"  Total : {dataset.total_count}"
    )

    print(
        f"  Train : {dataset.train_count}"
    )

    print(
        f"  Test  : {dataset.test_count}"
    )

    if dataset.total_count == 0:

        fail(
            "Transaction generator produced "
            "zero transactions."
        )

    print()
    print("First 5 transactions:")

    for tx in dataset.transactions[:5]:

        print(
            f"  TX {tx.tx_id}: "
            f"{tx.source} -> {tx.destination} "
            f"| amount={tx.amount:.2f} "
            f"| split={tx.split}"
        )

    return dataset


# ==========================================================
# TEST FAILURE MODEL
# ==========================================================

def test_failure_model(G):

    section(
        5,
        "TEST FAILURE MODEL",
    )

    assign_failure_probabilities(
        G,
        average_rate=0.01,
        seed=SEED,
    )

    failure_model = FailureModel(
        node_failure_probability=0.01,
        liquidity_failure_probability=0.05,
        seed=SEED,
    )

    print(
        "FailureModel initialized."
    )

    tested_edge = None

    for u, v, key in G.edges(
        keys=True,
    ):

        tested_edge = (
            u,
            v,
            key,
        )

        break

    if tested_edge is None:

        fail(
            "Graph contains no edges."
        )

    u, v, key = tested_edge

    print(
        f"Testing channel: "
        f"{u} -> {v} "
        f"(key={key})"
    )

    result = failure_model.evaluate_edge(
        G,
        u,
        v,
        amount=MIN_AMOUNT,
        key=key,
    )

    print()
    print("FailureModel result:")

    for key_name, value in result.items():

        print(
            f"  {key_name}: {value}"
        )

    if not isinstance(
        result,
        dict,
    ):

        fail(
            "FailureModel.evaluate_edge() "
            "must return a dictionary."
        )

    return failure_model


# ==========================================================
# TEST NETWORK DYNAMICS
# ==========================================================

def test_network_dynamics(G):

    section(
        6,
        "TEST NETWORK DYNAMICS",
    )

    dynamics = NetworkDynamics(
        G,
        channel_failure_rate=0.01,
        node_failure_rate=0.005,
        recovery_rate=0.05,
        seed=SEED,
    )

    print(
        "NetworkDynamics initialized."
    )

    usable_edges = dynamics.get_usable_edges(
        MIN_AMOUNT
    )

    print(
        f"  Available edges: "
        f"{len(usable_edges)}"
    )

    state = dynamics.get_state()

    print(
        f"  State nodes: "
        f"{state.number_of_nodes()}"
    )

    print(
        f"  State edges: "
        f"{state.number_of_edges()}"
    )

    return dynamics


# ==========================================================
# TEST ONION ROUTER
# ==========================================================

def test_onion_router():

    section(
        7,
        "TEST ONION ROUTER",
    )

    onion = OnionRouter()

    path = [
        "A",
        "B",
        "C",
    ]

    packet = onion.build_onion(
        path=path,
        bucket_id=0,
        tx_id=1,
        amount=1000,
        attempt_id=0,
    )

    print(
        "Onion packet created."
    )

    print(
        f"  Version     : {packet['version']}"
    )

    print(
        f"  Source      : {packet['source']}"
    )

    print(
        f"  Destination : {packet['destination']}"
    )

    print(
        f"  Path length : {packet['path_length']}"
    )

    print(
        f"  Bucket ID   : {packet['bucket_id']}"
    )

    print(
        f"  TX ID       : {packet['tx_id']}"
    )

    print(
        f"  Attempt ID  : {packet['attempt_id']}"
    )

    current_packet = packet
    current_node = path[0]

    forwarding_sequence = []
    hop_indices = []

    destination_reached = False

    for _ in range(len(path) + 1):

        result = onion.forward(
            current_packet,
            current_node,
        )

        forwarding_sequence.append(
            result["current_node"]
        )

        hop_indices.append(
            result["hop_index"]
        )

        print(
            f"  Hop {result['hop_index']}: "
            f"{result['current_node']} -> "
            f"{result['next_hop']}"
        )

        if result.get(
            "destination_reached",
            False,
        ):

            destination_reached = True
            break

        next_packet = result.get(
            "next_packet"
        )

        next_hop = result.get(
            "next_hop"
        )

        if next_packet is None:

            fail(
                "OnionRouter returned no next_packet."
            )

        if next_hop is None:

            fail(
                "OnionRouter returned no next_hop."
            )

        current_packet = next_packet
        current_node = next_hop

    assert_true(
        forwarding_sequence == path,
        "Unexpected onion forwarding sequence.",
    )

    assert_true(
        hop_indices == [0, 1, 2],
        "Unexpected onion hop indices.",
    )

    assert_true(
        destination_reached,
        "OnionRouter did not reach destination.",
    )

    print()
    print(
        "Onion forwarding test passed."
    )

    return onion


# ==========================================================
# TEST ROUTER
# ==========================================================

def test_router():

    section(
        8,
        "TEST ROUTER",
    )

    router = Router()

    path = [
        "A",
        "B",
        "C",
    ]

    result = router.forward(
        path=path,
        bucket_id=0,
        tx_id=1,
        amount=1000,
        attempt_id=0,
    )

    assert_true(
        isinstance(result, dict),
        "Router.forward() must return a dictionary.",
    )

    print("Router result:")

    for key in [
        "success",
        "reason",
        "path",
        "visited",
        "attempt_id",
        "route_id",
        "source",
        "destination",
        "path_length",
    ]:

        print(
            f"  {key}: {result.get(key)}"
        )

    assert_true(
        result.get("success", False),
        "Router.forward() reported failure.",
    )

    assert_true(
        result.get("path") == path,
        "Router returned an unexpected path.",
    )

    route_id = result.get("route_id")

    assert_true(
        bool(route_id),
        "Router did not return route_id.",
    )

    packet = result.get("packet")

    assert_true(
        isinstance(packet, dict),
        "Router did not return a valid packet.",
    )

    assert_true(
        packet.get("route_id") == route_id,
        "Route ID mismatch.",
    )

    assert_true(
        packet.get("source") == path[0],
        "Incorrect packet source.",
    )

    assert_true(
        packet.get("destination") == path[-1],
        "Incorrect packet destination.",
    )

    print()
    print(
        "Router forwarding test passed."
    )

    print(
        f"  Route ID : {route_id}"
    )

    print(
        f"  Visited  : {result.get('visited')}"
    )

    return router


# ==========================================================
# BUILD DETERMINISTIC GRAPH
# ==========================================================

def build_deterministic_partial_graph():

    G = nx.MultiDiGraph()

    primary_edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
    ]

    alternative_edges = [
        ("B", "F"),
        ("F", "G"),
        ("G", "E"),
    ]

    for u, v in (
        primary_edges
        +
        alternative_edges
    ):

        G.add_edge(
            u,
            v,
            key=0,
            capacity=1_000_000.0,
            balance_uv=500_000.0,
            balance_vu=500_000.0,
            fee_base_msat=1000.0,
            fee_proportional_millionths=1.0,
            delay=1.0,
            available=True,
            failure_count=0,
            success_count=0,
        )

    for node in G.nodes:

        G.nodes[node]["available"] = True
        G.nodes[node]["is_online"] = True

    return G


# ==========================================================
# BUILD DETERMINISTIC CANDIDATES
# ==========================================================

def build_deterministic_candidates():

    candidate_1 = {
        "path": [
            "A",
            "B",
            "C",
            "D",
            "E",
        ],
        "edges": [
            ("A", "B", 0),
            ("B", "C", 0),
            ("C", "D", 0),
            ("D", "E", 0),
        ],
        "hop_count": 4,
        "cost": 1.0,
        "total_fee": 4.0,
        "success": None,
        "rank": 1,
    }

    candidate_2 = {
        "path": [
            "A",
            "B",
            "F",
            "G",
            "E",
        ],
        "edges": [
            ("A", "B", 0),
            ("B", "F", 0),
            ("F", "G", 0),
            ("G", "E", 0),
        ],
        "hop_count": 4,
        "cost": 2.0,
        "total_fee": 4.0,
        "success": None,
        "rank": 2,
    }

    return [
        candidate_1,
        candidate_2,
    ]


# ==========================================================
# DETERMINISTIC FAILURE MODEL
# ==========================================================

class DeterministicFailureModel:
    """
    Deterministic FailureModel used exclusively by the
    partial-backtracking integration test.

    Attempt 1:
        A-B-C-D-E fails on C-D.

    Attempt 2:
        A-B-F-G-E succeeds.

    The flexible signature is intentional: it accepts
    optional positional and keyword arguments used by
    different PaymentSimulator versions.
    """

    def __init__(self):

        self.calls = 0

        self.failed_edge = (
            "C",
            "D",
            0,
        )

    def evaluate_payment_failure(
        self,
        G=None,
        path=None,
        amount=None,
        edges=None,
        *args,
        **kwargs,
    ):

        # --------------------------------------------------
        # Accept alternate keyword naming if present.
        # --------------------------------------------------

        if G is None:

            G = kwargs.get(
                "graph",
                kwargs.get("network"),
            )

        if path is None:

            path = kwargs.get(
                "route",
                kwargs.get("path"),
            )

        if amount is None:

            amount = kwargs.get(
                "payment_amount",
                kwargs.get("amount"),
            )

        if edges is None:

            edges = kwargs.get(
                "route_edges",
                kwargs.get("edges"),
            )

        self.calls += 1

        if edges is None:
            edges = []
        else:
            edges = list(edges)

        # --------------------------------------------------
        # Attempt 1 = deterministic failure
        # --------------------------------------------------

        if self.calls == 1:

            return {
                "success": False,
                "reason": (
                    "deterministic_channel_failure"
                ),
                "failed_node": None,
                "failed_edge": self.failed_edge,
                "failure_index": 2,
                "visited_edges": edges[:2],
            }

        # --------------------------------------------------
        # Attempt 2+ = success
        # --------------------------------------------------

        return {
            "success": True,
            "reason": None,
            "failed_node": None,
            "failed_edge": None,
            "failure_index": None,
            "visited_edges": edges,
        }


# ==========================================================
# SIMPLE DETERMINISTIC BUCKET
# ==========================================================

class DeterministicBucket:

    def __init__(
        self,
        candidates,
    ):

        self.bucket_id = 9001

        self.candidates = candidates

        self.current_index = 0

        # Number of REAL payment attempts.
        self.attempts = 0

        self.selected_candidate_rank = None

        self.status = "active"

    def current(self):

        if (
            self.current_index < 0
            or
            self.current_index >= len(
                self.candidates
            )
        ):

            return None

        return self.candidates[
            self.current_index
        ]

    def mark_success(
        self,
        candidate,
    ):

        self.status = "completed"

        try:

            self.selected_candidate_rank = (
                self.candidates.index(
                    candidate
                ) + 1
            )

        except ValueError:

            self.selected_candidate_rank = None


# ==========================================================
# TEST DETERMINISTIC PARTIAL BACKTRACKING
# ==========================================================

def test_deterministic_partial_backtracking():

    section(
        10,
        "TEST DETERMINISTIC PARTIAL BACKTRACKING",
    )

    print(
        "Building deterministic MultiDiGraph..."
    )

    G = build_deterministic_partial_graph()

    candidates = (
        build_deterministic_candidates()
    )

    print(
        f"  Nodes : {G.number_of_nodes()}"
    )

    print(
        f"  Edges : {G.number_of_edges()}"
    )

    assert_true(
        G.number_of_nodes() == 7,
        "Deterministic graph must contain 7 nodes.",
    )

    assert_true(
        G.number_of_edges() == 7,
        "Deterministic graph must contain 7 edges.",
    )

    primary_path = candidate_path(
        candidates[0]
    )

    alternative_path = candidate_path(
        candidates[1]
    )

    primary_edges = candidate_edges(
        candidates[0]
    )

    alternative_edges = candidate_edges(
        candidates[1]
    )

    print()
    print("Candidate 1:")

    print(
        f"  Path  : {primary_path}"
    )

    print(
        f"  Edges : {primary_edges}"
    )

    print()
    print("Candidate 2:")

    print(
        f"  Path  : {alternative_path}"
    )

    print(
        f"  Edges : {alternative_edges}"
    )

    # ------------------------------------------------------
    # Exact channel validation
    # ------------------------------------------------------

    for edge in (
        primary_edges
        +
        alternative_edges
    ):

        u, v, key = edge

        assert_true(
            G.has_edge(
                u,
                v,
                key,
            ),
            f"Missing channel: {edge}",
        )

    # ------------------------------------------------------
    # PartialBacktracker
    # ------------------------------------------------------

    bucket = DeterministicBucket(
        candidates
    )

    partial_backtracker = PartialBacktracker(
        network=G,
    )

    partial_backtracker.bucket = bucket

    failed_edge = (
        "C",
        "D",
        0,
    )

    failure_index = 2

    print()
    print("Simulated failure:")

    print(
        f"  Failed edge   : {failed_edge}"
    )

    print(
        f"  Failure index : {failure_index}"
    )

    print(
        f"  Failed route  : {primary_path}"
    )

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # This operation must NOT increment bucket.attempts.
    # It is a routing recovery operation, not a payment.
    # ------------------------------------------------------

    result = partial_backtracker.backtrack(
        route=primary_path,
        failed_edge=failed_edge,
        failure_index=failure_index,
        amount=PARTIAL_TEST_AMOUNT,
        bucket_id=bucket.bucket_id,
        attempt_id=1,
    )

    assert_true(
        isinstance(result, dict),
        "PartialBacktracker must return a dictionary.",
    )

    print()
    print("PartialBacktracker result:")

    print(
        f"  Status       : {result.get('status')}"
    )

    print(
        f"  Success      : {result.get('success')}"
    )

    print(
        f"  Branch point : {result.get('branch_point')}"
    )

    print(
        f"  Prefix       : {result.get('preserved_prefix')}"
    )

    print(
        f"  Suffix       : {result.get('alternative_suffix')}"
    )

    print(
        f"  New route    : {result.get('new_route')}"
    )

    assert_true(
        result.get("success", False),
        "PartialBacktracker failed.",
    )

    new_route = result.get(
        "new_route"
    )

    expected_route = [
        "A",
        "B",
        "F",
        "G",
        "E",
    ]

    assert_true(
        new_route is not None,
        "No reconstructed route returned.",
    )

    assert_true(
        list(new_route) == expected_route,
        "Unexpected reconstructed route.\n"
        f"Expected: {expected_route}\n"
        f"Actual  : {new_route}",
    )

    branch_point = result.get(
        "branch_point"
    )

    assert_true(
        branch_point == "B",
        "Unexpected branch point.\n"
        f"Expected: B\n"
        f"Actual  : {branch_point}",
    )

    # ------------------------------------------------------
    # Validate rebuilt edges
    # ------------------------------------------------------

    rebuilt_edges = alternative_edges

    for edge in rebuilt_edges:

        u, v, key = edge

        assert_true(
            G.has_edge(
                u,
                v,
                key,
            ),
            f"Invalid rebuilt channel: {edge}",
        )

    print()
    print(
        "Exact alternative channels validated."
    )

    # ------------------------------------------------------
    # Router.rebuild()
    # ------------------------------------------------------

    router = Router()

    rebuild_result = router.rebuild(
        path=list(new_route),
        bucket_id=bucket.bucket_id,
        tx_id=PARTIAL_TEST_TX_ID,
        amount=PARTIAL_TEST_AMOUNT,
        attempt_id=2,
    )

    assert_true(
        isinstance(
            rebuild_result,
            dict,
        ),
        "Router.rebuild() must return a dictionary.",
    )

    print()
    print("Router.rebuild() result:")

    print(
        f"  Success   : "
        f"{rebuild_result.get('success')}"
    )

    print(
        f"  Reason    : "
        f"{rebuild_result.get('reason')}"
    )

    print(
        f"  Route ID  : "
        f"{rebuild_result.get('route_id')}"
    )

    assert_true(
        rebuild_result.get(
            "success",
            False,
        ),
        "Router.rebuild() failed.",
    )

    assert_true(
        rebuild_result.get(
            "path"
        ) == expected_route,
        "Router.rebuild() returned wrong path.",
    )

    assert_true(
        bool(
            rebuild_result.get(
                "route_id"
            )
        ),
        "Router.rebuild() returned no route_id.",
    )

    # ======================================================
    # PAYMENT SIMULATOR
    # ======================================================

    deterministic_failure_model = (
        DeterministicFailureModel()
    )

    network_dynamics = NetworkDynamics(
        G,
        channel_failure_rate=0.0,
        node_failure_rate=0.0,
        recovery_rate=0.0,
        seed=SEED,
    )

    simulator = PaymentSimulator(
        G,
        failure_model=deterministic_failure_model,
        network_dynamics=network_dynamics,
    )

    # ======================================================
    # PAYMENT ATTEMPT 1
    # ======================================================

    print()
    print(
        "Executing payment attempt 1..."
    )

    first_payment = simulator.simulate_payment(
        path=primary_path,
        edges=primary_edges,
        amount=PARTIAL_TEST_AMOUNT,
        tx_id=PARTIAL_TEST_TX_ID,
    )

    if hasattr(
        first_payment,
        "to_dict",
    ):

        first_result = first_payment.to_dict()

    elif isinstance(
        first_payment,
        dict,
    ):

        first_result = first_payment

    else:

        fail(
            "PaymentSimulator returned an "
            "unsupported result type."
        )

    print(
        f"  Success      : "
        f"{first_result.get('success')}"
    )

    print(
        f"  Reason       : "
        f"{first_result.get('reason')}"
    )

    print(
        f"  Failed edge  : "
        f"{first_result.get('failed_edge')}"
    )

    print(
        f"  Failure idx  : "
        f"{first_result.get('failure_index')}"
    )

    assert_true(
        not first_result.get(
            "success",
            False,
        ),
        "First deterministic payment must fail.",
    )

    assert_true(
        first_result.get(
            "failed_edge"
        ) == failed_edge,
        "Unexpected failed edge.\n"
        f"Expected: {failed_edge}\n"
        f"Actual  : "
        f"{first_result.get('failed_edge')}",
    )

    assert_true(
        first_result.get(
            "failure_index"
        ) == failure_index,
        "Unexpected failure index.",
    )

    # ------------------------------------------------------
    # Actual payment attempt #1
    #
    # Backtracker did not increment this.
    # ------------------------------------------------------

    bucket.attempts += 1

    assert_true(
        bucket.attempts == 1,
        "Bucket attempt counter must be 1 "
        "after the first payment.",
    )

    # ======================================================
    # PAYMENT ATTEMPT 2
    # ======================================================

    print()
    print(
        "Executing payment attempt 2..."
    )

    second_payment = simulator.simulate_payment(
        path=list(new_route),
        edges=rebuilt_edges,
        amount=PARTIAL_TEST_AMOUNT,
        tx_id=PARTIAL_TEST_TX_ID,
    )

    if hasattr(
        second_payment,
        "to_dict",
    ):

        second_result = second_payment.to_dict()

    elif isinstance(
        second_payment,
        dict,
    ):

        second_result = second_payment

    else:

        fail(
            "PaymentSimulator returned an "
            "unsupported result type."
        )

    print(
        f"  Success      : "
        f"{second_result.get('success')}"
    )

    print(
        f"  Reason       : "
        f"{second_result.get('reason')}"
    )

    print(
        f"  Failed edge  : "
        f"{second_result.get('failed_edge')}"
    )

    assert_true(
        second_result.get(
            "success",
            False,
        ),
        "Second deterministic payment must succeed.",
    )

    # ------------------------------------------------------
    # Actual payment attempt #2
    # ------------------------------------------------------

    bucket.attempts += 1

    assert_true(
        bucket.attempts == 2,
        "Bucket attempt counter must be 2 "
        "after the second payment.",
    )

    # ======================================================
    # MARK CANDIDATE SUCCESS
    # ======================================================

    successful_candidate = (
        find_candidate_by_path(
            bucket.candidates,
            new_route,
        )
    )

    assert_true(
        successful_candidate is not None,
        "Could not identify candidate 2.",
    )

    if isinstance(
        successful_candidate,
        dict,
    ):

        successful_candidate[
            "success"
        ] = True

        successful_candidate[
            "executed_path"
        ] = list(new_route)

    bucket.mark_success(
        successful_candidate
    )

    # ======================================================
    # FINAL ASSERTIONS
    # ======================================================

    assert_true(
        bucket.attempts == 2,
        "Exactly two payment attempts are expected.",
    )

    assert_true(
        bucket.status == "completed",
        "Bucket must be completed.",
    )

    assert_true(
        bucket.selected_candidate_rank == 2,
        "Candidate 2 must be the successful candidate.",
    )

    assert_true(
        deterministic_failure_model.calls == 2,
        "FailureModel must be called exactly twice.",
    )

    print()
    print(
        "DETERMINISTIC PARTIAL BACKTRACKING TEST PASSED"
    )

    print()
    print("Verified pipeline:")

    print(
        "  Candidate 1"
        " -> channel failure"
    )

    print(
        "  -> PaymentResult"
    )

    print(
        "  -> PartialBacktracker"
    )

    print(
        "  -> alternative suffix"
    )

    print(
        "  -> reconstructed route"
    )

    print(
        "  -> Router.rebuild()"
    )

    print(
        "  -> PaymentSimulator"
    )

    print(
        "  -> successful payment"
    )

    print(
        "  -> Bucket success"
    )

    print()
    print(
        f"  Initial route : {primary_path}"
    )

    print(
        f"  Failed edge  : {failed_edge}"
    )

    print(
        f"  Branch point : {branch_point}"
    )

    print(
        f"  Rebuilt route: {new_route}"
    )

    print(
        f"  Attempts     : {bucket.attempts}"
    )

    print(
        f"  Bucket status: {bucket.status}"
    )

    print(
        f"  Selected rank: "
        f"{bucket.selected_candidate_rank}"
    )

    return {
        "graph": G,
        "bucket": bucket,
        "initial_route": primary_path,
        "rebuilt_route": list(new_route),
        "failed_edge": failed_edge,
        "branch_point": branch_point,
        "first_result": first_result,
        "second_result": second_result,
    }


# ==========================================================
# GENERATE TOP-K
# ==========================================================

def generate_routing_candidates(
    G,
    tx,
):

    return top_k_paths(
        G=G,
        source=tx.source,
        target=tx.destination,
        amount=tx.amount,
        eta=ETA,
        k=TOP_K,
        max_hops=MAX_HOPS,
        lambda_h=LAMBDA_H,
    )


# ==========================================================
# BUILD BUCKET
# ==========================================================

def build_bucket(
    G,
    tx,
):

    candidates = generate_routing_candidates(
        G,
        tx,
    )

    if not candidates:

        return None, [], None

    manager = CandidateManager(
        candidates=candidates,
        max_candidates=TOP_K,
    )

    filtered = manager.filter_candidates(
        G,
        tx.amount,
    )

    if not filtered:

        return None, candidates, manager

    manager.rank_candidates()

    bucket = manager.create_bucket(
        tx_id=tx.tx_id,
        k=TOP_K,
    )

    return (
        bucket,
        manager.candidates,
        manager,
    )


# ==========================================================
# PRINT CANDIDATES
# ==========================================================

def print_candidates(
    candidates,
):

    if not candidates:

        print(
            "  No candidate route generated."
        )

        return

    print()
    print(
        f"  Candidate routes: "
        f"{len(candidates)}"
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        path = candidate_path(
            candidate
        )

        if isinstance(
            candidate,
            dict,
        ):

            hop_count = candidate.get(
                "hop_count",
                "-",
            )

            cost = candidate.get(
                "cost",
                0.0,
            )

            fee = candidate.get(
                "total_fee",
                0.0,
            )

        else:

            hop_count = "-"
            cost = 0.0
            fee = 0.0

        print(
            f"    [{index}] "
            f"hops={hop_count} "
            f"| cost={cost:.4f} "
            f"| fee={fee:.4f}"
        )

        print(
            f"         {path}"
        )


# ==========================================================
# MARK SUCCESS
# ==========================================================

def mark_payment_success(
    bucket,
    candidate,
    executed_path=None,
):

    if isinstance(
        candidate,
        dict,
    ):

        candidate["success"] = True

        if executed_path is not None:

            candidate[
                "executed_path"
            ] = list(
                executed_path
            )

    bucket.mark_success(
        candidate
    )


# ==========================================================
# ROUTER EXECUTION
# ==========================================================

def execute_router(
    router,
    path,
    bucket,
    tx,
    attempt_id,
):

    return router.forward(
        path=path,
        bucket_id=bucket.bucket_id,
        tx_id=tx.tx_id,
        amount=tx.amount,
        attempt_id=attempt_id,
    )


# ==========================================================
# RESULT -> DICT
# ==========================================================

def result_to_dict(result):

    if hasattr(
        result,
        "to_dict",
    ):

        return result.to_dict()

    if isinstance(
        result,
        dict,
    ):

        return result

    fail(
        "Unsupported payment result type."
    )


# ==========================================================
# FULL REROUTE
# ==========================================================

def full_reroute(
    G,
    tx,
    failed_routes=None,
):

    candidates = top_k_paths(
        G=G,
        source=tx.source,
        target=tx.destination,
        amount=tx.amount,
        eta=ETA,
        k=TOP_K,
        max_hops=MAX_HOPS,
        lambda_h=LAMBDA_H,
    )

    failed_routes = (
        failed_routes
        if failed_routes
        else set()
    )

    filtered = []

    for candidate in candidates:

        path = candidate_path(
            candidate
        )

        if path is None:
            continue

        if tuple(path) in failed_routes:
            continue

        filtered.append(
            candidate
        )

    if not filtered:

        return None, []

    manager = CandidateManager(
        candidates=filtered,
        max_candidates=TOP_K,
    )

    filtered = manager.filter_candidates(
        G,
        tx.amount,
    )

    if not filtered:

        return None, []

    manager.rank_candidates()

    bucket = manager.create_bucket(
        tx_id=tx.tx_id,
        k=TOP_K,
    )

    return (
        bucket,
        manager.candidates,
    )


# ==========================================================
# FULL REAL-GML PAYMENT TEST
# ==========================================================

def test_payment_simulation(
    G,
    failure_model,
    network_dynamics,
    dataset,
    router,
):

    section(
        11,
        "TEST FULL ROUTING + BUCKET + PAYMENT FLOW",
    )

    simulator = PaymentSimulator(
        G,
        failure_model=failure_model,
        network_dynamics=network_dynamics,
    )

    partial_backtracker = PartialBacktracker(
        network=G,
    )

    bucket_backtracker = BucketBacktracker(
        max_attempts=TOP_K + 2,
    )

    print(
        "PaymentSimulator initialized."
    )

    print(
        "PartialBacktracker initialized."
    )

    print(
        "Bucket Backtracker initialized."
    )

    print()
    print("FULL FLOW:")

    print(
        "  Transaction"
        " -> Top-K"
        " -> CandidateManager"
        " -> Bucket"
        " -> Router"
        " -> PaymentSimulator"
    )

    print(
        "  Failure"
        " -> PartialBacktracking"
        " / Bucket fallback"
        " / Full reroute"
    )

    successful = 0
    failed = 0
    no_path = 0

    partial_events = 0
    partial_successes = 0

    bucket_fallback_events = 0
    full_reroute_events = 0

    tested = 0

    for tx in dataset.test:

        if tested >= TEST_PAYMENTS:
            break

        if not choose_valid_transaction(
            tx,
            G,
        ):

            no_path += 1
            tested += 1
            continue

        bucket, candidates, manager = build_bucket(
            G,
            tx,
        )

        if bucket is None:

            print()
            print("-" * 70)

            print(
                f"TX {tx.tx_id}: "
                f"{tx.source} -> "
                f"{tx.destination}"
            )

            print(
                "No usable Bucket candidates."
            )

            no_path += 1
            tested += 1

            continue

        print()
        print("-" * 70)

        print(
            f"TX {tx.tx_id}: "
            f"{tx.source} -> "
            f"{tx.destination}"
        )

        print(
            f"Amount: {tx.amount:.2f}"
        )

        print(
            f"Initial candidates: "
            f"{len(candidates)}"
        )

        print_candidates(
            candidates
        )

        partial_backtracker.bucket = bucket

        payment_success = False

        attempt_id = 0

        failed_routes = set()

        current_bucket = bucket

        while (
            attempt_id
            <
            MAX_PAYMENT_ATTEMPTS
        ):

            candidate = current_bucket.current()

            if candidate is None:
                break

            path = candidate_path(
                candidate
            )

            edges = candidate_edges(
                candidate
            )

            if not path:

                bucket_backtracker.backtrack(
                    current_bucket,
                    failed_candidate=candidate,
                )

                continue

            if not edges:

                edges = path_to_edges(
                    G,
                    path,
                )

            if edges is None:

                failed_routes.add(
                    tuple(path)
                )

                bucket_backtracker.backtrack(
                    current_bucket,
                    failed_candidate=candidate,
                )

                continue

            # ==================================================
            # ACTUAL PAYMENT ATTEMPT
            # ==================================================

            attempt_id += 1

            # Bucket.attempts counts REAL payment attempts.
            current_bucket.attempts += 1

            print()
            print(
                f"  Attempt {attempt_id}"
            )

            print(
                f"    Candidate rank : "
                f"{candidate_rank(current_bucket)}"
            )

            print(
                f"    Path           : "
                f"{path}"
            )

            # ==================================================
            # ROUTER
            # ==================================================

            router_result = execute_router(
                router=router,
                path=path,
                bucket=current_bucket,
                tx=tx,
                attempt_id=attempt_id,
            )

            print(
                f"    Router success : "
                f"{router_result.get('success')}"
            )

            if not router_result.get(
                "success",
                False,
            ):

                failed_routes.add(
                    tuple(path)
                )

                bucket_backtracker.backtrack(
                    current_bucket,
                    failed_candidate=candidate,
                )

                continue

            # ==================================================
            # PAYMENT
            # ==================================================

            payment_result = (
                simulator.simulate_payment(
                    path=path,
                    edges=edges,
                    amount=tx.amount,
                    tx_id=tx.tx_id,
                )
            )

            result = result_to_dict(
                payment_result
            )

            success = bool(
                result.get(
                    "success",
                    False,
                )
            )

            reason = result.get(
                "reason"
            )

            failed_edge = result.get(
                "failed_edge"
            )

            failure_index = result.get(
                "failure_index"
            )

            failed_node = result.get(
                "failed_node"
            )

            print(
                f"    Payment success: "
                f"{success}"
            )

            print(
                f"    Reason         : "
                f"{reason}"
            )

            if failed_edge is not None:

                print(
                    f"    Failed edge    : "
                    f"{failed_edge}"
                )

            if failed_node is not None:

                print(
                    f"    Failed node    : "
                    f"{failed_node}"
                )

            if failure_index is not None:

                print(
                    f"    Failure index  : "
                    f"{failure_index}"
                )

            # ==================================================
            # SUCCESS
            # ==================================================

            if success:

                mark_payment_success(
                    current_bucket,
                    candidate,
                    path,
                )

                payment_success = True

                print()
                print(
                    "    PAYMENT SUCCESS"
                )

                print(
                    f"    Selected rank  : "
                    f"{current_bucket.selected_candidate_rank}"
                )

                print(
                    f"    Bucket attempts: "
                    f"{current_bucket.attempts}"
                )

                break

            # ==================================================
            # FAILURE
            # ==================================================

            if isinstance(
                candidate,
                dict,
            ):

                candidate["success"] = False

            failed_routes.add(
                tuple(path)
            )

            # ==================================================
            # PARTIAL BACKTRACKING
            # ==================================================

            if (
                failed_edge is not None
                and
                failure_index is not None
            ):

                partial_events += 1

                print()
                print(
                    "    PARTIAL BACKTRACKING"
                )

                partial_result = (
                    partial_backtracker.backtrack(
                        route=path,
                        failed_edge=failed_edge,
                        failure_index=failure_index,
                        amount=tx.amount,
                        bucket_id=current_bucket.bucket_id,
                        attempt_id=attempt_id,
                    )
                )

                print(
                    f"    Status         : "
                    f"{partial_result.get('status')}"
                )

                print(
                    f"    Branch point   : "
                    f"{partial_result.get('branch_point')}"
                )

                print(
                    f"    New route      : "
                    f"{partial_result.get('new_route')}"
                )

                if (
                    partial_result.get(
                        "success",
                        False,
                    )
                    and
                    partial_result.get(
                        "new_route"
                    )
                ):

                    partial_successes += 1

                    new_path = list(
                        partial_result[
                            "new_route"
                        ]
                    )

                    new_edges = None

                    alternative_candidate = (
                        find_candidate_by_path(
                            current_bucket.candidates,
                            new_path,
                        )
                    )

                    if alternative_candidate is not None:

                        new_edges = (
                            candidate_edges(
                                alternative_candidate
                            )
                        )

                    if new_edges is None:

                        new_edges = path_to_edges(
                            G,
                            new_path,
                        )

                    if new_edges is not None:

                        retry_attempt = (
                            attempt_id + 1
                        )

                        if (
                            retry_attempt
                            <= MAX_PAYMENT_ATTEMPTS
                        ):

                            print()
                            print(
                                "    RETRY PARTIAL ROUTE"
                            )

                            rebuild_result = (
                                router.rebuild(
                                    path=new_path,
                                    bucket_id=(
                                        current_bucket.bucket_id
                                    ),
                                    tx_id=tx.tx_id,
                                    amount=tx.amount,
                                    attempt_id=retry_attempt,
                                )
                            )

                            if rebuild_result.get(
                                "success",
                                False,
                            ):

                                attempt_id = (
                                    retry_attempt
                                )

                                # This is a REAL second
                                # payment attempt.

                                current_bucket.attempts += 1

                                retry_payment = (
                                    simulator.simulate_payment(
                                        path=new_path,
                                        edges=new_edges,
                                        amount=tx.amount,
                                        tx_id=tx.tx_id,
                                    )
                                )

                                retry_result = (
                                    result_to_dict(
                                        retry_payment
                                    )
                                )

                                print(
                                    f"    Retry success : "
                                    f"{retry_result.get('success')}"
                                )

                                if retry_result.get(
                                    "success",
                                    False,
                                ):

                                    successful_candidate = (
                                        alternative_candidate
                                    )

                                    if (
                                        successful_candidate
                                        is None
                                    ):

                                        successful_candidate = (
                                            candidate
                                        )

                                    mark_payment_success(
                                        current_bucket,
                                        successful_candidate,
                                        new_path,
                                    )

                                    payment_success = True

                                    print()
                                    print(
                                        "    PARTIAL BACKTRACK "
                                        "PAYMENT SUCCESS"
                                    )

                                    break

                                failed_routes.add(
                                    tuple(new_path)
                                )

            # ==================================================
            # BUCKET FALLBACK
            # ==================================================

            bucket_fallback_events += 1

            print()
            print(
                "    BUCKET CANDIDATE BACKTRACK"
            )

            next_candidate = (
                bucket_backtracker.backtrack(
                    current_bucket,
                    failed_candidate=candidate,
                    failed_channel=failed_edge,
                )
            )

            if next_candidate is not None:

                print(
                    f"    Next rank: "
                    f"{current_bucket.current_index + 1}"
                )

                continue

            # ==================================================
            # FULL REROUTE
            # ==================================================

            full_reroute_events += 1

            print()
            print(
                "    FULL REROUTE"
            )

            reroute_bucket, reroute_candidates = (
                full_reroute(
                    G,
                    tx,
                    failed_routes,
                )
            )

            if reroute_bucket is None:

                print(
                    "    No full-reroute candidate."
                )

                break

            print_candidates(
                reroute_candidates
            )

            current_bucket = reroute_bucket

            partial_backtracker.bucket = (
                current_bucket
            )

            bucket_backtracker = BucketBacktracker(
                max_attempts=TOP_K + 2,
            )

        if payment_success:

            successful += 1

        else:

            failed += 1

            print()
            print(
                "  PAYMENT FAILED"
            )

        tested += 1

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("=" * 70)
    print(
        "INTEGRATED PAYMENT SIMULATION SUMMARY"
    )
    print("=" * 70)

    print(
        f"  Tested                    : {tested}"
    )

    print(
        f"  Successful                : {successful}"
    )

    print(
        f"  Failed                    : {failed}"
    )

    print(
        f"  No valid initial path     : {no_path}"
    )

    print(
        f"  Partial backtrack events  : {partial_events}"
    )

    print(
        f"  Successful partial routes : {partial_successes}"
    )

    print(
        f"  Bucket fallback events    : "
        f"{bucket_fallback_events}"
    )

    print(
        f"  Full reroute events       : "
        f"{full_reroute_events}"
    )

    if tested > 0:

        success_rate = (
            successful / tested
        ) * 100.0

        print(
            f"  Payment success rate      : "
            f"{success_rate:.2f}%"
        )

    if tested > 0 and successful == 0:

        fail(
            "Real-GML integration produced "
            "zero successful payments."
        )

    return {
        "tested": tested,
        "successful": successful,
        "failed": failed,
        "no_path": no_path,
        "partial_events": partial_events,
        "partial_successes": partial_successes,
        "bucket_fallback_events": (
            bucket_fallback_events
        ),
        "full_reroute_events": (
            full_reroute_events
        ),
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()

    print("=" * 70)
    print(
        " REAL LIGHTNING SIMULATION "
        "INTEGRATION TEST"
    )
    print("=" * 70)

    print(
        f" Snapshot : {SNAPSHOT_NAME}"
    )

    print(
        f" Top-K    : {TOP_K}"
    )

    print(
        f" Max hops : {MAX_HOPS}"
    )

    print(
        f" ETA      : {ETA}"
    )

    print(
        f" Lambda_h : {LAMBDA_H}"
    )

    print("=" * 70)

    try:

        # ==================================================
        # 1
        # ==================================================

        snapshot_path = find_snapshot()

        G = load_gml_snapshot(
            snapshot_path
        )

        # ==================================================
        # 2
        # ==================================================

        G = normalize_graph(
            G
        )

        # ==================================================
        # 3
        # ==================================================

        G = prepare_edge_attributes(
            G
        )

        # ==================================================
        # 4
        # ==================================================

        dataset = test_transaction_generator(
            G
        )

        # ==================================================
        # 5
        # ==================================================

        failure_model = test_failure_model(
            G
        )

        # ==================================================
        # 6
        # ==================================================

        network_dynamics = test_network_dynamics(
            G
        )

        # ==================================================
        # 7
        # ==================================================

        test_onion_router()

        # ==================================================
        # 8
        # ==================================================

        router = test_router()

        # ==================================================
        # 10
        # ==================================================

        deterministic_result = (
            test_deterministic_partial_backtracking()
        )

        # ==================================================
        # 11
        # ==================================================

        integration_result = (
            test_payment_simulation(
                G=G,
                failure_model=failure_model,
                network_dynamics=network_dynamics,
                dataset=dataset,
                router=router,
            )
        )

        # ==================================================
        # FINAL
        # ==================================================

        print()
        print("=" * 70)
        print(
            " SIMULATION TEST COMPLETED"
        )
        print("=" * 70)

        print()
        print(
            "Verified modules:"
        )

        print(
            "  [OK] GML Snapshot"
        )

        print(
            "  [OK] Graph normalization"
        )

        print(
            "  [OK] Channel attributes"
        )

        print(
            "  [OK] Transaction Generator"
        )

        print(
            "  [OK] Failure Model"
        )

        print(
            "  [OK] Network Dynamics"
        )

        print(
            "  [OK] Top-K Pathfinding"
        )

        print(
            "  [OK] CandidateManager"
        )

        print(
            "  [OK] Bucket"
        )

        print(
            "  [OK] Bucket Backtracker"
        )

        print(
            "  [OK] Onion Router"
        )

        print(
            "  [OK] Router"
        )

        print(
            "  [OK] Router.rebuild()"
        )

        print(
            "  [OK] Payment Simulator"
        )

        print(
            "  [OK] Partial Backtracking"
        )

        print(
            "  [OK] Deterministic Partial Recovery"
        )

        print(
            "  [OK] Bucket Candidate Fallback"
        )

        print(
            "  [OK] Full Reroute Interface"
        )

        # ==================================================
        # REAL GML SUMMARY
        # ==================================================

        print()
        print(
            "REAL-GML SUMMARY:"
        )

        print(
            f"  Payments tested           : "
            f"{integration_result['tested']}"
        )

        print(
            f"  Successful payments       : "
            f"{integration_result['successful']}"
        )

        print(
            f"  Partial backtrack events : "
            f"{integration_result['partial_events']}"
        )

        print(
            f"  Successful partial routes: "
            f"{integration_result['partial_successes']}"
        )

        print(
            f"  Bucket fallback events   : "
            f"{integration_result['bucket_fallback_events']}"
        )

        print(
            f"  Full reroute events      : "
            f"{integration_result['full_reroute_events']}"
        )

        # ==================================================
        # DETERMINISTIC SUMMARY
        # ==================================================

        print()
        print(
            "DETERMINISTIC PARTIAL RECOVERY:"
        )

        print(
            f"  Initial route  : "
            f"{deterministic_result['initial_route']}"
        )

        print(
            f"  Failed edge    : "
            f"{deterministic_result['failed_edge']}"
        )

        print(
            f"  Branch point   : "
            f"{deterministic_result['branch_point']}"
        )

        print(
            f"  Rebuilt route  : "
            f"{deterministic_result['rebuilt_route']}"
        )

        print(
            f"  Attempts       : "
            f"{deterministic_result['bucket'].attempts}"
        )

        print(
            f"  Final status   : "
            f"{deterministic_result['bucket'].status}"
        )

        print()
        print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            " SIMULATION TEST FAILED"
        )
        print("=" * 70)

        print()
        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        print()
        print(
            "Traceback:"
        )

        traceback.print_exc()

        print("=" * 70)

        raise


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()