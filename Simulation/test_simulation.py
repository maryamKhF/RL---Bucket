# Simulation/test_simulation.py

import sys
import traceback
from pathlib import Path

import networkx as nx


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# PATHFINDING IMPORTS
# ==========================================================

from Pathfinding.top_k_paths import (
    top_k_paths,
)


# ==========================================================
# BUCKET IMPORTS
# ==========================================================

from Bucket.candidate_manager import (
    CandidateManager,
)


# ==========================================================
# SIMULATION IMPORTS
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

# ----------------------------------------------------------
# Routing configuration
# ----------------------------------------------------------

TOP_K = 5
MAX_HOPS = 12

# Current routing heuristic action.
#
# This value is the interface through which PPO can later
# provide its learned action.
#
# It is NOT claimed to be a PPO-generated value here.
ETA = 0.0

LAMBDA_H = 1.0


# ==========================================================
# COMMON PRINT HELPERS
# ==========================================================

def section(number, title):

    print()
    print("=" * 70)
    print(f"{number}. {title}")
    print("=" * 70)


def fail(message):

    raise RuntimeError(message)


# ==========================================================
# SNAPSHOT
# ==========================================================

def find_snapshot():
    """
    Find the real Lightning Network GML snapshot.
    """

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
        f"  Graph type : "
        f"{type(G).__name__}"
    )

    print(
        f"  Nodes      : "
        f"{G.number_of_nodes()}"
    )

    print(
        f"  Edges      : "
        f"{G.number_of_edges()}"
    )

    return G


# ==========================================================
# GRAPH NORMALIZATION
# ==========================================================

def normalize_graph(G):
    """
    Convert an undirected GML snapshot into a directed
    MultiDiGraph.

    Both directions are created because the source snapshot
    does not explicitly represent directional channels.
    """

    section(
        2,
        "NORMALIZE GRAPH",
    )

    if G.is_directed():

        print(
            "Graph is already directed."
        )

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

            for u, v, attrs in G.edges(
                data=True
            ):

                DG.add_edge(
                    u,
                    v,
                    **attrs,
                )

            G = DG

        return G

    print(
        "GML graph is undirected."
    )

    print(
        "Creating directed representation..."
    )

    DG = nx.MultiDiGraph()

    for node, attrs in G.nodes(
        data=True
    ):

        DG.add_node(
            node,
            **attrs,
        )

    for u, v, attrs in G.edges(
        data=True
    ):

        attrs_uv = dict(attrs)
        attrs_vu = dict(attrs)

        DG.add_edge(
            u,
            v,
            **attrs_uv,
        )

        DG.add_edge(
            v,
            u,
            **attrs_vu,
        )

    print(
        "Directed graph created."
    )

    print(
        f"  Nodes : "
        f"{DG.number_of_nodes()}"
    )

    print(
        f"  Edges : "
        f"{DG.number_of_edges()}"
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

        if data["capacity"] <= 0:

            data["capacity"] = float(
                MAX_AMOUNT
            )

        # --------------------------------------------------
        # Fees
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

        # --------------------------------------------------
        # Directional balances
        # --------------------------------------------------

        if "balance_uv" not in data:

            data["balance_uv"] = (
                data["capacity"] / 2.0
            )

        if "balance_vu" not in data:

            data["balance_vu"] = (
                data["capacity"] / 2.0
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
        # Availability / counters
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
    """
    Convert a node path into exact directed edges.

    For MultiDiGraph:

        (u, v, key)

    The first currently available channel is selected.
    """

    if not path:
        return None

    if len(path) < 2:
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

            selected_key = None

            # Prefer an available channel that can
            # currently be used.
            for key, data in edge_data.items():

                if data.get(
                    "available",
                    True,
                ):

                    selected_key = key
                    break

            # If every channel is unavailable,
            # do not silently invent another channel.
            if selected_key is None:
                return None

            edges.append(
                (
                    u,
                    v,
                    selected_key,
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
# TEST 1
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
    print(
        "Transaction dataset generated."
    )

    print(
        f"  Total : "
        f"{dataset.total_count}"
    )

    print(
        f"  Train : "
        f"{dataset.train_count}"
    )

    print(
        f"  Test  : "
        f"{dataset.test_count}"
    )

    if dataset.total_count == 0:

        fail(
            "Transaction generator produced "
            "zero transactions."
        )

    print()
    print(
        "First 5 transactions:"
    )

    for tx in dataset.transactions[:5]:

        print(
            f"  TX {tx.tx_id}: "
            f"{tx.source} -> {tx.destination} "
            f"| amount={tx.amount:.2f} "
            f"| split={tx.split}"
        )

    return dataset


# ==========================================================
# TEST 2
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
    print(
        "FailureModel result:"
    )

    for key_name, value in result.items():

        print(
            f"  {key_name}: {value}"
        )

    return failure_model


# ==========================================================
# TEST 3
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
# TEST 4
# ONION ROUTER
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

    bucket_id = 0
    tx_id = 1
    amount = 1000
    attempt_id = 0

    packet = onion.build_onion(
        path=path,
        bucket_id=bucket_id,
        tx_id=tx_id,
        amount=amount,
        attempt_id=attempt_id,
    )

    print(
        "Onion packet created."
    )

    print(
        f"  Version     : "
        f"{packet['version']}"
    )

    print(
        f"  Source      : "
        f"{packet['source']}"
    )

    print(
        f"  Destination : "
        f"{packet['destination']}"
    )

    print(
        f"  Path length : "
        f"{packet['path_length']}"
    )

    print(
        f"  Bucket ID   : "
        f"{packet['bucket_id']}"
    )

    print(
        f"  TX ID       : "
        f"{packet['tx_id']}"
    )

    print(
        f"  Attempt ID  : "
        f"{packet['attempt_id']}"
    )

    current_packet = packet
    current_node = path[0]

    forwarding_sequence = []
    hop_indices = []

    destination_reached = False

    max_steps = len(path) + 1

    for _ in range(max_steps):

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
                "OnionRouter returned no next_packet "
                "before reaching the destination."
            )

        if next_hop is None:

            fail(
                "OnionRouter returned next_hop=None "
                "before reaching the destination."
            )

        current_packet = next_packet
        current_node = next_hop

    expected_sequence = [
        "A",
        "B",
        "C",
    ]

    expected_hops = [
        0,
        1,
        2,
    ]

    if forwarding_sequence != expected_sequence:

        fail(
            "Unexpected onion forwarding sequence.\n"
            f"Expected: {expected_sequence}\n"
            f"Actual  : {forwarding_sequence}"
        )

    if hop_indices != expected_hops:

        fail(
            "Unexpected onion hop indices.\n"
            f"Expected: {expected_hops}\n"
            f"Actual  : {hop_indices}"
        )

    if not destination_reached:

        fail(
            "OnionRouter did not report "
            "destination_reached=True."
        )

    print()
    print(
        "Onion forwarding test passed."
    )

    print(
        f"  Forwarding sequence: "
        f"{' -> '.join(forwarding_sequence)}"
    )

    print(
        f"  Hop indices: "
        f"{hop_indices}"
    )

    print(
        f"  Destination reached: "
        f"{destination_reached}"
    )

    return onion


# ==========================================================
# TEST 5
# ROUTER
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

    if not isinstance(
        result,
        dict,
    ):

        fail(
            "Router.forward() must return a dictionary."
        )

    print(
        "Router result:"
    )

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
            f"  {key}: "
            f"{result.get(key)}"
        )

    if not result.get(
        "success",
        False,
    ):

        fail(
            "Router.forward() reported failure."
        )

    returned_path = result.get(
        "path"
    )

    if returned_path != path:

        fail(
            "Router returned an unexpected path.\n"
            f"Expected: {path}\n"
            f"Actual  : {returned_path}"
        )

    route_id = result.get(
        "route_id"
    )

    if not route_id:

        fail(
            "Router did not return route_id."
        )

    packet = result.get(
        "packet"
    )

    if not isinstance(
        packet,
        dict,
    ):

        fail(
            "Router did not return a valid onion packet."
        )

    packet_route_id = packet.get(
        "route_id"
    )

    if not packet_route_id:

        fail(
            "Onion packet does not contain route_id."
        )

    if packet_route_id != route_id:

        fail(
            "Router route_id and packet route_id "
            "do not match.\n"
            f"Router : {route_id}\n"
            f"Packet : {packet_route_id}"
        )

    if packet.get(
        "source"
    ) != path[0]:

        fail(
            "Router packet source is incorrect."
        )

    if packet.get(
        "destination"
    ) != path[-1]:

        fail(
            "Router packet destination is incorrect."
        )

    if packet.get(
        "path_length"
    ) != len(path):

        fail(
            "Router packet path_length is incorrect."
        )

    result2 = router.forward(
        path=path,
        bucket_id=0,
        tx_id=2,
        amount=1000,
        attempt_id=0,
    )

    route_id_2 = result2.get(
        "route_id"
    )

    if route_id_2 != route_id:

        fail(
            "route_id is not deterministic for the same route.\n"
            f"First : {route_id}\n"
            f"Second: {route_id_2}"
        )

    print()
    print(
        "Router integration test passed."
    )

    print(
        f"  Route ID       : "
        f"{route_id}"
    )

    print(
        f"  Source         : "
        f"{packet['source']}"
    )

    print(
        f"  Destination    : "
        f"{packet['destination']}"
    )

    print(
        f"  Path length    : "
        f"{packet['path_length']}"
    )

    return router


# ==========================================================
# ROUTING CANDIDATES
# ==========================================================

def generate_routing_candidates(
    G,
    tx,
):
    """
    Generate Top-K candidate routes using the actual
    Pathfinding module.

    Routing flow:

        Network
           |
           v
        eta
           |
           v
    Adaptive Heuristic
           |
           v
    Modified Cost
           |
           v
       Top-K Paths
    """

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

    return candidates


# ==========================================================
# BUCKET CREATION
# ==========================================================

def build_bucket(
    G,
    tx,
):
    """
    Generate, filter and rank candidates, then create Bucket.

    CandidateManager is deliberately used before Bucket.
    """

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

    return bucket, manager.candidates, manager


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

        path = candidate.get(
            "path",
            [],
        )

        print(
            f"    [{index}] "
            f"rank={index} "
            f"| hops={candidate.get('hop_count')} "
            f"| cost={candidate.get('cost', 0.0):.4f} "
            f"| fee={candidate.get('total_fee', 0.0):.4f} "
            f"| reliability={candidate.get('reliability', 0.0):.6f}"
        )

        print(
            f"         "
            f"{path}"
        )


# ==========================================================
# UPDATE CANDIDATE SUCCESS STATUS
# ==========================================================

def mark_candidate_success(
    bucket,
    candidate,
    success,
):
    """
    Update the candidate's runtime success status.

    This does not alter the routing cost.
    """

    if isinstance(
        candidate,
        dict,
    ):

        candidate["success"] = bool(
            success
        )

    if success:

        bucket.mark_success(
            candidate
        )


# ==========================================================
# FULL REROUTE
# ==========================================================

def full_reroute(
    G,
    tx,
    failed_routes=None,
):
    """
    Generate a fresh candidate set from the current network state.

    This is the fallback when Bucket cannot provide a valid
    partial-backtracking suffix.
    """

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

    if failed_routes:

        filtered = []

        for candidate in candidates:

            path = tuple(
                candidate.get(
                    "path",
                    [],
                )
            )

            if path in failed_routes:
                continue

            filtered.append(
                candidate
            )

        candidates = filtered

    manager = CandidateManager(
        candidates=candidates,
        max_candidates=TOP_K,
    )

    candidates = manager.filter_candidates(
        G,
        tx.amount,
    )

    manager.rank_candidates()

    bucket = manager.create_bucket(
        tx_id=tx.tx_id,
        k=TOP_K,
    )

    return bucket, candidates


# ==========================================================
# TEST 6
# INTEGRATED PAYMENT SIMULATION
# ==========================================================

def test_payment_simulation(
    G,
    failure_model,
    network_dynamics,
    dataset,
):

    section(
        9,
        "TEST INTEGRATED ROUTING + BUCKET + PAYMENT SIMULATION",
    )

    simulator = PaymentSimulator(
        G,
        failure_model=failure_model,
        network_dynamics=network_dynamics,
    )

    backtracker = PartialBacktracker(
        network=G,
    )

    print(
        "PaymentSimulator initialized."
    )

    print(
        "PartialBacktracker initialized."
    )

    print()
    print(
        "Routing pipeline:"
    )

    print(
        "  Transaction"
    )

    print(
        "      -> Top-K Pathfinding"
    )

    print(
        "      -> CandidateManager"
    )

    print(
        "      -> Bucket"
    )

    print(
        "      -> PaymentSimulator"
    )

    print(
        "      -> Failure"
    )

    print(
        "      -> Partial Backtracking"
    )

    print(
        "      -> Bucket Alternative"
    )

    print(
        "      -> Retry"
    )

    print(
        "      -> Full Reroute if required"
    )

    successful = 0
    failed = 0
    no_path = 0

    partial_backtrack_count = 0
    partial_backtrack_success = 0
    full_reroute_count = 0

    tested = 0

    for tx in dataset.test:

        if tested >= TEST_PAYMENTS:
            break

        # --------------------------------------------------
        # Validate transaction
        # --------------------------------------------------

        if not choose_valid_transaction(
            tx,
            G,
        ):

            no_path += 1
            continue

        # --------------------------------------------------
        # Generate initial candidates
        # --------------------------------------------------

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
                f"Amount: "
                f"{tx.amount:.2f}"
            )

            print(
                "No usable candidate routes."
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
            f"Amount: "
            f"{tx.amount:.2f}"
        )

        print(
            f"Initial candidates: "
            f"{len(candidates)}"
        )

        print_candidates(
            candidates
        )

        # --------------------------------------------------
        # Create a Backtracker using the real Bucket.
        # --------------------------------------------------

        backtracker.bucket = bucket

        # --------------------------------------------------
        # Attempt loop
        # --------------------------------------------------

        attempt_id = 0

        failed_routes = set()

        payment_success = False

        current_bucket = bucket

        max_attempts = (
            max(
                1,
                TOP_K
            )
            + 2
        )

        while (
            attempt_id < max_attempts
        ):

            # --------------------------------------------------
            # Current candidate
            # --------------------------------------------------

            candidate = current_bucket.current()

            if candidate is None:
                break

            if isinstance(
                candidate,
                dict,
            ):

                path = candidate.get(
                    "path"
                )

                candidate_edges = candidate.get(
                    "edges"
                )

            else:

                # Legacy compatibility
                try:

                    path = candidate[0]
                    candidate_edges = candidate[1]

                except (
                    TypeError,
                    IndexError,
                ):

                    path = None
                    candidate_edges = None

            if not path:

                current_bucket.next_candidate()
                continue

            # --------------------------------------------------
            # Prefer exact edges stored by Top-K Pathfinding.
            #
            # If they are unavailable, resolve the current
            # route against the live graph.
            # --------------------------------------------------

            edges = candidate_edges

            if not edges:

                edges = path_to_edges(
                    G,
                    path,
                )

            if edges is None:

                failed_routes.add(
                    tuple(path)
                )

                current_bucket.backtrack(
                    failed_candidate=candidate
                )

                continue

            attempt_id += 1

            print()
            print(
                f"  Attempt {attempt_id}"
            )

            print(
                f"    Candidate rank : "
                f"{current_bucket.current_index + 1}"
            )

            print(
                f"    Path           : "
                f"{path}"
            )

            print(
                f"    Edges          : "
                f"{edges}"
            )

            # --------------------------------------------------
            # Execute payment
            # --------------------------------------------------

            result = simulator.simulate_payment(
                path=path,
                edges=edges,
                amount=tx.amount,
                tx_id=tx.tx_id,
            )

            if hasattr(
                result,
                "to_dict",
            ):

                result_dict = result.to_dict()

            else:

                result_dict = result

            success = result_dict.get(
                "success",
                False,
            )

            reason = result_dict.get(
                "reason"
            )

            failed_edge = result_dict.get(
                "failed_edge"
            )

            failed_node = result_dict.get(
                "failed_node"
            )

            failure_index = result_dict.get(
                "failure_index"
            )

            print(
                f"    Success        : "
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

                mark_candidate_success(
                    current_bucket,
                    candidate,
                    True,
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
                    f"    Attempts       : "
                    f"{attempt_id}"
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

            # --------------------------------------------------
            # If failure information is unavailable,
            # move directly to next Bucket candidate.
            # --------------------------------------------------

            if (
                failed_edge is None
                and
                failure_index is None
            ):

                print(
                    "    No failure-edge information."
                )

                current_bucket.backtrack(
                    failed_candidate=candidate
                )

                continue

            # ==================================================
            # PARTIAL BACKTRACKING
            # ==================================================

            partial_backtrack_count += 1

            print()
            print(
                "    PARTIAL BACKTRACKING"
            )

            backtrack_result = backtracker.backtrack(
                route=path,
                failed_edge=failed_edge,
                failure_index=failure_index,
                amount=tx.amount,
                bucket_id=current_bucket.bucket_id,
                attempt_id=attempt_id,
            )

            status = backtrack_result.get(
                "status"
            )

            print(
                f"    Status         : "
                f"{status}"
            )

            print(
                f"    Branch point   : "
                f"{backtrack_result.get('branch_point')}"
            )

            print(
                f"    Failed edge    : "
                f"{backtrack_result.get('failed_edge')}"
            )

            print(
                f"    Prefix         : "
                f"{backtrack_result.get('preserved_prefix')}"
            )

            print(
                f"    Suffix         : "
                f"{backtrack_result.get('alternative_suffix')}"
            )

            print(
                f"    New route      : "
                f"{backtrack_result.get('new_route')}"
            )

            # --------------------------------------------------
            # Alternative suffix found
            # --------------------------------------------------

            if (
                status == "alternative_found"
                and
                backtrack_result.get(
                    "success",
                    False,
                )
            ):

                partial_backtrack_success += 1

                new_path = backtrack_result.get(
                    "new_route"
                )

                if not new_path:

                    print(
                        "    Backtracker returned "
                        "an empty route."
                    )

                    current_bucket.backtrack(
                        failed_candidate=candidate
                    )

                    continue

                new_edges = path_to_edges(
                    G,
                    new_path,
                )

                if new_edges is None:

                    print(
                        "    Alternative route "
                        "is no longer executable."
                    )

                    current_bucket.backtrack(
                        failed_candidate=candidate
                    )

                    continue

                # --------------------------------------------------
                # Execute the new partial-backtracked route.
                # --------------------------------------------------

                retry_attempt = attempt_id + 1

                print()
                print(
                    "    RETRY ALTERNATIVE ROUTE"
                )

                print(
                    f"    Retry attempt : "
                    f"{retry_attempt}"
                )

                print(
                    f"    Route         : "
                    f"{new_path}"
                )

                retry_result = simulator.simulate_payment(
                    path=new_path,
                    edges=new_edges,
                    amount=tx.amount,
                    tx_id=tx.tx_id,
                )

                if hasattr(
                    retry_result,
                    "to_dict",
                ):

                    retry_dict = (
                        retry_result.to_dict()
                    )

                else:

                    retry_dict = retry_result

                retry_success = retry_dict.get(
                    "success",
                    False,
                )

                retry_reason = retry_dict.get(
                    "reason"
                )

                print(
                    f"    Retry success : "
                    f"{retry_success}"
                )

                print(
                    f"    Retry reason  : "
                    f"{retry_reason}"
                )

                if retry_success:

                    payment_success = True

                    print()
                    print(
                        "    PARTIAL BACKTRACKING "
                        "PAYMENT SUCCESS"
                    )

                    break

                # --------------------------------------------------
                # Alternative suffix also failed.
                #
                # Move Bucket forward and allow another candidate.
                # --------------------------------------------------

                print(
                    "    Alternative route failed."
                )

                current_bucket.backtrack(
                    failed_candidate=candidate
                )

                continue

            # ==================================================
            # NO VALID BUCKET ALTERNATIVE
            # ==================================================

            print()
            print(
                "    No valid Bucket alternative."
            )

            print(
                "    Full reroute required."
            )

            full_reroute_count += 1

            reroute_bucket, reroute_candidates = (
                full_reroute(
                    G,
                    tx,
                    failed_routes=failed_routes,
                )
            )

            if reroute_bucket is None:

                print(
                    "    Full reroute produced "
                    "no usable route."
                )

                break

            print()
            print(
                "    FULL REROUTE CANDIDATES"
            )

            print_candidates(
                reroute_candidates
            )

            # --------------------------------------------------
            # Replace current Bucket.
            # --------------------------------------------------

            current_bucket = reroute_bucket

            backtracker.bucket = current_bucket

            # --------------------------------------------------
            # Continue from the new Bucket.
            # --------------------------------------------------

        # ==================================================
        # Transaction result
        # ==================================================

        if payment_success:

            successful += 1

        else:

            failed += 1

            print()
            print(
                "  PAYMENT FAILED"
            )

        tested += 1

    # ==========================================================
    # SUMMARY
    # ==========================================================

    print()
    print("=" * 70)

    print(
        "INTEGRATED PAYMENT SIMULATION SUMMARY"
    )

    print("=" * 70)

    print(
        f"  Tested                    : "
        f"{tested}"
    )

    print(
        f"  Successful                : "
        f"{successful}"
    )

    print(
        f"  Failed                    : "
        f"{failed}"
    )

    print(
        f"  No valid initial path     : "
        f"{no_path}"
    )

    print(
        f"  Partial backtrack events  : "
        f"{partial_backtrack_count}"
    )

    print(
        f"  Successful partial routes : "
        f"{partial_backtrack_success}"
    )

    print(
        f"  Full reroute events       : "
        f"{full_reroute_count}"
    )

    if tested > 0:

        success_rate = (
            successful / tested
        ) * 100.0

        print(
            f"  Payment success rate      : "
            f"{success_rate:.2f}%"
        )

    if partial_backtrack_count > 0:

        backtrack_rate = (
            partial_backtrack_success
            /
            partial_backtrack_count
        ) * 100.0

        print(
            f"  Partial backtrack success : "
            f"{backtrack_rate:.2f}%"
        )

    return simulator


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
        f" Snapshot: "
        f"{SNAPSHOT_NAME}"
    )

    print(
        f" Top-K: "
        f"{TOP_K}"
    )

    print(
        f" Max hops: "
        f"{MAX_HOPS}"
    )

    print(
        f" ETA: "
        f"{ETA}"
    )

    print(
        f" Lambda_h: "
        f"{LAMBDA_H}"
    )

    print("=" * 70)

    try:

        # --------------------------------------------------
        # 1. Snapshot
        # --------------------------------------------------

        snapshot_path = find_snapshot()

        G = load_gml_snapshot(
            snapshot_path
        )

        # --------------------------------------------------
        # 2. Normalize
        # --------------------------------------------------

        G = normalize_graph(
            G
        )

        # --------------------------------------------------
        # 3. Channel attributes
        # --------------------------------------------------

        G = prepare_edge_attributes(
            G
        )

        # --------------------------------------------------
        # 4. Transactions
        # --------------------------------------------------

        dataset = test_transaction_generator(
            G
        )

        # --------------------------------------------------
        # 5. Failure model
        # --------------------------------------------------

        failure_model = test_failure_model(
            G
        )

        # --------------------------------------------------
        # 6. Network dynamics
        # --------------------------------------------------

        network_dynamics = test_network_dynamics(
            G
        )

        # --------------------------------------------------
        # 7. Onion
        # --------------------------------------------------

        test_onion_router()

        # --------------------------------------------------
        # 8. Router
        # --------------------------------------------------

        test_router()

        # --------------------------------------------------
        # 9. Integrated routing/payment simulation
        # --------------------------------------------------

        test_payment_simulation(
            G,
            failure_model,
            network_dynamics,
            dataset,
        )

        # --------------------------------------------------
        # Final success
        # --------------------------------------------------

        print()
        print("=" * 70)

        print(
            " SIMULATION TEST COMPLETED"
        )

        print("=" * 70)

        print()
        print(
            "Modules tested:"
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
            "  [OK] Onion Router"
        )

        print(
            "  [OK] Router"
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
            "  [OK] Payment Simulator"
        )

        print(
            "  [OK] Partial Backtracking"
        )

        print(
            "  [OK] Full Reroute Fallback"
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
            f"Error type: "
            f"{type(exc).__name__}"
        )

        print(
            f"Error: "
            f"{exc}"
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