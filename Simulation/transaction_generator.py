# Simulation/transaction_generator.py

"""
Transaction generation for RL + Bucket + Lightning routing.

Pipeline:

    LN Snapshot
        |
        v
    Graph Construction
        |
        v
    Transaction Generation
        |
        v
    Train/Test Split
        |
        v
    Neighborhood Observation
        |
        v
    PPO + Routing


Each transaction is:

    t = (u, v, A_t)

where:

    u   = source
    v   = destination
    A_t = payment amount

This module is responsible ONLY for generating payment requests.

It does NOT:

    - find routes
    - run Dijkstra/LND
    - run PPO
    - access Bucket
    - simulate failures
    - modify channel balances
    - execute payments
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx


# ==============================================================
# Transaction
# ==============================================================

@dataclass
class Transaction:
    """
    Represents one Lightning payment request.

    Core transaction:

        (source, destination, amount)

    Runtime fields are kept separate from generated
    transaction attributes.
    """

    tx_id: int

    source: Any

    destination: Any

    amount: float

    timestamp: int

    # ----------------------------------------------------------
    # Dataset information
    # ----------------------------------------------------------

    split: str = "train"
    # train | test

    # ----------------------------------------------------------
    # Runtime information
    # ----------------------------------------------------------

    status: str = "PENDING"
    # PENDING | SUCCESS | FAILED

    attempts: int = 0

    path: Optional[List[Any]] = None

    fee: float = 0.0

    delay: float = 0.0

    carbon: float = 0.0

    failure_reason: Optional[str] = None

    failed_edge: Optional[Tuple] = None

    # ----------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # ==========================================================
    # Reset runtime state
    # ==========================================================

    def reset_runtime(self):
        """
        Reset simulation-related fields.
        """

        self.status = "PENDING"

        self.attempts = 0

        self.path = None

        self.fee = 0.0

        self.delay = 0.0

        self.carbon = 0.0

        self.failure_reason = None

        self.failed_edge = None


# ==============================================================
# Transaction Dataset
# ==============================================================

@dataclass
class TransactionDataset:
    """
    Container for training and testing transactions.

    Default split:

        70% train
        30% test
    """

    train: List[Transaction]

    test: List[Transaction]

    # ==========================================================
    # All transactions
    # ==========================================================

    @property
    def all(self) -> List[Transaction]:

        return self.train + self.test

    # ==========================================================
    # Sizes
    # ==========================================================

    @property
    def train_size(self) -> int:

        return len(self.train)

    @property
    def test_size(self) -> int:

        return len(self.test)

    @property
    def total_size(self) -> int:

        return (
            len(self.train)
            +
            len(self.test)
        )

    # ----------------------------------------------------------
    # Compatibility aliases
    # ----------------------------------------------------------

    @property
    def transactions(self) -> List[Transaction]:

        return self.all

    @property
    def total_count(self) -> int:

        return self.total_size

    @property
    def train_count(self) -> int:

        return self.train_size

    @property
    def test_count(self) -> int:

        return self.test_size

    # ==========================================================
    # Summary
    # ==========================================================

    def summary(self):

        return {

            "total":
                self.total_size,

            "train":
                self.train_size,

            "test":
                self.test_size,

            "train_ratio":
                (
                    self.train_size
                    /
                    self.total_size
                    if self.total_size
                    else 0.0
                ),

            "test_ratio":
                (
                    self.test_size
                    /
                    self.total_size
                    if self.total_size
                    else 0.0
                )

        }


# ==============================================================
# Connected Components
# ==============================================================

def _build_connectivity_graph(G):
    """
    Build the graph used ONLY for determining whether two
    nodes belong to the same connectivity component.

    For directed Lightning graphs, the undirected projection
    is used here.

    Important:
        This does NOT perform payment routing.

    Actual routing remains the responsibility of the
    Pathfinding/RL/Bucket layer.
    """

    if G is None:

        raise ValueError(
            "G must not be None."
        )

    if G.is_directed():

        return G.to_undirected()

    return G


def _get_connected_components(G):
    """
    Return connected components with at least two nodes.

    Components are sorted from largest to smallest.

    Returns
    -------
    list[list]
        Each item contains the nodes of one component.
    """

    connectivity_graph = _build_connectivity_graph(G)

    components = [

        list(component)

        for component in nx.connected_components(
            connectivity_graph
        )

        if len(component) >= 2

    ]

    components.sort(
        key=len,
        reverse=True
    )

    return components


# ==============================================================
# Component Selection
# ==============================================================

def _select_component(
    components,
    rng,
    prefer_largest=True
):
    """
    Select a connected component from which source and
    destination will be sampled.

    By default the largest component is used.

    This is appropriate for a real Lightning Network snapshot
    because it avoids generating transactions between isolated
    or tiny disconnected components.
    """

    if not components:

        raise RuntimeError(
            "No connected component with at least "
            "two nodes was found."
        )

    if prefer_largest:

        return components[0]

    # ----------------------------------------------------------
    # Size-weighted component selection
    # ----------------------------------------------------------

    sizes = np.asarray(
        [len(component) for component in components],
        dtype=float
    )

    probabilities = (
        sizes
        /
        sizes.sum()
    )

    index = rng.choice(
        len(components),
        p=probabilities
    )

    return components[int(index)]


# ==============================================================
# Generate transactions
# ==============================================================

def generate_transactions(
    G,
    n: int = 10_000,
    seed: int = 42,
    min_amount: float = 1_000,
    max_amount: float = 1_000_000,
    train_ratio: float = 0.70,
    timestamp_start: int = 0,
    require_path: bool = True,
    prefer_largest_component: bool = True,
) -> List[Transaction]:
    """
    Generate Lightning payment requests.

    Parameters
    ----------
    G : networkx.Graph
        Lightning Network graph.

    n : int
        Number of transactions.

    seed : int
        Random seed.

    min_amount : float
        Minimum payment amount.

    max_amount : float
        Maximum payment amount.

    train_ratio : float
        Training fraction.

        Default = 0.70

    timestamp_start : int
        Starting transaction timestamp/index.

    require_path : bool
        If True, source and destination are selected from
        the same connected component.

    prefer_largest_component : bool
        If True, transactions are generated from the largest
        connected component.

        This is the recommended setting for a real Lightning
        Network snapshot.

    Returns
    -------
    list[Transaction]
        Generated transactions.

    Notes
    -----
    This function does not perform actual route finding.

    Connectivity is used only to avoid obviously impossible
    source/destination pairs caused by disconnected graph
    components.
    """

    # ==========================================================
    # Validation
    # ==========================================================

    if G is None:

        raise ValueError(
            "G must not be None."
        )

    if n <= 0:

        raise ValueError(
            "n must be greater than zero."
        )

    if min_amount <= 0:

        raise ValueError(
            "min_amount must be greater than zero."
        )

    if max_amount < min_amount:

        raise ValueError(
            "max_amount must be greater than or equal "
            "to min_amount."
        )

    if not 0.0 < train_ratio < 1.0:

        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not isinstance(
        timestamp_start,
        (int, np.integer)
    ):

        raise ValueError(
            "timestamp_start must be an integer."
        )

    # ==========================================================
    # Graph validation
    # ==========================================================

    if G.number_of_nodes() < 2:

        raise ValueError(
            "Graph must contain at least two nodes."
        )

    # ==========================================================
    # Random generator
    # ==========================================================

    rng = np.random.default_rng(
        seed
    )

    # ==========================================================
    # Determine sampling pool
    # ==========================================================

    if require_path:

        components = _get_connected_components(
            G
        )

        if not components:

            raise RuntimeError(
                "The graph contains no connected component "
                "with at least two nodes."
            )

        largest_component = components[0]

        # ------------------------------------------------------
        # Diagnostic information
        # ------------------------------------------------------

        print(
            f"[TransactionGenerator] "
            f"Connected components: {len(components)}"
        )

        print(
            f"[TransactionGenerator] "
            f"Largest component: {len(largest_component)} nodes"
        )

        if len(components) > 1:

            print(
                f"[TransactionGenerator] "
                f"Using {'largest' if prefer_largest_component else 'weighted'} "
                f"component for transaction generation."
            )

        # ------------------------------------------------------
        # Recommended behavior:
        # sample from largest connected component.
        # ------------------------------------------------------

        if prefer_largest_component:

            sampling_nodes = largest_component

            sampling_components = [
                largest_component
            ]

        else:

            sampling_nodes = None

            sampling_components = components

    else:

        sampling_nodes = list(
            G.nodes
        )

        sampling_components = None

    # ==========================================================
    # Additional validation
    # ==========================================================

    if require_path and prefer_largest_component:

        if len(sampling_nodes) < 2:

            raise RuntimeError(
                "The selected connected component contains "
                "fewer than two nodes."
            )

    elif not require_path:

        if len(sampling_nodes) < 2:

            raise ValueError(
                "Graph must contain at least two nodes."
            )

    # ==========================================================
    # Generate transactions
    # ==========================================================

    transactions: List[Transaction] = []

    # ----------------------------------------------------------
    # We no longer need a huge retry loop.
    #
    # source/destination are sampled directly from a valid
    # connected component.
    # ----------------------------------------------------------

    for tx_id in range(n):

        # ======================================================
        # Select source and destination
        # ======================================================

        if require_path:

            if prefer_largest_component:

                component_nodes = sampling_nodes

            else:

                component_nodes = _select_component(
                    sampling_components,
                    rng=rng,
                    prefer_largest=False
                )

            source, destination = rng.choice(
                component_nodes,
                size=2,
                replace=False
            )

        else:

            source, destination = rng.choice(
                sampling_nodes,
                size=2,
                replace=False
            )

        # ------------------------------------------------------
        # Preserve original graph node type.
        # ------------------------------------------------------

        source = source.item() if hasattr(
            source,
            "item"
        ) else source

        destination = destination.item() if hasattr(
            destination,
            "item"
        ) else destination

        # ======================================================
        # Generate amount
        # ======================================================

        amount = _generate_amount(
            rng=rng,
            min_amount=min_amount,
            max_amount=max_amount
        )

        # ======================================================
        # Timestamp
        # ======================================================

        timestamp = (
            int(timestamp_start)
            +
            tx_id
        )

        # ======================================================
        # Metadata
        # ======================================================

        metadata = {

            "generator_seed":
                seed,

            "connectivity_required":
                require_path,

            "component_based_sampling":
                require_path,

            "sampling_largest_component":
                (
                    prefer_largest_component
                    if require_path
                    else False
                )

        }

        # ======================================================
        # Create transaction
        # ======================================================

        transaction = Transaction(

            tx_id=tx_id,

            source=source,

            destination=destination,

            amount=amount,

            timestamp=timestamp,

            split="train",

            status="PENDING",

            metadata=metadata

        )

        transactions.append(
            transaction
        )

    # ==========================================================
    # Train/Test split
    # ==========================================================

    assign_split(
        transactions,
        train_ratio=train_ratio
    )

    return transactions


# ==============================================================
# Generate payment amount
# ==============================================================

def _generate_amount(
    rng,
    min_amount: float,
    max_amount: float,
    sigma: float = 1.3
) -> float:
    """
    Generate a payment amount using a log-normal distribution.

    The resulting amount is clipped to the requested range.
    """

    mean = np.log(
        min_amount
    )

    amount = rng.lognormal(
        mean=mean,
        sigma=sigma
    )

    amount = np.clip(
        amount,
        min_amount,
        max_amount
    )

    return float(
        amount
    )


# ==============================================================
# Assign train/test split
# ==============================================================

def assign_split(
    transactions: List[Transaction],
    train_ratio: float = 0.70
):
    """
    Assign transactions to training and testing sets.

    Default:

        70% train
        30% test

    The original transaction order is preserved.

    Returns
    -------
    tuple[list[Transaction], list[Transaction]]
    """

    if not transactions:

        return [], []

    if not 0.0 < train_ratio < 1.0:

        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    split_index = int(
        len(transactions)
        *
        train_ratio
    )

    # ----------------------------------------------------------
    # Keep both datasets non-empty when possible.
    # ----------------------------------------------------------

    if len(transactions) >= 2:

        split_index = min(
            max(
                split_index,
                1
            ),
            len(transactions) - 1
        )

    train = transactions[
        :split_index
    ]

    test = transactions[
        split_index:
    ]

    for tx in train:

        tx.split = "train"

    for tx in test:

        tx.split = "test"

    return (
        train,
        test
    )


# ==============================================================
# Generate train/test dataset
# ==============================================================

def generate_transaction_dataset(
    G,
    n: int = 10_000,
    seed: int = 42,
    min_amount: float = 1_000,
    max_amount: float = 1_000_000,
    train_ratio: float = 0.70,
    timestamp_start: int = 0,
    require_path: bool = True,
    prefer_largest_component: bool = True,
) -> TransactionDataset:
    """
    Generate a complete train/test transaction dataset.
    """

    transactions = generate_transactions(

        G=G,

        n=n,

        seed=seed,

        min_amount=min_amount,

        max_amount=max_amount,

        train_ratio=train_ratio,

        timestamp_start=timestamp_start,

        require_path=require_path,

        prefer_largest_component=(
            prefer_largest_component
        )

    )

    train, test = assign_split(

        transactions,

        train_ratio=train_ratio

    )

    return TransactionDataset(

        train=train,

        test=test

    )


# ==============================================================
# Convert transactions to dictionaries
# ==============================================================

def transactions_to_dicts(
    transactions
):
    """
    Convert Transaction objects into dictionaries.

    Useful for Simulation/environment.py.
    """

    return [

        {

            "tx_id":
                tx.tx_id,

            "source":
                tx.source,

            "destination":
                tx.destination,

            "amount":
                tx.amount,

            "timestamp":
                tx.timestamp,

            "split":
                tx.split,

            "status":
                tx.status

        }

        for tx in transactions

    ]


# ==============================================================
# Filter by split
# ==============================================================

def get_train_transactions(
    transactions
):
    """
    Return only training transactions.
    """

    return [

        tx

        for tx in transactions

        if tx.split == "train"

    ]


def get_test_transactions(
    transactions
):
    """
    Return only testing transactions.
    """

    return [

        tx

        for tx in transactions

        if tx.split == "test"

    ]


# ==============================================================
# Simple test
# ==============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # Small Lightning-like graph
    # ----------------------------------------------------------

    G = nx.DiGraph()

    G.add_edges_from([

        ("A", "B"),
        ("B", "C"),
        ("C", "D"),

        ("D", "A"),

        ("A", "C"),
        ("B", "D")

    ])

    # ----------------------------------------------------------
    # Generate dataset
    # ----------------------------------------------------------

    dataset = generate_transaction_dataset(

        G=G,

        n=100,

        seed=42,

        min_amount=1_000,

        max_amount=1_000_000,

        train_ratio=0.70,

        require_path=True,

        prefer_largest_component=True

    )

    # ----------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------

    print("=" * 70)
    print("TRANSACTION GENERATOR TEST")
    print("=" * 70)

    print(
        "Dataset:",
        dataset.summary()
    )

    print()

    print(
        "First training transactions:"
    )

    for tx in dataset.train[:5]:

        print(tx)

    print()

    print(
        "First testing transactions:"
    )

    for tx in dataset.test[:5]:

        print(tx)