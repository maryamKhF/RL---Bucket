import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Transaction:
    """
    Represents a payment request in the Lightning Network.
    """

    tx_id: int
    source: str
    destination: str
    amount: float
    timestamp: int

    # Runtime information (updated during simulation)
    status: str = "PENDING"          # PENDING | SUCCESS | FAILED
    attempts: int = 0                # Number of routing attempts
    path: Optional[list[str]] = None # Selected routing path
    fee: float = 0.0                 # Total routing fee


def generate_transactions(
    G,
    n: int = 10_000,
    seed: int = 42,
    min_amount: float = 1_000,
    max_amount: float = 1_000_000,
):
    """
    Generate random payment requests for the Lightning Network.

    Parameters
    ----------
    G : networkx.Graph
        Lightning Network graph.

    n : int
        Number of transactions to generate.

    seed : int
        Random seed for reproducibility.

    min_amount : float
        Minimum payment amount (satoshi).

    max_amount : float
        Maximum payment amount (satoshi).

    Returns
    -------
    list[Transaction]
        Generated transactions.
    """

    rng = np.random.default_rng(seed)
    nodes = list(G.nodes)

    transactions = []

    for tx_id in range(n):

        source, destination = rng.choice(
            nodes,
            size=2,
            replace=False,
        )

        # Generate payment amount using a log-normal distribution
        amount = rng.lognormal(
            mean=np.log(min_amount),
            sigma=1.3,
        )

        amount = float(
            np.clip(
                amount,
                min_amount,
                max_amount,
            )
        )

        transactions.append(
            Transaction(
                tx_id=tx_id,
                source=str(source),
                destination=str(destination),
                amount=amount,
                timestamp=tx_id,
            )
        )

    return transactions