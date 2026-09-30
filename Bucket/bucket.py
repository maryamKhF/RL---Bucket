"""
Bucket/bucket.py

Bucket container and candidate execution logic.

Responsibilities
----------------
- Store routing candidates
- Manage candidate attempts
- Count failed candidates
- Track failed channels
- Identify selected candidate rank
- Handle candidate backtracking
- Validate candidate availability
- Support candidate dictionaries produced by Top-K Pathfinding

Important
---------
Bucket does NOT perform the actual payment.

The actual payment execution and stochastic failure handling
should be performed by PaymentSimulator / FailureModel.

Unknown directional liquidity is NOT treated as zero liquidity.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================
# Bucket
# ============================================================

@dataclass
class Bucket:
    """
    Container for routing candidates of one transaction.

    Candidates are normally dictionaries produced by
    Pathfinding.top_k_paths(), for example:

        {
            "path": [...],
            "edges": [...],
            "cost": ...,
            "hop_count": ...,
            "total_fee": ...,
            "total_delay": ...,
            "reliability": ...,
            "failure_probability": ...,
            "eta": ...,
            "lambda_h": ...,
            "candidate": True,
            "success": None
        }

    The Bucket does not execute the payment itself.
    It manages candidate selection and records failures.
    """

    bucket_id: Any

    transaction_id: Any

    candidates: list = field(default_factory=list)

    # Zero-based index of the candidate currently being attempted.
    current_index: int = 0

    # Number of candidate attempts.
    attempts: int = 0

    # Bucket status.
    status: str = "active"

    # Successfully selected candidate.
    selected_candidate: Any = None

    # Number of failed candidates.
    failed_candidates_count: int = 0

    # Zero-based index of successfully selected candidate.
    selected_candidate_index: Optional[int] = None

    # One-based rank of successfully selected candidate.
    selected_candidate_rank: Optional[int] = None

    # Failed candidate objects.
    failed_candidates: list = field(default_factory=list)

    # Failed channel identifiers.
    failed_channels: list = field(default_factory=list)

    # ========================================================
    # Current Candidate
    # ========================================================

    def current(self):
        """
        Return the current candidate.

        Returns
        -------
        candidate or None
        """

        if not self.candidates:
            return None

        if self.current_index < 0:
            return None

        if self.current_index >= len(self.candidates):
            return None

        return self.candidates[self.current_index]

    # ========================================================
    # Candidate Movement
    # ========================================================

    def next_candidate(self):
        """
        Move to the next candidate.

        Returns
        -------
        candidate or None
        """

        if self.current_index < len(self.candidates):
            self.current_index += 1

        return self.current()

    def backtrack(
        self,
        failed_candidate=None,
        failed_channel=None
    ):
        """
        Record a failed candidate and move to the next one.

        Parameters
        ----------
        failed_candidate:
            Candidate that failed.

        failed_channel:
            Channel responsible for the failure.
        """

        self.attempts += 1

        self.failed_candidates_count += 1

        if failed_candidate is not None:
            self.failed_candidates.append(
                failed_candidate
            )

        if failed_channel is not None:
            if failed_channel not in self.failed_channels:
                self.failed_channels.append(
                    failed_channel
                )

        return self.next_candidate()

    # ========================================================
    # Status Management
    # ========================================================

    def mark_success(self, candidate):
        """
        Mark the bucket as successfully completed.

        Stores both:
        - zero-based candidate index
        - one-based candidate rank
        """

        self.status = "completed"

        self.selected_candidate = candidate

        self.selected_candidate_index = (
            self.current_index
        )

        self.selected_candidate_rank = (
            self.current_index + 1
        )

    def fail(self):
        """
        Mark the bucket as failed.
        """

        self.status = "failed"

        self.selected_candidate = None
        self.selected_candidate_index = None
        self.selected_candidate_rank = None

    def finished(self):
        """
        Check whether bucket execution has terminated.
        """

        return self.status in {
            "completed",
            "failed"
        }

    # ========================================================
    # Information
    # ========================================================

    def info(self):
        """
        Return complete Bucket information.
        """

        return {
            "bucket_id": self.bucket_id,

            "transaction_id": self.transaction_id,

            "candidate_count": len(
                self.candidates
            ),

            "current_index": self.current_index,

            "attempts": self.attempts,

            "status": self.status,

            "selected_candidate_index":
                self.selected_candidate_index,

            "selected_candidate_rank":
                self.selected_candidate_rank,

            "failed_candidates_count":
                self.failed_candidates_count,

            "failed_candidates":
                self.failed_candidates,

            "failed_channels_count":
                len(self.failed_channels),

            "failed_channels":
                self.failed_channels,

            "selected_candidate":
                self.selected_candidate
        }


# ============================================================
# Candidate Helpers
# ============================================================

def _candidate_path(candidate):
    """
    Extract path from a candidate.

    Supports the current dictionary-based candidate format
    and also keeps backward compatibility with tuple/list
    candidates.

    Returns
    -------
    list or None
    """

    if candidate is None:
        return None

    if isinstance(candidate, dict):
        return candidate.get("path")

    if isinstance(candidate, (tuple, list)):
        if len(candidate) >= 1:
            return candidate[0]

    return None


def _candidate_edges(candidate):
    """
    Extract edge list from a candidate.

    Current format:
        candidate["edges"]

    Backward-compatible tuple format:
        (path, edges, score)
    """

    if candidate is None:
        return None

    if isinstance(candidate, dict):
        return candidate.get("edges")

    if isinstance(candidate, (tuple, list)):
        if len(candidate) >= 2:
            return candidate[1]

    return None


def _normalize_edge(edge):
    """
    Normalize an edge identifier.

    Supported forms:
        (u, v)
        (u, v, key)

    Returns
    -------
    tuple or None
    """

    if edge is None:
        return None

    if isinstance(edge, (tuple, list)):

        if len(edge) >= 3:
            return (
                edge[0],
                edge[1],
                edge[2]
            )

        if len(edge) == 2:
            return (
                edge[0],
                edge[1]
            )

    return None


def _get_channel(G, edge):
    """
    Safely retrieve channel data from a NetworkX graph.

    Supports:
        Graph
        DiGraph
        MultiGraph
        MultiDiGraph

    Returns
    -------
    dict or None
    """

    normalized = _normalize_edge(edge)

    if normalized is None:
        return None

    if len(normalized) == 3:

        u, v, key = normalized

        try:
            if G.is_multigraph():
                return G.edges[u, v, key]
        except (
            KeyError,
            IndexError,
            TypeError
        ):
            return None

    elif len(normalized) == 2:

        u, v = normalized

        try:
            return G.edges[u, v]
        except (
            KeyError,
            IndexError,
            TypeError
        ):
            return None

    return None


def _is_channel_available(channel):
    """
    Check channel availability.

    Missing 'available' means available unless another
    module explicitly marks the channel unavailable.
    """

    if channel is None:
        return False

    return bool(
        channel.get(
            "available",
            True
        )
    )


def _get_known_liquidity(channel):
    """
    Return known directional liquidity.

    IMPORTANT
    ---------
    Missing liquidity is represented by None.

    It must NOT be converted to zero.

    Priority:
        1. estimated_liquidity
        2. liquidity_uv
        3. balance_uv

    Channel capacity is intentionally NOT interpreted
    as directional liquidity.
    """

    if channel is None:
        return None

    liquidity_keys = (
        "estimated_liquidity",
        "liquidity_uv",
        "balance_uv"
    )

    for key in liquidity_keys:

        value = channel.get(key)

        if value is None:
            continue

        try:
            value = float(value)
        except (
            TypeError,
            ValueError
        ):
            continue

        if value < 0:
            value = 0.0

        return value

    return None


def _liquidity_is_sufficient(
    channel,
    amount
):
    """
    Check whether a channel can carry the payment amount.

    Returns
    -------
    bool

    Logic
    -----
    Known liquidity:
        liquidity >= amount → True
        liquidity < amount  → False

    Unknown liquidity:
        True

    Unknown liquidity is accepted here because the actual
    payment attempt and stochastic failure are handled by
    PaymentSimulator / FailureModel.
    """

    if channel is None:
        return False

    liquidity = _get_known_liquidity(
        channel
    )

    # Directional liquidity is unknown.
    if liquidity is None:
        return True

    try:
        amount = float(amount)
    except (
        TypeError,
        ValueError
    ):
        return False

    if amount < 0:
        return False

    return liquidity >= amount


# ============================================================
# Candidate Validation
# ============================================================

def validate_candidate(
    G,
    candidate,
    amount
):
    """
    Validate a routing candidate before selection.

    This function validates structural/channel conditions only.

    It does NOT simulate payment failure.

    Returns
    -------
    tuple
        (
            valid,
            failed_channel,
            reason
        )

    Possible reasons:
        None
        "invalid_candidate"
        "missing_path"
        "missing_edges"
        "invalid_edge"
        "missing_channel"
        "channel_unavailable"
        "insufficient_liquidity"
    """

    if candidate is None:
        return (
            False,
            None,
            "invalid_candidate"
        )

    path = _candidate_path(
        candidate
    )

    edges = _candidate_edges(
        candidate
    )

    if path is None:
        return (
            False,
            None,
            "missing_path"
        )

    if edges is None:
        return (
            False,
            None,
            "missing_edges"
        )

    if not isinstance(
        edges,
        (list, tuple)
    ):
        return (
            False,
            None,
            "invalid_edges"
        )

    try:
        amount = float(amount)
    except (
        TypeError,
        ValueError
    ):
        return (
            False,
            None,
            "invalid_amount"
        )

    if amount < 0:
        return (
            False,
            None,
            "invalid_amount"
        )

    # --------------------------------------------------------
    # Validate every channel
    # --------------------------------------------------------

    for edge in edges:

        normalized_edge = _normalize_edge(
            edge
        )

        if normalized_edge is None:

            return (
                False,
                edge,
                "invalid_edge"
            )

        channel = _get_channel(
            G,
            normalized_edge
        )

        if channel is None:

            return (
                False,
                normalized_edge,
                "missing_channel"
            )

        # ----------------------------------------------------
        # Availability
        # ----------------------------------------------------

        if not _is_channel_available(
            channel
        ):

            return (
                False,
                normalized_edge,
                "channel_unavailable"
            )

        # ----------------------------------------------------
        # Known liquidity
        # ----------------------------------------------------

        if not _liquidity_is_sufficient(
            channel,
            amount
        ):

            return (
                False,
                normalized_edge,
                "insufficient_liquidity"
            )

    return (
        True,
        None,
        None
    )


# ============================================================
# Bucket Execution Engine
# ============================================================

def execute_bucket(
    G,
    bucket,
    amount,
    rng=None
):
    """
    Execute candidate selection from a Bucket.

    IMPORTANT
    ---------
    This function does NOT execute the actual payment.

    It only:
        1. Reads the current candidate.
        2. Checks channel availability.
        3. Checks known directional liquidity.
        4. Accepts candidates with unknown liquidity.
        5. Moves to the next candidate if validation fails.

    The actual payment should subsequently be sent to
    PaymentSimulator, where FailureModel can determine
    stochastic node/channel/liquidity failures.

    Candidate format
    ----------------
    Preferred format:

        {
            "path": [...],
            "edges": [...],
            "cost": ...,
            ...
        }

    Backward-compatible format:

        (
            path,
            edges,
            score
        )

    Parameters
    ----------
    G:
        NetworkX graph.

    bucket:
        Bucket instance.

    amount:
        Payment amount.

    rng:
        Optional random generator.

        Currently not used directly because stochastic
        payment failure belongs to FailureModel.

    Returns
    -------
    tuple
        (
            success,
            selected_candidate,
            attempts
        )
    """

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if bucket is None:

        return (
            False,
            None,
            0
        )

    if bucket.finished():

        return (
            bucket.status == "completed",
            bucket.selected_candidate,
            bucket.attempts
        )

    attempts = 0

    # --------------------------------------------------------
    # Candidate execution loop
    # --------------------------------------------------------

    while not bucket.finished():

        candidate = bucket.current()

        # ----------------------------------------------------
        # No candidate remaining
        # ----------------------------------------------------

        if candidate is None:

            bucket.fail()

            break

        attempts += 1

        valid, failed_channel, reason = (
            validate_candidate(
                G,
                candidate,
                amount
            )
        )

        # ----------------------------------------------------
        # Candidate accepted
        # ----------------------------------------------------

        if valid:

            bucket.mark_success(
                candidate
            )

            return (
                True,
                candidate,
                attempts
            )

        # ----------------------------------------------------
        # Candidate rejected
        # ----------------------------------------------------

        # Store structured failure information only through
        # the failed candidate list and channel list.
        #
        # The candidate itself remains intact so that later
        # backtracking logic can inspect its route.

        bucket.backtrack(
            failed_candidate=candidate,
            failed_channel=failed_channel
        )

    # --------------------------------------------------------
    # All candidates failed validation
    # --------------------------------------------------------

    return (
        False,
        None,
        attempts
    )