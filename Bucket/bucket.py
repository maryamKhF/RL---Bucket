"""
Bucket/bucket.py

Bucket container and candidate lifecycle management.

Responsibilities
----------------
- Store routing candidates
- Manage current candidate
- Select a structurally valid candidate
- Record actual payment attempts
- Record failed candidates
- Track failed channels
- Track failure reasons
- Track selected candidate rank
- Support candidate backtracking
- Validate candidate availability
- Preserve exact MultiDiGraph channel keys

Important lifecycle
-------------------
Candidate selection and payment success are different events.

    Candidate validation
            |
            v
    Candidate selected
            |
            v
    Payment attempt
            |
       +----+----+
       |         |
    success    failure
       |         |
       v         v
   completed   backtrack
                 |
                 v
           next candidate

Therefore:

    execute_bucket()
        -> selects candidate only

    record_attempt()
        -> counts actual payment attempt

    mark_success()
        -> called only after successful payment

    backtrack()
        -> records failed candidate and moves forward

Important semantics
-------------------
- Bucket.attempts counts ACTUAL payment attempts only.
- Candidate validation does NOT increment attempts.
- Backtracking does NOT increment attempts.
- Bucket does NOT execute payments.
- PaymentSimulator performs payment execution.
- FailureModel performs stochastic failure evaluation.
- Capacity is NOT directional liquidity.
- Unknown directional liquidity is accepted by structural validation.
- Exact MultiDiGraph channel keys are preserved whenever available.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import math


# ============================================================
# Bucket
# ============================================================

@dataclass
class Bucket:
    """
    Container for routing candidates belonging to one transaction.

    A candidate is normally a dictionary produced by
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

    The Bucket manages candidate lifecycle only.

    It does NOT:
        - execute payments
        - simulate channel failure
        - simulate liquidity failure
        - invoke PPO
        - perform routing
    """

    bucket_id: Any
    transaction_id: Any

    candidates: list = field(default_factory=list)

    # --------------------------------------------------------
    # Candidate pointer
    # --------------------------------------------------------

    # Zero-based index of the currently active candidate.
    current_index: int = 0

    # --------------------------------------------------------
    # Actual payment attempts
    # --------------------------------------------------------

    # IMPORTANT:
    # This counter represents actual calls to PaymentSimulator.
    #
    # Candidate validation and backtracking do NOT increment it.
    attempts: int = 0

    # --------------------------------------------------------
    # Bucket lifecycle
    # --------------------------------------------------------

    # Possible states:
    #
    # active
    # completed
    # failed
    #
    # "active" includes the period in which a candidate has
    # been selected but its payment has not yet completed.
    status: str = "active"

    # --------------------------------------------------------
    # Currently / successfully selected candidate
    # --------------------------------------------------------

    # Candidate currently selected for payment, or the candidate
    # that successfully completed payment.
    selected_candidate: Any = None

    # Zero-based index of the successfully completed candidate.
    #
    # This is populated only by mark_success().
    selected_candidate_index: Optional[int] = None

    # One-based rank of the successfully completed candidate.
    #
    # This is populated only by mark_success().
    selected_candidate_rank: Optional[int] = None

    # --------------------------------------------------------
    # Failure tracking
    # --------------------------------------------------------

    failed_candidates_count: int = 0

    failed_candidates: list = field(default_factory=list)

    failed_channels: list = field(default_factory=list)

    failure_reasons: list = field(default_factory=list)

    # ========================================================
    # Initialization validation
    # ========================================================

    def __post_init__(self):
        """
        Validate initial Bucket state.
        """

        if self.current_index < 0:
            self.current_index = 0

        if self.attempts < 0:
            self.attempts = 0

        if self.failed_candidates_count < 0:
            self.failed_candidates_count = 0

        if self.status not in {
            "active",
            "completed",
            "failed"
        }:
            self.status = "active"

    # ========================================================
    # Current Candidate
    # ========================================================

    def current(self):
        """
        Return the currently active candidate.

        Returns
        -------
        candidate or None
        """

        if self.status != "active":
            return None

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

        This operation does NOT count as a payment attempt.

        Returns
        -------
        candidate or None
        """

        if self.status != "active":
            return None

        if self.current_index < len(self.candidates):
            self.current_index += 1

        candidate = self.current()

        if candidate is None and self.current_index >= len(
            self.candidates
        ):
            self.status = "failed"

        return candidate

    # ========================================================
    # Candidate Selection
    # ========================================================

    def select_candidate(self, candidate=None):
        """
        Select the current candidate for payment execution.

        IMPORTANT
        ---------
        Selection is NOT payment success.

        Therefore this method does NOT set:

            status = "completed"

        The Bucket remains active until PaymentSimulator reports
        successful payment and mark_success() is called.

        Returns
        -------
        candidate or None
        """

        if self.status != "active":
            return None

        current_candidate = self.current()

        if current_candidate is None:
            return None

        if candidate is not None:

            if candidate is not current_candidate:

                # Structural equality fallback.
                if candidate != current_candidate:
                    return None

        self.selected_candidate = current_candidate

        # Do not populate successful-candidate index/rank yet.
        #
        # Those fields describe the candidate whose PAYMENT
        # actually succeeded.

        return current_candidate

    # ========================================================
    # Payment Attempt Tracking
    # ========================================================

    def record_attempt(self):
        """
        Record one actual payment attempt.

        This method must be called immediately before:

            PaymentSimulator.simulate_payment(...)

        Candidate validation does NOT count.

        Backtracking does NOT count.

        Returns
        -------
        int
            Updated number of actual payment attempts.
        """

        if self.finished():
            return self.attempts

        candidate = self.current()

        if candidate is None:
            return self.attempts

        self.attempts += 1

        self.selected_candidate = candidate

        return self.attempts

    # ========================================================
    # Payment Success
    # ========================================================

    def mark_success(self, candidate=None):
        """
        Mark the Bucket as successfully completed.

        This method must be called ONLY after the actual payment
        succeeds.

        Parameters
        ----------
        candidate:
            Candidate whose payment succeeded.

            If omitted, the current candidate is used.

        Returns
        -------
        bool
            True if success state was recorded.
        """

        if self.status != "active":
            return False

        current_candidate = self.current()

        if current_candidate is None:
            return False

        if candidate is not None:

            if candidate is not current_candidate:

                if candidate != current_candidate:
                    return False

        self.selected_candidate = current_candidate

        self.selected_candidate_index = self.current_index

        self.selected_candidate_rank = (
            self.current_index + 1
        )

        self.status = "completed"

        return True

    # ========================================================
    # Backtracking
    # ========================================================

    def backtrack(
        self,
        failed_candidate=None,
        failed_channel=None,
        reason=None
    ):
        """
        Record a failed candidate and move to the next candidate.

        IMPORTANT
        ---------
        Backtracking does NOT increment attempts.

        attempts = actual payment attempts

        failed_candidates_count = failed candidate count

        These are intentionally separate quantities.

        Parameters
        ----------
        failed_candidate:
            Candidate that failed.

        failed_channel:
            Channel responsible for the failure.

        reason:
            Failure or rejection reason.

        Returns
        -------
        candidate or None
            Next active candidate.
        """

        if self.status != "active":
            return None

        # ----------------------------------------------------
        # Determine candidate being failed
        # ----------------------------------------------------

        candidate_index = self.current_index

        if failed_candidate is None:
            failed_candidate = self.current()

        # ----------------------------------------------------
        # Record failed candidate
        # ----------------------------------------------------

        if failed_candidate is not None:

            if not self._candidate_already_failed(
                failed_candidate
            ):

                self.failed_candidates.append(
                    failed_candidate
                )

                self.failed_candidates_count += 1

        # ----------------------------------------------------
        # Record failed channel
        # ----------------------------------------------------

        if failed_channel is not None:

            normalized_channel = _normalize_edge(
                failed_channel
            )

            channel_value = (
                normalized_channel
                if normalized_channel is not None
                else failed_channel
            )

            if channel_value not in self.failed_channels:

                self.failed_channels.append(
                    channel_value
                )

        # ----------------------------------------------------
        # Record reason
        # ----------------------------------------------------

        if reason is not None:

            self.failure_reasons.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_rank": candidate_index + 1,
                    "reason": reason,
                    "failed_channel": (
                        _normalize_edge(failed_channel)
                        if failed_channel is not None
                        else None
                    )
                }
            )

        # ----------------------------------------------------
        # Clear currently selected candidate
        # ----------------------------------------------------

        self.selected_candidate = None

        # ----------------------------------------------------
        # Move to next candidate
        # ----------------------------------------------------

        return self.next_candidate()

    # ========================================================
    # Failure
    # ========================================================

    def fail(self):
        """
        Mark Bucket as exhausted/failed.

        No candidate remains usable.
        """

        self.status = "failed"

        self.selected_candidate = None

        self.selected_candidate_index = None

        self.selected_candidate_rank = None

        return None

    # ========================================================
    # Status
    # ========================================================

    def finished(self):
        """
        Return True when Bucket execution has terminated.
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
        Return complete Bucket state.
        """

        current_candidate = self.current()

        return {
            "bucket_id":
                self.bucket_id,

            "transaction_id":
                self.transaction_id,

            "candidate_count":
                len(self.candidates),

            "current_index":
                self.current_index,

            "current_rank":
                (
                    self.current_index + 1
                    if current_candidate is not None
                    else None
                ),

            # Actual payment attempts only.
            "attempts":
                self.attempts,

            "status":
                self.status,

            "selected_candidate_index":
                self.selected_candidate_index,

            "selected_candidate_rank":
                self.selected_candidate_rank,

            "failed_candidates_count":
                self.failed_candidates_count,

            "failed_candidates":
                list(self.failed_candidates),

            "failed_channels_count":
                len(self.failed_channels),

            "failed_channels":
                list(self.failed_channels),

            "failure_reasons":
                list(self.failure_reasons),

            "selected_candidate":
                self.selected_candidate
        }

    # ========================================================
    # Internal Failure Duplicate Check
    # ========================================================

    @staticmethod
    def _candidate_already_failed(candidate, failed_candidates=None):
        """
        Check whether a candidate has already been recorded as failed.

        This helper supports identity first and equality second.
        """

        if failed_candidates is None:
            return False

        for failed in failed_candidates:

            if failed is candidate:
                return True

            try:

                if failed == candidate:
                    return True

            except Exception:

                continue

        return False


# ============================================================
# Candidate Helpers
# ============================================================

def _candidate_path(candidate):
    """
    Extract node path from a candidate.

    Supported forms:

        candidate["path"]

    or backward-compatible:

        (path, edges, score)

    Returns
    -------
    list or None
    """

    if candidate is None:
        return None

    if isinstance(candidate, dict):

        path = candidate.get("path")

        if path is None:
            return None

        if not isinstance(path, (list, tuple)):
            return None

        return list(path)

    if isinstance(candidate, (tuple, list)):

        if len(candidate) < 1:
            return None

        first = candidate[0]

        if isinstance(first, (list, tuple)):
            return list(first)

    return None


def _candidate_edges(candidate):
    """
    Extract edge list from a candidate.

    Supported forms:

        candidate["edges"]

    or:

        (path, edges, score)

    Returns
    -------
    list or None
    """

    if candidate is None:
        return None

    if isinstance(candidate, dict):

        edges = candidate.get("edges")

        if edges is None:
            return None

        if not isinstance(edges, (list, tuple)):
            return None

        return list(edges)

    if isinstance(candidate, (tuple, list)):

        if len(candidate) < 2:
            return None

        edges = candidate[1]

        if isinstance(edges, (list, tuple)):
            return list(edges)

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

    if not isinstance(edge, (tuple, list)):
        return None

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


# ============================================================
# Graph Channel Access
# ============================================================

def _get_channel(G, edge):
    """
    Safely retrieve one exact channel.

    Supports:

        Graph
        DiGraph
        MultiGraph
        MultiDiGraph

    For MultiGraph/MultiDiGraph, a three-element edge:

        (u, v, key)

    is resolved exactly.

    A two-element edge in a multigraph is intentionally NOT
    resolved to an arbitrary parallel channel, because that
    would destroy channel identity.
    """

    if G is None:
        return None

    normalized = _normalize_edge(edge)

    if normalized is None:
        return None

    # --------------------------------------------------------
    # Exact keyed channel
    # --------------------------------------------------------

    if len(normalized) == 3:

        u, v, key = normalized

        try:

            if not G.is_multigraph():
                return None

            if not G.has_edge(u, v, key):
                return None

            return G.edges[u, v, key]

        except (
            KeyError,
            IndexError,
            TypeError
        ):

            return None

    # --------------------------------------------------------
    # Endpoint-only channel
    # --------------------------------------------------------

    u, v = normalized

    try:

        if G.is_multigraph():

            # Do not silently choose one arbitrary parallel channel.
            return None

        if not G.has_edge(u, v):
            return None

        return G.edges[u, v]

    except (
        KeyError,
        IndexError,
        TypeError
    ):

        return None


# ============================================================
# Channel Availability
# ============================================================

def _is_channel_available(channel):
    """
    Check channel runtime availability.

    Missing ``available`` means available.

    Explicit False means unavailable.
    """

    if channel is None:
        return False

    return (
        channel.get("available", True) is not False
    )


# ============================================================
# Directional Liquidity
# ============================================================

def _get_known_liquidity(channel):
    """
    Return known directional liquidity.

    IMPORTANT
    ---------
    Missing liquidity is represented by None.

    It is NEVER converted to zero.

    Priority:

        1. estimated_liquidity
        2. liquidity_uv
        3. balance_uv
        4. liquidity

    Channel ``capacity`` is deliberately excluded.
    """

    if channel is None:
        return None

    liquidity_keys = (
        "estimated_liquidity",
        "liquidity_uv",
        "balance_uv",
        "liquidity"
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

        if not math.isfinite(value):
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
    Check whether known directional liquidity can carry amount.

    Known liquidity:

        liquidity >= amount -> True
        liquidity < amount  -> False

    Unknown liquidity:

        -> True

    Unknown liquidity is accepted during structural validation.
    The actual payment attempt is responsible for determining
    stochastic failure.
    """

    if channel is None:
        return False

    try:

        amount = float(amount)

    except (
        TypeError,
        ValueError
    ):

        return False

    if not math.isfinite(amount):
        return False

    if amount <= 0:
        return False

    liquidity = _get_known_liquidity(channel)

    # Unknown liquidity is not interpreted as zero.
    if liquidity is None:
        return True

    return liquidity >= amount


# ============================================================
# Candidate Structural Validation
# ============================================================

def validate_candidate(
    G,
    candidate,
    amount
):
    """
    Validate a routing candidate before payment execution.

    This function performs only deterministic structural
    validation.

    It does NOT:
        - execute payment
        - invoke FailureModel
        - apply stochastic failure
        - increment Bucket.attempts

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
        invalid_candidate
        missing_path
        invalid_path
        missing_edges
        invalid_edges
        invalid_amount
        invalid_edge
        path_edge_count_mismatch
        path_edge_mismatch
        missing_channel
        channel_unavailable
        insufficient_liquidity
    """

    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    if candidate is None:

        return (
            False,
            None,
            "invalid_candidate"
        )

    # --------------------------------------------------------
    # Path
    # --------------------------------------------------------

    path = _candidate_path(candidate)

    if path is None:

        return (
            False,
            None,
            "missing_path"
        )

    if len(path) < 2:

        return (
            False,
            None,
            "invalid_path"
        )

    try:

        if len(path) != len(set(path)):

            return (
                False,
                None,
                "invalid_path"
            )

    except TypeError:

        return (
            False,
            None,
            "invalid_path"
        )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    edges = _candidate_edges(candidate)

    if edges is None:

        return (
            False,
            None,
            "missing_edges"
        )

    if len(edges) != len(path) - 1:

        return (
            False,
            None,
            "path_edge_count_mismatch"
        )

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

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

    if not math.isfinite(amount) or amount <= 0:

        return (
            False,
            None,
            "invalid_amount"
        )

    # --------------------------------------------------------
    # Validate exact path <-> edge correspondence
    # --------------------------------------------------------

    for index, edge in enumerate(edges):

        normalized_edge = _normalize_edge(edge)

        if normalized_edge is None:

            return (
                False,
                edge,
                "invalid_edge"
            )

        u = normalized_edge[0]
        v = normalized_edge[1]

        expected_u = path[index]
        expected_v = path[index + 1]

        if (
            u != expected_u
            or
            v != expected_v
        ):

            return (
                False,
                normalized_edge,
                "path_edge_mismatch"
            )

        # ----------------------------------------------------
        # Exact channel retrieval
        # ----------------------------------------------------

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
        # Runtime availability
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
        # Known directional liquidity
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

    # --------------------------------------------------------
    # Candidate is structurally valid
    # --------------------------------------------------------

    return (
        True,
        None,
        None
    )


# ============================================================
# Bucket Candidate Selection
# ============================================================

def execute_bucket(
    G,
    bucket,
    amount,
    rng=None
):
    """
    Select the first structurally valid candidate.

    IMPORTANT
    ---------
    This function does NOT execute payment.

    It performs:

        candidate
            |
            v
        structural validation
            |
        +---+---+
        |       |
      valid   invalid
        |       |
        v       v
      select  backtrack
                |
                v
          next candidate

    The selected candidate remains in an ACTIVE Bucket.

    Actual payment execution must then be performed by
    PaymentSimulator.

    Correct lifecycle:

        execute_bucket(...)
                |
                v
        selected candidate
                |
                v
        bucket.record_attempt()
                |
                v
        PaymentSimulator
                |
          +-----+-----+
          |           |
       success      failure
          |           |
          v           v
    mark_success()  backtrack()

    Parameters
    ----------
    G:
        NetworkX graph.

    bucket:
        Bucket instance.

    amount:
        Payment amount.

    rng:
        Retained for backward compatibility.

        It is intentionally unused because stochastic failure
        belongs to FailureModel.

    Returns
    -------
    tuple
        (
            success,
            selected_candidate,
            attempts
        )

    IMPORTANT
    ---------
    The variable ``success`` here means:

        "a structurally valid candidate was selected"

    It does NOT mean:

        "payment succeeded"

    Therefore callers must NOT interpret this as final payment
    success.
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

    # --------------------------------------------------------
    # Candidate selection loop
    # --------------------------------------------------------

    while not bucket.finished():

        candidate = bucket.current()

        # ----------------------------------------------------
        # No candidate remains
        # ----------------------------------------------------

        if candidate is None:

            bucket.fail()

            break

        # ----------------------------------------------------
        # Validate candidate
        # ----------------------------------------------------

        valid, failed_channel, reason = (
            validate_candidate(
                G,
                candidate,
                amount
            )
        )

        # ----------------------------------------------------
        # Structurally valid candidate
        # ----------------------------------------------------

        if valid:

            bucket.select_candidate(
                candidate
            )

            return (
                True,
                candidate,
                bucket.attempts
            )

        # ----------------------------------------------------
        # Candidate rejected
        # ----------------------------------------------------

        bucket.backtrack(
            failed_candidate=candidate,
            failed_channel=failed_channel,
            reason=reason
        )

    # --------------------------------------------------------
    # No valid candidate remains
    # --------------------------------------------------------

    return (
        False,
        None,
        bucket.attempts
    )


# ============================================================
# Standalone Diagnostic Test
# ============================================================

if __name__ == "__main__":

    import networkx as nx

    print("=" * 70)
    print("BUCKET MODULE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Build MultiDiGraph
    # --------------------------------------------------------

    G = nx.MultiDiGraph()

    nodes = [
        "A",
        "B",
        "C",
        "D",
        "F",
        "G",
        "E"
    ]

    for node in nodes:

        G.add_node(
            node,
            available=True,
            is_online=True
        )

    # Main route:
    #
    # A -> B -> C -> D -> E
    #
    # Alternative:
    #
    # A -> B -> F -> G -> E

    edges = [

        ("A", "B", 0, 10000),

        ("B", "C", 0, 10000),

        ("C", "D", 0, 10000),

        ("D", "E", 0, 10000),

        ("B", "F", 0, 10000),

        ("F", "G", 0, 10000),

        ("G", "E", 0, 10000)
    ]

    for u, v, key, liquidity in edges:

        G.add_edge(
            u,
            v,
            key=key,
            capacity=10000,
            balance_uv=liquidity,
            available=True
        )

    # --------------------------------------------------------
    # 2. Candidate definitions
    # --------------------------------------------------------

    candidate_1 = {

        "path": [
            "A",
            "B",
            "C",
            "D",
            "E"
        ],

        "edges": [
            ("A", "B", 0),
            ("B", "C", 0),
            ("C", "D", 0),
            ("D", "E", 0)
        ],

        "cost": 10.0,
        "candidate": True,
        "success": None
    }

    candidate_2 = {

        "path": [
            "A",
            "B",
            "F",
            "G",
            "E"
        ],

        "edges": [
            ("A", "B", 0),
            ("B", "F", 0),
            ("F", "G", 0),
            ("G", "E", 0)
        ],

        "cost": 12.0,
        "candidate": True,
        "success": None
    }

    bucket = Bucket(
        bucket_id="B001",
        transaction_id="TX001",
        candidates=[
            candidate_1,
            candidate_2
        ]
    )

    # --------------------------------------------------------
    # 3. Select first candidate
    # --------------------------------------------------------

    selected_ok, selected, attempts = execute_bucket(
        G,
        bucket,
        amount=1000
    )

    print()
    print("TEST 1: Candidate Selection")
    print("-" * 70)

    print(
        f"Selection result : {selected_ok}"
    )

    print(
        f"Selected path    : "
        f"{selected.get('path') if selected else None}"
    )

    print(
        f"Bucket status    : "
        f"{bucket.status}"
    )

    print(
        f"Attempts         : "
        f"{bucket.attempts}"
    )

    assert selected_ok is True
    assert selected is candidate_1

    # Selection must NOT mean payment success.
    assert bucket.status == "active"

    # No payment has happened yet.
    assert bucket.attempts == 0

    print(
        "TEST 1 : PASS"
    )

    # --------------------------------------------------------
    # 4. Record actual payment attempt
    # --------------------------------------------------------

    bucket.record_attempt()

    print()
    print("TEST 2: Actual Payment Attempt")
    print("-" * 70)

    print(
        f"Attempts after record_attempt(): "
        f"{bucket.attempts}"
    )

    assert bucket.attempts == 1
    assert bucket.status == "active"

    print(
        "TEST 2 : PASS"
    )

    # --------------------------------------------------------
    # 5. Simulate payment failure externally
    # --------------------------------------------------------

    bucket.backtrack(
        failed_candidate=candidate_1,
        failed_channel=("C", "D", 0),
        reason="channel_failure"
    )

    print()
    print("TEST 3: Backtracking")
    print("-" * 70)

    print(
        f"Current index     : "
        f"{bucket.current_index}"
    )

    print(
        f"Current rank      : "
        f"{bucket.current_index + 1}"
    )

    print(
        f"Attempts          : "
        f"{bucket.attempts}"
    )

    print(
        f"Failed candidates : "
        f"{bucket.failed_candidates_count}"
    )

    print(
        f"Failed channels   : "
        f"{bucket.failed_channels}"
    )

    assert bucket.attempts == 1
    assert bucket.failed_candidates_count == 1
    assert bucket.current_index == 1
    assert bucket.current() is candidate_2
    assert bucket.status == "active"

    print(
        "TEST 3 : PASS"
    )

    # --------------------------------------------------------
    # 6. Select alternative candidate
    # --------------------------------------------------------

    selected_ok, selected, attempts = execute_bucket(
        G,
        bucket,
        amount=1000
    )

    print()
    print("TEST 4: Alternative Candidate Selection")
    print("-" * 70)

    print(
        f"Selection result : {selected_ok}"
    )

    print(
        f"Selected path    : "
        f"{selected.get('path') if selected else None}"
    )

    print(
        f"Bucket status    : "
        f"{bucket.status}"
    )

    print(
        f"Attempts         : "
        f"{bucket.attempts}"
    )

    assert selected_ok is True
    assert selected is candidate_2

    # Still not payment success.
    assert bucket.status == "active"

    # No second payment attempt has happened yet.
    assert bucket.attempts == 1

    print(
        "TEST 4 : PASS"
    )

    # --------------------------------------------------------
    # 7. Record second actual attempt
    # --------------------------------------------------------

    bucket.record_attempt()

    # --------------------------------------------------------
    # 8. Mark actual payment success
    # --------------------------------------------------------

    success_recorded = bucket.mark_success(
        candidate_2
    )

    print()
    print("TEST 5: Actual Payment Success")
    print("-" * 70)

    print(
        f"Success recorded : {success_recorded}"
    )

    print(
        f"Bucket status    : "
        f"{bucket.status}"
    )

    print(
        f"Attempts         : "
        f"{bucket.attempts}"
    )

    print(
        f"Selected index   : "
        f"{bucket.selected_candidate_index}"
    )

    print(
        f"Selected rank    : "
        f"{bucket.selected_candidate_rank}"
    )

    assert success_recorded is True
    assert bucket.status == "completed"
    assert bucket.attempts == 2
    assert bucket.selected_candidate is candidate_2
    assert bucket.selected_candidate_index == 1
    assert bucket.selected_candidate_rank == 2

    print(
        "TEST 5 : PASS"
    )

    # --------------------------------------------------------
    # 9. Final state
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL BUCKET STATE")
    print("=" * 70)

    info = bucket.info()

    for key, value in info.items():

        print(
            f"{key:28}: {value}"
        )

    print()
    print(
        "BUCKET MODULE STATUS : SUCCESS"
    )
    print("=" * 70)