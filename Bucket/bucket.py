"""
Bucket/bucket.py

Deterministic Bucket container and candidate lifecycle management.

Responsibilities
----------------
- Store routing candidates
- Preserve candidate ordering
- Select the first structurally valid candidate
- Record actual payment attempts
- Record failed candidates
- Record failed channels
- Record failure reasons
- Track successful candidate index/rank
- Support deterministic candidate backtracking
- Validate candidate structure
- Validate exact MultiDiGraph channel identity
- Validate known directional liquidity
- Preserve the separation between routing and payment execution

Lifecycle
---------
    Candidate validation
            |
            v
    Candidate selected
            |
            v
    record_attempt()
            |
            v
    PaymentSimulator
         /       \
     success    failure
       |          |
       v          v
    success    backtrack()
                   |
                   v
             next candidate

Important semantics
-------------------
Bucket does NOT:

- execute payments
- simulate stochastic failure
- invoke FailureModel
- invoke PPO
- perform pathfinding
- modify candidate cost
- modify candidate ordering

Bucket.attempts counts actual payment attempts only.

Candidate validation and backtracking do not increment attempts.

Directional liquidity:
----------------------
Known directional liquidity is treated as a hard structural
feasibility constraint.

Unknown directional liquidity is NOT interpreted as zero.
It is accepted by structural validation and remains the
responsibility of the actual payment execution / FailureModel.

Capacity is NOT directional liquidity and is deliberately excluded.

MultiDiGraph:
-------------
A parallel channel must be referenced by its exact key.

For a MultiGraph / MultiDiGraph:

    (u, v, key)

is required.

An endpoint-only edge:

    (u, v)

is rejected because selecting an arbitrary parallel channel
would destroy channel identity.

Candidate edge formats supported
--------------------------------
1. Top-K edge dictionary:

    {
        "source": u,
        "target": v,
        "channel_key": key,
        ...
    }

2. Equivalent aliases:

    {
        "u": u,
        "v": v,
        "key": key,
        ...
    }

3. Tuple:

    (u, v)

4. Keyed tuple:

    (u, v, key)

For MultiDiGraph, only the keyed form is accepted.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import math


# ============================================================
# Validation Helpers
# ============================================================

def _require_bool(value, field_name):
    """
    Require an actual Python bool.

    Strings such as "false" and integers such as 0/1 are rejected.
    """
    if not isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be a boolean, "
            f"got {type(value).__name__}"
        )

    return value


def _validate_positive_amount(amount):
    """
    Validate payment amount.

    Returns
    -------
    float
        Validated positive finite amount.
    """
    if isinstance(amount, bool):
        raise ValueError("amount must be numeric, not bool")

    try:
        value = float(amount)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"amount must be numeric, got {type(amount).__name__}"
        ) from exc

    if not math.isfinite(value):
        raise ValueError("amount must be finite")

    if value <= 0:
        raise ValueError("amount must be > 0")

    return value


def _validate_bucket_status(status):
    """
    Validate Bucket lifecycle status.
    """
    if status not in {
        "active",
        "completed",
        "failed",
    }:
        raise ValueError(
            f"invalid Bucket status: {status!r}"
        )


# ============================================================
# Candidate Access
# ============================================================

def _candidate_path(candidate):
    """
    Extract node path from a candidate.

    Supported:
        candidate["path"]

    Legacy tuple/list:
        (path, edges, score)
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
    Extract physical edges from a candidate.

    Supported:
        candidate["edges"]

    Legacy tuple/list:
        (path, edges, score)
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


# ============================================================
# Edge Normalization
# ============================================================

def _normalize_edge(edge):
    """
    Normalize an edge identifier.

    Supported:

        (u, v)
        (u, v, key)

    or dictionaries:

        {
            "source": u,
            "target": v,
            "channel_key": key
        }

    or:

        {
            "u": u,
            "v": v,
            "key": key
        }

    Returns
    -------
    tuple or None

        (u, v)

    or

        (u, v, key)
    """
    if edge is None:
        return None

    # --------------------------------------------------------
    # Dictionary representation
    # --------------------------------------------------------

    if isinstance(edge, dict):

        if "source" in edge:
            u = edge["source"]
        elif "u" in edge:
            u = edge["u"]
        else:
            return None

        if "target" in edge:
            v = edge["target"]
        elif "v" in edge:
            v = edge["v"]
        else:
            return None

        if "channel_key" in edge:
            key = edge["channel_key"]
            return (u, v, key)

        if "channel_key" not in edge and "key" in edge:
            key = edge["key"]
            return (u, v, key)

        # SCID is NOT a substitute for a MultiDiGraph key.
        return (u, v)

    # --------------------------------------------------------
    # Tuple/list representation
    # --------------------------------------------------------

    if isinstance(edge, (tuple, list)):

        if len(edge) >= 3:
            return (
                edge[0],
                edge[1],
                edge[2],
            )

        if len(edge) == 2:
            return (
                edge[0],
                edge[1],
            )

    return None


def _edge_channel_key(edge):
    """
    Return exact channel key when present.
    """
    normalized = _normalize_edge(edge)

    if normalized is None:
        return None

    if len(normalized) == 3:
        return normalized[2]

    return None


# ============================================================
# Graph Channel Access
# ============================================================

def _get_channel(G, edge):
    """
    Retrieve exactly one physical channel.

    MultiGraph / MultiDiGraph:
        keyed edge is mandatory.

    Graph / DiGraph:
        endpoint-only edge is allowed.
    """
    if G is None:
        return None

    normalized = _normalize_edge(edge)

    if normalized is None:
        return None

    # --------------------------------------------------------
    # Keyed channel
    # --------------------------------------------------------

    if len(normalized) == 3:

        u, v, key = normalized

        if not G.is_multigraph():
            return None

        try:
            if not G.has_edge(u, v, key):
                return None

            return G.edges[u, v, key]

        except (KeyError, IndexError, TypeError):
            return None

    # --------------------------------------------------------
    # Endpoint-only edge
    # --------------------------------------------------------

    u, v = normalized

    if G.is_multigraph():
        # Never choose arbitrary parallel channel.
        return None

    try:
        if not G.has_edge(u, v):
            return None

        return G.edges[u, v]

    except (KeyError, IndexError, TypeError):
        return None


# ============================================================
# Channel Availability
# ============================================================

def _is_channel_available(channel):
    """
    Validate and check channel availability.

    Missing `available` is allowed because availability is an
    optional runtime-state attribute.

    If present, it MUST be bool.
    """
    if channel is None:
        return False

    if "available" not in channel:
        return True

    available = _require_bool(
        channel["available"],
        "channel.available",
    )

    return available


# ============================================================
# Node Availability
# ============================================================

def _is_node_available(G, node):
    """
    Validate optional node runtime-state flags.

    Supported fields:

        available
        is_online
        online

    Missing fields are allowed.

    If present, each field MUST be bool.
    """
    if G is None or node not in G:
        return False

    data = G.nodes[node]

    for field_name in (
        "available",
        "is_online",
        "online",
    ):

        if field_name not in data:
            continue

        value = _require_bool(
            data[field_name],
            f"node[{node!r}].{field_name}",
        )

        if not value:
            return False

    return True


# ============================================================
# Directional Liquidity
# ============================================================

def _get_known_liquidity(channel):
    """
    Return known directional liquidity.

    Priority:

        1. estimated_liquidity
        2. liquidity_uv
        4. liquidity

    Missing all fields:
        -> None

    Invalid present value:
        -> ValueError

    Capacity is deliberately excluded.
    """
    if channel is None:
        return None

    liquidity_keys = (
        "estimated_liquidity",
        "liquidity_uv",
        "liquidity",
    )

    for key in liquidity_keys:

        if key not in channel:
            continue

        value = channel[key]

        if value is None:
            continue

        if isinstance(value, bool):
            raise ValueError(
                f"{key} must be numeric, not bool"
            )

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key} must be numeric, "
                f"got {type(channel[key]).__name__}"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"{key} must be finite"
            )

        if value < 0:
            raise ValueError(
                f"{key} must be >= 0"
            )

        return value

    return None


def _liquidity_is_sufficient(channel, amount):
    """
    Check directional liquidity.

    Known liquidity:
        liquidity >= amount -> True
        liquidity < amount  -> False

    Unknown liquidity:
        -> True

    Unknown liquidity is NOT converted to zero.
    """
    amount = _validate_positive_amount(amount)

    if channel is None:
        return False

    liquidity = _get_known_liquidity(channel)

    if liquidity is None:
        return True

    return liquidity >= amount


# ============================================================
# Candidate Structural Validation
# ============================================================

def validate_candidate(G, candidate, amount):
    """
    Deterministically validate one routing candidate.

    Returns
    -------
    tuple
        (
            valid,
            failed_channel,
            reason
        )

    This function NEVER:
        - increments attempts
        - executes payment
        - invokes FailureModel
        - modifies candidate
    """

    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    if candidate is None:
        return (
            False,
            None,
            "invalid_candidate",
        )

    # --------------------------------------------------------
    # Path
    # --------------------------------------------------------

    path = _candidate_path(candidate)

    if path is None:
        return (
            False,
            None,
            "missing_path",
        )

    if len(path) < 2:
        return (
            False,
            None,
            "invalid_path",
        )

    try:
        if len(path) != len(set(path)):
            return (
                False,
                None,
                "invalid_path",
            )
    except TypeError:
        return (
            False,
            None,
            "invalid_path",
        )

    # --------------------------------------------------------
    # Graph / node validation
    # --------------------------------------------------------

    if G is None:
        return (
            False,
            None,
            "missing_graph",
        )

    for node in path:

        if node not in G:
            return (
                False,
                None,
                "missing_node",
            )

        if not _is_node_available(G, node):
            return (
                False,
                None,
                "node_unavailable",
            )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    edges = _candidate_edges(candidate)

    if edges is None:
        return (
            False,
            None,
            "missing_edges",
        )

    if len(edges) != len(path) - 1:
        return (
            False,
            None,
            "path_edge_count_mismatch",
        )

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    try:
        amount = _validate_positive_amount(amount)
    except ValueError:
        return (
            False,
            None,
            "invalid_amount",
        )

    # --------------------------------------------------------
    # Path / edge correspondence
    # --------------------------------------------------------

    for index, edge in enumerate(edges):

        normalized_edge = _normalize_edge(edge)

        if normalized_edge is None:
            return (
                False,
                edge,
                "invalid_edge",
            )

        u = normalized_edge[0]
        v = normalized_edge[1]

        expected_u = path[index]
        expected_v = path[index + 1]

        if u != expected_u or v != expected_v:
            return (
                False,
                normalized_edge,
                "path_edge_mismatch",
            )

        # ----------------------------------------------------
        # Exact channel
        # ----------------------------------------------------

        channel = _get_channel(
            G,
            normalized_edge,
        )

        if channel is None:
            return (
                False,
                normalized_edge,
                "missing_channel",
            )

        # ----------------------------------------------------
        # Availability
        # ----------------------------------------------------

        if not _is_channel_available(channel):
            return (
                False,
                normalized_edge,
                "channel_unavailable",
            )

        # ----------------------------------------------------
        # Directional liquidity
        # ----------------------------------------------------

        if not _liquidity_is_sufficient(
            channel,
            amount,
        ):
            return (
                False,
                normalized_edge,
                "insufficient_liquidity",
            )

    return (
        True,
        None,
        None,
    )


# ============================================================
# Bucket
# ============================================================

@dataclass
class Bucket:
    """
    Deterministic container for routing candidates belonging
    to one transaction.
    """

    bucket_id: Any
    transaction_id: Any

    candidates: list = field(default_factory=list)

    # Zero-based active candidate index.
    current_index: int = 0

    # Actual payment attempts only.
    attempts: int = 0

    # active / completed / failed
    status: str = "active"

    # Current candidate during payment lifecycle.
    selected_candidate: Any = None

    # Successful candidate only.
    selected_candidate_index: Optional[int] = None
    selected_candidate_rank: Optional[int] = None

    # Failure tracking.
    failed_candidates_count: int = 0
    failed_candidates: list = field(default_factory=list)
    failed_channels: list = field(default_factory=list)
    failure_reasons: list = field(default_factory=list)

    # ========================================================
    # Initialization
    # ========================================================

    def __post_init__(self):
        """
        Strictly validate initial state.
        """

        if not isinstance(self.candidates, list):
            self.candidates = list(self.candidates)

        if isinstance(self.current_index, bool):
            raise ValueError(
                "current_index must be integer"
            )

        if not isinstance(self.current_index, int):
            raise ValueError(
                "current_index must be integer"
            )

        if self.current_index < 0:
            raise ValueError(
                "current_index must be >= 0"
            )

        if isinstance(self.attempts, bool):
            raise ValueError(
                "attempts must be integer"
            )

        if not isinstance(self.attempts, int):
            raise ValueError(
                "attempts must be integer"
            )

        if self.attempts < 0:
            raise ValueError(
                "attempts must be >= 0"
            )

        if isinstance(
            self.failed_candidates_count,
            bool,
        ):
            raise ValueError(
                "failed_candidates_count must be integer"
            )

        if not isinstance(
            self.failed_candidates_count,
            int,
        ):
            raise ValueError(
                "failed_candidates_count must be integer"
            )

        if self.failed_candidates_count < 0:
            raise ValueError(
                "failed_candidates_count must be >= 0"
            )

        _validate_bucket_status(
            self.status
        )

    # ========================================================
    # Current Candidate
    # ========================================================

    def current(self):
        """
        Return current candidate when Bucket is active.
        """
        if self.status != "active":
            return None

        if not self.candidates:
            return None

        if self.current_index < 0:
            return None

        if self.current_index >= len(
            self.candidates
        ):
            return None

        return self.candidates[
            self.current_index
        ]

    # ========================================================
    # Candidate Movement
    # ========================================================

    def next_candidate(self):
        """
        Move to the next candidate.

        Does not increment attempts.
        """
        if self.status != "active":
            return None

        self.selected_candidate = None

        if self.current_index < len(
            self.candidates
        ):
            self.current_index += 1

        candidate = self.current()

        if candidate is None:
            self.status = "failed"

        return candidate

    # ========================================================
    # Candidate Selection
    # ========================================================

    def select_candidate(self, candidate=None):
        """
        Select the current structurally valid candidate.

        Selection does NOT imply payment success.
        """
        if self.status != "active":
            return None

        current_candidate = self.current()

        if current_candidate is None:
            return None

        if candidate is not None:

            if candidate is not current_candidate:

                if not _safe_equal(
                    candidate,
                    current_candidate,
                ):
                    return None

        self.selected_candidate = (
            current_candidate
        )

        return current_candidate

    # ========================================================
    # Payment Attempt
    # ========================================================

    def record_attempt(self):
        """
        Record exactly one actual payment attempt.

        This should be called immediately before
        PaymentSimulator execution.
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
        Mark actual payment success.

        This must only be called after PaymentSimulator
        reports successful payment.
        """
        if self.status != "active":
            return False

        current_candidate = self.current()

        if current_candidate is None:
            return False

        if candidate is not None:

            if candidate is not current_candidate:

                if not _safe_equal(
                    candidate,
                    current_candidate,
                ):
                    return False

        self.selected_candidate = (
            current_candidate
        )

        self.selected_candidate_index = (
            self.current_index
        )

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
        reason=None,
    ):
        """
        Record a failed candidate and move to the next one.

        Backtracking never increments attempts.
        """
        if self.status != "active":
            return None

        candidate_index = self.current_index

        if failed_candidate is None:
            failed_candidate = self.current()

        # ----------------------------------------------------
        # Failed candidate
        # ----------------------------------------------------

        if failed_candidate is not None:

            if not self._candidate_already_failed(
                failed_candidate,
                self.failed_candidates,
            ):

                self.failed_candidates.append(
                    failed_candidate
                )

                self.failed_candidates_count += 1

        # ----------------------------------------------------
        # Failed channel
        # ----------------------------------------------------

        normalized_channel = None

        if failed_channel is not None:

            normalized_channel = _normalize_edge(
                failed_channel
            )

            if normalized_channel is None:
                raise ValueError(
                    "failed_channel must be a valid edge"
                )

            if normalized_channel not in (
                self.failed_channels
            ):
                self.failed_channels.append(
                    normalized_channel
                )

        # ----------------------------------------------------
        # Failure reason
        # ----------------------------------------------------

        if reason is not None:

            self.failure_reasons.append(
                {
                    "candidate_index":
                        candidate_index,

                    "candidate_rank":
                        candidate_index + 1,

                    "reason":
                        reason,

                    "failed_channel":
                        normalized_channel,
                }
            )

        self.selected_candidate = None

        return self.next_candidate()

    # ========================================================
    # Failure
    # ========================================================

    def fail(self):
        """
        Mark Bucket as exhausted.
        """
        if self.status == "completed":
            return None

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
        Return True if lifecycle has terminated.
        """
        return self.status in {
            "completed",
            "failed",
        }

    # ========================================================
    # Information
    # ========================================================

    def info(self):
        """
        Return a snapshot of Bucket state.
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
                self.selected_candidate,
        }

    # ========================================================
    # Duplicate Failure Detection
    # ========================================================

    @staticmethod
    def _candidate_already_failed(
        candidate,
        failed_candidates=None,
    ):
        """
        Identity-first, equality-second duplicate check.
        """
        if failed_candidates is None:
            return False

        for failed in failed_candidates:

            if failed is candidate:
                return True

            if _safe_equal(
                failed,
                candidate,
            ):
                return True

        return False


# ============================================================
# Safe Equality
# ============================================================

def _safe_equal(left, right):
    """
    Compare two candidate objects safely.

    Prevents malformed equality implementations from breaking
    Bucket lifecycle operations.
    """
    if left is right:
        return True

    try:
        result = left == right

        if isinstance(result, bool):
            return result

        # Some objects return array-like equality results.
        try:
            return bool(result)
        except (TypeError, ValueError):
            return False

    except Exception:
        return False


# ============================================================
# Bucket Execution
# ============================================================

def execute_bucket(
    G,
    bucket,
    amount,
    rng=None,
):
    """
    Select the first structurally valid candidate.

    This function is deterministic.

    It does NOT:
        - execute payment
        - increment attempts
        - invoke FailureModel
        - perform stochastic routing

    Returns
    -------
    tuple
        (
            selection_success,
            selected_candidate,
            attempts
        )

    Important:
        selection_success == True means only that a structurally
        valid candidate was selected.

        It does NOT mean payment succeeded.
    """

    # `rng` is retained solely for API compatibility.
    # Stochastic behavior belongs to FailureModel.
    _ = rng

    if bucket is None:
        return (
            False,
            None,
            0,
        )

    if not isinstance(
        bucket,
        Bucket,
    ):
        raise TypeError(
            "bucket must be a Bucket instance"
        )

    if bucket.finished():

        return (
            bucket.status == "completed",
            bucket.selected_candidate,
            bucket.attempts,
        )

    # --------------------------------------------------------
    # Validate amount before scanning candidates
    # --------------------------------------------------------

    try:
        amount = _validate_positive_amount(
            amount
        )
    except ValueError:
        return (
            False,
            None,
            bucket.attempts,
        )

    # --------------------------------------------------------
    # Candidate scan
    # --------------------------------------------------------

    while not bucket.finished():

        candidate = bucket.current()

        if candidate is None:

            bucket.fail()

            break

        valid, failed_channel, reason = (
            validate_candidate(
                G,
                candidate,
                amount,
            )
        )

        # ----------------------------------------------------
        # Valid candidate
        # ----------------------------------------------------

        if valid:

            selected = (
                bucket.select_candidate(
                    candidate
                )
            )

            if selected is None:
                raise RuntimeError(
                    "Candidate validation succeeded "
                    "but candidate selection failed"
                )

            return (
                True,
                selected,
                bucket.attempts,
            )

        # ----------------------------------------------------
        # Invalid candidate
        # ----------------------------------------------------

        bucket.backtrack(
            failed_candidate=candidate,
            failed_channel=failed_channel,
            reason=reason,
        )

    return (
        False,
        None,
        bucket.attempts,
    )


# ============================================================
# Public API
# ============================================================

__all__ = [
    "Bucket",
    "execute_bucket",
    "validate_candidate",
    "_candidate_path",
    "_candidate_edges",
    "_normalize_edge",
    "_get_channel",
    "_get_known_liquidity",
    "_liquidity_is_sufficient",
]
