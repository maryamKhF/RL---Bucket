"""
Bucket/test_real_gml.py

Integration test for the Bucket module using the real
Lightning Network topology:

    20190501.gml.geo

Tested components
-----------------
1. Real GML loading
2. Graph normalization
3. Real Top-K candidate generation
4. CandidateManager
5. Bucket creation
6. Candidate validation
7. Forced channel failure
8. Bucket Backtracker
9. Candidate fallback
10. Final Bucket statistics

Important
---------
This test uses the existing Pathfinding module for candidate
generation.

It does NOT use:
- PPO / RL
- PaymentSimulator
- FailureModel
- PartialBacktracker
- Onion routing

The purpose of this test is specifically to verify:

    Real GML
        ↓
    Pathfinding Top-K
        ↓
    CandidateManager
        ↓
    Bucket
        ↓
    Backtracker
        ↓
    Alternative Candidate
"""

import random
from pathlib import Path

import networkx as nx

from Bucket.bucket import Bucket
from Bucket.candidate_manager import CandidateManager
from Bucket.backtrack import Backtracker

from Pathfinding.top_k_paths import top_k_paths
from Pathfinding.heuristics import lnd_cost


# ==========================================================
# Configuration
# ==========================================================

GML_FILE = (
    Path(__file__).resolve().parent.parent
    / "20190501.gml.geo"
)

K_CANDIDATES = 5

PAYMENT_AMOUNT = 10_000

MAX_HOPS = 12

ETA = 0.5

LAMBDA_H = 1.0

SEED = 42


# ==========================================================
# Real GML Bucket Test
# ==========================================================

class BucketRealGMLTest:
    """
    Integration test for:

        Pathfinding
            ↓
        CandidateManager
            ↓
        Bucket
            ↓
        Backtracker

    using the real Lightning Network GML snapshot.
    """

    def __init__(
        self,
        gml_path=GML_FILE,
        k=K_CANDIDATES,
        amount=PAYMENT_AMOUNT,
        max_hops=MAX_HOPS,
        eta=ETA,
        lambda_h=LAMBDA_H,
        seed=SEED
    ):

        self.gml_path = Path(
            gml_path
        )

        self.k = int(k)

        self.amount = float(amount)

        self.max_hops = int(max_hops)

        self.eta = float(eta)

        self.lambda_h = float(lambda_h)

        self.seed = seed

        self.rng = random.Random(
            seed
        )

        self.G = None

        self.source = None

        self.target = None

        self.candidates = []

        self.manager = None

        self.bucket = None

        self.backtracker = None

        self.forced_failed_edge = None

    # ======================================================
    # 1. Load GML
    # ======================================================

    def load_graph(self):

        print(
            "\n===== 1. LOAD GML GRAPH ====="
        )

        if not self.gml_path.exists():

            raise FileNotFoundError(
                f"GML file not found:\n"
                f"{self.gml_path}"
            )

        print(
            f"Loading:\n"
            f"  {self.gml_path}"
        )

        G = nx.read_gml(
            self.gml_path,
            label=None
        )

        # --------------------------------------------------
        # Normalize to MultiDiGraph
        # --------------------------------------------------

        if isinstance(
            G,
            nx.MultiDiGraph
        ):

            self.G = G

        elif isinstance(
            G,
            nx.MultiGraph
        ):

            self.G = nx.MultiDiGraph(
                G
            )

        elif isinstance(
            G,
            nx.DiGraph
        ):

            self.G = nx.MultiDiGraph(
                G
            )

        else:

            self.G = nx.MultiDiGraph(
                G
            )

        self._normalize_graph_attributes()

        print(
            "\nGraph loaded successfully."
        )

        print(
            f"Nodes   : "
            f"{self.G.number_of_nodes()}"
        )

        print(
            f"Channels: "
            f"{self.G.number_of_edges()}"
        )

        return self.G

    # ======================================================
    # 2. Normalize Graph Attributes
    # ======================================================

    def _normalize_graph_attributes(self):
        """
        Normalize only attributes required for Bucket testing.

        IMPORTANT
        ---------
        Unknown directional liquidity remains unknown.

        We intentionally DO NOT create:

            balance_uv = capacity

        because channel capacity is not directional liquidity.
        """

        # --------------------------------------------------
        # Node attributes
        # --------------------------------------------------

        for node in self.G.nodes:

            data = self.G.nodes[node]

            data.setdefault(
                "online",
                True
            )

            data.setdefault(
                "available",
                True
            )

        # --------------------------------------------------
        # Channel attributes
        # --------------------------------------------------

        for (
            u,
            v,
            k,
            data
        ) in self.G.edges(
            keys=True,
            data=True
        ):

            data.setdefault(
                "available",
                True
            )

            # Capacity may exist in the GML file.
            #
            # It is retained as channel capacity but is NOT
            # used as directional liquidity.

            capacity = self._get_number(
                data,
                [
                    "capacity",
                    "capacity_sat",
                    "channel_capacity"
                ],
                default=None
            )

            if capacity is not None:

                data["capacity"] = capacity

            # ------------------------------------------------
            # Directional liquidity
            # ------------------------------------------------
            #
            # DO NOT add:
            #
            # balance_uv = capacity
            #
            # because that would turn unknown liquidity into
            # artificial full liquidity.
            #
            # If the real GML contains a directional balance,
            # preserve it.
            #

            if "balance_uv" in data:

                data["balance_uv"] = (
                    self._safe_float_or_none(
                        data["balance_uv"]
                    )
                )

            if "balance_vu" in data:

                data["balance_vu"] = (
                    self._safe_float_or_none(
                        data["balance_vu"]
                    )
                )

            # ------------------------------------------------
            # Failure statistics
            # ------------------------------------------------

            data.setdefault(
                "failure_count",
                0
            )

            data.setdefault(
                "success_count",
                0
            )

    # ======================================================
    # Utility: Safe Number
    # ======================================================

    @staticmethod
    def _safe_float_or_none(
        value
    ):

        try:

            value = float(value)

        except (
            TypeError,
            ValueError
        ):

            return None

        if value < 0:

            return 0.0

        return value

    # ======================================================
    # Utility: Numeric Attribute
    # ======================================================

    @classmethod
    def _get_number(
        cls,
        data,
        keys,
        default=None
    ):

        for key in keys:

            if key not in data:
                continue

            value = cls._safe_float_or_none(
                data[key]
            )

            if value is not None:

                return value

        return default

    # ======================================================
    # 3. Find Source / Target
    # ======================================================

    def find_source_target(self):

        print(
            "\n===== 2. SELECT SOURCE / TARGET ====="
        )

        # --------------------------------------------------
        # Prefer largest strongly connected component
        # --------------------------------------------------

        try:

            components = (
                list(
                    nx.strongly_connected_components(
                        self.G
                    )
                )
            )

        except nx.NetworkXNotImplemented:

            components = []

        if components:

            largest_component = max(
                components,
                key=len
            )

            candidate_nodes = list(
                largest_component
            )

        else:

            candidate_nodes = list(
                self.G.nodes()
            )

        if len(candidate_nodes) < 2:

            raise RuntimeError(
                "Graph does not contain "
                "enough connected nodes."
            )

        # --------------------------------------------------
        # Prefer high-degree nodes
        # --------------------------------------------------

        degree_nodes = sorted(
            candidate_nodes,
            key=lambda n: self.G.degree(n),
            reverse=True
        )

        search_nodes = degree_nodes[
            : min(100, len(degree_nodes))
        ]

        source = None

        target = None

        # --------------------------------------------------
        # Find a pair with directed connectivity
        # --------------------------------------------------

        for s in search_nodes:

            for t in search_nodes:

                if s == t:
                    continue

                if nx.has_path(
                    self.G,
                    s,
                    t
                ):

                    source = s

                    target = t

                    break

            if source is not None:
                break

        if source is None:

            raise RuntimeError(
                "Could not find a connected "
                "source/target pair."
            )

        self.source = source

        self.target = target

        print(
            f"Source: {source}"
        )

        print(
            f"Target: {target}"
        )

        print(
            f"Source degree: "
            f"{self.G.degree(source)}"
        )

        print(
            f"Target degree: "
            f"{self.G.degree(target)}"
        )

        return (
            source,
            target
        )

    # ======================================================
    # 4. Generate Candidates Using Pathfinding
    # ======================================================

    def generate_candidates(
        self,
        source,
        target
    ):

        print(
            "\n===== 3. GENERATE TOP-K CANDIDATES ====="
        )

        print(
            "Using Pathfinding.top_k_paths()"
        )

        print(
            f"Amount   : {self.amount}"
        )

        print(
            f"K        : {self.k}"
        )

        print(
            f"Max hops : {self.max_hops}"
        )

        print(
            f"ETA      : {self.eta}"
        )

        print(
            f"Lambda_h : {self.lambda_h}"
        )

        # --------------------------------------------------
        # Generate candidates through the real Pathfinding
        # implementation.
        # --------------------------------------------------

        self.candidates = top_k_paths(
            G=self.G,
            source=source,
            target=target,
            amount=self.amount,
            heuristic_fn=lnd_cost,
            eta=self.eta,
            k=self.k,
            max_hops=self.max_hops,
            lambda_h=self.lambda_h
        )

        if not self.candidates:

            raise RuntimeError(
                "Pathfinding returned no "
                "candidate paths."
            )

        print(
            "\nGenerated candidates: "
            f"{len(self.candidates)}"
        )

        # --------------------------------------------------
        # Candidate validation
        # --------------------------------------------------

        for index, candidate in enumerate(
            self.candidates
        ):

            if not isinstance(
                candidate,
                dict
            ):

                raise TypeError(
                    "Pathfinding candidate must "
                    "be a dictionary."
                )

            path = candidate.get(
                "path"
            )

            edges = candidate.get(
                "edges"
            )

            cost = candidate.get(
                "cost"
            )

            if path is None:

                raise ValueError(
                    f"Candidate {index + 1} "
                    "has no path."
                )

            if edges is None:

                raise ValueError(
                    f"Candidate {index + 1} "
                    "has no edges."
                )

            if len(path) - 1 > self.max_hops:

                raise AssertionError(
                    f"Candidate {index + 1} "
                    "exceeds max_hops."
                )

            print(
                f"\nCandidate {index + 1}"
            )

            print(
                f"  Path       : {path}"
            )

            print(
                f"  Hops       : "
                f"{len(path) - 1}"
            )

            print(
                f"  Cost       : {cost}"
            )

            print(
                f"  Total fee  : "
                f"{candidate.get('total_fee')}"
            )

            print(
                f"  Reliability: "
                f"{candidate.get('reliability')}"
            )

        return self.candidates

    # ======================================================
    # 5. CandidateManager
    # ======================================================

    def prepare_candidates(self):

        print(
            "\n===== 4. CANDIDATE MANAGER ====="
        )

        self.manager = CandidateManager(
            candidates=self.candidates,
            max_candidates=self.k
        )

        print(
            f"Initial candidates: "
            f"{len(self.manager.candidates)}"
        )

        # --------------------------------------------------
        # Filter
        # --------------------------------------------------
        #
        # CandidateManager is responsible for candidate-level
        # filtering. Unknown liquidity must not automatically
        # be interpreted as zero.
        #

        self.manager.filter_candidates(
            self.G,
            self.amount
        )

        print(
            f"After filtering: "
            f"{len(self.manager.candidates)}"
        )

        if not self.manager.candidates:

            raise RuntimeError(
                "CandidateManager removed all "
                "candidate paths."
            )

        # --------------------------------------------------
        # Rank
        # --------------------------------------------------

        self.manager.rank_candidates()

        print(
            "\nRanked candidates:"
        )

        for index, candidate in enumerate(
            self.manager.candidates
        ):

            if isinstance(
                candidate,
                dict
            ):

                path = candidate.get(
                    "path"
                )

                cost = candidate.get(
                    "cost"
                )

                print(
                    f"  Rank {index + 1}: "
                    f"path={path}, "
                    f"cost={cost}"
                )

            else:

                print(
                    f"  Rank {index + 1}: "
                    f"{candidate}"
                )

        return self.manager

    # ======================================================
    # 6. Create Bucket
    # ======================================================

    def create_bucket(
        self,
        tx_id="real-gml-tx-001"
    ):

        print(
            "\n===== 5. CREATE BUCKET ====="
        )

        self.bucket = (
            self.manager.create_bucket(
                tx_id=tx_id,
                k=self.k
            )
        )

        if self.bucket is None:

            raise RuntimeError(
                "CandidateManager failed "
                "to create Bucket."
            )

        print(
            f"Bucket ID: "
            f"{self.bucket.bucket_id}"
        )

        print(
            f"Transaction ID: "
            f"{self.bucket.transaction_id}"
        )

        print(
            f"Candidates in Bucket: "
            f"{len(self.bucket.candidates)}"
        )

        if not self.bucket.candidates:

            raise RuntimeError(
                "Bucket contains no candidates."
            )

        for index, candidate in enumerate(
            self.bucket.candidates
        ):

            if isinstance(
                candidate,
                dict
            ):

                print(
                    f"  Candidate {index + 1}: "
                    f"{candidate.get('path')}"
                )

            else:

                print(
                    f"  Candidate {index + 1}: "
                    f"{candidate}"
                )

        return self.bucket

    # ======================================================
    # 7. Force First Candidate Failure
    # ======================================================

    def force_first_candidate_failure(self):

        print(
            "\n===== 6. FORCE CHANNEL FAILURE ====="
        )

        if self.bucket is None:

            raise RuntimeError(
                "Bucket has not been created."
            )

        if not self.bucket.candidates:

            raise RuntimeError(
                "Bucket contains no candidates."
            )

        first_candidate = (
            self.bucket.candidates[0]
        )

        if not isinstance(
            first_candidate,
            dict
        ):

            raise TypeError(
                "Bucket candidates must use "
                "the dictionary format."
            )

        edges = first_candidate.get(
            "edges"
        )

        path = first_candidate.get(
            "path"
        )

        if not edges:

            raise RuntimeError(
                "First candidate has no edges."
            )

        failed_edge = edges[0]

        if len(failed_edge) != 3:

            raise RuntimeError(
                "Expected directed edge "
                "format (u, v, key)."
            )

        u, v, k = failed_edge

        channel = self.G.edges[
            u,
            v,
            k
        ]

        # --------------------------------------------------
        # Force channel unavailable.
        # --------------------------------------------------

        channel["available"] = False

        self.forced_failed_edge = (
            u,
            v,
            k
        )

        print(
            "Forced failure:"
        )

        print(
            "  Candidate rank: 1"
        )

        print(
            f"  Channel: "
            f"{self.forced_failed_edge}"
        )

        print(
            f"  Path: {path}"
        )

        print(
            "  available: False"
        )

        return self.forced_failed_edge

    # ======================================================
    # 8. Backtracking Test
    # ======================================================

    def test_backtracking(self):

        print(
            "\n===== 7. BACKTRACKING ====="
        )

        if self.bucket is None:

            raise RuntimeError(
                "Bucket has not been created."
            )

        self.backtracker = Backtracker(
            max_attempts=max(
                1,
                len(
                    self.bucket.candidates
                )
            )
        )

        print(
            f"Initial candidate rank: "
            f"{self.bucket.current_index + 1}"
        )

        # --------------------------------------------------
        # Iterate through Bucket candidates.
        # --------------------------------------------------

        while not self.bucket.finished():

            candidate = self.bucket.current()

            if candidate is None:

                self.bucket.fail()

                break

            if not isinstance(
                candidate,
                dict
            ):

                raise TypeError(
                    "Bucket candidate must "
                    "be a dictionary."
                )

            path = candidate.get(
                "path"
            )

            edges = candidate.get(
                "edges"
            )

            rank = (
                self.bucket.current_index + 1
            )

            print(
                f"\nTrying Candidate {rank}"
            )

            print(
                f"Path: {path}"
            )

            # ------------------------------------------------
            # Validate candidate
            # ------------------------------------------------

            valid = True

            failed_channel = None

            failure_reason = None

            for edge in edges:

                if len(edge) != 3:

                    valid = False

                    failed_channel = edge

                    failure_reason = (
                        "invalid_edge"
                    )

                    break

                u, v, k = edge

                if not self.G.has_edge(
                    u,
                    v,
                    k
                ):

                    valid = False

                    failed_channel = (
                        u,
                        v,
                        k
                    )

                    failure_reason = (
                        "missing_channel"
                    )

                    break

                channel = self.G.edges[
                    u,
                    v,
                    k
                ]

                # --------------------------------------------
                # Availability
                # --------------------------------------------

                if not channel.get(
                    "available",
                    True
                ):

                    valid = False

                    failed_channel = (
                        u,
                        v,
                        k
                    )

                    failure_reason = (
                        "channel_unavailable"
                    )

                    break

                # --------------------------------------------
                # Known directional liquidity
                # --------------------------------------------
                #
                # IMPORTANT:
                # Missing balance_uv means UNKNOWN,
                # not zero.
                #

                liquidity = None

                for key in (
                    "estimated_liquidity",
                    "liquidity_uv",
                    "balance_uv"
                ):

                    if key not in channel:
                        continue

                    liquidity = (
                        self._safe_float_or_none(
                            channel[key]
                        )
                    )

                    if liquidity is not None:
                        break

                if (
                    liquidity is not None
                    and liquidity < self.amount
                ):

                    valid = False

                    failed_channel = (
                        u,
                        v,
                        k
                    )

                    failure_reason = (
                        "insufficient_liquidity"
                    )

                    break

            # ------------------------------------------------
            # Candidate success
            # ------------------------------------------------

            if valid:

                self.bucket.mark_success(
                    candidate
                )

                print(
                    "\nCandidate ACCEPTED"
                )

                print(
                    f"Selected rank: "
                    f"{self.bucket.selected_candidate_rank}"
                )

                break

            # ------------------------------------------------
            # Candidate failure
            # ------------------------------------------------

            print(
                "Candidate FAILED"
            )

            print(
                f"Failure reason: "
                f"{failure_reason}"
            )

            print(
                f"Failed channel: "
                f"{failed_channel}"
            )

            next_candidate = (
                self.backtracker.backtrack(
                    self.bucket,
                    failed_candidate=candidate,
                    failed_channel=failed_channel
                )
            )

            if next_candidate is None:

                break

            print(
                f"Backtracking to Candidate "
                f"{self.bucket.current_index + 1}"
            )

        return self.bucket

    # ======================================================
    # 9. Final Report
    # ======================================================

    def print_report(self):

        print(
            "\n===== 8. FINAL BUCKET REPORT ====="
        )

        if self.bucket is None:

            print(
                "Bucket does not exist."
            )

            return

        info = self.bucket.info()

        print(
            f"Bucket ID: "
            f"{info['bucket_id']}"
        )

        print(
            f"Transaction ID: "
            f"{info['transaction_id']}"
        )

        print(
            f"Total candidates: "
            f"{info['candidate_count']}"
        )

        print(
            f"Failed candidates: "
            f"{info['failed_candidates_count']}"
        )

        print(
            f"Failed channels: "
            f"{info['failed_channels_count']}"
        )

        print(
            f"Attempts: "
            f"{info['attempts']}"
        )

        print(
            f"Status: "
            f"{info['status']}"
        )

        print(
            f"Current index: "
            f"{info['current_index']}"
        )

        print(
            f"Selected candidate index: "
            f"{info['selected_candidate_index']}"
        )

        print(
            f"Selected candidate rank: "
            f"{info['selected_candidate_rank']}"
        )

        # --------------------------------------------------
        # Selected candidate
        # --------------------------------------------------

        selected = info[
            "selected_candidate"
        ]

        if selected is not None:

            if isinstance(
                selected,
                dict
            ):

                print(
                    f"Selected path: "
                    f"{selected.get('path')}"
                )

                print(
                    f"Selected cost: "
                    f"{selected.get('cost')}"
                )

            else:

                print(
                    f"Selected candidate: "
                    f"{selected}"
                )

        # --------------------------------------------------
        # Failed channels
        # --------------------------------------------------

        print(
            "\nFailed channels:"
        )

        if not info[
            "failed_channels"
        ]:

            print(
                "  None"
            )

        else:

            for channel in info[
                "failed_channels"
            ]:

                print(
                    f"  {channel}"
                )

        # --------------------------------------------------
        # Failed candidates
        # --------------------------------------------------

        print(
            "\nFailed candidates:"
        )

        if not info[
            "failed_candidates"
        ]:

            print(
                "  None"
            )

        else:

            for index, candidate in enumerate(
                info["failed_candidates"]
            ):

                if isinstance(
                    candidate,
                    dict
                ):

                    print(
                        f"  Candidate "
                        f"{index + 1}: "
                        f"{candidate.get('path')}"
                    )

                else:

                    print(
                        f"  Candidate "
                        f"{index + 1}: "
                        f"{candidate}"
                    )

    # ======================================================
    # 10. Validation
    # ======================================================

    def validate_result(self):

        print(
            "\n===== 9. VALIDATION ====="
        )

        if self.bucket is None:

            raise AssertionError(
                "Bucket was not created."
            )

        # --------------------------------------------------
        # Candidate existence
        # --------------------------------------------------

        if len(
            self.bucket.candidates
        ) < 2:

            raise AssertionError(
                "At least two candidates are "
                "required for fallback testing."
            )

        # --------------------------------------------------
        # First candidate must be recorded
        # as failed because we forced its
        # first channel unavailable.
        # --------------------------------------------------

        if not self.bucket.failed_candidates:

            raise AssertionError(
                "No failed candidate was recorded."
            )

        if self.forced_failed_edge is not None:

            if (
                self.forced_failed_edge
                not in self.bucket.failed_channels
            ):

                raise AssertionError(
                    "Forced failed channel was "
                    "not recorded by Bucket."
                )

        # --------------------------------------------------
        # Backtracker attempt
        # --------------------------------------------------

        if self.backtracker is None:

            raise AssertionError(
                "Backtracker was not created."
            )

        if self.backtracker.attempts < 1:

            raise AssertionError(
                "Backtracker did not perform "
                "any backtracking attempt."
            )

        # --------------------------------------------------
        # Selected candidate
        # --------------------------------------------------

        if self.bucket.status != "completed":

            raise AssertionError(
                "Bucket did not complete "
                "successfully after fallback."
            )

        if self.bucket.selected_candidate is None:

            raise AssertionError(
                "No selected candidate was recorded."
            )

        if (
            self.bucket.selected_candidate_rank
            is None
        ):

            raise AssertionError(
                "Selected candidate rank "
                "was not recorded."
            )

        if (
            self.bucket.selected_candidate_rank
            <= 1
        ):

            raise AssertionError(
                "The first candidate was forced "
                "to fail, so a later candidate "
                "should have been selected."
            )

        print(
            "Candidate generation : PASS"
        )

        print(
            "CandidateManager      : PASS"
        )

        print(
            "Bucket creation       : PASS"
        )

        print(
            "Forced failure        : PASS"
        )

        print(
            "Backtracking          : PASS"
        )

        print(
            "Candidate fallback    : PASS"
        )

        print(
            "Final state           : PASS"
        )

    # ======================================================
    # 11. Run Complete Test
    # ======================================================

    def run(self):

        print(
            "======================================================"
        )

        print(
            " REAL GML BUCKET INTEGRATION TEST"
        )

        print(
            "======================================================"
        )

        print(
            f"Snapshot : {self.gml_path.name}"
        )

        print(
            f"Amount   : {self.amount}"
        )

        print(
            f"K        : {self.k}"
        )

        print(
            f"Max hops : {self.max_hops}"
        )

        print(
            f"ETA      : {self.eta}"
        )

        print(
            f"Lambda_h : {self.lambda_h}"
        )

        print(
            f"Seed     : {self.seed}"
        )

        # --------------------------------------------------
        # 1. Load graph
        # --------------------------------------------------

        self.load_graph()

        # --------------------------------------------------
        # 2. Source / target
        # --------------------------------------------------

        source, target = (
            self.find_source_target()
        )

        # --------------------------------------------------
        # 3. Pathfinding Top-K
        # --------------------------------------------------

        self.generate_candidates(
            source,
            target
        )

        # --------------------------------------------------
        # 4. CandidateManager
        # --------------------------------------------------

        self.prepare_candidates()

        # --------------------------------------------------
        # 5. Bucket
        # --------------------------------------------------

        self.create_bucket()

        # --------------------------------------------------
        # Require at least two candidates
        # --------------------------------------------------

        if len(
            self.bucket.candidates
        ) < 2:

            raise RuntimeError(
                "At least two candidates are "
                "required to test Bucket "
                "backtracking."
            )

        # --------------------------------------------------
        # 6. Force first candidate failure
        # --------------------------------------------------

        self.force_first_candidate_failure()

        # --------------------------------------------------
        # 7. Backtracking
        # --------------------------------------------------

        self.test_backtracking()

        # --------------------------------------------------
        # 8. Report
        # --------------------------------------------------

        self.print_report()

        # --------------------------------------------------
        # 9. Validation
        # --------------------------------------------------

        self.validate_result()

        print(
            "\n======================================================"
        )

        print(
            " BUCKET TEST STATUS : SUCCESS"
        )

        print(
            "======================================================"
        )

        return self.bucket


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    test = BucketRealGMLTest()

    test.run()