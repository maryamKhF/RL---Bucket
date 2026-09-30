# Bucket/test_real_gml.py

"""
Test the complete Bucket module using the real
Lightning Network topology:

    20190501.gml.geo

Tested components:

    1. GML loading
    2. Candidate generation
    3. CandidateManager
    4. Bucket
    5. Backtracker
    6. Channel failure
    7. Candidate fallback
    8. Final Bucket statistics
"""

import random
from pathlib import Path

import networkx as nx

from Bucket.bucket import Bucket
from Bucket.candidate_manager import CandidateManager
from Bucket.backtrack import Backtracker


# ==========================================================
# Configuration
# ==========================================================

GML_FILE = (
    Path(__file__).resolve().parent.parent
    / "20190501.gml.geo"
)

K_CANDIDATES = 5

PAYMENT_AMOUNT = 50_000

MAX_HOPS = 8

SEED = 42


# ==========================================================
# Real GML Test Class
# ==========================================================

class BucketRealGMLTest:
    """
    Test the complete Bucket module on the real
    Lightning Network GML topology.
    """

    def __init__(
        self,
        gml_path=GML_FILE,
        k=K_CANDIDATES,
        amount=PAYMENT_AMOUNT,
        max_hops=MAX_HOPS,
        seed=SEED
    ):

        self.gml_path = Path(gml_path)

        self.k = k

        self.amount = amount

        self.max_hops = max_hops

        self.rng = random.Random(seed)

        self.G = None

        self.candidates = []

        self.manager = None

        self.bucket = None

        self.backtracker = None

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
            f"Loading:\n{self.gml_path}"
        )

        # NetworkX reads GML directly.
        G = nx.read_gml(
            self.gml_path,
            label=None
        )

        # Convert to MultiDiGraph so that the
        # Bucket module can use:
        #
        # G.edges[u, v, k]

        if isinstance(
            G,
            nx.MultiDiGraph
        ):

            self.G = G

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
    # 2. Normalize Channel Attributes
    # ======================================================

    def _normalize_graph_attributes(self):
        """
        Make sure the attributes required by the
        Bucket module exist.

        The real GML snapshot may not contain all
        simulation-specific attributes such as:

            available
            balance_uv
            capacity
        """

        for node in self.G.nodes:

            self.G.nodes[node].setdefault(
                "online",
                True
            )

        for u, v, k, data in self.G.edges(
            keys=True,
            data=True
        ):

            # ------------------------------------------
            # Channel availability
            # ------------------------------------------

            data.setdefault(
                "available",
                True
            )

            # ------------------------------------------
            # Capacity
            # ------------------------------------------

            capacity = self._get_number(
                data,
                [
                    "capacity",
                    "capacity_sat",
                    "channel_capacity"
                ],
                default=1_000_000
            )

            data["capacity"] = capacity

            # ------------------------------------------
            # Directional liquidity
            # ------------------------------------------

            data.setdefault(
                "balance_uv",
                capacity
            )

            data.setdefault(
                "balance_vu",
                capacity
            )

            # ------------------------------------------
            # Failure statistics
            # ------------------------------------------

            data.setdefault(
                "failure_count",
                0
            )

            data.setdefault(
                "success_count",
                0
            )

    # ======================================================
    # Utility: Numeric Attribute
    # ======================================================

    @staticmethod
    def _get_number(
        data,
        keys,
        default=0
    ):

        for key in keys:

            if key not in data:
                continue

            try:

                return float(
                    data[key]
                )

            except (
                TypeError,
                ValueError
            ):

                continue

        return float(default)

    # ======================================================
    # 3. Find Source and Target
    # ======================================================

    def find_source_target(self):

        print(
            "\n===== 2. SELECT SOURCE / TARGET ====="
        )

        nodes = list(
            self.G.nodes()
        )

        if len(nodes) < 2:

            raise RuntimeError(
                "Graph does not contain "
                "enough nodes."
            )

        # Prefer high-degree nodes because they
        # are more likely to have multiple
        # alternative routes.

        degree_nodes = sorted(
            nodes,
            key=lambda n: self.G.degree(n),
            reverse=True
        )

        source = None

        target = None

        # Search among high-degree nodes.
        search_nodes = degree_nodes[:50]

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

        return source, target

    # ======================================================
    # 4. Generate Candidate Paths
    # ======================================================

    def generate_candidates(
        self,
        source,
        target
    ):

        print(
            "\n===== 3. GENERATE CANDIDATES ====="
        )

        # NetworkX's simple path generator is used
        # only to create alternative paths.
        #
        # CandidateManager remains responsible for
        # candidate management.

        paths = nx.shortest_simple_paths(
            nx.DiGraph(self.G),
            source,
            target
        )

        candidates = []

        for path in paths:

            if len(path) - 1 > self.max_hops:

                continue

            edges = []

            total_score = 0.0

            valid = True

            for u, v in zip(
                path[:-1],
                path[1:]
            ):

                if not self.G.has_edge(
                    u,
                    v
                ):

                    valid = False

                    break

                # Select the first available channel
                # between u and v.

                selected_key = None

                selected_data = None

                for key, data in self.G[
                    u
                ][v].items():

                    if data.get(
                        "available",
                        True
                    ):

                        selected_key = key

                        selected_data = data

                        break

                if selected_key is None:

                    valid = False

                    break

                edges.append(
                    (
                        u,
                        v,
                        selected_key
                    )
                )

                # Simple path score:
                # number of hops.
                #
                # CandidateManager can later rank
                # candidates using this score.

                total_score += 1.0

            if not valid:
                continue

            candidates.append(
                (
                    path,
                    edges,
                    total_score
                )
            )

            if len(candidates) >= self.k:
                break

        self.candidates = candidates

        if not self.candidates:

            raise RuntimeError(
                "No candidate paths were found."
            )

        print(
            f"Generated candidates: "
            f"{len(self.candidates)}"
        )

        for i, candidate in enumerate(
            self.candidates
        ):

            path, edges, score = candidate

            print(
                f"\nCandidate {i + 1}"
            )

            print(
                f"  Path : {path}"
            )

            print(
                f"  Hops : {len(path) - 1}"
            )

            print(
                f"  Score: {score}"
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

        # Filter candidates according to the
        # requested payment amount.

        self.manager.filter_candidates(
            self.G,
            self.amount
        )

        print(
            f"After filtering: "
            f"{len(self.manager.candidates)}"
        )

        # Rank candidates.

        self.manager.rank_candidates()

        print(
            "\nRanked candidates:"
        )

        for i, candidate in enumerate(
            self.manager.candidates
        ):

            path, edges, score = candidate

            print(
                f"  Rank {i + 1}: "
                f"path={path}, "
                f"score={score}"
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

        self.bucket = self.manager.create_bucket(
            tx_id=tx_id,
            k=self.k
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

        for i, candidate in enumerate(
            self.bucket.candidates
        ):

            print(
                f"  Candidate "
                f"{i + 1}: "
                f"{candidate[0]}"
            )

        return self.bucket

    # ======================================================
    # 7. Force Channel Failure
    # ======================================================

    def force_first_candidate_failure(self):

        print(
            "\n===== 6. FORCE CHANNEL FAILURE ====="
        )

        if not self.bucket.candidates:

            raise RuntimeError(
                "Bucket contains no candidates."
            )

        first_candidate = (
            self.bucket.candidates[0]
        )

        path, edges, score = first_candidate

        if not edges:

            raise RuntimeError(
                "First candidate has no edges."
            )

        failed_edge = edges[0]

        u, v, k = failed_edge

        channel = self.G.edges[
            u,
            v,
            k
        ]

        channel["available"] = False

        print(
            "Forced failure:"
        )

        print(
            f"  Candidate rank: 1"
        )

        print(
            f"  Channel: "
            f"{failed_edge}"
        )

        print(
            f"  Path: {path}"
        )

        return failed_edge

    # ======================================================
    # 8. Backtracking Test
    # ======================================================

    def test_backtracking(self):

        print(
            "\n===== 7. BACKTRACKING ====="
        )

        self.backtracker = Backtracker(
            max_attempts=len(
                self.bucket.candidates
            )
        )

        print(
            f"Initial candidate rank: "
            f"{self.bucket.current_index + 1}"
        )

        while not self.bucket.finished():

            candidate = self.bucket.current()

            if candidate is None:

                self.bucket.fail()

                break

            path, edges, score = candidate

            print(
                f"\nTrying Candidate "
                f"{self.bucket.current_index + 1}"
            )

            print(
                f"Path: {path}"
            )

            # ------------------------------------------
            # Validate current candidate
            # ------------------------------------------

            valid = True

            failed_channel = None

            for u, v, k in edges:

                channel = self.G.edges[
                    u,
                    v,
                    k
                ]

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

                    break

                if channel.get(
                    "balance_uv",
                    0
                ) < self.amount:

                    valid = False

                    failed_channel = (
                        u,
                        v,
                        k
                    )

                    break

            # ------------------------------------------
            # Success
            # ------------------------------------------

            if valid:

                self.bucket.mark_success(
                    candidate
                )

                print(
                    "\nSUCCESS"
                )

                break

            # ------------------------------------------
            # Failure
            # ------------------------------------------

            print(
                "Candidate FAILED"
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

    
    # 9. Final Report
   

    def print_report(self):

        print(
            "\n===== 8. FINAL BUCKET REPORT ====="
        )

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
            f"Selected candidate index: "
            f"{info['selected_candidate_index']}"
        )

        print(
            f"Selected candidate rank: "
            f"{info['selected_candidate_rank']}"
        )

        if info[
            "selected_candidate"
        ] is not None:

            path = info[
                "selected_candidate"
            ][0]

            print(
                f"Selected path: "
                f"{path}"
            )

        print(
            "\nFailed channels:"
        )

        for channel in info[
            "failed_channels"
        ]:

            print(
                f"  {channel}"
            )

        print(
            "\nFailed candidates:"
        )

        for i, candidate in enumerate(
            info["failed_candidates"]
        ):

            print(
                f"  Candidate "
                f"{i + 1}: "
                f"{candidate[0]}"
            )

   
    # 10. Run Complete Test
    

    def run(self):

        print(
            "=========================================="
        )

        print(
            " REAL GML BUCKET MODULE TEST"
        )

        print(
            "=========================================="
        )

        # 1. Load real topology

        self.load_graph()

        # 2. Find source / target

        source, target = (
            self.find_source_target()
        )

        # 3. Generate candidate paths

        self.generate_candidates(
            source,
            target
        )

        # 4. CandidateManager

        self.prepare_candidates()

        # 5. Bucket

        self.create_bucket()

        # Need at least two candidates
        # to test backtracking.

        if len(
            self.bucket.candidates
        ) < 2:

            raise RuntimeError(
                "At least two candidates are "
                "required to test Bucket "
                "backtracking."
            )

        # 6. Force failure

        self.force_first_candidate_failure()

        # 7. Backtracking

        self.test_backtracking()

        # 8. Final report

        self.print_report()

        print(
            "\n=========================================="
        )

        print(
            " TEST FINISHED"
        )

        print(
            "=========================================="
        )

        return self.bucket



# Main

if __name__ == "__main__":

    test = BucketRealGMLTest()

    test.run()