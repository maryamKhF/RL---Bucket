"""Reporting tests only; no routing, model inference or experiment execution."""

import copy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from Evaluations.xai_report import human_explanation


class RecoveryReportTests(unittest.TestCase):
    def record(self):
        path = ["A", "B", "C", "D"]
        edges = [["A", "B", "ab"], ["B", "C", "bc"], ["C", "D", "cd"]]
        return {"success": True, "explanation": {"policy": {}, "routing": {
            "decision": {"success": True, "path": path, "attempt_count": 2,
                         "successful_bucket_rank": None, "successful_route_source": "partial_backtrack",
                         "partial_backtrack_retry_count": 1, "source_retry_count": 0,
                         "preserved_prefix_hops": 1, "backtracked_hops": 0,
                         "recovery_events": [{"type": "partial_backtrack", "attempt_id": 2,
                                              "new_route": path, "branch_node": "B",
                                              "preserved_prefix_hops": 1, "backtracked_hops": 0,
                                              "success": True, "reason": "success"}]},
            "buckets": [{"bucket_id": 7, "candidates": [{"rank": 3, "path": path, "channels": copy.deepcopy(edges)}]}],
            "payment_attempts": [{"channels": []}, {"channels": [{"channel": e} for e in edges]}],
        }}}

    def test_candidate_rank_and_payment_attempt_number_are_distinct(self):
        record = self.record()
        before = copy.deepcopy(record)
        human = human_explanation(record)
        self.assertEqual(human["successful_bucket_rank"], 3)
        self.assertEqual(human["payment_attempt_count"], 2)
        self.assertIn("payment attempt #2", human["recovery_summary"])
        self.assertIn("candidate rank=3", human["recovery_summary"])
        self.assertIn("no backward hops", human["recovery_summary"])
        self.assertEqual(record, before)  # Reports must not modify scientific results.

    def test_same_node_path_with_different_parallel_channel_has_no_matching_rank(self):
        record = self.record()
        record["explanation"]["routing"]["buckets"][0]["candidates"][0]["channels"][1][2] = "different-channel"
        human = human_explanation(record)
        self.assertIsNone(human["successful_bucket_rank"])
        self.assertIn("combined/recomputed", human["bucket_rank_status"])

    def test_failed_recovery_searches_do_not_disappear(self):
        record = self.record()
        record["success"] = False
        decision = record["explanation"]["routing"]["decision"]
        decision.update(success=False, path=[], recovery_events=[
            {"type": "recovery_unavailable", "attempt_id": 1, "reason": "no_valid_bucket_alternative"},
            {"type": "full_reroute", "after_attempt": 1, "success": False, "reason": "bucket_exhausted"}])
        human = human_explanation(record)
        self.assertIn("Recovery unavailable", human["recovery_summary"])
        self.assertIn("route_found=False", human["recovery_summary"])
        self.assertNotIn("No recovery attempt", human["recovery_summary"])
        self.assertIsNone(human["successful_bucket_rank"])

    def test_ambiguous_candidates_are_not_assigned_an_arbitrary_rank(self):
        record = self.record()
        candidates = record["explanation"]["routing"]["buckets"][0]["candidates"]
        other = copy.deepcopy(candidates[0])
        other["rank"] = 4
        candidates.append(other)
        human = human_explanation(record)
        self.assertIsNone(human["successful_bucket_rank"])
        self.assertIn("ambiguous", human["bucket_rank_status"])

    def test_legacy_records_use_the_recorded_rank_without_inventing_zero_counts(self):
        record = self.record()
        routing = record["explanation"]["routing"]
        routing.pop("payment_attempts")
        routing["decision"] = {"success": True, "successful_bucket_rank": 1, "attempt_count": 1}
        human = human_explanation(record)
        self.assertEqual(human["successful_bucket_rank"], 1)
        self.assertIsNone(human["backtracked_hops"])
        self.assertEqual(human["recovery_summary"], "No recovery attempt recorded.")

    def test_flat_export_refresh_changes_only_the_explanation_column(self):
        from Evaluations.round_one_report import refresh_trial_explanations
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = self.record()
            record["payment_seed"] = 123
            with gzip.open(root / "evaluation_explanations.jsonl.gz", "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            original = {"payment_seed": "123", "reward": "1.234500", "human_explanation": "stale"}
            path = root / "evaluation_trials.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(original))
                writer.writeheader(); writer.writerow(original)
            refresh_trial_explanations(root)
            with path.open(encoding="utf-8", newline="") as handle:
                corrected = next(csv.DictReader(handle))
            self.assertEqual(corrected["reward"], original["reward"])
            self.assertEqual(corrected["payment_seed"], original["payment_seed"])
            self.assertIn("Bucket rank=3", corrected["human_explanation"])
            self.assertIn("payment attempt #2", corrected["human_explanation"])
            # A mismatched trace must not silently attach prose to another trial.
            before = path.read_bytes()
            record["payment_seed"] = 999
            with gzip.open(root / "evaluation_explanations.jsonl.gz", "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "identities/order differ"):
                refresh_trial_explanations(root)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
