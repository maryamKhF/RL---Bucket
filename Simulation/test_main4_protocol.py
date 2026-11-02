import unittest

from main4 import (
    PAPER_EVALUATION_COUNT,
    PAPER_REPETITIONS,
    PAPER_TRAINING_COUNT,
    PAPER_TRANSACTION_COUNT,
    _mean_ci95,
    _validate_protocol,
)


def valid_config():
    return {
        "n_transactions": PAPER_TRANSACTION_COUNT,
        "train_ratio": 0.70,
        "test_ratio": 0.30,
        "evaluation": {
            "transaction_count": PAPER_EVALUATION_COUNT,
            "repetitions": PAPER_REPETITIONS,
        },
    }


class Main4ProtocolTests(unittest.TestCase):
    def test_accepts_article_protocol(self):
        _validate_protocol(valid_config(), [0.01, 0.03, 0.06])

    def test_rejects_non_article_transaction_pool(self):
        config = valid_config()
        config["n_transactions"] = 30
        with self.assertRaisesRegex(ValueError, "n_transactions"):
            _validate_protocol(config, [0.03])

    def test_rejects_wrong_test_count_and_repetition_count(self):
        config = valid_config()
        config["evaluation"]["transaction_count"] = 10
        config["evaluation"]["repetitions"] = 1
        with self.assertRaisesRegex(ValueError, "transaction_count"):
            _validate_protocol(config, [0.03])

    def test_rejects_invalid_failure_rate(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            _validate_protocol(valid_config(), [1.1])

    def test_confidence_interval_requires_multiple_seed_values(self):
        single = _mean_ci95([0.8])
        four_seeds = _mean_ci95([0.7, 0.8, 0.9, 0.8])
        repeated = _mean_ci95([0.7, 0.8, 0.9, 0.8, 0.8])
        self.assertIsNone(single["ci_low"])
        self.assertEqual(four_seeds["n"], 4)
        self.assertEqual(repeated["n"], PAPER_REPETITIONS)
        self.assertLess(repeated["ci_low"], repeated["mean"])
        self.assertGreater(repeated["ci_high"], repeated["mean"])


if __name__ == "__main__":
    unittest.main()
