import unittest

from Simulation.transaction_split import (
    TRAIN_RATIO,
    normalize_evaluation_count,
    split_transaction_pool,
    training_count_for_evaluation_count,
)


class TransactionSplitTests(unittest.TestCase):

    def test_configured_default_is_exactly_seventy_thirty(self):
        training_count = training_count_for_evaluation_count(3000)
        self.assertEqual(training_count, 7000)
        self.assertEqual(training_count / (training_count + 3000), TRAIN_RATIO)

    def test_arbitrary_target_is_normalized_for_an_exact_split(self):
        self.assertEqual(normalize_evaluation_count(10), 9)
        self.assertEqual(training_count_for_evaluation_count(10), 21)
        self.assertEqual(normalize_evaluation_count(100), 99)
        self.assertEqual(training_count_for_evaluation_count(100), 231)
        self.assertEqual(normalize_evaluation_count(11), 12)

    def test_exact_ratio_holds_for_a_range_of_requested_counts(self):
        for requested_count in range(1, 201):
            evaluation_count = normalize_evaluation_count(requested_count)
            training_count = training_count_for_evaluation_count(requested_count)
            total_count = training_count + evaluation_count

            self.assertEqual(evaluation_count % 3, 0)
            self.assertEqual(training_count / total_count, TRAIN_RATIO)

    def test_split_uses_disjoint_ordered_slices(self):
        training_count = training_count_for_evaluation_count(10)
        pool = list(range(training_count + normalize_evaluation_count(10)))

        training, evaluation = split_transaction_pool(pool, 10)

        self.assertEqual(training, list(range(21)))
        self.assertEqual(evaluation, list(range(21, 30)))
        self.assertFalse(set(training).intersection(evaluation))
        self.assertEqual(len(training) / len(pool), TRAIN_RATIO)

    def test_split_rejects_incomplete_pool(self):
        with self.assertRaises(ValueError):
            split_transaction_pool(list(range(29)), 10)

    def test_nonpositive_evaluation_count_is_rejected(self):
        with self.assertRaises(ValueError):
            training_count_for_evaluation_count(0)


if __name__ == "__main__":
    unittest.main()
