import unittest

from RL.reward import calculate_reward


class RewardFormulaTests(unittest.TestCase):
    def test_paper_baseline_uses_route_node_count_over_total_carbon(self):
        self.assertAlmostEqual(
            calculate_reward(True, path_length=2, carbon_intensity=100.0),
            10.0,
        )

    def test_amount_variant_uses_fixed_reference_normalization(self):
        value = calculate_reward(
            True, 2, 100.0, formula="amount", payment_amount=500_000
        )
        self.assertAlmostEqual(value, 5.0)

    def test_liquidity_margin_and_persistence_multiply_base(self):
        baseline = calculate_reward(True, 2, 100.0)
        self.assertAlmostEqual(
            calculate_reward(
                True, 2, 100.0, formula="liquidity_margin", liquidity_margin=1.5
            ),
            baseline * 1.5,
        )
        self.assertAlmostEqual(
            calculate_reward(True, 2, 100.0, formula="persistence", persistence=0.8),
            baseline * 0.8,
        )

    def test_fee_delay_and_hops_use_carbon_equivalent_denominators(self):
        fee = calculate_reward(True, 2, 100.0, fee=1000, formula="fee")
        delay = calculate_reward(True, 2, 100.0, delay=10, formula="delay")
        hops = calculate_reward(True, 2, 100.0, formula="hops")
        self.assertAlmostEqual(fee, 3000.0 / 400.0)
        self.assertAlmostEqual(delay, 3000.0 / 400.0)
        self.assertAlmostEqual(hops, 3000.0 / 320.0)

    def test_unsuccessful_payment_has_zero_reward_for_every_formula(self):
        for formula in (
            "paper_base",
            "amount",
            "liquidity_margin",
            "persistence",
            "fee",
            "delay",
            "hops",
        ):
            with self.subTest(formula=formula):
                self.assertEqual(
                    calculate_reward(False, 2, 100.0, formula=formula), 0.0
                )

    def test_unknown_formula_is_rejected_for_successful_payment(self):
        with self.assertRaisesRegex(ValueError, "Unknown reward formula"):
            calculate_reward(True, 2, 100.0, formula="not-a-formula")


if __name__ == "__main__":
    unittest.main()
