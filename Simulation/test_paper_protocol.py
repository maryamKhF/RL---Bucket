import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Runtime.paper_protocol import (
    PAPER_EVALUATION_COUNT,
    PAPER_REPETITIONS,
    PAPER_TRAINING_COUNT,
    PAPER_TRANSACTION_COUNT,
    _mean_ci95,
    _run_trial,
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
        _validate_protocol(valid_config(), [0.01, 0.03, 0.06], PAPER_REPETITIONS)

    def test_rejects_non_article_transaction_pool(self):
        config = valid_config()
        config["n_transactions"] = 30
        with self.assertRaisesRegex(ValueError, "n_transactions"):
            _validate_protocol(config, [0.03], PAPER_REPETITIONS)

    def test_rejects_wrong_test_count_and_repetition_count(self):
        config = valid_config()
        config["evaluation"]["transaction_count"] = 10
        config["evaluation"]["repetitions"] = 1
        with self.assertRaisesRegex(ValueError, "transaction_count"):
            _validate_protocol(config, [0.03], PAPER_REPETITIONS)

    def test_rejects_invalid_failure_rate(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            _validate_protocol(valid_config(), [1.1], PAPER_REPETITIONS)

    def test_confidence_interval_requires_multiple_seed_values(self):
        single = _mean_ci95([0.8])
        four_seeds = _mean_ci95([0.7, 0.8, 0.9, 0.8])
        repeated = _mean_ci95([0.7, 0.8, 0.9, 0.8, 0.8])
        self.assertIsNone(single["ci_low"])
        self.assertEqual(four_seeds["n"], 4)
        self.assertEqual(repeated["n"], PAPER_REPETITIONS)
        self.assertLess(repeated["ci_low"], repeated["mean"])
        self.assertGreater(repeated["ci_high"], repeated["mean"])


class Main4ParallelEvaluationTests(unittest.TestCase):
    def test_single_trial_after_parallel_training_and_reload(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from Pathfinding.heuristics import lnd_cost
        from RL.environment import RoutingEnv
        from RL.ppo_agent import build_ppo
        from RL.test_environment import make_graph, make_transaction
        from RL.test_ppo_agent import valid_config

        graph = make_graph()
        for _, _, attrs in graph.edges(data=True):
            attrs.update(
                fee_base=1.0, fee_rate=0.0, delay=1.0,
                simulator_failure_probability=0.0,
            )
        transaction = make_transaction()
        cfg = valid_config()
        cfg["seed"] = 42
        cfg["graph"] = {"max_hops": 12, "neighborhood_m": 5}
        cfg["rl"].update(n_steps=2, batch_size=2, n_epochs=1)

        def make_env():
            return RoutingEnv(
                copy.deepcopy(graph), [copy.deepcopy(transaction)], lnd_cost,
                copy.deepcopy(cfg), mode="train",
            )

        with tempfile.TemporaryDirectory(prefix="main4_evaluation_") as tmp:
            with patch.dict("os.environ", {
                "RL_REPORT_FILE": str(Path(tmp) / "report.txt"),
                "RL_FAST_ENV": "0",
            }):
                from Runtime import experiment_core as core

                # Include the exact worker count from the failed server run.
                for workers in (1, 25):
                    with self.subTest(workers=workers):
                        env = DummyVecEnv([make_env for _ in range(workers)])
                        try:
                            model = build_ppo(env, cfg, seed=42)
                            model.learn(total_timesteps=2 * workers)
                            saved = Path(tmp) / f"model_{workers}.zip"
                            model.save(str(saved))
                            with patch(
                                "stable_baselines3.common.base_class.load_from_zip_file",
                                core._load_from_zip_file_bytesio,
                            ):
                                restored = PPO.load(str(saved), device="cpu")
                            for policy in (model, restored):
                                attached_env = policy.get_env()
                                with patch.object(policy, "set_env", side_effect=AssertionError(
                                    "Evaluation must not replace PPO's training environment"
                                )):
                                    results = [
                                        _run_trial(
                                            core, graph, cfg, policy, transaction,
                                            index, 42, 1, 0.06, 0,
                                        )
                                        for index in (1, 2)
                                    ]
                                self.assertEqual(policy.n_envs, workers)
                                self.assertIs(policy.get_env(), attached_env)
                                self.assertEqual(len(results), 2)
                                self.assertEqual(results[0]["transaction_id"], transaction.tx_id)
                                self.assertEqual(results[0]["evaluation_seed"], 10_000_043)
                                self.assertEqual(results[1]["evaluation_seed"], 10_000_044)
                                self.assertIsInstance(results[0]["success"], bool)
                                self.assertEqual(results[0]["top_k"], 5)
                        finally:
                            env.close()


if __name__ == "__main__":
    unittest.main()
