"""Regression checks for server-sized PPO rollouts and subprocess training."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from Pathfinding.heuristics import lnd_cost
from RL.environment import RoutingEnv
from RL.ppo_agent import build_ppo
from RL.test_environment import make_graph, make_transactions
from RL.test_ppo_agent import DummyRoutingEnv, valid_config
from RL.train import resolve_vectorized_training_configuration, train_agent


class ParallelTrainingTests(unittest.TestCase):
    def test_continual_target_keeps_the_previous_stage_rollout_and_minibatch_budget(self):
        resolved = resolve_vectorized_training_configuration(
            {"total_timesteps": 41780, "n_steps": 2048, "batch_size": 64},
            30, rollout_timesteps=20000,
        )
        self.assertEqual(resolved["total_timesteps"], 41780)
        self.assertEqual(resolved["n_steps"], 66)
        self.assertEqual(resolved["batch_size"], 60)

    def test_server_rollout_has_complete_minibatches(self):
        resolved = resolve_vectorized_training_configuration(
            {"total_timesteps": 20000, "n_steps": 2048, "batch_size": 64}, 25
        )
        self.assertEqual(resolved["n_steps"], 80)
        self.assertEqual(resolved["batch_size"], 50)
        self.assertEqual(resolved["total_timesteps"], 20000)
        self.assertEqual((80 * 25) % 50, 0)

    def test_single_environment_configuration_is_preserved(self):
        cfg = {"total_timesteps": 20000, "n_steps": 2048, "batch_size": 64}
        self.assertEqual(resolve_vectorized_training_configuration(cfg, 1), cfg)

    def test_server_configuration_builds_a_vectorized_ppo(self):
        env = DummyVecEnv([DummyRoutingEnv for _ in range(25)])
        cfg = valid_config()
        cfg["rl"].update(resolve_vectorized_training_configuration(
            {"total_timesteps": 20000, "n_steps": 2048, "batch_size": 64}, 25
        ))
        try:
            model = build_ppo(env, cfg, seed=42)
            self.assertEqual(model.rollout_buffer.buffer_size * model.n_envs, 2000)
            self.assertEqual(model.batch_size, 50)
        finally:
            env.close()

    def test_batch_can_span_multiple_environments(self):
        env = DummyVecEnv([DummyRoutingEnv for _ in range(4)])
        cfg = valid_config()
        cfg["rl"].update(n_steps=8, batch_size=32)
        try:
            model = build_ppo(env, cfg, seed=42)
            self.assertEqual(model.batch_size, 32)
        finally:
            env.close()

    def test_incomplete_combined_minibatch_is_rejected(self):
        env = DummyVecEnv([DummyRoutingEnv for _ in range(25)])
        cfg = valid_config()
        cfg["rl"].update(n_steps=80, batch_size=64)
        try:
            with self.assertRaisesRegex(ValueError, "rollout size"):
                build_ppo(env, cfg, seed=42)
        finally:
            env.close()

    def test_real_routing_trains_in_two_processes_and_saves_model(self):
        graph = make_graph()
        for _, _, attrs in graph.edges(data=True):
            attrs.update(failure_probability=0.06, fee_base=1.0, fee_rate=0.0, delay=1.0)
        transactions = make_transactions(4)
        cfg = valid_config()
        cfg["graph"] = {"max_hops": 12, "neighborhood_m": 5}
        cfg["rl"].update(
            total_timesteps=40, n_steps=8, batch_size=8, n_epochs=1, checkpoint_freq=16
        )
        with tempfile.TemporaryDirectory(prefix="rl_parallel_regression_") as tmp:
            root = Path(tmp)
            cfg["rl"].update(
                model_dir=str(root / "models"),
                log_dir=str(root / "logs"),
                checkpoint_dir=str(root / "checkpoints"),
            )
            with patch.dict("os.environ", {
                "RL_ENV_WORKERS": "2",
                "RL_FAST_ENV": "0",
            }):
                model = train_agent(
                    graph, transactions, lnd_cost, cfg,
                    "parallel_regression", seed=42, fast_training=False,
                )
            self.assertEqual(model.num_timesteps, 40)
            self.assertGreater(model._n_updates, 0)
            saved = root / "models" / "parallel_regression.zip"
            self.assertTrue(saved.is_file())
            self.assertTrue(list((root / "checkpoints").glob("*.zip")))
            metadata = json.loads((root / "models" / "parallel_regression_metadata.json").read_text())
            self.assertEqual(metadata["n_envs"], 2)
            self.assertEqual(metadata["batch_size"], 4)
            # Use the maintained ZIP compatibility loader, also used by stages.
            from Evaluations.explainable_routing import load_policy
            restored = load_policy(saved, device="cpu")
            checkpoint = sorted((root / "checkpoints").glob("*.zip"))[-1]
            restored_checkpoint = load_policy(checkpoint, device="cpu")
            self.assertGreater(restored_checkpoint.num_timesteps, 0)
            env = RoutingEnv(graph, transactions, lnd_cost, cfg, mode="eval")
            try:
                observation, _ = env.reset(seed=42)
                action, _ = restored.predict(observation, deterministic=True)
                self.assertEqual(action.shape, (1,))
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
