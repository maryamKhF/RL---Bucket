"""Exercise real subprocess scheduling, PPO trajectories and interrupted work."""

from functools import partial
import time
import unittest

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from RL.ready_rollouts import ReadyPPO, ReadySubprocVecEnv


class DelayedEnv(gym.Env):
    def __init__(self, rank, flip=False, delay=.08):
        self.rank, self.flip, self.delay = rank, flip, delay
        self.observation_space = spaces.Box(-100, 100, (2,), dtype=np.float32)
        self.action_space = spaces.Box(-.1, .1, (1,), dtype=np.float32)
        self.level = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([self.rank, self.level], dtype=np.float32), {}

    def step(self, action):
        if (self.level + self.rank + int(self.flip)) % 2 == 0:
            time.sleep(self.delay)
        level = self.level
        self.level += 1
        return (np.array([self.rank, self.level], dtype=np.float32),
                float(10 * self.rank + level), self.level % 3 == 0, False,
                {"received_action": float(action[0]), "level": level})


class Audit(BaseCallback):
    def __init__(self, stop=False):
        super().__init__()
        self.rows, self.stop = [], stop

    def _on_step(self):
        self.rows.extend(self.locals["infos"])
        return not self.stop


class ReadyRolloutTests(unittest.TestCase):
    def collect(self, ready=True, flip=False, delay=.08, stop=False):
        torch.set_num_threads(1)
        env = ReadySubprocVecEnv([partial(DelayedEnv, rank, flip, delay) for rank in range(2)],
                                 start_method="spawn")
        model = (ReadyPPO if ready else PPO)("MlpPolicy", env, n_steps=6, batch_size=6,
                   n_epochs=1, seed=45, device="cpu", policy_kwargs={"net_arch": [8, 8]})
        audit = Audit(stop)
        _, callback = model._setup_learn(12, audit)
        callback.on_training_start({}, {})
        before = {key: value.clone() for key, value in model.policy.state_dict().items()}
        started = time.perf_counter()
        try:
            if stop:
                with self.assertRaises(InterruptedError):
                    model.collect_rollouts(env, callback, model.rollout_buffer, 6)
                self.assertTrue(env.closed)
                self.assertFalse(model.rollout_buffer.full)
                self.assertEqual(model.num_timesteps, 0)
            else:
                self.assertTrue(model.collect_rollouts(env, callback, model.rollout_buffer, 6))
            elapsed = time.perf_counter() - started
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.policy.state_dict()[key]), key)
            return model, audit, elapsed
        finally:
            env.close()

    def test_equal_worker_budget_frozen_policy_correct_gae_and_unclipped_log_probs(self):
        model, audit, _ = self.collect(delay=0)
        buffer = model.rollout_buffer
        self.assertTrue(buffer.full)
        self.assertEqual(model.num_timesteps, 12)
        self.assertEqual(audit.n_calls, 6)
        self.assertEqual(model.last_rollout_performance["per_worker"], [6, 6])
        for rank in range(2):
            np.testing.assert_array_equal(buffer.observations[:, rank, 1], np.arange(6))
            self.assertEqual(sorted(row["level"] for row in audit.rows if row["worker_rank"] == rank), list(range(6)))
            for row in (row for row in audit.rows if row["worker_rank"] == rank):
                self.assertEqual(row["audit_timestep"], (row["rollout_step"] + 1) * 2)
                action = buffer.actions[row["rollout_step"], rank, 0]
                self.assertAlmostEqual(row["received_action"], float(np.clip(action, -.1, .1)), places=6)
        with torch.no_grad():
            values, probabilities, _ = model.policy.evaluate_actions(
                torch.as_tensor(buffer.observations.reshape(-1, 2)),
                torch.as_tensor(buffer.actions.reshape(-1, 1)))
        np.testing.assert_allclose(buffer.values, values.numpy().reshape(6, 2), atol=1e-6)
        np.testing.assert_allclose(buffer.log_probs, probabilities.numpy().reshape(6, 2), atol=1e-6)
        expected = np.zeros_like(buffer.advantages)
        for rank in range(2):
            tail = 0.0
            for step in reversed(range(6)):
                nonterminal = float((step + 1) % 3 != 0)
                next_value = buffer.values[step + 1, rank] if step < 5 else 0.0
                delta = buffer.rewards[step, rank] + model.gamma * next_value * nonterminal - buffer.values[step, rank]
                tail = delta + model.gamma * model.gae_lambda * nonterminal * tail
                expected[step, rank] = tail
        np.testing.assert_allclose(buffer.advantages, expected, atol=1e-5)

    def test_completion_order_does_not_change_noise_trajectories_or_returns(self):
        first, _, _ = self.collect(flip=False, delay=.02)
        second, _, _ = self.collect(flip=True, delay=.02)
        for name in ("observations", "actions", "values", "log_probs", "rewards", "returns", "episode_starts"):
            np.testing.assert_allclose(getattr(first.rollout_buffer, name),
                                       getattr(second.rollout_buffer, name), atol=1e-6, err_msg=name)

    def test_removes_per_step_barriers_on_alternating_slow_payments(self):
        _, _, synchronous = self.collect(ready=False, delay=.15)
        _, _, ready = self.collect(delay=.15)
        print(f"[SCHEDULER BENCHMARK] synchronous={synchronous:.3f}s; ready={ready:.3f}s; speedup={synchronous/ready:.2f}x", flush=True)
        self.assertLess(ready, synchronous * .8)

    def test_partial_rollout_is_never_marked_as_learned(self):
        self.collect(delay=.01, stop=True)


if __name__ == "__main__":
    unittest.main()
