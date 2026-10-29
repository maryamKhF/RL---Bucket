"""Report measured bottlenecks without treating cumulative steps as new work."""

from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from RL.train import PPOProgressCallback


class ProgressTimingTests(unittest.TestCase):
    def test_continued_run_reports_real_speed_and_slowest_worker(self):
        env = SimpleNamespace(reset_infos=[{"reset_seconds": .2}, {"reset_seconds": .3}])
        model = SimpleNamespace(num_timesteps=43560, get_env=lambda: env)
        callback = PPOProgressCallback(63560)
        callback.init_callback(model)
        output = io.StringIO()
        infos = [{"performance": {"step_seconds": 20, "step_cpu_seconds": 19}},
                 {"performance": {"step_seconds": 70, "step_cpu_seconds": 31},
                  "path_search_attempts": 4, "path_search_elapsed_seconds": 69.5,
                  "path_search_timeout_stages": ["candidate_search"]}]
        with redirect_stdout(output), patch("RL.train.time.time", side_effect=[0, 76]):
            callback.on_training_start({"infos": infos}, {})
            model.num_timesteps += 30
            self.assertTrue(callback.on_step())
        text = output.getvalue()
        self.assertIn("Speed      : 0.395 timesteps/sec", text)
        self.assertIn("Slowest worker: 1; payment + reset: 70.30s", text)
        self.assertIn("payment CPU: 31.00s; reset: 0.30s", text)
        self.assertIn("Routing searches: 4; search time: 69.50s", text)


if __name__ == "__main__":
    unittest.main()
