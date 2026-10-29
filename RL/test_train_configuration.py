import unittest

from RL.train import resolve_training_configuration


class TrainingConfigurationTests(unittest.TestCase):

    def setUp(self):
        self.rl_cfg = {
            "total_timesteps": 20000,
            "n_steps": 1024,
            "batch_size": 64,
        }

    def test_fast_training_can_be_requested_explicitly(self):
        resolved = resolve_training_configuration(
            self.rl_cfg,
            fast_training=True,
        )

        self.assertEqual(resolved["mode"], "FAST TRAINING")
        self.assertEqual(resolved["total_timesteps"], 16)
        self.assertEqual(resolved["n_steps"], 8)
        self.assertEqual(resolved["batch_size"], 8)

    def test_normal_training_keeps_configured_parameters(self):
        resolved = resolve_training_configuration(
            self.rl_cfg,
            fast_training=False,
        )

        self.assertEqual(resolved["mode"], "NORMAL TRAINING")
        self.assertEqual(resolved["total_timesteps"], 20000)
        self.assertEqual(resolved["n_steps"], 1024)
        self.assertEqual(resolved["batch_size"], 64)


if __name__ == "__main__":
    unittest.main()
