"""Lightweight integration smoke test for main5 on the real snapshot.

This test loads the full snapshot and exercises reset, PPO construction,
prediction, and reward dispatch once for each main5 arm. It deliberately does
not train PPO, search/execute a route, or evaluate the 3,000-row test split.

Run from the project root:
    python -m unittest Simulation.test_main5_smoke -v
"""

import copy
import gc
import math
import os
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "20190501.gml.geo"
CONFIG = ROOT / "Configs" / "config.yaml"


class Main5RealSnapshotSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SNAPSHOT.is_file():
            raise FileNotFoundError(f"Real snapshot not found: {SNAPSHOT}")
        if not CONFIG.is_file():
            raise FileNotFoundError(f"Project config not found: {CONFIG}")

        os.environ["RL_FAST_VALIDATION"] = "0"
        os.environ["RL_FAST_TRAINING"] = "0"
        os.environ["RL_FAST_ENV"] = "0"
        os.environ["RL_REPORT_FILE"] = str(
            Path(tempfile.gettempdir()) / "rl_bucket_main5_smoke_report.txt"
        )

        import main as core
        import main5

        cls.core = core
        cls.main5 = main5
        cls.cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        cls.graph = core.LNGraphBuilder(
            default_capacity=cls.cfg.get("channel", {}).get("default_capacity")
        ).from_data(core.geo_to_json(SNAPSHOT))
        if cls.graph.number_of_nodes() == 0 or cls.graph.number_of_edges() == 0:
            raise AssertionError("The full snapshot produced an empty graph.")

        core.set_seed(int(cls.cfg["seed"]))
        transactions = core.generate_transactions(
            cls.graph,
            1,
            int(cls.cfg["seed"]),
            cls.cfg["simulation"]["min_amount"],
            cls.cfg["simulation"]["max_amount"],
        )
        if len(transactions) != 1:
            raise AssertionError("Could not generate one real-snapshot transaction.")
        cls.transaction = transactions[0]

    def test_each_reward_arm_builds_and_routes_on_full_snapshot(self):
        core = self.core
        seed = int(self.cfg["seed"])
        for formula, label in self.main5.ARMS:
            with self.subTest(formula=formula):
                cfg = copy.deepcopy(self.cfg)
                cfg.setdefault("reward", {})["formula"] = formula
                graph = copy.deepcopy(self.graph)
                core.assign_failure_probabilities(graph, 0.03, seed)
                dynamics = core.NetworkDynamics(
                    graph,
                    seed=seed,
                    default_capacity=cfg.get("channel", {}).get("default_capacity"),
                )
                failure_model = core.FailureModel(seed=seed)
                env = core.RoutingEnv(
                    G=graph,
                    transactions=[copy.deepcopy(self.transaction)],
                    heuristic_fn=core.lnd_cost,
                    config=cfg,
                    mode="eval",
                    failure_model=failure_model,
                    network_dynamics=dynamics,
                )
                model = None
                try:
                    observation, _info = env.reset(seed=seed)
                    model = core.build_ppo(env=env, cfg=cfg, seed=seed)
                    action, _state = model.predict(observation, deterministic=True)
                    self.assertTrue(env.action_space.contains(action))
                    reward = env._calculate_environment_reward(
                        success=True,
                        path_length=2,
                        carbon_intensity=100.0,
                        fee=1000.0,
                        delay=10.0,
                        partial_backtrack_count=0,
                        full_reroute_count=0,
                        attempt_count=1,
                        payment_amount=500_000.0,
                        liquidity_margin=1.5,
                        persistence=0.75,
                    )
                    self.assertTrue(math.isfinite(float(reward)))
                    self.assertEqual(cfg["reward"]["formula"], formula)
                    print(
                        f"[main5 smoke OK] {label}: "
                        f"nodes={graph.number_of_nodes()}, "
                        f"directed_channels={graph.number_of_edges()}, "
                        f"action={action}, reward_dispatch={reward:.6g}"
                    )
                finally:
                    if model is not None:
                        model.get_env().close()
                    else:
                        env.close()
                    del model, env, dynamics, failure_model, graph
                    gc.collect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
