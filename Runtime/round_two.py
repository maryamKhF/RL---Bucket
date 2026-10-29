"""Continue the previous stage's PPO agent on the same fixed data and compute budget."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from Runtime.recovery_safety import atomic_json, file_hash
from Runtime.round_one import MainOneOfFive
from Runtime.round_one_recovery import restore_round_archive


class MainTwoOfFive(MainOneOfFive):
    """Carry the learned policy, optimizer and cumulative step count into stage two."""

    round_number = 2
    previous_round_number = 1
    previous_runner_class = MainOneOfFive

    def __init__(self, *args, previous_run=None, previous_archive=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_run = Path(previous_run).resolve() if previous_run else None
        self.previous_archive = Path(previous_archive).resolve() if previous_archive else None

    def _prepare(self):
        # The previous stage supplies the trained agent; --resume-* instead
        # restores a coherent checkpoint from within this new stage.
        if bool(self.previous_run) == bool(self.previous_archive):
            raise ValueError(f"Select exactly one completed round-{self.previous_round_number} source: previous run or archive.")
        if self.previous_archive:
            self.previous_run = restore_round_archive(
                self.previous_archive, self.output_dir / "previous_round", round_number=self.previous_round_number)
        previous = self.previous_runner_class("unused", "unused", self.output_dir, resume_run=self.previous_run)
        previous.workers = self.workers
        previous._resume()
        self.parent_model, _reference, self.parent_step = previous.trained_artifacts()
        self.parent_hash = file_hash(self.parent_model)
        inputs = self.previous_run / "inputs"
        # Require sealed caches: regenerating them under the new seed changes
        # the test set and makes stage comparisons use different payments.
        manifest = json.loads((inputs / "data_manifest.json").read_text(encoding="utf-8"))
        for name in ("network.pkl", "transactions.pkl", "transaction_split.csv"):
            if manifest.get(name) != file_hash(inputs / name):
                raise ValueError(f"Previous-stage fixed dataset checksum mismatch: {name}")
        self.previous = previous
        self.snapshot = inputs / "network_snapshot.gml.geo"
        self.config = inputs / "config.yaml"
        self.carbon_source = inputs / "global_energy_mix.json"
        if self.failure_rates is not None and self.failure_rates != previous.rates:
            raise ValueError("Keep the previous stage's failure-rate conditions for stage comparison.")
        self.failure_rates = list(previous.rates)
        super()._prepare()
        metadata_path = self.run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parent_metadata = json.loads((self.previous_run / "run_metadata.json").read_text(encoding="utf-8"))
        ancestors = list(parent_metadata.get("agent_ancestors", []))
        # Older stage-two artifacts predate this compact lineage index. They
        # already record the stage-one model hash and cumulative starting step.
        if not ancestors and parent_metadata.get("previous_round"):
            ancestors.append({
                "round": parent_metadata["previous_round"],
                "training_seed": parent_metadata["previous_training_seed"],
                "actual_training_transitions": parent_metadata["initial_training_transitions"],
                "model_sha256": parent_metadata["previous_model_sha256"],
            })
        ancestors.append({"round": self.previous_round_number,
                          "training_seed": int(previous.cfg["seed"]),
                          "actual_training_transitions": self.parent_step,
                          "model_sha256": self.parent_hash})
        metadata.update(
            previous_round=self.previous_round_number, previous_training_seed=int(previous.cfg["seed"]),
            transaction_seed=int(previous.cfg.get("transaction_seed", previous.cfg["seed"])),
            transaction_split_sha256=manifest["transaction_split.csv"],
            previous_model_sha256=self.parent_hash,
            agent_ancestors=ancestors,
            training_mode="continual", initial_training_transitions=self.parent_step,
            requested_training_transitions=self.cfg["rl"]["round_training_timesteps"],
            cumulative_training_target=self.cfg["rl"]["total_timesteps"],
            model_initialization="load previous PPO actor, critic, optimizer and timestep counter",
            seed_rule="next stage environment RNG seed + 1; learned parameters and optimizer are retained",
            evaluation_updates_policy=False,
        )
        atomic_json(metadata_path, metadata)
        print(f"[{self.label}] Continue the saved PPO agent: previous actual steps={self.parent_step:,}; "
              f"additional requested steps={self.cfg['rl']['round_training_timesteps']:,}; "
              f"cumulative target={self.cfg['rl']['total_timesteps']:,}.", flush=True)

    def _configuration(self):
        import copy
        cfg = copy.deepcopy(self.previous.cfg)
        cfg["transaction_seed"] = int(cfg.get("transaction_seed", cfg["seed"]))
        cfg["seed"] = int(cfg["seed"]) + 1
        cfg["rl"]["seed"] = cfg["seed"]
        budget = int(cfg["rl"].get("round_training_timesteps", cfg["rl"]["total_timesteps"]))
        cfg["rl"].update(total_timesteps=self.parent_step + budget,
                          round_training_timesteps=budget,
                          initial_policy="inputs/parent_policy.zip",
                          initial_policy_sha256=self.parent_hash)
        # Retain all PPO, timeout, sampling and training failure settings.
        return cfg

    def _copy_fixed_dataset(self, inputs):
        for name in ("network.pkl", "transactions.pkl", "transaction_split.csv", "data_manifest.json"):
            shutil.copy2(self.previous_run / "inputs" / name, inputs / name)
        shutil.copy2(self.parent_model, inputs / "parent_policy.zip")

    def _resume(self):
        super()._resume()
        rl = self.cfg["rl"]
        if rl.get("initial_policy") != "inputs/parent_policy.zip":
            raise ValueError(f"This round-{self.round_number} archive is not a continuation of the previous agent.")
        if file_hash(self.run_dir / rl["initial_policy"]) != rl["initial_policy_sha256"]:
            raise ValueError("Previous-stage policy checksum mismatch.")
