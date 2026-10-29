"""Balanced on-policy rollouts without a barrier after every vector step.

Each worker still contributes exactly n_steps transitions. Buffer indices keep
the original (time, worker) trajectories, including GAE and timeout bootstrap.
The policy is frozen until ALL workers finish that rollout. Gaussian noise is
drawn in advance by (time, worker), independent of completion order.
"""

from multiprocessing.connection import wait
import time

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv


COLLECTOR = "ready-balanced-v1"


class ReadySubprocVecEnv(SubprocVecEnv):
    def abort_collection(self):
        """Discard in-flight work; never wait minutes or commit partial PPO work."""
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2)
        for remote in self.remotes:
            remote.close()
        self.waiting = False
        self.closed = True


class ReadyPPO(PPO):
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if not isinstance(env, ReadySubprocVecEnv):
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if (not isinstance(self.action_space, spaces.Box) or self.use_sde
                or self.policy.squash_output
                or not isinstance(self.policy.action_dist, DiagGaussianDistribution)
                or not isinstance(self.observation_space, spaces.Box)):
            raise ValueError("Ready rollouts require the existing unsquashed Gaussian MLP policy and Box observations.")
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()  # All workers idle: coherent recovery boundary.
        started = time.perf_counter()
        initial_steps = int(self.num_timesteps)
        width = env.num_envs
        completed = np.zeros(width, dtype=int)
        pending = {}
        records = []
        env.reset_infos = list(env.reset_infos)
        inference_seconds = audit_seconds = 0.0
        # Policy std is state independent for this DiagGaussian policy. Keeping
        # noise indexed by trajectory avoids scheduling-dependent exploration.
        noise = torch.randn((n_rollout_steps, width, self.action_space.shape[0]), device=self.device)

        def dispatch(ranks):
            nonlocal inference_seconds
            if not ranks:
                return
            tick = time.perf_counter()
            observations = np.array(self._last_obs[ranks], copy=True)
            episode_starts = np.array(self._last_episode_starts[ranks], copy=True)
            with torch.no_grad():
                tensor = obs_as_tensor(observations, self.device)
                distribution = self.policy.get_distribution(tensor)
                gaussian = distribution.distribution
                selected_noise = torch.stack([noise[completed[rank], rank] for rank in ranks])
                action_tensor = gaussian.mean + gaussian.stddev * selected_noise
                values = self.policy.predict_values(tensor).flatten().cpu().numpy()
                probabilities = distribution.log_prob(action_tensor).cpu().numpy()
                actions = action_tensor.cpu().numpy()
            clipped = np.clip(actions, self.action_space.low, self.action_space.high)
            inference_seconds += time.perf_counter() - tick
            for index, rank in enumerate(ranks):
                env.remotes[rank].send(("step", clipped[index]))
                pending[env.remotes[rank]] = (rank, int(completed[rank]), observations[index],
                    actions[index], episode_starts[index], values[index], probabilities[index])

        try:
            dispatch(list(range(width)))
            while pending:
                # Start the next payment as soon as its worker is free. Never
                # resubmit a worker beyond its equal per-rollout contribution.
                ready = wait(list(pending), timeout=1)
                next_ranks = []
                for remote in sorted(ready, key=lambda pipe: pending[pipe][0]):
                    rank, step, observation, action, episode_start, value, probability = pending.pop(remote)
                    new_obs, reward, done, info, reset_info = remote.recv()
                    info["worker_rank"] = rank
                    info["rollout_step"] = step
                    info["audit_timestep"] = initial_steps + (step + 1) * width
                    if "performance" in info:
                        info["performance"]["reset_seconds"] = (reset_info or {}).get("reset_seconds", 0.0)
                    env.reset_infos[rank] = reset_info
                    adjusted_reward = float(reward)
                    if done and info.get("TimeLimit.truncated", False) and info.get("terminal_observation") is not None:
                        with torch.no_grad():
                            terminal = self.policy.obs_to_tensor(info["terminal_observation"])[0]
                            adjusted_reward += self.gamma * float(self.policy.predict_values(terminal)[0].item())
                    rollout_buffer.observations[step, rank] = observation
                    rollout_buffer.actions[step, rank] = action
                    rollout_buffer.rewards[step, rank] = adjusted_reward
                    rollout_buffer.episode_starts[step, rank] = episode_start
                    rollout_buffer.values[step, rank] = value
                    rollout_buffer.log_probs[step, rank] = probability
                    self._last_obs[rank] = new_obs
                    self._last_episode_starts[rank] = done
                    completed[rank] += 1
                    records.append((observation, action, reward, done, info, value, probability))
                    if completed[rank] < n_rollout_steps:
                        next_ranks.append(rank)
                dispatch(next_ranks)
                # Preserve the callback batch size and number of calls. Audit
                # real worker identities explicitly, not their arrival indices.
                while len(records) >= width:
                    batch, records = records[:width], records[width:]
                    observations, actions, rewards, dones, infos, values, log_probs = zip(*batch)
                    obs_tensor = obs_as_tensor(np.asarray(observations), self.device)
                    actions, rewards, dones = np.asarray(actions), np.asarray(rewards), np.asarray(dones)
                    values = torch.as_tensor(values, device=self.device)
                    log_probs = torch.as_tensor(log_probs, device=self.device)
                    self.num_timesteps += width
                    self._update_info_buffer(infos, dones)
                    tick = time.perf_counter()
                    callback.update_locals({"obs_tensor": obs_tensor, "actions": actions,
                        "rewards": rewards, "dones": dones, "infos": infos,
                        "values": values, "log_probs": log_probs})
                    if not callback.on_step():
                        raise InterruptedError("Partial rollout discarded; retain the last completed PPO checkpoint.")
                    audit_seconds += time.perf_counter() - tick
            if records or not np.all(completed == n_rollout_steps):
                raise RuntimeError("Unbalanced or incomplete ready rollout; PPO update refused.")
            rollout_buffer.pos = n_rollout_steps
            rollout_buffer.full = True
            with torch.no_grad():
                last_values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))
            rollout_buffer.compute_returns_and_advantage(last_values, self._last_episode_starts)
            elapsed = time.perf_counter() - started
            self.last_rollout_performance = {
                "collector": COLLECTOR, "transitions": int(completed.sum()),
                "per_worker": completed.tolist(), "seconds": elapsed,
                "inference_seconds": inference_seconds, "audit_seconds": audit_seconds,
                "transitions_per_second": int(completed.sum()) / max(elapsed, 1e-9),
            }
            print(f"[ROLLOUT] {COLLECTOR}; {n_rollout_steps} x {width} transitions; "
                  f"seconds={elapsed:.2f}; inference={inference_seconds:.2f}s; "
                  f"audit={audit_seconds:.2f}s; "
                  f"speed={self.last_rollout_performance['transitions_per_second']:.3f}", flush=True)
            callback.update_locals({"rollout_buffer": rollout_buffer, "new_obs": self._last_obs,
                                    "dones": self._last_episode_starts, "values": last_values})
            callback.on_rollout_end()
            return True
        except BaseException:
            self.num_timesteps = initial_steps
            rollout_buffer.reset()
            env.abort_collection()
            raise
