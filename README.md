# RL + Bucket: Adaptive Routing and Failure-Resilient Payment Simulation

A modular Python research implementation for studying adaptive routing in Lightning-style payment channel networks using **Proximal Policy Optimization (PPO)**, heuristic-based pathfinding, **Top-K candidate route generation**, **Bucket-based route management**, and **partial backtracking after route failures**.

The repository is designed for reproducible experimentation on real Lightning Network graph snapshots as well as controlled simulation scenarios.

---

## Overview

The project models an off-chain payment network as a directed, weighted graph:

$$G=(V,E)$$

where:

- $V$ represents network nodes.
- $E$ represents directed payment channels.
- Channel attributes include capacity, fees, CLTV-related information, status, and optional geographic metadata.

For a payment from source $u$ to destination $v$, the routing pipeline is:

```
Network Snapshot
      |
      v
Graph Construction
      |
      v
Transaction Generation
      |
      v
Local Network Observation
      |
      v
State Representation
      |
      v
PPO Agent
      |
      v
Adaptive Heuristic Parameter
      |
      v
Heuristic-Aware Pathfinding
      |
      v
Top-K Candidate Routes
      |
      v
Bucket Route Management
      |
      v
Best Candidate Selection
      |
      v
Onion-Based Forwarding
      |
      +------> Success
      |
      v
     Failure
      |
      v
Partial Backtracking
      |
      v
Alternative Route / Suffix
      |
      v
Continue Payment
```

The key design principle is that PPO does **not** directly generate a complete route. Instead, the RL component provides an adaptive routing signal or heuristic parameter, while the pathfinding component remains responsible for constructing candidate paths.

---

## Research Objectives

The implementation is organized around four main objectives:

1. **Adaptive routing:** use PPO to adapt routing behavior according to the observed network state.
2. **Candidate route generation:** generate multiple feasible routes rather than relying on a single path.
3. **Failure-resilient forwarding:** retain alternative routes in a Bucket and recover from failures using partial backtracking.
4. **Reproducible evaluation:** evaluate routing behavior under different failure rates and compare it with baseline strategies.

The repository separates these responsibilities into independent modules so that individual components can be tested and replaced without redesigning the entire system.

---

## Main Components

### 1. Network Modeling

The `Network/` package contains the graph and channel abstractions.

Main files:

- `Network/node.py` — node representation.
- `Network/channel.py` — payment channel representation.
- `Network/graph_builder.py` — graph construction and snapshot processing.
- `Network/topology.py` — topology-related utilities.
- `Network/environment.py` — network-level environment support.
- `Network/test_network.py` — network module tests.

The repository includes real GML-based network snapshots:

- `20190501.gml.geo`
- `20230618.gml.geo`

These files can be used to construct graphs from real network data rather than relying only on synthetic topologies.

---

### 2. Pathfinding

The `Pathfinding/` package implements the route-generation layer.

Main files:

- `Pathfinding/dijkstra.py` — Dijkstra-based pathfinding.
- `Pathfinding/heuristics.py` — routing heuristic calculations.
- `Pathfinding/top_k_paths.py` — Top-K candidate route generation.
- `Pathfinding/test_pathfinding.py` — pathfinding tests.

The pathfinding layer is deliberately separated from the RL layer. This allows the routing algorithm to remain interpretable while PPO controls the adaptive component of the routing criterion.

Conceptually:

```
State -> PPO -> adaptive parameter
                    |
                    v
             routing heuristic
                    |
                    v
             Dijkstra / path search
                    |
                    v
              Top-K routes
```

---

### 3. Reinforcement Learning

The `RL/` package implements the PPO-based adaptive routing component.

Main files:

- `RL/state.py` — state construction and local network observations.
- `RL/environment.py` — Gymnasium-compatible RL environment.
- `RL/reward.py` — reward calculation.
- `RL/ppo_agent.py` — PPO agent construction and inference.
- `RL/train.py` — training workflow.
- `RL/test_rl.py` — RL integration tests.

The environment follows the standard reinforcement-learning formulation:

```
M = (S, A, T, R, gamma)
```

where:

- $S$ is the state space.
- $A$ is the action space.
- $T$ is the transition function.
- $R$ is the reward function.
- $gamma$ is the discount factor.

The current configuration uses PPO with a continuous adaptive action. The action is interpreted as an adaptive routing parameter rather than a direct node-selection action.

This distinction is important: **PPO learns routing behavior; it does not replace the pathfinding algorithm.**

---

### 4. Bucket and Partial Backtracking

The `Bucket/` package implements candidate-route storage and failure recovery.

Main files:

- `Bucket/bucket.py` — Bucket data structure and route management.
- `Bucket/candidate_manager.py` — candidate route handling and ranking.
- `Bucket/backtrack.py` — partial backtracking logic.
- `Bucket/test_real_gml.py` — Bucket integration test using a real GML snapshot.

The Bucket stores a bounded set of candidate routes. The active route can be replaced when a channel or path segment fails.

The intended recovery behavior is:

```
Active Route
     |
     v
Failure Detected
     |
     v
Find Valid Recovery Point
     |
     v
Partial Backtrack
     |
     v
Alternative Route Suffix
     |
     v
Continue Forwarding
```

This avoids treating every failure as a reason to restart the complete routing process from the original source.

The current configuration enables backtracking and limits the number of stored candidates.

---

### 5. Payment Simulation

The `Simulation/` package provides an end-to-end payment simulation environment.

Main files:

- `Simulation/transaction_generator.py` — transaction generation.
- `Simulation/router.py` — routing orchestration.
- `Simulation/payment_simulator.py` — payment execution.
- `Simulation/failure_model.py` — probabilistic failure modeling.
- `Simulation/network_dynamics.py` — network-state changes.
- `Simulation/onion.py` — Onion-style forwarding representation.
- `Simulation/backtrack.py` — simulation-level recovery logic.
- `Simulation/environment.py` — integrated simulation environment.
- `Simulation/test_simulation.py` — simulation integration test.

The simulator supports different failure rates and evaluates whether payments succeed under changing routing conditions.

---

### 6. Evaluation

The `Evaluations/` package provides metrics and baseline comparisons.

Main files:

- `Evaluations/metrics.py` — evaluation metrics.
- `Evaluations/evaluate.py` — evaluation workflow.
- `Evaluations/baselines.py` — baseline routing strategies.
- `Evaluations/test_evaluations.py` — evaluation tests.

The evaluation layer is kept separate from the implementation so that routing algorithms can be compared using the same simulation conditions.

Typical metrics include:

- Payment success rate
- Route length
- Routing cost / fee
- Failure recovery behavior
- Backtracking behavior
- Candidate-route usage
- Computational or routing overhead

The exact metrics and aggregation procedure should be interpreted together with the active configuration and evaluation code.

---

## Configuration

Global parameters are centralized in:

```
Configs/config.yaml
```

Important current settings include:

| Category | Parameter | Current value |
|---|---|---:|
| Reproducibility | Seed | 42 |
| Dataset | Transactions | 10,000 |
| Dataset | Train/Test split | 70% / 30% |
| Network | Neighborhood K | 15 |
| Network | Neighborhood radius M | 5 |
| Routing | Top-K paths | 5 |
| Bucket | Maximum candidates | 5 |
| Bucket | Backtracking | Enabled |
| Bucket | Maximum backtracking depth | 3 |
| Failures | Failure rates | 1%–6% |
| RL | Algorithm | PPO |
| RL | Total timesteps | 20,000 |
| RL | Learning rate | 0.0003 |
| RL | Gamma | 0.99 |
| RL | GAE lambda | 0.95 |
| Evaluation | Repetitions | 10 |
| Evaluation | Confidence interval | 95% |

These values are configuration defaults, not universal requirements. Experiments can be changed through `Configs/config.yaml`.

---

## Repository Structure

```
RL---Bucket/
|
+-- 20190501.gml.geo
+-- 20230618.gml.geo
|
+-- Network/
|   +-- node.py
|   +-- channel.py
|   +-- graph_builder.py
|   +-- topology.py
|   +-- environment.py
|   +-- test_network.py
|
+-- Pathfinding/
|   +-- dijkstra.py
|   +-- heuristics.py
|   +-- top_k_paths.py
|   +-- test_pathfinding.py
|
+-- RL/
|   +-- state.py
|   +-- environment.py
|   +-- reward.py
|   +-- ppo_agent.py
|   +-- train.py
|   +-- test_rl.py
|
+-- Bucket/
|   +-- bucket.py
|   +-- candidate_manager.py
|   +-- backtrack.py
|   +-- test_real_gml.py
|
+-- Simulation/
|   +-- transaction_generator.py
|   +-- router.py
|   +-- payment_simulator.py
|   +-- failure_model.py
|   +-- network_dynamics.py
|   +-- onion.py
|   +-- backtrack.py
|   +-- environment.py
|   +-- test_simulation.py
|
+-- Evaluations/
|   +-- metrics.py
|   +-- evaluate.py
|   +-- baselines.py
|   +-- test_evaluations.py
|
+-- Visualization/
|   +-- plots.py
|
+-- Configs/
|   +-- config.yaml
|
+-- main.py
+-- test_main.py
+-- requirements.txt
+-- README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/maryamKhF/RL---Bucket.git
cd RL---Bucket
```

### 2. Create a virtual environment

Windows:

```bash
py -m venv .venv
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The current dependency file includes:

- NumPy
- pandas
- NetworkX
- Gymnasium
- Stable-Baselines3
- PyTorch
- Matplotlib
- PyYAML
- SciPy

---

## Running the Project

### Run the main integration workflow

```bash
python main.py
```

### Run the main integration test

```bash
python test_main.py
```

### Test the network module

```bash
python -m Network.test_network
```

### Test pathfinding

```bash
python -m Pathfinding.test_pathfinding
```

### Test the PPO/RL module

```bash
python -m RL.test_rl
```

### Test the Bucket module with a real GML snapshot

```bash
python -m Bucket.test_real_gml
```

### Test the integrated simulation

```bash
python -m Simulation.test_simulation
```

### Test evaluation components

```bash
python -m Evaluations.test_evaluations
```

---

## Real GML Snapshots

The repository contains two GML-based network snapshots:

```
20190501.gml.geo
20230618.gml.geo
```

The network-processing modules are intended to load graph topology and channel attributes from these snapshots.

A typical workflow is:

```
GML Snapshot
     |
     v
Graph Builder
     |
     v
NetworkX Graph
     |
     +----------------+
     |                |
     v                v
Pathfinding          RL State
     |                |
     +-------+--------+
             |
             v
        Route Selection
```

The snapshots are useful for testing the implementation against real network topology rather than only synthetic graphs.

---

## RL State Representation

The RL state is based on local network information around the payment endpoints rather than passing the complete network graph to the policy.

A local observation can be represented as a fixed-size matrix:

```
s(u,v) in R^(K x D)
```

where:

- $K$ is the maximum number of observed channels.
- $D$ is the number of channel features.

The current configuration uses:

```
K = 15
M = 5
```

Padding and fixed-size representations can be used when the observed neighborhood contains fewer than $K$ usable entries.

This design keeps the policy input bounded and avoids requiring the PPO network to process the entire Lightning topology at every decision.

---

## Adaptive Routing Action

The RL action is an adaptive scalar:

```
eta in [-1, 1]
```

The PPO policy produces this parameter from the current state. The parameter is then passed to the routing heuristic.

The conceptual separation is:

```
RL policy
   |
   v
adaptive parameter eta
   |
   v
routing heuristic
   |
   v
pathfinding
   |
   v
candidate routes
```

This design preserves a clear boundary between learning and route construction.

---

## Top-K Candidate Routes

For each source-destination pair, the pathfinding layer can generate multiple candidate routes:

```
P = {p1, p2, ..., pK}
```

The candidates are evaluated using the routing cost and managed by the Bucket layer.

The configured maximum number of stored candidates is:

```
5
```

The separation between **candidate generation** and **candidate storage** is intentional:

- Pathfinding determines which candidate routes are available.
- Candidate management ranks and filters them.
- Bucket stores the selected candidates.
- Backtracking uses the stored candidates during failure recovery.

---

## Failure Model

The simulation supports probabilistic channel failures.

The current configuration evaluates:

```
1%, 2%, 3%, 4%, 5%, 6%
```

failure-rate scenarios.

A simplified execution sequence is:

```
Select Route
    |
    v
Forward Payment
    |
    +---- success ----> Complete
    |
    v
Channel Failure
    |
    v
Update Channel / Route State
    |
    v
Search Bucket
    |
    +---- valid alternative ----> Partial Backtrack
    |                                  |
    |                                  v
    |                           Continue Payment
    |
    +---- no valid alternative ----> Re-routing
```

This makes it possible to measure the effect of route caching and partial recovery under different failure conditions.

---

## Onion-Based Forwarding

The simulation includes an Onion Routing component.

The route is selected before the forwarding representation is constructed. Each forwarding step receives only the information needed for the next stage rather than the complete route.

The implementation is intended for simulation and algorithmic evaluation. It should not be interpreted as a production implementation of the Lightning Network's cryptographic protocol.

---

## Experimental Design

The configuration currently defines:

- 10,000 generated transactions.
- 70% training data.
- 30% testing data.
- Multiple failure-rate scenarios from 1% to 6%.
- Ten evaluation repetitions.
- A 95% confidence-interval setting.

For reproducible experiments:

1. Keep the random seed fixed.
2. Keep training and testing transactions separated.
3. Use the same network snapshot for comparable experiments.
4. Keep failure-model parameters fixed when comparing routing methods.
5. Record configuration changes together with experimental results.

---

## Baselines

The repository contains a dedicated baseline module:

```
Evaluations/baselines.py
```

Baselines provide a reference point for evaluating the contribution of the adaptive routing and Bucket components.

A fair comparison should use:

- The same graph snapshot.
- The same source-destination transaction set.
- The same payment amounts.
- The same failure scenarios.
- The same evaluation repetitions.

Only the routing strategy should change when isolating algorithmic effects.

---

## Visualization

The `Visualization/` package currently contains:

```
Visualization/plots.py
```

The visualization layer can be used to generate plots from experimental results, including comparisons across routing strategies, failure scenarios, and performance metrics.

Plot parameters such as DPI, figure size, and output directories are configurable in `Configs/config.yaml`.

---

## Reproducibility

The repository is structured to support reproducible research.

Important reproducibility controls include:

- Global random seed.
- Explicit train/test split.
- Centralized YAML configuration.
- Fixed failure-rate scenarios.
- Configurable transaction count.
- Configurable PPO hyperparameters.
- Configurable Bucket limits.
- Repeated evaluation runs.

When reporting results, the exact configuration file used for an experiment should be preserved.

---

## Design Principles

The implementation follows several architectural principles:

### Separation of concerns

Network construction, state generation, reinforcement learning, pathfinding, Bucket management, payment simulation, and evaluation are implemented as separate modules.

### RL does not directly replace routing

PPO produces an adaptive routing signal. The pathfinding layer remains responsible for constructing candidate paths.

### Failure recovery is explicit

A failed route is not automatically treated as a complete transaction failure. The system can inspect stored alternatives and attempt partial recovery.

### Real and synthetic testing

The project supports real GML snapshots while retaining modular components that can be used with controlled synthetic network scenarios.

### Configuration-driven experiments

Important experimental parameters are centralized in `Configs/config.yaml` rather than being hard-coded across the project.

---

## Limitations and Scope

This repository is a research and simulation implementation, not a production Lightning Network node or routing daemon.

In particular:

- The payment process is simulated.
- Channel failures are modeled probabilistically.
- The Onion component is intended for simulation and does not replace production Lightning cryptography.
- Network snapshots represent particular points in time and do not constitute a live network view.
- Results depend on the selected snapshot, transaction distribution, failure model, and configuration.
- The current implementation should be validated experimentally before being used to support claims about real-world network performance.

---

## Recommended Research Workflow

A typical experiment can follow this sequence:

```
1. Select GML snapshot
        |
        v
2. Build NetworkX graph
        |
        v
3. Generate / load transactions
        |
        v
4. Construct local observations
        |
        v
5. Train PPO
        |
        v
6. Generate adaptive routing parameters
        |
        v
7. Generate Top-K candidate routes
        |
        v
8. Populate Bucket
        |
        v
9. Simulate payment forwarding
        |
        +---- Success
        |
        +---- Failure
                 |
                 v
          Partial Backtracking
                 |
                 v
          Alternative Route
                 |
                 v
          Continue / Re-route
        |
        v
10. Compute evaluation metrics
        |
        v
11. Compare with baselines
        |
        v
12. Visualize and report results
```

---

## Project Status

The repository is organized as a modular research prototype with separate implementations for:

- Real GML graph loading
- Network modeling
- Pathfinding
- Top-K route generation
- PPO-based adaptive routing
- Bucket management
- Partial backtracking
- Payment simulation
- Failure modeling
- Onion-style forwarding
- Evaluation and baselines
- Visualization

The individual test modules can be executed independently to validate each layer before running the complete integration workflow.

---

## Citation

If this repository is used in academic work, please cite the corresponding thesis, paper, or research project associated with the implementation.

---

## License

No explicit open-source license is currently specified in the repository. Unless a license is added, users should not assume permissions beyond those granted by applicable copyright law.

---

## Repository

**GitHub:** https://github.com/maryamKhF/RL---Bucket.git
