# Improved RL + Bucket Lightning Network

Python research prototype combining:
1. NetworkX Lightning-like topology
2. LND/CLN/ECL-style heuristic routing
3. PPO learning of eta
4. Modified Dijkstra
5. Top-K candidate paths
6. Bucket-style alternative-path execution
7. Simulated balances and channel failures
8. Train/test separation and evaluation

## Install
```bash
pip install -r requirements.txt
```

## Run without snapshot
```bash
python main.py --train --evaluate
```

This uses a synthetic LN-like graph so the project is runnable immediately.

## Optional generic JSON snapshot
```bash
python main.py --snapshot data/snapshots/network.json --train --evaluate
```

Expected generic structure:
```json
{
  "nodes": [
    {"id":"A","country":"US","latitude":40,"longitude":-74,"carbon_intensity":300}
  ],
  "channels": [
    {"source":"A","target":"B","capacity":1000000,"fee_base":1000,"fee_rate":10,"delay":1}
  ]
}
```

Important: real LN private channel balances are not public in snapshots. The simulator therefore initializes/updates directional balances and failure probabilities as experimental state.
