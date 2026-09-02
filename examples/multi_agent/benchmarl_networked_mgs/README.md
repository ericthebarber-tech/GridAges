# GridAges with official BenchMARL algorithms

This experiment runs the GridAges networked microgrid environment with the
official BenchMARL implementations of:

- MAPPO
- MADDPG
- MASAC

All three microgrids belong to one BenchMARL agent group. Each actor receives
its local observation, while the centralized critic can use the concatenated
global state (and joint actions where required by the algorithm).

## Installation

From the GridAges repository root:

```bash
source .venv/bin/activate
pip install -r examples/multi_agent/benchmarl_networked_mgs/requirements.txt
```

The versions are pinned because this project currently resolves BenchMARL
1.4.0 with TorchRL/TensorDict 0.7.2. The warning about unavailable TorchRL C++
extensions is harmless here because prioritized replay is disabled.

## Quick test

Run one short experiment first:

```bash
python examples/multi_agent/benchmarl_networked_mgs/train_mappo.py \
  --total-frames 2400 \
  --eval-frequency 1200 \
  --eval-episodes 2 \
  --device cpu
```

Replace `train_mappo.py` with `train_maddpg.py` or `train_masac.py`. The
default output is:

```text
examples/multi_agent/benchmarl_networked_mgs/logs/<algorithm>_seed<seed>/
```

BenchMARL creates a timestamped experiment directory inside that folder.
`evaluation.csv` contains the raw physical reward, operating cost, safety
violation, and power-flow convergence rate. `updates.csv` contains algorithm
losses and optimization diagnostics.

## Full multi-seed comparison

The following command runs all three algorithms for seeds 42, 43, and 44:

```bash
python examples/multi_agent/benchmarl_networked_mgs/run_suite.py \
  --seeds 42,43,44 \
  --total-frames 500000 \
  --eval-frequency 10000 \
  --eval-episodes 5
```

This is nine sequential experiments and may take a long time. For an initial
check, use `--total-frames 10000`.

Plot the mean and one-standard-deviation band across all available seeds:

```bash
python examples/multi_agent/benchmarl_networked_mgs/plot_results.py
```

This writes `benchmarl_comparison.png` and
`benchmarl_comparison_final_metrics.csv` inside the local `logs` folder. If
only one seed is complete, that algorithm is plotted without a deviation
band. Re-running a seed selects its newest `evaluation.csv`.

To keep the mean-across-seeds figure and additionally save a separate
four-panel comparison for every available seed:

```bash
python examples/multi_agent/benchmarl_networked_mgs/plot_results.py \
  --per-seed
```

The additional files are named `benchmarl_comparison_seed42.png`,
`benchmarl_comparison_seed43.png`, and so on.

## Reward and parameter sharing

Shared reward is enabled by default:

```bash
python examples/multi_agent/benchmarl_networked_mgs/train_masac.py \
  --share-reward
```

Use `--no-share-reward` for individual microgrid rewards.

Actors use the same network architecture but separate weights by default.
Use `--share-policy-params` to share actor weights across MG1, MG2, and MG3.
Critic parameters are shared within the group by default; this can be changed
with `--no-share-critic-params`.

Examples:

```bash
# Same actor architecture and shared weights
python examples/multi_agent/benchmarl_networked_mgs/train_mappo.py \
  --share-policy-params \
  --actor-hidden 128,128 \
  --critic-hidden 256,256

# Same architecture but independent actor weights
python examples/multi_agent/benchmarl_networked_mgs/train_maddpg.py \
  --no-share-policy-params
```

BenchMARL builds one model configuration per agent group, so agents in this
centralized group have the same layer layout. They can still have independent
weights. Giving every agent a different layer layout would require splitting
them into separate groups, which would also split the standard centralized
critic and is therefore not used here.

## Homogeneous and heterogeneous grids

The default configuration gives MG1, MG2, and MG3 different load, generator,
storage, and renewable settings:

```text
configs/heterogeneous_env.json
```

For the same grid parameters:

```bash
python examples/multi_agent/benchmarl_networked_mgs/train_mappo.py \
  --env-config examples/multi_agent/benchmarl_networked_mgs/configs/homogeneous_env.json
```

The adapter pads local observations and actions to a common size because a
BenchMARL agent group requires stackable tensors. Policy actions are normalized
to `[-1, 1]` and mapped back to each microgrid's original physical limits
before the GridAges environment is stepped.

Generated logs and checkpoints are ignored by Git and do not need to be
committed.
