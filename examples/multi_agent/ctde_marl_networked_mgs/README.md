# CTDE MARL for networked microgrids

This experiment suite trains separate actors for MG1, MG2, and MG3 using
centralized training and decentralized execution (CTDE):

- MAPPO: stochastic decentralized actors and centralized value critics.
- MADDPG: deterministic decentralized actors and centralized action-value critics.
- MASAC: stochastic decentralized actors, twin centralized critics, automatic entropy.
- MATD3: deterministic decentralized actors, twin centralized critics, target smoothing, delayed actor updates.

Unlike the centralized SB3 baseline, each actor receives only its own local
observation at execution time. Critics receive joint observations and, for
off-policy methods, joint actions during training.

The four algorithms in this directory are implemented directly with PyTorch;
they do not call Stable-Baselines3 or RLlib algorithm classes. GridAges still
provides the PettingZoo environment and pandapower simulation. Install the
additional experiment dependencies, if needed, with:

```bash
python -m pip install -r \
  examples/multi_agent/ctde_marl_networked_mgs/requirements.txt
```

## Quick smoke tests

Run from the repository root:

```bash
python examples/multi_agent/ctde_marl_networked_mgs/train_mappo.py \
  --total-timesteps 256 --rollout-steps 128 --eval-frequency 256 \
  --eval-episodes 1 --batch-size 64

python examples/multi_agent/ctde_marl_networked_mgs/train_maddpg.py \
  --total-timesteps 100 --learning-starts 32 --batch-size 32 \
  --eval-frequency 100 --eval-episodes 1
```

Replace `train_maddpg.py` with `train_masac.py` or `train_matd3.py` for the
other off-policy methods.

## Shared versus local rewards

Shared cooperative reward is the default:

```bash
python examples/multi_agent/ctde_marl_networked_mgs/train_mappo.py --share-reward
```

Use each microgrid's own reward:

```bash
python examples/multi_agent/ctde_marl_networked_mgs/train_mappo.py --no-share-reward
```

## Same or different neural-network configurations

Use the same actor/critic architecture:

```bash
--network-config examples/multi_agent/ctde_marl_networked_mgs/configs/homogeneous_networks.json
```

Use different architectures for MG1, MG2, and MG3:

```bash
--network-config examples/multi_agent/ctde_marl_networked_mgs/configs/heterogeneous_networks.json
```

These files control architecture, not parameter sharing. Every microgrid keeps
its own actor so execution remains decentralized.

## Same or different microgrid configurations

Use equal device parameters:

```bash
--env-config examples/multi_agent/ctde_marl_networked_mgs/configs/homogeneous_grids.json
```

Use different load scales, capacities, capability curves, and costs:

```bash
--env-config examples/multi_agent/ctde_marl_networked_mgs/configs/heterogeneous_grids.json
```

Use different feeder topologies (MG1/MG2 use IEEE 13-bus and MG3 uses IEEE
34-bus):

```bash
--env-config examples/multi_agent/ctde_marl_networked_mgs/configs/heterogeneous_topologies.json
```

`ConfigurableMultiAgentMicrogrids` also accepts a Python import path for each
agent's `network_factory`, plus device and connection bus names. A completely
different PettingZoo grid environment can be selected with:

```bash
--env-class package.module:EnvironmentClass
```

## Full multi-seed suite

```bash
python examples/multi_agent/ctde_marl_networked_mgs/run_suite.py \
  --seeds 42,43,44 \
  --total-timesteps 500000 \
  --eval-frequency 10000 \
  --eval-episodes 10 \
  --env-config examples/multi_agent/ctde_marl_networked_mgs/configs/heterogeneous_grids.json \
  --network-config examples/multi_agent/ctde_marl_networked_mgs/configs/heterogeneous_networks.json
```

Runs are saved under
`examples/multi_agent/ctde_marl_networked_mgs/logs/<algorithm>_seed<seed>/`,
including `config.json`,
`training.csv`, `evaluation.csv`, `updates.csv`, checkpoints, and
`final_model.pt`. `updates.csv` contains per-agent actor loss, critic loss,
and (for MASAC) entropy-temperature metrics averaged every
`--log-frequency` environment steps.

## Plot mean and standard deviation

```bash
python examples/multi_agent/ctde_marl_networked_mgs/plot_results.py \
  examples/multi_agent/ctde_marl_networked_mgs/logs/*_seed* \
  --save examples/multi_agent/ctde_marl_networked_mgs/logs/ctde_comparison.png
```

The learning curve uses deterministic evaluation episodes, not training
rewards. Shading is one standard deviation across seeds. The summary panels
report final evaluation system return (higher is better), total microgrid
operating cost (lower), summed safety violations (lower), and mean AC
power-flow convergence rate (higher). Rewards in the CSV files are unscaled;
`--reward-scale` is used only for optimization.

See `OPEN_SOURCE_REVIEW.md` for the library and power-grid environment review.
