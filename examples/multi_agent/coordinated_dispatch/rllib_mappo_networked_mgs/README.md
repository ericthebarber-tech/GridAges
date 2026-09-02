# Networked microgrid RL benchmarks

This directory contains two kinds of benchmarks:

- `multi_agent_mgs.py`: the original RLlib MAPPO-style example with one policy per microgrid.
- `train_{ppo,ddpg,sac,td3}.py`: Stable-Baselines3 centralized-controller baselines matching the algorithms in `examples/single_agent/microgrid_ems`.

The centralized wrapper concatenates the observations and continuous actions for MG1,
MG2, and MG3. It applies all three actions simultaneously to the original PettingZoo
environment and uses the sum of their rewards as the system reward. Evaluation still
records metrics for every microgrid as well as the complete system.

Install the example dependencies if they are not already available:

```bash
pip install "stable-baselines3[extra]" matplotlib tensorboard
```

Run the four algorithms from the repository root:

```bash
python3 examples/multi_agent/coordinated_dispatch/rllib_mappo_networked_mgs/train_ppo.py
python3 examples/multi_agent/coordinated_dispatch/rllib_mappo_networked_mgs/train_ddpg.py
python3 examples/multi_agent/coordinated_dispatch/rllib_mappo_networked_mgs/train_sac.py
python3 examples/multi_agent/coordinated_dispatch/rllib_mappo_networked_mgs/train_td3.py
```

Useful options are `--total-timesteps`, `--eval-freq`, `--eval-episodes`, `--seed`,
`--output-dir`, and `--progress-bar`. By default, results are written under
`examples/multi_agent/coordinated_dispatch/rllib_mappo_networked_mgs/logs/sb3/<algorithm>/`
and include TensorBoard logs, checkpoints,
`evaluations.npz`, and `evaluation.csv`.

Compare all completed runs:

```bash
python3 examples/multi_agent/coordinated_dispatch/rllib_mappo_networked_mgs/plot_metrics.py \
  examples/multi_agent/coordinated_dispatch/rllib_mappo_networked_mgs/logs/sb3/ppo \
  examples/multi_agent/coordinated_dispatch/rllib_mappo_networked_mgs/logs/sb3/ddpg \
  examples/multi_agent/coordinated_dispatch/rllib_mappo_networked_mgs/logs/sb3/sac \
  examples/multi_agent/coordinated_dispatch/rllib_mappo_networked_mgs/logs/sb3/td3
```

The comparison figure includes training return, final evaluation return, operating
cost, safety violations, and power-flow convergence rate.
