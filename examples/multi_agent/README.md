# Multi-agent networked-microgrid experiments

This directory contains three complementary experiment suites:

| Suite | Control architecture | Algorithms | Implementation |
|---|---|---|---|
| `coordinated_dispatch/rllib_mappo_networked_mgs` | Centralized controller | PPO, DDPG, SAC, TD3 | Stable-Baselines3 baselines plus the original RLlib example |
| `benchmarl_networked_mgs` | CTDE | MAPPO, MADDPG, MASAC | Official BenchMARL algorithms |
| `ctde_marl_networked_mgs` | CTDE | MAPPO, MADDPG, MASAC (and an experimental MATD3 implementation) | Direct PyTorch implementations |

The experiment design, figures, numerical summaries, and interpretation are
collected in
[`experiment_report/multi_agent_experiment_report.ipynb`](experiment_report/multi_agent_experiment_report.ipynb).

Training logs, checkpoints, TensorBoard event files, and generated evaluations
remain local under each suite's `logs/` directory and are intentionally ignored
by Git. Only compact final figures in `experiment_report/assets/` are tracked.
