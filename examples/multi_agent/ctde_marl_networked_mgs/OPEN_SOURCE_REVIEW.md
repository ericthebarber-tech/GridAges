# Open-source library review for this grid MARL case

## Recommended algorithm layer

**[BenchMARL](https://github.com/facebookresearch/BenchMARL) /
[TorchRL](https://docs.pytorch.org/rl/main/index.html)** is the closest
maintained external framework for this
project. BenchMARL provides MAPPO/IPPO, MADDPG/IDDPG, and MASAC/ISAC, supports
continuous actions, centralized critics, parameter sharing, and PettingZoo
tasks. TorchRL documents both multi-agent PPO and DDPG with decentralized
actors and centralized critics. It does not currently provide MATD3 as a
standard BenchMARL algorithm, which is why this directory implements all four
methods on one common project-local CTDE data path.

**[AgileRL](https://docs.agilerl.com/en/stable/multi_agent_training/index.html)**
is another useful option when MADDPG and MATD3 are the focus. It
supports PettingZoo parallel environments, but it does not provide the same
single-framework MAPPO/MASAC coverage needed by this four-way comparison.

**[EPyMARL](https://github.com/uoe-agents/epymarl)** has native PettingZoo
integration, MAPPO/MADDPG, common or individual rewards, and standardized
multi-seed logging. It does not cover MASAC and MATD3, so it cannot run the
requested four-way comparison without additional implementations.

**[MARLlib](https://github.com/Replicable-MARL/MARLlib)** has broad RLlib-based
MARL coverage and flexible policy sharing. Its official installation notes
still target Linux, Python 3.8/3.9, old Gym, and patched RLlib, making it a poor
fit for this macOS/Python 3.12 GridAges workspace. It also does not provide the
same exact four-algorithm set.

**[HARL](https://github.com/PKU-MARL/HARL)** is especially relevant if the
scientific question shifts toward heterogeneous-agent sequential updates. Its
HAPPO/HADDPG/HASAC/HATD3 family is not an exact MAPPO/MADDPG/MASAC/MATD3
comparison, but it is a strong follow-up benchmark.

## Power-system environment projects

| Project | Fit for this case | Recommendation |
|---|---|---|
| [PowerGridworld](https://github.com/NREL/PowerGridworld) | Explicitly multi-agent, modular power-system environments; prior examples use MADDPG and RLlib PPO. | Best external environment reference for architecture and validation. Porting would replace, rather than train, the current GridAges environment. |
| [MAPDN](https://github.com/Future-Power-Networks/MAPDN) | Multi-agent active voltage control with continuous actions, partial observations, global state, and AC power flow. | Best benchmark to add as a second environment when studying voltage control. Its objective is narrower than the current microgrid EMS dispatch problem. |
| [Gym-ANM](https://github.com/robinhenry/gym-anm) | Modern Gymnasium framework for active network management and customizable distribution networks. | Good single-agent/centralized ANM comparison, but not natively the full MARL algorithm layer required here. |
| [PowerGym](https://github.com/siemens/powergym) | IEEE 13/34/123/8500-node Volt-VAR benchmark. | Useful voltage-control benchmark; not a direct substitute for coordinated microgrid economic dispatch. |
| [Grid2Op](https://github.com/Grid2op/grid2op) | Mature transmission-grid operation platform with topology actions, maintenance, and L2RPN benchmarks. | Strong for transmission operation and topology control, but a different problem from distribution-level networked microgrids. |
| [OPF-Gym](https://github.com/Digitalized-Energy-Systems/opfgym) | Gymnasium + pandapower environments for OPF, SimBench grids, and custom OPF tasks. | Useful centralized OPF baseline and environment-design reference; not natively multi-agent. |

## Practical conclusion

Keep GridAges as the primary environment because it already contains the
networked IEEE34/IEEE13 microgrids, device costs, storage dynamics, renewable
time series, and PettingZoo parallel API required by this study. Use
BenchMARL/TorchRL as the main external algorithm reference, PowerGridworld as
the closest power-system MARL environment reference, and MAPDN as an optional
second benchmark for active voltage control.
