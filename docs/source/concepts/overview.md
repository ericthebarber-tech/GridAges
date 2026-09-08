# Conceptual Overview

**GridAges** is an agent-centric simulation framework designed for **Reinforcement Learning (RL)** and **Multi-Agent Reinforcement Learning (MARL)** in modern power distribution systems and microgrids.

---

## Core Pillars

1. **Agent-Centric Modeling**: Power grid components (DERs, microgrids, aggregators, substations) are modeled as autonomous entities capable of observation, local optimization, and collective coordination.
2. **Physics-Based Simulation**: Leverages **pandapower** to solve full AC power flow equations at each step, accurately calculating line loadings, voltage magnitudes, and losses.
3. **Gymnasium & PettingZoo Integration**: Standardized interfaces allow seamless training with popular RL libraries like Stable-Baselines3, CleanRL, Ray RLlib, and Tianshou.
4. **Realistic Time-Series Uncertainty**: Integrated dataset loaders simulate realistic fluctuations in solar irradiance, wind speed, consumer electricity demand, and real-time wholesale power prices.
