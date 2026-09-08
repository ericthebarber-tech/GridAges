# IEEE 13 Microgrid EMS Quickstart

This guide gets you running a reinforcement learning agent on the IEEE 13-bus microgrid Energy Management System (EMS) in minutes.

---

## 1. Installation

```bash
pip install -e .
pip install gymnasium pandapower
```

---

## 2. Quick Simulation

```python
from gridages.envs.single_agent import IEEE13Env

# Create the IEEE-13 microgrid environment
env = IEEE13Env()

# Reset environment to get initial observation
obs, info = env.reset(seed=42)
print("Initial observation shape:", obs.shape)

# Step with a sample action
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}, Terminated: {terminated}")
```
