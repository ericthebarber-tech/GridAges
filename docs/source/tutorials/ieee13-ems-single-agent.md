# IEEE 13 Microgrid EMS Tutorial

This tutorial demonstrates how to instantiate the IEEE 13-Bus microgrid environment, inspect its observation and action spaces, and run a standard reinforcement learning rollout loop.

---

## 1. Initializing the Environment

```python
import gymnasium as gym
import gridages
from gridages.envs.single_agent import IEEE13Env
from gridages.devices import ESS, DG, RES

# 1. Define DERs
devices = [
    ESS(name="Battery_671", bus="671", capacity_mwh=1.0, max_p_mw=0.5),
    RES(name="SolarPV_675", bus="675", max_p_mw=0.8, type="solar"),
    DG(name="Diesel_680", bus="680", min_p_mw=0.1, max_p_mw=0.6),
]

# 2. Instantiate IEEE 13 Microgrid EMS Environment
env = IEEE13Env(devices=devices)
```

---

## 2. Inspecting Action & Observation Spaces

```python
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
```

---

## 3. Running an Episode Loop

```python
obs, info = env.reset(seed=42)
done = False
total_reward = 0.0

while not done:
    # Sample random action or query trained policy
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode Completed. Total Reward: {total_reward:.2f}")
```
