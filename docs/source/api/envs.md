# Environments API (`gridages.envs`)

GridAges provides standard **Gymnasium** and **PettingZoo** compatible environments for single-agent and multi-agent reinforcement learning in power systems.

---

## Single-Agent Environments

Single-agent environments inherit from `GridBaseEnv` and implement the Gymnasium `(obs, reward, terminated, truncated, info)` step interface.

### Base Environment

```{eval-rst}
.. autoclass:: gridages.envs.single_agent.base_env.GridBaseEnv
   :members:
   :undoc-members:
   :show-inheritance:
```

### IEEE 13-Bus Microgrid Environment

```{eval-rst}
.. autoclass:: gridages.envs.single_agent.microgrid_ems.ieee13_mg.IEEE13Env
   :members:
   :undoc-members:
   :show-inheritance:
```

### IEEE 34-Bus Microgrid Environment

```{eval-rst}
.. autoclass:: gridages.envs.single_agent.microgrid_ems.ieee34_mg.IEEE34Env
   :members:
   :undoc-members:
   :show-inheritance:
```

### CIGRE Medium Voltage Environment

```{eval-rst}
.. autoclass:: gridages.envs.single_agent.cigre_mv.CIGREMVEnv
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Multi-Agent Environments

Multi-agent environments enable coordinated Energy Management System (EMS) and multi-microgrid trading scenarios.

### Base Multi-Agent Environments

```{eval-rst}
.. autoclass:: gridages.envs.multi_agent.base_env.GridEnv
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: gridages.envs.multi_agent.base_env.NetworkedGridEnv
   :members:
   :undoc-members:
   :show-inheritance:
```

### Multi-Agent Microgrids (IEEE 34 / 13)

```{eval-rst}
.. autoclass:: gridages.envs.multi_agent.ieee34_ieee13.MultiAgentMicrogrids
   :members:
   :undoc-members:
   :show-inheritance:
```
