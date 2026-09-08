# GridAges Documentation

**GridAges** is an agent-centric power grid simulator for **Reinforcement Learning (RL)** and **Multi-Agent RL (MARL)**, built on **pandapower**.

- ⚡ **Agent-Centric**: Compose microgrids, DER clusters, and distribution feeders as modular, autonomous agents.
- 🔬 **Physics-Based**: Accurate AC power flow, voltage magnitudes, and line thermal limits via pandapower.
- 🎮 **RL-Ready**: Full compatibility with **Gymnasium** (single-agent) and **PettingZoo** (multi-agent).
- 📈 **Data-Driven**: Real-world time-series profiles for loads, solar PV, wind generation, and dynamic electricity prices.
- 🎨 **Interactive Tools**: Built-in Drag-and-Drop Microgrid & DER Builder for rapid scenario prototyping.

---

## 🚀 Interactive Tools & Tutorials

```{toctree}
:maxdepth: 2

tutorials/interactive-builder
tutorials/ieee13-ems-single-agent
```

---

## 🛠️ Getting Started

```{toctree}
:maxdepth: 2

getting-started/installation
getting-started/ieee13-quickstart
getting-started/dataset
```

---

## 💡 Concepts

```{toctree}
:maxdepth: 2

concepts/overview
concepts/agent-centric
concepts/ems-opf-mdp
concepts/constraints-and-safety
concepts/uncertainty-and-data
```

---

## 📚 API Reference

```{toctree}
:maxdepth: 2

api/devices
api/envs
api/networks
api/core
```

---

## 🤝 Community & Project Info

```{toctree}
:maxdepth: 2

contributing/contributing
contributing/roadmap
changelog
```