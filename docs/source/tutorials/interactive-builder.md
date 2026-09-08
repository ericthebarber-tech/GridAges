# Interactive Microgrid & DER Builder

Use this visual drag-and-drop tool to compose microgrid topologies and distribute **Distributed Energy Resources (DERs)** across power grid buses. 

As you drag and configure devices, the matching `GridAges` Python environment code is generated live below.

---

```{raw} html
<div class="grid-builder-container gridages-interactive-builder"></div>
```

---

## How to Use the Builder

1. **Select a Grid Topology**: Choose between the standard IEEE 13-bus radial feeder, the CIGRE Medium Voltage benchmark, or a modular 3-bus microgrid.
2. **Drag and Drop DERs**: Pick any component from the left palette (such as **Battery Storage**, **Solar PV**, **Wind Turbine**, **Diesel Generator**, or **Loads**) and drop it onto any target bus card.
3. **Configure Parameters**: Click on any placed device pill to edit rated capacities ($P_{\max}$, $Q_{\max}$), round-trip efficiency, minimum generation limits, or cost curve coefficients.
4. **Copy & Run**: Click **📋 Copy Code** and paste the snippet directly into your Python script or Jupyter Notebook to start training reinforcement learning agents with `gymnasium` and `pandapower`.

---

## Supported DER Device Classes

| Device | Class | Key Parameters |
| :--- | :--- | :--- |
| **Battery ESS** | {class}`gridages.devices.storage.ESS` | `capacity_mwh`, `max_p_mw`, `efficiency`, `soc_min`, `soc_max` |
| **Solar PV** | {class}`gridages.devices.generator.RES` | `max_p_mw`, `type="solar"`, profile series |
| **Wind Turbine** | {class}`gridages.devices.generator.RES` | `max_p_mw`, `type="wind"`, cut-in/cut-out speeds |
| **Diesel/Gas DG** | {class}`gridages.devices.generator.DG` | `min_p_mw`, `max_p_mw`, `cost_curve_coefs`, `startup_cost` |
| **Interconnection** | {class}`gridages.devices.grid.Grid` | `bus`, `max_p_mw`, electricity market pricing |
