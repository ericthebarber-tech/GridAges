# Agent-Centric Modeling

In classical power grid simulations, components are often treated merely as passive nodes and branches inside a global admittance matrix. 

In **GridAges**, each device is modeled as an **Agent** with its own:
- **Local State**: Voltage, active/reactive power output, State-of-Charge (SOC), or tap position.
- **Action Space**: Continuous power setpoints ($P$, $Q$), discrete dispatch switches, or curtailment factors.
- **Operational Constraints**: Maximum apparent power ($S_{\text{rated}}$), power factor limits, ramp rates, and storage energy bounds.
- **Objective / Cost Function**: Fuel costs, degradation costs, or peak shaving rewards.

---

## Single-Agent vs. Multi-Agent Topologies

- **Single-Agent EMS**: A centralized Energy Management System controls all DERs within a microgrid to minimize total operating costs while maintaining voltage stability.
- **Multi-Agent Networked Microgrids**: Autonomous microgrids trade power at points of common coupling (PCC), negotiating imports and exports while competing or cooperating in local energy markets.
