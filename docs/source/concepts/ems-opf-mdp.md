# EMS, Optimal Power Flow & MDP

GridAges formulates microgrid energy management and Optimal Power Flow (OPF) as a **Markov Decision Process (MDP)**.

---

## The MDP Formulation

A GridAges environment is defined as a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

- **State Space ($\mathcal{S}$)**:
  - Time of day / step index $t \in [0, T]$.
  - Current electricity import/export price $c_t$.
  - Forecasted / current renewable generation (solar $P_{\text{pv}}$, wind $P_{\text{wind}}$).
  - Net aggregate feeder load $P_{\text{load}}, Q_{\text{load}}$.
  - Battery State of Charge $\text{SOC}_t \in [0, 1]$.
  - Generator operational status and startup/shutdown timers.

- **Action Space ($\mathcal{A}$)**:
  - Battery charging / discharging active power setpoint $P_{\text{ess}}$ and reactive power $Q_{\text{ess}}$.
  - Controllable generation setpoints $P_{\text{dg}}, Q_{\text{dg}}$.
  - Renewable curtailment ratios $\alpha \in [0, 1]$.
  - Transformer tap changer positions $n_{\text{tap}} \in \{-16, \dots, +16\}$.

- **Reward Function ($\mathcal{R}$)**:
  $$\mathcal{R}_t = - \left( C_{\text{grid}}(P_{\text{grid}}) + C_{\text{gen}}(P_{\text{dg}}) + C_{\text{deg}}(\Delta \text{SOC}) + \lambda_v \text{Penalty}_{\text{voltage}} + \lambda_l \text{Penalty}_{\text{line}} \right)$$
