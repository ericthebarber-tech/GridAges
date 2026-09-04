# Constraints & Safety

In power systems reinforcement learning, ensuring physical safety and compliance with grid operating limits is paramount.

---

## Key Physical Constraints

1. **Voltage Limits**:
   $$V_{\min} \le |V_i| \le V_{\max} \quad (\text{typically } 0.95 \text{ p.u.} \le |V_i| \le 1.05 \text{ p.u.})$$

2. **Branch Thermal Limits**:
   $$I_{ij} \le I_{ij}^{\max} \quad (\text{or Line Loading } \le 100\%)$$

3. **Inverter Apparent Power Capacity**:
   $$P_{\text{inv}}^2 + Q_{\text{inv}}^2 \le S_{\text{rated}}^2$$

4. **Battery SOC Bounds**:
   $$\text{SOC}_{\min} \le \text{SOC}_t \le \text{SOC}_{\max}$$

---

## Safety Penalty Functions in GridAges

GridAges includes utility functions in `gridages.utils.safety` to compute smooth quadratic or barrier penalties for violations during RL training.
