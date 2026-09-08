# Datasets & Uncertainty

GridAges supports dataset-driven time-series uncertainty modeling for load demands, solar irradiance, wind generation, and real-time electricity pricing.

---

## Dataset Structure

Time-series profiles are stored in CSV format within the `data/` directory:

- **Active & Reactive Load Profiles**: Normal or normalized kW/kVar demands across 24-hour dispatch horizons or 8760-hour annual profiles.
- **Solar Irradiance ($G_t$)**: Global Horizontal Irradiance (GHI) and temperature time series.
- **Wind Speed ($v_t$)**: Hourly or minute-level wind speed series.
- **Electricity Market Prices ($c_t$)**: Real-time pricing (RTP) and time-of-use (TOU) tariff curves.

---

## Loading Profiles into Environments

Environments load these datasets during initialization or sample distinct episode days randomly upon calling `env.reset()`:

```python
from gridages.envs.single_agent import IEEE13Env

# Initialize environment with dataset sampling enabled
env = IEEE13Env(uncertainty=True)
obs, info = env.reset(seed=42)
```
