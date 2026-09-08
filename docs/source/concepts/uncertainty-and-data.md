# Uncertainty & Time-Series Data

Reinforcement learning policies must generalize to unpredictable real-world dynamics. GridAges integrates stochastic modeling for multiple uncertainty sources:

---

## Sources of Uncertainty

1. **Renewable Generation**: Non-stationary solar irradiance curves with cloud transients and wind gust fluctuations.
2. **Net Demand**: Residential, commercial, and industrial load patterns exhibiting daily and seasonal trends.
3. **Wholesale Price Dynamics**: Volatile day-ahead and real-time locational marginal prices (LMP).

---

## Data Integration Workflow

When `uncertainty=True` is passed to environments, GridAges samples episodic daily trajectories with added Gaussian or autoregressive noise to train robust controllers.
