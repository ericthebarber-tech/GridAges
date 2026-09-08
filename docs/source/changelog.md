# Changelog

All notable changes to the **GridAges** project will be documented in this file.

---

## [0.1.0] - 2026-09-02

### Added
- **Interactive Drag-and-Drop Microgrid & DER Builder**: Visual topology and device composer with live Python code generation in documentation.
- **Full API Documentation**: Automated docstring generation for `devices`, `envs`, `networks`, and `core`.
- **Benchmark Environments**:
  - `IEEE13Env`: 13-bus distribution feeder with distributed storage and solar PV.
  - `IEEE34Env`: 34-bus rural distribution feeder with voltage regulators.
  - `CIGREMVEnv`: European benchmark medium-voltage network with high DER penetration.
  - `MultiAgentMicrogrids`: PettingZoo-compatible networked microgrids.
- **Standardized Core Abstractions**: `DeviceState`, `GridState`, and `Action` spaces.
