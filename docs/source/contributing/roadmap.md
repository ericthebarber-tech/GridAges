# GridAges Roadmap

Here is the upcoming development roadmap and milestones for the GridAges simulator:

---

## Current Release (v0.1.0)

- [x] Standard Gymnasium single-agent environments for IEEE 13, IEEE 34, and CIGRE MV.
- [x] PettingZoo multi-agent environments for networked microgrids.
- [x] Modular device abstractions (`ESS`, `DG`, `RES`, `Grid`, `Transformer`, `Shunt`).
- [x] Dataset-driven time-series uncertainty.
- [x] Interactive web-based Drag-and-Drop Microgrid Builder in Sphinx documentation.

---

## Near-Term Goals (v0.2.0)

- [ ] Support for 3-phase unbalanced power flow formulations and harmonic analysis.
- [ ] Safe RL and Constrained Markov Decision Process (CMDP) wrappers (e.g. Lagrangian multipliers, control barrier functions).
- [ ] Advanced grid-forming and grid-following inverter dynamics.
- [ ] Direct export from drag-and-drop builder to pandapower network JSON.
