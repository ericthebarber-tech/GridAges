# Devices API (`gridages.devices`)

The `gridages.devices` module defines the physical and operational models for Distributed Energy Resources (DERs), storage systems, transformers, and grid interfaces.

All devices inherit from the base {class}`~gridages.devices.base.Device` class and maintain internal state, action bounds, and physics constraints.

---

## Base Device

```{eval-rst}
.. autoclass:: gridages.devices.base.Device
   :members:
   :show-inheritance:
```

---

## Energy Storage System (ESS)

```{eval-rst}
.. autoclass:: gridages.devices.storage.ESS
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Distributed Generation (DG & RES)

```{eval-rst}
.. autoclass:: gridages.devices.generator.DG
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: gridages.devices.generator.RES
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Grid Interconnection

```{eval-rst}
.. autoclass:: gridages.devices.grid.Grid
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## Transformers & Reactive Power Compensation

```{eval-rst}
.. autoclass:: gridages.devices.transformer.Transformer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: gridages.devices.compensation.Shunt
   :members:
   :undoc-members:
   :show-inheritance:
```
