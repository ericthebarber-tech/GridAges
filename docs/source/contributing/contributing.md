# Contributing to GridAges

We welcome contributions to GridAges! Whether you are implementing new grid benchmarks, creating DER device models, adding multi-agent RL algorithms, or improving documentation, here is how to get started.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hepengli/GridAges.git
   cd GridAges
   ```

2. **Install in editable mode with development dependencies**:
   ```bash
   pip install -e .
   pip install sphinx furo myst-parser sphinx-copybutton sphinx-design pytest
   ```

3. **Building the Documentation**:
   ```bash
   sphinx-build -b html docs/source docs/build/html
   ```

---

## Code Style & Testing

- We adhere to standard PEP 8 Python formatting and type annotations.
- Run tests with `pytest`:
  ```bash
  pytest gridages/test/
  ```
