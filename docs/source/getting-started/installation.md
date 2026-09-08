# Installation

## Requirements
- Python 3.12+
- pandapower
- gymnasium

## Environment Setup

Create and activate the `powergrid` conda environment:

```bash
conda create -n powergrid python=3.12 -y
conda activate powergrid
pip install -U pip
```

## Install GridAges

From the repo root (`GridAges/`), choose the appropriate option for your workflow:

```bash
# Base installation
pip install -e .

# Development dependencies
pip install -e ".[dev]"

# Documentation dependencies
pip install -e ".[docs]"

# Full installation (Development and Documentation)
pip install -e ".[dev,docs]"
```

## Running Documentation

To run and view the Sphinx documentation for GridAges, use the `powergrid` conda environment where Sphinx and all documentation plugins are installed.

### Step 1: Activate the Environment

Open your terminal (PowerShell / Command Prompt) and activate the environment:

```bash
conda activate powergrid
```

### Step 2: Live Preview Server with Auto-Reload

From the root project folder (`GridAges/`), run:

```bash
sphinx-autobuild docs/source docs/build/
```

### Step 3: View the Documentation

Once the live server starts, open your browser and navigate to:

```text
http://localhost:8000
```
*(or `http://127.0.0.1:8000`)*