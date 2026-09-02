# Multi-agent experiment report

Open `multi_agent_experiment_report.ipynb` with VS Code, JupyterLab, or Jupyter
Notebook. The notebook compares:

1. centralized coordinated dispatch with Stable-Baselines3;
2. official BenchMARL CTDE algorithms; and
3. custom PyTorch CTDE algorithms.

The notebook renders without access to the ignored training directories because
the three final figures and `final_metrics.csv` are stored in this report folder.
Code cells require `pandas` and a Jupyter kernel if you want to execute them.

To install notebook support in the active virtual environment:

```bash
python -m pip install jupyter ipykernel pandas
```

Raw logs and model checkpoints are deliberately not committed. Regenerate the
source figures with the plotting commands documented in each experiment suite's
README, then update the compact report assets when publishing new results.
