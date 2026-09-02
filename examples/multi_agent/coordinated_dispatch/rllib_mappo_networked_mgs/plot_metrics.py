"""Plot and compare PPO/DDPG/SAC/TD3 multi-agent experiments."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_SB3_LOG_DIR = EXPERIMENT_DIR / "logs" / "sb3"


def read_run(directory):
    history = {"timesteps": [], "rewards": []}
    history_file = directory / "evaluations.npz"
    if history_file.exists():
        archive = np.load(history_file)
        history = {
            "timesteps": archive["timesteps"].tolist(),
            "rewards": np.mean(archive["results"], axis=1).tolist(),
        }
    evaluation = []
    with (directory / "evaluation.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            evaluation.append({
                "agent": row["agent"],
                "reward": float(row["reward"]),
                "operating_cost": float(row["operating_cost"]),
                "safety": float(row["safety"]),
                "convergence_rate": float(row["convergence_rate"]),
            })
    return history, evaluation


def agent_means(rows, metric):
    values = defaultdict(list)
    for row in rows:
        values[row["agent"]].append(row[metric])
    return {agent: float(np.mean(samples)) for agent, samples in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_SB3_LOG_DIR / "algorithm_comparison.png",
    )
    args = parser.parse_args()
    runs = {directory.name.upper(): read_run(directory) for directory in args.directories}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for name, (history, _) in runs.items():
        if history["timesteps"]:
            axes[0, 0].plot(
                history["timesteps"], history["rewards"],
                label=name,
            )
    axes[0, 0].set_title("Training episode return")
    axes[0, 0].set_xlabel("environment steps")
    axes[0, 0].legend()

    algorithms = list(runs)
    agents = sorted({row["agent"] for _, rows in runs.values() for row in rows})
    x = np.arange(len(algorithms))
    width = 0.8 / max(len(agents), 1)
    panels = [
        ("reward", "Evaluation return (higher is better)"),
        ("operating_cost", "Operating cost (lower is better)"),
        ("safety", "Safety violation (lower is better)"),
        ("convergence_rate", "Power-flow convergence rate"),
    ]
    for axis, (metric, title) in zip(axes.flat[1:], panels):
        for index, agent in enumerate(agents):
            values = [agent_means(runs[name][1], metric).get(agent, np.nan)
                      for name in algorithms]
            offsets = x + (index - (len(agents) - 1) / 2) * width
            axis.bar(offsets, values, width=width, label=agent)
        axis.set_xticks(x)
        axis.set_xticklabels(algorithms)
        axis.set_title(title)
        axis.legend()

    axes[1, 2].axis("off")
    for axis in axes.flat[:5]:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=180)
    print(args.save)


if __name__ == "__main__":
    main()
