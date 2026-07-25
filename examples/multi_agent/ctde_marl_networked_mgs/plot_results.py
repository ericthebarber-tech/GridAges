"""Plot mean and standard deviation across CTDE algorithms and seeds."""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("reward", "System return", True),
    ("operating_cost", "Operating cost", False),
    ("safety", "Safety violation", False),
    ("convergence_rate", "Power-flow convergence", True),
)
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = EXPERIMENT_DIR / "logs"


def identity(directory):
    match = re.match(r"^(mappo|maddpg|masac|matd3)_seed(\d+)$",
                     directory.name.lower())
    return (match.group(1).upper(), int(match.group(2))) if match else None


def read_evaluations(directory):
    path = directory / "evaluation.csv"
    if not path.exists():
        return None
    with path.open(newline="") as stream:
        rows = [
            {
                **row,
                "timestep": int(row["timestep"]),
                **{metric: float(row[metric]) for metric, _, _ in METRICS},
            }
            for row in csv.DictReader(stream)
            if row["agent"] == "system"
        ]
    return rows or None


def timestep_means(rows, metric):
    values = defaultdict(list)
    for row in rows:
        values[row["timestep"]].append(row[metric])
    steps = np.asarray(sorted(values), dtype=float)
    means = np.asarray([np.mean(values[int(step)]) for step in steps])
    return steps, means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument(
        "--save", type=Path, default=DEFAULT_LOG_DIR / "ctde_comparison.png"
    )
    args = parser.parse_args()
    grouped = defaultdict(dict)
    for directory in args.directories:
        key, rows = identity(directory), read_evaluations(directory)
        if key and rows:
            grouped[key[0]][key[1]] = rows

    algorithms = [name for name in ("MAPPO", "MADDPG", "MASAC", "MATD3")
                  if grouped[name]]
    if not algorithms:
        raise SystemExit("No complete algorithm_seedN directories found")
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    curve_axis = axes[0, 0]
    for algorithm in algorithms:
        curves = [timestep_means(rows, "reward")
                  for rows in grouped[algorithm].values()]
        start = max(curve[0][0] for curve in curves)
        end = min(curve[0][-1] for curve in curves)
        grid = np.linspace(start, end, min(200, min(len(c[0]) for c in curves)))
        values = np.vstack([np.interp(grid, steps, means)
                            for steps, means in curves])
        mean, std = values.mean(0), values.std(0)
        line = curve_axis.plot(
            grid,
            mean,
            marker="o" if len(grid) == 1 else None,
            label=f"{algorithm} (n={len(values)})",
        )[0]
        curve_axis.fill_between(grid, mean - std, mean + std,
                                color=line.get_color(), alpha=0.2)
    curve_axis.set_title("Evaluation return: mean ± std across seeds")
    curve_axis.set_xlabel("environment steps")
    curve_axis.legend()

    x = np.arange(len(algorithms))
    for axis, (metric, title, higher) in zip(axes.flat[1:], METRICS):
        means, errors = [], []
        for algorithm in algorithms:
            seed_values = []
            for rows in grouped[algorithm].values():
                last_step = max(row["timestep"] for row in rows)
                seed_values.append(np.mean(
                    [row[metric] for row in rows if row["timestep"] == last_step]
                ))
            means.append(np.mean(seed_values))
            errors.append(np.std(seed_values))
        axis.bar(x, means, yerr=errors, capsize=5)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [f"{name}\nn={len(grouped[name])}" for name in algorithms]
        )
        axis.set_title(f"{title} ({'higher' if higher else 'lower'} is better)")
        axis.grid(axis="y", alpha=0.25)
    axes[1, 2].axis("off")
    figure.tight_layout()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.save, dpi=180)
    print("Saved:", args.save)
    for algorithm in algorithms:
        print(algorithm, "seeds:", sorted(grouped[algorithm]))


if __name__ == "__main__":
    main()
