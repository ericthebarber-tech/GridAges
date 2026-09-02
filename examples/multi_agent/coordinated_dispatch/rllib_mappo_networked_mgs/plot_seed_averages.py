"""Plot algorithm means and standard deviations across random seeds."""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("PPO", "DDPG", "SAC", "TD3")
METRICS = (
    ("reward", "Evaluation return", True),
    ("operating_cost", "Operating cost", False),
    ("safety", "Safety violation", False),
    ("convergence_rate", "Power-flow convergence", True),
)
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_SB3_LOG_DIR = EXPERIMENT_DIR / "logs" / "sb3"


def identify_run(directory):
    match = re.match(r"^(ppo|ddpg|sac|td3)_seed(\d+)", directory.name.lower())
    if not match:
        return None
    return match.group(1).upper(), int(match.group(2))


def read_run(directory):
    evaluation_file = directory / "evaluation.csv"
    history_file = directory / "evaluations.npz"
    if not evaluation_file.exists() or not history_file.exists():
        return None

    with evaluation_file.open(newline="") as stream:
        system_rows = [row for row in csv.DictReader(stream) if row["agent"] == "system"]
    if not system_rows:
        return None

    evaluation = {
        metric: float(np.mean([float(row[metric]) for row in system_rows]))
        for metric, _, _ in METRICS
    }
    archive = np.load(history_file)
    return {
        "timesteps": np.asarray(archive["timesteps"], dtype=float),
        "rewards": np.mean(np.asarray(archive["results"], dtype=float), axis=1),
        "evaluation": evaluation,
    }


def aggregate_curves(runs):
    max_start = max(run["timesteps"][0] for run in runs)
    min_end = min(run["timesteps"][-1] for run in runs)
    points = min(200, min(len(run["timesteps"]) for run in runs))
    grid = np.linspace(max_start, min_end, points)
    curves = np.vstack([
        np.interp(grid, run["timesteps"], run["rewards"]) for run in runs
    ])
    return grid, np.mean(curves, axis=0), np.std(curves, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_SB3_LOG_DIR / "seed_average_comparison.png",
    )
    args = parser.parse_args()

    grouped = defaultdict(dict)
    skipped = []
    for directory in args.directories:
        identity = identify_run(directory)
        run = read_run(directory) if identity else None
        if identity and run:
            algorithm, seed = identity
            # Prefer a completed "stable" rerun if duplicate seed directories exist.
            grouped[algorithm][seed] = run
        else:
            skipped.append(str(directory))

    algorithms = [name for name in ALGORITHMS if grouped[name]]
    if not algorithms:
        raise SystemExit("No complete seeded runs were found.")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    curve_axis = axes[0, 0]
    for algorithm in algorithms:
        runs = list(grouped[algorithm].values())
        steps, mean, std = aggregate_curves(runs)
        line = curve_axis.plot(steps, mean, label="{} (n={})".format(
            algorithm, len(runs)))[0]
        curve_axis.fill_between(steps, mean - std, mean + std,
                                color=line.get_color(), alpha=0.2)
    curve_axis.set_title("Mean evaluation return across seeds")
    curve_axis.set_xlabel("environment steps")
    curve_axis.set_ylabel("return")
    curve_axis.legend()

    x = np.arange(len(algorithms))
    for axis, (metric, title, higher_is_better) in zip(axes.flat[1:], METRICS):
        means, stds = [], []
        for algorithm in algorithms:
            values = [run["evaluation"][metric] for run in grouped[algorithm].values()]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
        axis.bar(x, means, yerr=stds, capsize=5)
        axis.set_xticks(x)
        axis.set_xticklabels([
            "{}\nn={}".format(name, len(grouped[name])) for name in algorithms
        ])
        axis.set_title("{} ({} is better)".format(
            title, "higher" if higher_is_better else "lower"
        ))

    axes[1, 2].axis("off")
    for axis in axes.flat[:5]:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=180)

    print("Saved:", args.save)
    for algorithm in algorithms:
        print("{} seeds: {}".format(algorithm, sorted(grouped[algorithm])))
    if skipped:
        print("Skipped incomplete/unrecognized runs:")
        for directory in skipped:
            print(" -", directory)


if __name__ == "__main__":
    main()
