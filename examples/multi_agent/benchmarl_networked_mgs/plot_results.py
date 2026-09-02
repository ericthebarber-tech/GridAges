"""Plot mean evaluation metrics with a standard-deviation band across seeds."""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = EXPERIMENT_DIR / "logs"
RUN_PATTERN = re.compile(r"^(mappo|maddpg|masac)_seed(\d+)$", re.IGNORECASE)
ALGORITHM_ORDER = ("mappo", "maddpg", "masac")
METRICS = (
    ("reward", "Episode return", True),
    ("operating_cost", "Operating cost", False),
    ("safety", "Safety violation", False),
    ("convergence_rate", "Power-flow convergence rate", True),
)


def identify_run(path):
    for part in reversed(path.parts):
        match = RUN_PATTERN.match(part)
        if match:
            return match.group(1).lower(), int(match.group(2))
    return None


def discover_evaluations(root):
    """Select the newest completed evaluation file for each algorithm/seed."""
    selected = {}
    for path in root.rglob("evaluation.csv"):
        identity = identify_run(path)
        if identity is None:
            continue
        if identity not in selected or path.stat().st_mtime > selected[
            identity
        ].stat().st_mtime:
            selected[identity] = path
    return selected


def read_evaluations(path, agent):
    values = defaultdict(lambda: defaultdict(list))
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["agent"] != agent:
                continue
            timestep = int(float(row["timestep"]))
            for metric, _, _ in METRICS:
                values[timestep][metric].append(float(row[metric]))
    return {
        timestep: {
            metric: float(np.mean(metric_values))
            for metric, metric_values in metrics.items()
        }
        for timestep, metrics in values.items()
    }


def collect(root, agent):
    result = defaultdict(dict)
    for (algorithm, seed), path in discover_evaluations(root).items():
        run = read_evaluations(path, agent)
        if run:
            result[algorithm][seed] = run
    if not result:
        raise FileNotFoundError(
            f"No evaluation.csv files below {root}. Expected paths containing "
            "mappo_seed42, maddpg_seed42, or masac_seed42."
        )
    return result


def summarize_at_timesteps(seed_runs, metric):
    timesteps = sorted(
        {step for run in seed_runs.values() for step in run}
    )
    means, standard_deviations, counts = [], [], []
    for timestep in timesteps:
        values = [
            run[timestep][metric]
            for run in seed_runs.values()
            if timestep in run and metric in run[timestep]
        ]
        means.append(float(np.mean(values)))
        standard_deviations.append(
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
        counts.append(len(values))
    return (
        np.asarray(timesteps),
        np.asarray(means),
        np.asarray(standard_deviations),
        np.asarray(counts),
    )


def write_final_summary(data, path):
    rows = []
    for algorithm in ALGORITHM_ORDER:
        seed_runs = data.get(algorithm, {})
        if not seed_runs:
            continue
        common_steps = set.intersection(
            *(set(run) for run in seed_runs.values() if run)
        )
        if not common_steps:
            continue
        common_final = max(common_steps)
        for metric, _, higher_is_better in METRICS:
            values = [
                run[common_final][metric]
                for run in seed_runs.values()
                if common_final in run
            ]
            rows.append(
                {
                    "algorithm": algorithm.upper(),
                    "timestep": common_final,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "seeds": len(values),
                    "better": "higher" if higher_is_better else "lower",
                }
            )
    if not rows:
        raise ValueError(
            "Evaluation files were found, but they have no common timesteps "
            "for the requested agent."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(data, output, title):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {"mappo": "#377eb8", "maddpg": "#e41a1c", "masac": "#4daf4a"}

    for axis, (metric, label, _) in zip(axes.flat, METRICS):
        for algorithm in ALGORITHM_ORDER:
            seed_runs = data.get(algorithm, {})
            if not seed_runs:
                continue
            x, mean, std, counts = summarize_at_timesteps(seed_runs, metric)
            axis.plot(
                x,
                mean,
                linewidth=2,
                marker="o",
                markersize=3,
                color=colors[algorithm],
                label=(
                    f"{algorithm.upper()} ({len(seed_runs)} "
                    f"{'seed' if len(seed_runs) == 1 else 'seeds'})"
                ),
            )
            if np.any(counts > 1):
                axis.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=colors[algorithm],
                    alpha=0.18,
                )
        axis.set_title(label)
        axis.set_xlabel("Environment frames")
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        if metric == "convergence_rate":
            axis.set_ylim(-0.05, 1.05)

    axes[0, 0].set_ylabel("Mean ± 1 SD")
    axes[1, 0].set_ylabel("Mean ± 1 SD")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_seed(data, seed, output, agent):
    """Plot all available algorithms for one random seed."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {"mappo": "#377eb8", "maddpg": "#e41a1c", "masac": "#4daf4a"}

    for axis, (metric, label, _) in zip(axes.flat, METRICS):
        for algorithm in ALGORITHM_ORDER:
            run = data.get(algorithm, {}).get(seed)
            if not run:
                continue
            timesteps = np.asarray(sorted(run))
            values = np.asarray([run[step][metric] for step in timesteps])
            axis.plot(
                timesteps,
                values,
                linewidth=2,
                marker="o",
                markersize=3,
                color=colors[algorithm],
                label=algorithm.upper(),
            )
        axis.set_title(label)
        axis.set_xlabel("Environment frames")
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        if metric == "convergence_rate":
            axis.set_ylim(-0.05, 1.05)

    axes[0, 0].set_ylabel(f"Seed {seed} evaluation mean")
    axes[1, 0].set_ylabel(f"Seed {seed} evaluation mean")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        f"GridAges BenchMARL evaluation — seed {seed}, agent={agent}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--agent", default="system")
    parser.add_argument(
        "--per-seed",
        action="store_true",
        help="Also save one four-panel comparison figure for each seed.",
    )
    parser.add_argument(
        "--title",
        default="GridAges BenchMARL evaluation (mean across seeds)",
    )
    args = parser.parse_args()

    output = args.output or args.log_dir / "benchmarl_comparison.png"
    data = collect(args.log_dir, args.agent)
    plot(data, output, args.title)
    summary = output.with_name(f"{output.stem}_final_metrics.csv")
    write_final_summary(data, summary)
    seed_outputs = []
    if args.per_seed:
        seeds = sorted(
            {seed for algorithm_runs in data.values() for seed in algorithm_runs}
        )
        suffix = output.suffix or ".png"
        for seed in seeds:
            seed_output = output.with_name(
                f"{output.stem}_seed{seed}{suffix}"
            )
            plot_seed(data, seed, seed_output, args.agent)
            seed_outputs.append(seed_output)
    discovered = ", ".join(
        f"{algorithm.upper()}={len(seeds)}"
        for algorithm, seeds in data.items()
    )
    print(f"Runs: {discovered}")
    print(f"Plot: {output.resolve()}")
    for seed_output in seed_outputs:
        print(f"Seed plot: {seed_output.resolve()}")
    print(f"Final metrics: {summary.resolve()}")


if __name__ == "__main__":
    main()
