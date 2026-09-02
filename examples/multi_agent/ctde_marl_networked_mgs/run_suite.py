"""Run all four CTDE algorithms sequentially for one or more seeds."""
import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = EXPERIMENT_DIR / "logs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--eval-frequency", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--network-config", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--share-reward", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    for algorithm in ("mappo", "maddpg", "masac", "matd3"):
        for seed in [int(item) for item in args.seeds.split(",")]:
            command = [
                sys.executable,
                str(directory / "train.py"),
                "--algorithm",
                algorithm,
                "--seed",
                str(seed),
                "--total-timesteps",
                str(args.total_timesteps),
                "--eval-frequency",
                str(args.eval_frequency),
                "--eval-episodes",
                str(args.eval_episodes),
                "--output-dir",
                str(args.output_root / f"{algorithm}_seed{seed}"),
                "--share-reward" if args.share_reward else "--no-share-reward",
            ]
            if args.env_config:
                command.extend(["--env-config", str(args.env_config)])
            if args.network_config:
                command.extend(["--network-config", str(args.network_config)])
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
