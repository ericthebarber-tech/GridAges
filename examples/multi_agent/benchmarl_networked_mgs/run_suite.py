"""Run all three official BenchMARL algorithms over multiple seeds."""
import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = EXPERIMENT_DIR / "logs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--total-frames", type=int, default=500_000)
    parser.add_argument("--frames-per-batch", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--optimizer-steps", type=int, default=240)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--eval-frequency", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--share-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--share-policy-params",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    for algorithm in ("mappo", "maddpg", "masac"):
        for seed in [int(item) for item in args.seeds.split(",")]:
            command = [
                sys.executable,
                str(EXPERIMENT_DIR / "train.py"),
                "--algorithm",
                algorithm,
                "--seed",
                str(seed),
                "--total-frames",
                str(args.total_frames),
                "--frames-per-batch",
                str(args.frames_per_batch),
                "--batch-size",
                str(args.batch_size),
                "--num-envs",
                str(args.num_envs),
                "--update-epochs",
                str(args.update_epochs),
                "--optimizer-steps",
                str(args.optimizer_steps),
                "--learning-starts",
                str(args.learning_starts),
                "--buffer-size",
                str(args.buffer_size),
                "--eval-frequency",
                str(args.eval_frequency),
                "--eval-episodes",
                str(args.eval_episodes),
                "--device",
                args.device,
                "--output-dir",
                str(args.output_root / f"{algorithm}_seed{seed}"),
                (
                    "--share-reward"
                    if args.share_reward
                    else "--no-share-reward"
                ),
                (
                    "--share-policy-params"
                    if args.share_policy_params
                    else "--no-share-policy-params"
                ),
            ]
            if args.env_config:
                command.extend(["--env-config", str(args.env_config)])
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
