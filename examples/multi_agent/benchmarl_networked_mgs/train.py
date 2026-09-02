"""Train official BenchMARL MAPPO, MADDPG, or MASAC on GridAges."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from benchmarl.algorithms import MaddpgConfig, MappoConfig, MasacConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

from callbacks import GridMetricsCallback
from task import make_task


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = EXPERIMENT_DIR / "logs"
DEFAULT_ENV_CONFIG = EXPERIMENT_DIR / "configs" / "heterogeneous_env.json"
ALGORITHM_CONFIGS = {
    "mappo": MappoConfig,
    "maddpg": MaddpgConfig,
    "masac": MasacConfig,
}


def load_json(path):
    if path is None:
        return {}
    with Path(path).open() as stream:
        return json.load(stream)


def parse_hidden(value):
    widths = [int(item) for item in value.split(",") if item.strip()]
    if not widths:
        raise ValueError("At least one hidden layer is required")
    return widths


def effective_interval(requested, frames_per_batch):
    return int(math.ceil(requested / frames_per_batch) * frames_per_batch)


def build_experiment(args):
    output = args.output_dir or DEFAULT_LOG_DIR / (
        f"{args.algorithm}_seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)

    task_config = {
        "env_config": {
            **load_json(args.env_config),
            "penalty": args.penalty,
            "share_reward": args.share_reward,
        },
        "reward_scale": args.reward_scale,
        "observation_clip": args.observation_clip,
        "episode_length": 24,
    }
    task = make_task(task_config)

    algorithm_config = ALGORITHM_CONFIGS[args.algorithm].get_from_yaml()
    algorithm_config.share_param_critic = args.share_critic_params
    if args.algorithm == "mappo":
        algorithm_config.entropy_coef = args.entropy_coefficient
        algorithm_config.clip_epsilon = args.clip_epsilon
        algorithm_config.lmbda = args.gae_lambda
    elif args.algorithm == "masac":
        algorithm_config.alpha_init = args.alpha_init
        algorithm_config.fixed_alpha = args.fixed_alpha

    actor_model = MlpConfig.get_from_yaml()
    actor_model.num_cells = parse_hidden(args.actor_hidden)
    actor_model.activation_class = torch.nn.ReLU
    critic_model = MlpConfig.get_from_yaml()
    critic_model.num_cells = parse_hidden(args.critic_hidden)
    critic_model.activation_class = torch.nn.ReLU

    config = ExperimentConfig.get_from_yaml()
    config.sampling_device = args.device
    config.train_device = args.device
    config.buffer_device = "cpu"
    config.share_policy_params = args.share_policy_params
    config.gamma = args.gamma
    config.lr = args.learning_rate
    config.max_n_frames = args.total_frames
    config.max_n_iters = None
    config.parallel_collection = False
    config.render = False
    config.loggers = ["csv"]
    config.create_json = True
    config.save_folder = str(output.resolve())
    config.checkpoint_interval = 0
    config.checkpoint_at_end = args.checkpoint_at_end
    config.keep_checkpoints_num = 1

    config.on_policy_collected_frames_per_batch = args.frames_per_batch
    config.on_policy_n_envs_per_worker = args.num_envs
    config.on_policy_n_minibatch_iters = args.update_epochs
    config.on_policy_minibatch_size = min(
        args.batch_size, args.frames_per_batch
    )

    config.off_policy_collected_frames_per_batch = args.frames_per_batch
    config.off_policy_n_envs_per_worker = args.num_envs
    config.off_policy_n_optimizer_steps = args.optimizer_steps
    config.off_policy_train_batch_size = args.batch_size
    config.off_policy_memory_size = args.buffer_size
    config.off_policy_init_random_frames = args.learning_starts
    config.off_policy_use_prioritized_replay_buffer = False

    interval = effective_interval(args.eval_frequency, args.frames_per_batch)
    config.evaluation = args.eval_episodes > 0
    config.evaluation_interval = interval
    config.evaluation_episodes = max(args.eval_episodes, 1)
    config.evaluation_deterministic_actions = True
    if interval != args.eval_frequency:
        print(
            f"Evaluation interval adjusted from {args.eval_frequency} to "
            f"{interval}; BenchMARL requires a multiple of frames-per-batch."
        )

    resolved = {
        "algorithm": args.algorithm,
        "seed": args.seed,
        "task": task_config,
        "experiment": config.__dict__,
        "algorithm_config": algorithm_config.__dict__,
        "actor_model": actor_model.__dict__,
        "critic_model": critic_model.__dict__,
        "benchmarl_version": "1.4.0",
        "torchrl_version": "0.7.2",
    }
    with (output / "run_config.json").open("w") as stream:
        json.dump(resolved, stream, indent=2, default=str)

    return Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model,
        critic_model_config=critic_model,
        seed=args.seed,
        config=config,
        callbacks=[GridMetricsCallback()],
    )


def train(args):
    experiment = build_experiment(args)
    print("BenchMARL output:", experiment.folder_name)
    experiment.run()


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--algorithm", choices=ALGORITHM_CONFIGS, default="mappo")
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--total-frames", type=int, default=500_000)
    value.add_argument("--frames-per-batch", type=int, default=240)
    value.add_argument("--batch-size", type=int, default=128)
    value.add_argument("--num-envs", type=int, default=1)
    value.add_argument("--update-epochs", type=int, default=10)
    value.add_argument("--optimizer-steps", type=int, default=240)
    value.add_argument("--learning-starts", type=int, default=1_000)
    value.add_argument("--buffer-size", type=int, default=200_000)
    value.add_argument("--eval-frequency", type=int, default=10_000)
    value.add_argument("--eval-episodes", type=int, default=5)
    value.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG)
    value.add_argument("--output-dir", type=Path)
    value.add_argument(
        "--share-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    value.add_argument(
        "--share-policy-params",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    value.add_argument(
        "--share-critic-params",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    value.add_argument(
        "--checkpoint-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    value.add_argument("--actor-hidden", default="128,128")
    value.add_argument("--critic-hidden", default="256,256")
    value.add_argument("--learning-rate", type=float, default=3e-4)
    value.add_argument("--gamma", type=float, default=0.99)
    value.add_argument("--penalty", type=float, default=10.0)
    value.add_argument("--reward-scale", type=float, default=0.01)
    value.add_argument("--observation-clip", type=float, default=10.0)
    value.add_argument("--clip-epsilon", type=float, default=0.2)
    value.add_argument("--gae-lambda", type=float, default=0.95)
    value.add_argument("--entropy-coefficient", type=float, default=0.01)
    value.add_argument("--alpha-init", type=float, default=1.0)
    value.add_argument(
        "--fixed-alpha",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    value.add_argument(
        "--device",
        default="cpu",
    )
    return value


if __name__ == "__main__":
    train(parser().parse_args())
