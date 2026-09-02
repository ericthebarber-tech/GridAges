"""Train MAPPO, MADDPG, MASAC, or MATD3 on a PettingZoo grid environment."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from algorithms import MADDPG, MAPPO, MASAC, MATD3
from core import MultiAgentAdapter, MultiAgentReplayBuffer, load_json


ALGORITHMS = {
    "mappo": MAPPO,
    "maddpg": MADDPG,
    "masac": MASAC,
    "matd3": MATD3,
}
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = EXPERIMENT_DIR / "logs"
DEFAULT_ENV = (
    "configurable_env:ConfigurableMultiAgentMicrogrids"
)


def append_rows(path, rows):
    rows = list(rows)
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def episode_rows(timestep, episode, raw_rewards, infos, steps, phase):
    rows = []
    for agent, reward in raw_rewards.items():
        rows.append(
            {
                "timestep": timestep,
                "episode": episode,
                "phase": phase,
                "agent": agent,
                "reward": reward,
                "operating_cost": infos[agent]["operating_cost"],
                "safety": infos[agent]["safety"],
                "convergence_rate": infos[agent]["convergence_rate"],
                "steps": steps,
            }
        )
    rows.append(
        {
            "timestep": timestep,
            "episode": episode,
            "phase": phase,
            "agent": "system",
            "reward": sum(raw_rewards.values()),
            "operating_cost": sum(v["operating_cost"] for v in infos.values()),
            "safety": sum(v["safety"] for v in infos.values()),
            "convergence_rate": float(
                np.mean([v["convergence_rate"] for v in infos.values()])
            ),
            "steps": steps,
        }
    )
    return rows


def evaluate(algorithm, adapter, episodes, timestep):
    rows = []
    for episode in range(episodes):
        observations, _ = adapter.reset()
        rewards = {a: 0.0 for a in adapter.agent_ids}
        metrics = {
            a: {"operating_cost": 0.0, "safety": 0.0, "convergence_rate": 0.0}
            for a in adapter.agent_ids
        }
        for step in range(adapter.env.max_episode_steps):
            selected = algorithm.select_actions(observations, deterministic=True)
            actions = selected[0] if isinstance(selected, tuple) else selected
            observations, _, done, infos, raw_rewards = adapter.step(actions)
            for agent in adapter.agent_ids:
                rewards[agent] += float(raw_rewards[agent])
                metrics[agent]["operating_cost"] += float(
                    infos[agent].get("operating_cost", 0.0)
                )
                metrics[agent]["safety"] += float(infos[agent].get("safety", 0.0))
                metrics[agent]["convergence_rate"] += float(
                    infos[agent].get("converged", False)
                )
            if done:
                break
        for agent in adapter.agent_ids:
            metrics[agent]["convergence_rate"] /= float(step + 1)
        rows.extend(
            episode_rows(
                timestep, episode, rewards, metrics, step + 1, "evaluation"
            )
        )
    return rows


def save_checkpoint(path, algorithm, adapter, timestep, config):
    torch.save(
        {
            "timestep": timestep,
            "algorithm": config["algorithm"],
            "model": algorithm.checkpoint(),
            "normalization": adapter.normalization_state(),
            "config": config,
        },
        path,
    )


def make_algorithm(name, adapter, network_config, args):
    common = dict(
        adapter=adapter,
        network_config=network_config,
        device=args.device,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
    )
    if name == "mappo":
        common.update(
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            entropy_coefficient=args.entropy_coefficient,
            update_epochs=args.update_epochs,
            minibatch_size=args.batch_size,
        )
    elif name == "masac":
        common["alpha_lr"] = args.alpha_lr
    elif name == "matd3":
        common.update(
            policy_delay=args.policy_delay,
            target_noise=args.target_noise,
            noise_clip=args.noise_clip,
        )
    return ALGORITHMS[name](**common)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_config = load_json(args.env_config)
    env_config.update(
        {
            "penalty": args.penalty,
            "share_reward": args.share_reward,
        }
    )
    network_config = load_json(args.network_config)
    if not network_config:
        network_config = {
            "default": {
                "actor": [int(v) for v in args.actor_hidden.split(",")],
                "critic": [int(v) for v in args.critic_hidden.split(",")],
            }
        }

    output = args.output_dir or DEFAULT_LOG_DIR / (
        f"{args.algorithm}_seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(output)
    config["env_config_resolved"] = env_config
    config["network_config_resolved"] = network_config
    with (output / "config.json").open("w") as stream:
        json.dump(config, stream, indent=2, default=str)

    train_env = MultiAgentAdapter(
        args.env_class,
        env_config,
        train=True,
        seed=args.seed,
        reward_scale=args.reward_scale,
    )
    eval_env = MultiAgentAdapter(
        args.env_class,
        env_config,
        train=False,
        seed=args.seed + 10_000,
        reward_scale=args.reward_scale,
        update_normalization=False,
    )
    algorithm = make_algorithm(
        args.algorithm, train_env, network_config, args
    )
    replay = None
    if args.algorithm != "mappo":
        replay = MultiAgentReplayBuffer(
            args.buffer_size, train_env.specs, train_env.agent_ids
        )

    observations, _ = train_env.reset(seed=args.seed)
    episode = 0
    episode_rewards = {a: 0.0 for a in train_env.agent_ids}
    episode_metrics = {
        a: {"operating_cost": 0.0, "safety": 0.0, "convergence_rate": 0.0}
        for a in train_env.agent_ids
    }
    episode_steps = 0
    next_evaluation = args.eval_frequency
    next_update_log = args.log_frequency
    update_metrics = defaultdict(list)
    start_time = time.time()

    try:
        for timestep in range(1, args.total_timesteps + 1):
            if args.algorithm == "mappo":
                actions, log_probs, values = algorithm.select_actions(observations)
            elif timestep <= args.learning_starts:
                actions = {
                    a: np.random.uniform(-1, 1, train_env.specs[a].action_dim)
                    .astype(np.float32)
                    for a in train_env.agent_ids
                }
            else:
                actions = algorithm.select_actions(
                    observations, noise_std=args.exploration_noise
                )

            next_observations, scaled_rewards, done, infos, raw_rewards = (
                train_env.step(actions)
            )
            if args.algorithm == "mappo":
                algorithm.add_transition(
                    observations,
                    actions,
                    log_probs,
                    values,
                    scaled_rewards,
                    done,
                )
                if len(algorithm.rollout) >= args.rollout_steps:
                    metrics = algorithm.update(next_observations, done)
                    for key, value in metrics.items():
                        update_metrics[key].append(value)
            else:
                replay.add(
                    observations,
                    actions,
                    scaled_rewards,
                    next_observations,
                    done,
                )
                if timestep > args.learning_starts and len(replay) >= args.batch_size:
                    for _ in range(args.gradient_steps):
                        metrics = algorithm.update(
                            replay.sample(args.batch_size, algorithm.device)
                        )
                        for key, value in metrics.items():
                            update_metrics[key].append(value)

            episode_steps += 1
            for agent in train_env.agent_ids:
                episode_rewards[agent] += float(raw_rewards[agent])
                episode_metrics[agent]["operating_cost"] += float(
                    infos[agent].get("operating_cost", 0.0)
                )
                episode_metrics[agent]["safety"] += float(
                    infos[agent].get("safety", 0.0)
                )
                episode_metrics[agent]["convergence_rate"] += float(
                    infos[agent].get("converged", False)
                )
            observations = next_observations

            if done:
                for agent in train_env.agent_ids:
                    episode_metrics[agent]["convergence_rate"] /= episode_steps
                append_rows(
                    output / "training.csv",
                    episode_rows(
                        timestep,
                        episode,
                        episode_rewards,
                        episode_metrics,
                        episode_steps,
                        "training",
                    ),
                )
                episode += 1
                observations, _ = train_env.reset()
                episode_rewards = {a: 0.0 for a in train_env.agent_ids}
                episode_metrics = {
                    a: {
                        "operating_cost": 0.0,
                        "safety": 0.0,
                        "convergence_rate": 0.0,
                    }
                    for a in train_env.agent_ids
                }
                episode_steps = 0

            if timestep >= next_update_log and update_metrics:
                append_rows(
                    output / "updates.csv",
                    (
                        {
                            "timestep": timestep,
                            "metric": key,
                            "value": float(np.mean(values)),
                        }
                        for key, values in sorted(update_metrics.items())
                    ),
                )
                update_metrics.clear()
                next_update_log += args.log_frequency

            if timestep >= next_evaluation or timestep == args.total_timesteps:
                eval_env.copy_normalization_from(train_env)
                rows = evaluate(
                    algorithm, eval_env, args.eval_episodes, timestep
                )
                append_rows(output / "evaluation.csv", rows)
                save_checkpoint(
                    checkpoints / f"step_{timestep}.pt",
                    algorithm,
                    train_env,
                    timestep,
                    config,
                )
                system = [row for row in rows if row["agent"] == "system"]
                print(
                    f"{args.algorithm.upper()} step={timestep} "
                    f"return={np.mean([r['reward'] for r in system]):.3f} "
                    f"elapsed={time.time() - start_time:.1f}s"
                )
                next_evaluation += args.eval_frequency

        if args.algorithm == "mappo" and algorithm.rollout:
            metrics = algorithm.update(observations, done)
            for key, value in metrics.items():
                update_metrics[key].append(value)
        if update_metrics:
            append_rows(
                output / "updates.csv",
                (
                    {
                        "timestep": args.total_timesteps,
                        "metric": key,
                        "value": float(np.mean(values)),
                    }
                    for key, values in sorted(update_metrics.items())
                ),
            )
        save_checkpoint(
            output / "final_model.pt",
            algorithm,
            train_env,
            args.total_timesteps,
            config,
        )
    finally:
        train_env.close()
        eval_env.close()


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--algorithm", choices=ALGORITHMS, default="mappo")
    value.add_argument("--env-class", default=DEFAULT_ENV)
    value.add_argument("--env-config", type=Path)
    value.add_argument("--network-config", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--total-timesteps", type=int, default=500_000)
    value.add_argument("--eval-frequency", type=int, default=10_000)
    value.add_argument("--eval-episodes", type=int, default=10)
    value.add_argument("--log-frequency", type=int, default=1_000)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--share-reward", action=argparse.BooleanOptionalAction,
                       default=True)
    value.add_argument("--penalty", type=float, default=10.0)
    value.add_argument("--reward-scale", type=float, default=0.01)
    value.add_argument("--actor-hidden", default="128,128")
    value.add_argument("--critic-hidden", default="256,256")
    value.add_argument("--actor-lr", type=float, default=3e-4)
    value.add_argument("--critic-lr", type=float, default=3e-4)
    value.add_argument("--alpha-lr", type=float, default=3e-4)
    value.add_argument("--gamma", type=float, default=0.99)
    value.add_argument("--tau", type=float, default=0.005)
    value.add_argument("--buffer-size", type=int, default=1_000_000)
    value.add_argument("--batch-size", type=int, default=256)
    value.add_argument("--learning-starts", type=int, default=10_000)
    value.add_argument("--gradient-steps", type=int, default=1)
    value.add_argument("--exploration-noise", type=float, default=0.1)
    value.add_argument("--rollout-steps", type=int, default=2_048)
    value.add_argument("--gae-lambda", type=float, default=0.95)
    value.add_argument("--clip-ratio", type=float, default=0.2)
    value.add_argument("--entropy-coefficient", type=float, default=0.01)
    value.add_argument("--update-epochs", type=int, default=10)
    value.add_argument("--policy-delay", type=int, default=2)
    value.add_argument("--target-noise", type=float, default=0.2)
    value.add_argument("--noise-clip", type=float, default=0.5)
    value.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return value


if __name__ == "__main__":
    train(parser().parse_args())
