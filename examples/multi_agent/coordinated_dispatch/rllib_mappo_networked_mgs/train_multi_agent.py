"""Shared SB3 trainer for centralized PPO, DDPG, SAC, and TD3 baselines."""
import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization

from centralized_env import CentralizedMicrogridsEnv


ALGORITHMS = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC, "TD3": TD3}
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_SB3_LOG_DIR = EXPERIMENT_DIR / "logs" / "sb3"


class GridMetricsCallback(BaseCallback):
    """Write step and episode grid metrics to the SB3 logger/TensorBoard."""
    keys = ("operating_cost", "safety", "convergence_rate")

    def __init__(self):
        super().__init__()
        self.episode_values = {key: [] for key in self.keys}

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info in infos:
            for key in self.keys:
                if key in info:
                    value = float(info[key])
                    self.logger.record("grid/{}".format(key), value)
                    self.episode_values[key].append(value)
        if np.any(dones):
            for key, values in self.episode_values.items():
                if values:
                    self.logger.record(
                        "grid/{}_ep_mean".format(key), float(np.mean(values))
                    )
            self.episode_values = {key: [] for key in self.keys}
        return True


def make_env(train, seed):
    env = CentralizedMicrogridsEnv({
        "train": train, "penalty": 10.0, "share_reward": False,
    })
    env.reset(seed=seed)
    return Monitor(env)


def make_vec_env(train, seed, normalize_reward):
    vec_env = DummyVecEnv([lambda: make_env(train, seed)])
    return VecNormalize(
        vec_env,
        training=train,
        norm_obs=True,
        norm_reward=normalize_reward,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )


def model_kwargs(name, env, tensorboard_log, seed):
    common = dict(
        policy="MlpPolicy", env=env, tensorboard_log=str(tensorboard_log),
        seed=seed, verbose=1, device="auto",
        policy_kwargs={"net_arch": [128, 128]},
    )
    if name == "PPO":
        common.update(learning_rate=1e-4, n_steps=2048, batch_size=64,
                      gamma=0.99, gae_lambda=0.95, n_epochs=10, clip_range=0.2,
                      max_grad_norm=0.5)
    else:
        common.update(buffer_size=1_000_000, learning_starts=10_000,
                      batch_size=256, tau=0.005, gamma=0.99,
                      train_freq=1, gradient_steps=1)
        if name == "SAC":
            common.update(learning_rate=3e-4, ent_coef="auto")
        else:
            common.update(
                learning_rate=1e-3,
                action_noise=NormalActionNoise(
                    mean=np.zeros(env.action_space.shape[-1]),
                    sigma=0.1 * np.ones(env.action_space.shape[-1]),
                ),
            )
        if name == "TD3":
            common.update(policy_delay=2, target_policy_noise=0.2,
                          target_noise_clip=0.5)
    return common


def evaluate(model, env, episodes):
    rows = []
    for episode in range(episodes):
        observation = env.reset()
        agent_ids = env.get_attr("agent_ids")[0]
        rewards = {agent: 0.0 for agent in agent_ids}
        safety = {agent: 0.0 for agent in agent_ids}
        convergence = {agent: 0.0 for agent in agent_ids}
        for step in range(24):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, infos = env.step(action)
            info = infos[0]
            for agent in agent_ids:
                rewards[agent] += float(info["agent_rewards"][agent])
                safety[agent] += float(info["agent_safety"][agent])
                convergence[agent] += float(
                    info["agents"][agent].get("converged", False)
                )
            if done[0]:
                break
        for agent in agent_ids:
            rows.append({
                "episode": episode, "agent": agent, "reward": rewards[agent],
                "operating_cost": -rewards[agent] - 10.0 * safety[agent],
                "safety": safety[agent],
                "convergence_rate": convergence[agent] / float(step + 1),
                "steps": step + 1,
            })
        rows.append({
            "episode": episode, "agent": "system",
            "reward": sum(rewards.values()),
            "operating_cost": sum(-rewards[a] - 10.0 * safety[a] for a in agent_ids),
            "safety": sum(safety.values()),
            "convergence_rate": np.mean([
                convergence[a] / float(step + 1) for a in agent_ids
            ]),
            "steps": step + 1,
        })
    return rows


def run(name, args):
    output = args.output_dir or DEFAULT_SB3_LOG_DIR / name.lower()
    model_dir, tensorboard_dir = output / "models", output / "tb"
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    train_env = make_vec_env(True, args.seed, normalize_reward=True)
    eval_env = make_vec_env(False, args.seed + 10_000, normalize_reward=False)
    model = ALGORITHMS[name](**model_kwargs(name, train_env, tensorboard_dir, args.seed))
    callbacks = [
        EvalCallback(eval_env, best_model_save_path=str(model_dir), log_path=str(output),
                     eval_freq=args.eval_freq, n_eval_episodes=args.eval_episodes,
                     deterministic=True),
        CheckpointCallback(save_freq=args.eval_freq, save_path=str(model_dir),
                           name_prefix=name.lower(), save_vecnormalize=True),
        GridMetricsCallback(),
    ]
    progress_bar = args.progress_bar
    if progress_bar and (
        importlib.util.find_spec("tqdm") is None
        or importlib.util.find_spec("rich") is None
    ):
        print(
            "Progress bar disabled: install it with "
            "`python -m pip install tqdm rich`."
        )
        progress_bar = False
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                    progress_bar=progress_bar)
        model.save(model_dir / "final_model")
        train_env.save(str(model_dir / "vecnormalize.pkl"))
        sync_envs_normalization(train_env, eval_env)
        rows = evaluate(model, eval_env, args.eval_episodes)
        with (output / "evaluation.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    finally:
        train_env.close()
        eval_env.close()


def main(default_algo=None):
    parser = argparse.ArgumentParser()
    if default_algo is None:
        parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="PPO")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()
    run(default_algo or args.algo, args)


if __name__ == "__main__":
    main()
