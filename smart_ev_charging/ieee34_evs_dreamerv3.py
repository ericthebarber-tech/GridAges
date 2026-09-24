"""
Train a PPO agent on the GridAges IEEE34 EV smart-charging environment.

The training environment matches the uncontrolled-evaluation setup except for:
    * train=True
    * control_mode="rl"
    * ev_level="high"

Episodes are 48 half-hour steps:
    08:00 today -> 07:30 next day
with the terminal boundary at 08:00 the following day.

This script uses RLlib's current PPOConfig / EnvRunner / Learner API.
"""

import os
from pathlib import Path

import numpy as np

from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import register_env
from torch.utils.tensorboard import SummaryWriter

from gridages.envs.single_agent.ev_charging.ieee34_evs import IEEE34Env


# ---------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------
NUM_CPUS_PER_ENV_RUNNER = 1
NUM_ENV_RUNNERS = 1

# Set to 1 if a CUDA-capable GPU is available.
# The command-line --num-gpus option can override the default below.
DEFAULT_NUM_GPUS = 0


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
def make_env(env_config):
    return IEEE34Env(env_config=env_config)


register_env("gridages-ieee34-ev", make_env)


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------
parser = add_rllib_example_script_args(
    default_iters=100000,
    default_reward=np.inf,
    default_timesteps=1_000_000,
)

parser.set_defaults(
    checkpoint_freq=1000,
    verbose=0,
    no_tune=False,
    num_cpus_per_learner=NUM_CPUS_PER_ENV_RUNNER,
    num_gpus_per_learner=DEFAULT_NUM_GPUS,
)

args = parser.parse_args()


# ---------------------------------------------------------------------
# Callback / TensorBoard logging
# ---------------------------------------------------------------------
def _unwrap_single_env(env):
    """Return IEEE34Env from RLlib's single-env/vector-env wrapper."""
    current = env

    # RLlib commonly passes a vector env with one underlying Gymnasium env.
    if hasattr(current, "envs") and len(current.envs) > 0:
        current = current.envs[0]

    if hasattr(current, "unwrapped"):
        current = current.unwrapped

    return current


class IEEE34TrainingMetrics(RLlibCallback):
    """Log GridAges-specific training metrics.

    RLlib already logs episode return. These extra metrics make it easier to
    monitor the physical objective during training:
        * raw environment reward
        * safety penalty
        * penalized return
        * final daily PVR
        * daily peak grid demand
        * minimum/maximum bus voltage
        * maximum line loading
    """

    def __init__(self):
        super().__init__()
        self.writer = None
        self.global_step = 0

    def _init_writer(self):
        if self.writer is None:
            pid = os.getpid()
            logdir = (
                Path.home()
                / "ray_results"
                / "ieee34_evs"
                / "ppo"
                / f"pid{pid}"
            )
            logdir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(logdir))

    def on_episode_created(self, *, episode, **kwargs):
        self._init_writer()

        episode.custom_data["reward"] = []
        episode.custom_data["safety"] = []
        episode.custom_data["penalized_return"] = []
        episode.custom_data["cost"] = []
        episode.custom_data["min_voltage"] = []
        episode.custom_data["max_voltage"] = []
        episode.custom_data["max_line_loading"] = []

    def on_episode_step(self, *, episode, env, **kwargs):
        base_env = _unwrap_single_env(env)

        reward = np.sum(list(base_env.reward.values()))
        safety = np.sum(list(base_env.safety.values()))
        cost = base_env.devices.get("Grid").cost

        reward_scale = base_env.cfg.get("reward_scale", 1.0)
        safety_scale = base_env.cfg.get("safety_scale", 0.0)
        penalize_safety = base_env.cfg.get("penalize_safety", False)

        reward *= reward_scale
        safety *= safety_scale

        if penalize_safety:
            penalized_return = reward - safety
        else:
            penalized_return = reward

        episode.custom_data["reward"].append(reward)
        episode.custom_data["safety"].append(safety)
        episode.custom_data["cost"].append(cost)
        episode.custom_data["penalized_return"].append(penalized_return)

        if base_env.net.get("converged", False):
            if len(base_env.net.res_bus):
                vm = base_env.net.res_bus["vm_pu"].to_numpy(dtype=float)
                episode.custom_data["min_voltage"].append(float(np.nanmin(vm)))
                episode.custom_data["max_voltage"].append(float(np.nanmax(vm)))

            if len(base_env.net.res_line):
                loading = base_env.net.res_line[
                    "loading_percent"
                ].to_numpy(dtype=float)
                episode.custom_data["max_line_loading"].append(
                    float(np.nanmax(loading))
                )

        # if self.writer is not None:
        #     self.writer.add_scalar(
        #         "train_step/reward",
        #         reward,
        #         self.global_step,
        #     )
        #     self.writer.add_scalar(
        #         "train_step/safety",
        #         safety,
        #         self.global_step,
        #     )
        #     self.writer.add_scalar(
        #         "train_step/penalized_return",
        #         penalized_return,
        #         self.global_step,
        #     )
        #     # self.writer.add_scalar(
        #     #     "train_step/residental_011_ev_soc",
        #     #     base_env.devices['RESIDENTIAL_EV_011'].state.soc,
        #     #     self.global_step,
        #     # )

        self.global_step += 1

    def on_episode_end(self, *, episode, env, metrics_logger, **kwargs):
        base_env = _unwrap_single_env(env)

        ep_reward = float(np.sum(episode.custom_data["reward"]))
        ep_safety = float(np.sum(episode.custom_data["safety"]))
        ep_cost = float(np.sum(episode.custom_data["cost"]))
        ep_return = float(np.sum(episode.custom_data["penalized_return"]))

        # _grid_p_history is maintained by IEEE34Env._reward_and_safety().
        if len(base_env._episode_grid_p):
            p = np.asarray(base_env._episode_grid_p, dtype=float)
            # Daily mean demand.
            p_mean = np.mean(p)
            # Mean squared deviation from daily mean.
            load_variance = np.mean((p - p_mean) ** 2)
            peak_grid_p_mw = float(np.nanmax(p))
        else:
            load_variance = np.nan
            peak_grid_p_mw = np.nan

        min_voltage = (
            float(np.min(episode.custom_data["min_voltage"]))
            if episode.custom_data["min_voltage"]
            else np.nan
        )
        max_voltage = (
            float(np.max(episode.custom_data["max_voltage"]))
            if episode.custom_data["max_voltage"]
            else np.nan
        )
        max_line_loading = (
            float(np.max(episode.custom_data["max_line_loading"]))
            if episode.custom_data["max_line_loading"]
            else np.nan
        )

        # These appear under result["env_runners"][...].
        metrics_logger.log_value(
            "ep_reward",
            ep_reward,
            reduce="mean",
            window=20,
        )
        metrics_logger.log_value(
            "ep_safety",
            ep_safety,
            reduce="mean",
            window=20,
        )
        metrics_logger.log_value(
            "ep_cost",
            ep_safety,
            reduce="mean",
            window=20,
        )
        metrics_logger.log_value(
            "ep_penalized_return",
            ep_return,
            reduce="mean",
            window=20,
        )
        metrics_logger.log_value(
            "ep_load_variance",
            load_variance,
            reduce="mean",
            window=20,
        )
        metrics_logger.log_value(
            "ep_peak_grid_p_mw",
            peak_grid_p_mw,
            reduce="mean",
            window=20,
        )
        metrics_logger.log_value(
            "ep_min_voltage_pu",
            min_voltage,
            reduce="mean",
            window=20,
        )
        metrics_logger.log_value(
            "ep_max_voltage_pu",
            max_voltage,
            reduce="mean",
            window=20,
        )
        metrics_logger.log_value(
            "ep_max_line_loading_percent",
            max_line_loading,
            reduce="mean",
            window=20,
        )

        if self.writer is not None:
            episode_id = self.global_step
            self.writer.add_scalar(
                "train_episode/reward",
                ep_reward,
                episode_id,
            )
            self.writer.add_scalar(
                "train_episode/safety",
                ep_safety,
                episode_id,
            )
            self.writer.add_scalar(
                "train_episode/cost",
                ep_cost,
                episode_id,
            )
            self.writer.add_scalar(
                "train_episode/penalized_return",
                ep_return,
                episode_id,
            )

            if np.isfinite(load_variance):
                self.writer.add_scalar(
                    "train_episode/load_variance",
                    load_variance,
                    episode_id,
                )
            if np.isfinite(peak_grid_p_mw):
                self.writer.add_scalar(
                    "train_episode/peak_grid_p_mw",
                    peak_grid_p_mw,
                    episode_id,
                )
            if np.isfinite(min_voltage):
                self.writer.add_scalar(
                    "train_episode/min_voltage_pu",
                    min_voltage,
                    episode_id,
                )
            if np.isfinite(max_voltage):
                self.writer.add_scalar(
                    "train_episode/max_voltage_pu",
                    max_voltage,
                    episode_id,
                )
            self.writer.add_scalar(
                "train_episode/residential_001_ev_soc",
                base_env.devices['RESIDENTIAL_EV_001'].state.soc,
                episode_id,
            )

    def on_algorithm_end(self, **kwargs):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------
# This intentionally mirrors evaluate_uncontrolled_ieee34.py, except:
#   train=False            -> train=True
#   control_mode=uncontrolled -> control_mode=rl
#
# The high-EV scenario and 0.5-1.5 load scaling are kept identical.
ENV_CONFIG = {
    "train": True,
    "ev_level": "low",
    "control_mode": "rl",
    "episode_length": 48,
    "dt": 0.5,
    "start_hour": 8.0,
    "store_info": [
        "bus_voltage",
        "line_loading",
        "operating_cost",
    ],
    "store_arrays": False,
    "store_summaries": True,
    "reward_scale": 1.0,
    "safety_scale": 10000.0,
    "peak_weight": 10.0,
    "shape_weight": 1000.0,
    "price_weight": 0.0,
    "penalize_safety": True,
}

# Current RLlib uses the new RLModule/Learner and EnvRunner stack by default.
#
# A 4096-step train batch contains ~85 complete 48-step episodes.
# complete_episodes keeps each 08:00->07:30 trajectory intact.
config = (
    DreamerV3Config()
    .environment(
        env="gridages-ieee34-ev",
        env_config=ENV_CONFIG,
        disable_env_checking=False,
    )
    .env_runners(
        num_env_runners=NUM_ENV_RUNNERS,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=NUM_CPUS_PER_ENV_RUNNER,
        num_gpus_per_env_runner=0,
    )
    .learners(
        num_cpus_per_learner=1,
        num_gpus_per_learner=1 if args.num_gpus_per_learner > 0 else 0,
    )
    .training(
        model_size="XS",
    )
    .callbacks(
        callbacks_class=IEEE34TrainingMetrics,
    )
)


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_rllib_example_script_experiment(config, args)
