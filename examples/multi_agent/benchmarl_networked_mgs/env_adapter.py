"""PettingZoo adapter that makes heterogeneous GridAges agents stackable."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box
from pettingzoo.utils.env import ParallelEnv


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CTDE_EXAMPLE = REPOSITORY_ROOT / "examples" / "multi_agent" / (
    "ctde_marl_networked_mgs"
)
for path in (REPOSITORY_ROOT, CTDE_EXAMPLE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configurable_env import ConfigurableMultiAgentMicrogrids  # noqa: E402


class BenchMARLMicrogrids(ParallelEnv):
    """Pad local spaces so all microgrids can form one BenchMARL agent group.

    The underlying environment still receives each agent's original physical
    action. Padding only exists at the PettingZoo/TorchRL boundary.
    """

    metadata = {"name": "gridages_benchmarl_v0", "render_modes": []}

    def __init__(
        self,
        env_config,
        *,
        train,
        seed,
        reward_scale=0.01,
        observation_clip=10.0,
    ):
        config = dict(env_config)
        config.update({"train": bool(train), "seed": int(seed)})
        self.env = ConfigurableMultiAgentMicrogrids(config)
        self.possible_agents = list(self.env.possible_agents)
        self.agents = self.possible_agents[:]
        self.max_episode_steps = self.env.max_episode_steps
        self.reward_scale = float(reward_scale)
        self.observation_clip = float(observation_clip)

        self._observation_dims = {
            agent: int(np.prod(self.env.observation_space(agent).shape))
            for agent in self.possible_agents
        }
        self._action_dims = {
            agent: int(np.prod(self.env.action_space(agent).shape))
            for agent in self.possible_agents
        }
        self.observation_dim = max(self._observation_dims.values())
        self.action_dim = max(self._action_dims.values())

        observation_box = Box(
            low=-self.observation_clip,
            high=self.observation_clip,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.observation_spaces = {
            agent: observation_box for agent in self.possible_agents
        }
        # BenchMARL stacks all agents in one group and therefore requires an
        # identical action spec for every member. Policies operate in [-1, 1];
        # step() maps the active prefix back to each microgrid's physical bounds.
        action_box = Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.action_spaces = {
            agent: action_box for agent in self.possible_agents
        }

        self.state_space = Box(
            low=-self.observation_clip,
            high=self.observation_clip,
            shape=(len(self.possible_agents) * self.observation_dim,),
            dtype=np.float32,
        )
        self._last_observations = {
            agent: np.zeros(self.observation_dim, dtype=np.float32)
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _pad_observation(self, agent, observation):
        value = np.nan_to_num(
            np.asarray(observation, dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=self.observation_clip,
            neginf=-self.observation_clip,
        )
        padded = np.zeros(self.observation_dim, dtype=np.float32)
        padded[: self._observation_dims[agent]] = value
        return np.clip(
            padded, -self.observation_clip, self.observation_clip
        ).astype(np.float32)

    @staticmethod
    def _stable_info(info, raw_reward=0.0):
        return {
            "operating_cost": float(info.get("operating_cost", 0.0)),
            "safety": float(info.get("safety", 0.0)),
            "converged": float(info.get("converged", False)),
            "raw_reward": float(raw_reward),
        }

    def state(self):
        return np.concatenate(
            [self._last_observations[a] for a in self.possible_agents]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        padded = {
            agent: self._pad_observation(agent, observations[agent])
            for agent in self.possible_agents
        }
        self._last_observations = padded
        stable_infos = {
            agent: self._stable_info(infos.get(agent, {}))
            for agent in self.possible_agents
        }
        return padded, stable_infos

    def step(self, actions):
        physical_actions = {}
        for agent in self.possible_agents:
            value = np.asarray(actions[agent], dtype=np.float32).reshape(-1)
            if not np.isfinite(value).all():
                raise FloatingPointError(
                    f"BenchMARL produced a non-finite action for {agent}: {value}"
                )
            size = self._action_dims[agent]
            original_space = self.env.action_space(agent)
            normalized = np.clip(value[:size], -1.0, 1.0)
            low = original_space.low.reshape(-1)
            high = original_space.high.reshape(-1)
            physical_actions[agent] = (
                low + 0.5 * (normalized + 1.0) * (high - low)
            ).astype(np.float32)

        observations, rewards, terminations, truncations, infos = self.env.step(
            physical_actions
        )
        padded = {
            agent: self._pad_observation(agent, observations[agent])
            for agent in self.possible_agents
        }
        self._last_observations = padded
        stable_infos = {
            agent: self._stable_info(infos.get(agent, {}), rewards[agent])
            for agent in self.possible_agents
        }
        scaled_rewards = {
            agent: float(
                np.nan_to_num(
                    rewards[agent], nan=-1e6, posinf=1e6, neginf=-1e6
                )
            )
            * self.reward_scale
            for agent in self.possible_agents
        }
        self.agents = list(self.env.agents)
        return (
            padded,
            scaled_rewards,
            terminations,
            truncations,
            stable_infos,
        )

    def close(self):
        self.env.close()
