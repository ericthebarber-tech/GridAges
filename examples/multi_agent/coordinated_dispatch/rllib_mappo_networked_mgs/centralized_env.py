"""Gymnasium view of the coupled PettingZoo microgrids for SB3 baselines."""
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from gridages.envs.multi_agent.ieee34_ieee13 import MultiAgentMicrogrids


class CentralizedMicrogridsEnv(gym.Env):
    """Concatenate all agent observations/actions and sum their rewards.

    This is a centralized-controller baseline. The underlying grid remains a
    three-agent PettingZoo environment and all actions are applied together.
    """
    metadata = {"render_modes": []}

    def __init__(self, env_config=None):
        super().__init__()
        self.env_config = dict(env_config or {})
        self.parallel_env = MultiAgentMicrogrids(self.env_config)
        self.agent_ids = list(self.parallel_env.possible_agents)
        self._action_slices = {}

        obs_lows, obs_highs, action_lows, action_highs = [], [], [], []
        action_start = 0
        for agent in self.agent_ids:
            obs_space = self.parallel_env.observation_space(agent)
            action_space = self.parallel_env.action_space(agent)
            if not isinstance(obs_space, Box) or not isinstance(action_space, Box):
                raise TypeError("The SB3 baselines require Box observations and actions")
            obs_lows.append(obs_space.low.reshape(-1).astype(np.float32))
            obs_highs.append(obs_space.high.reshape(-1).astype(np.float32))
            action_size = int(np.prod(action_space.shape))
            self._action_slices[agent] = slice(action_start, action_start + action_size)
            action_start += action_size
            action_lows.append(action_space.low.reshape(-1).astype(np.float32))
            action_highs.append(action_space.high.reshape(-1).astype(np.float32))

        self.observation_space = Box(
            low=np.concatenate(obs_lows), high=np.concatenate(obs_highs), dtype=np.float32
        )
        self.action_space = Box(
            low=np.concatenate(action_lows), high=np.concatenate(action_highs),
            dtype=np.float32,
        )

    def _observation(self, observations):
        observation = np.concatenate([
            np.asarray(observations[agent], dtype=np.float32).reshape(-1)
            for agent in self.agent_ids
        ])
        # Power-flow failures or invalid device states must never poison a
        # neural-network update. VecNormalize handles scaling after this guard.
        return np.clip(
            np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6),
            -1e6, 1e6,
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observations, infos = self.parallel_env.reset(seed=seed, options=options)
        return self._observation(observations), {"agents": infos}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        actions = {
            agent: action[self._action_slices[agent]].reshape(
                self.parallel_env.action_space(agent).shape
            )
            for agent in self.agent_ids
        }
        observations, rewards, terminated, truncated, infos = self.parallel_env.step(actions)
        safety = {agent: float(infos[agent].get("safety", 0.0)) for agent in self.agent_ids}
        convergence = {
            agent: bool(infos[agent].get("converged", False)) for agent in self.agent_ids
        }
        reward = float(sum(rewards.values()))
        if not np.isfinite(reward):
            reward = -1e6
        penalty = float(self.env_config.get("penalty", 0.0))
        info = {
            "agents": infos,
            "agent_rewards": rewards,
            "agent_safety": safety,
            "operating_cost": -reward - penalty * sum(safety.values()),
            "safety": sum(safety.values()),
            "convergence_rate": float(np.mean(list(convergence.values()))),
        }
        return (
            self._observation(observations), reward,
            bool(terminated and all(terminated.values())),
            bool(truncated and all(truncated.values())), info,
        )

    def close(self):
        self.parallel_env.close()
