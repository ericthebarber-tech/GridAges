"""Shared environment adapter, networks, and replay storage for CTDE MARL."""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from gymnasium.spaces import Box
from torch import nn
from torch.distributions import Normal


def parse_hidden(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def load_json(path: str | Path | None) -> dict:
    if path is None:
        return {}
    with Path(path).open() as stream:
        return json.load(stream)


def import_object(path: str):
    module_name, object_name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, values):
        values = np.nan_to_num(
            np.asarray(values, dtype=np.float64),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        if values.ndim == self.mean.ndim:
            values = values[None, ...]
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean, self.var, self.count = new_mean, m2 / total, total

    def normalize(self, value, clip=10.0):
        value = np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip((value - self.mean) / np.sqrt(self.var + 1e-8),
                       -clip, clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"])
        self.var = np.asarray(state["var"])
        self.count = float(state["count"])


@dataclass(frozen=True)
class AgentSpec:
    observation_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray


class MultiAgentAdapter:
    """Normalizes local observations and maps actor actions from [-1, 1]."""

    def __init__(
        self,
        env_class: str,
        env_config: Mapping,
        *,
        train: bool,
        seed: int,
        reward_scale: float = 0.01,
        update_normalization: bool = True,
    ):
        env_type = import_object(env_class)
        config = dict(env_config)
        config["train"] = train
        config["seed"] = seed
        self.env = env_type(config)
        self.agent_ids = list(self.env.possible_agents)
        self.reward_scale = float(reward_scale)
        self.update_normalization = update_normalization
        self.specs: Dict[str, AgentSpec] = {}
        self.normalizers = {}
        for agent in self.agent_ids:
            obs_space = self.env.observation_space(agent)
            action_space = self.env.action_space(agent)
            if not isinstance(obs_space, Box) or not isinstance(action_space, Box):
                raise TypeError("CTDE implementations require continuous Box spaces")
            self.specs[agent] = AgentSpec(
                observation_dim=int(np.prod(obs_space.shape)),
                action_dim=int(np.prod(action_space.shape)),
                action_low=action_space.low.reshape(-1).astype(np.float32),
                action_high=action_space.high.reshape(-1).astype(np.float32),
            )
            self.normalizers[agent] = RunningMeanStd(
                (self.specs[agent].observation_dim,)
            )
        self.last_raw_observations = None

    @property
    def global_observation_dim(self):
        return sum(spec.observation_dim for spec in self.specs.values())

    @property
    def joint_action_dim(self):
        return sum(spec.action_dim for spec in self.specs.values())

    def _normalize(self, observations):
        result = {}
        for agent in self.agent_ids:
            value = np.asarray(observations[agent], dtype=np.float32).reshape(-1)
            if self.update_normalization:
                self.normalizers[agent].update(value)
            result[agent] = self.normalizers[agent].normalize(value)
        return result

    def reset(self, seed=None):
        observations, infos = self.env.reset(seed=seed)
        self.last_raw_observations = observations
        return self._normalize(observations), infos

    def scale_action(self, agent, normalized_action):
        spec = self.specs[agent]
        action = np.asarray(normalized_action, dtype=np.float32)
        if not np.isfinite(action).all():
            raise FloatingPointError(
                f"Actor for {agent} produced a non-finite action: {action}"
            )
        action = np.clip(action, -1.0, 1.0)
        return spec.action_low + 0.5 * (action + 1.0) * (
            spec.action_high - spec.action_low
        )

    def step(self, normalized_actions):
        actions = {
            agent: self.scale_action(agent, normalized_actions[agent])
            for agent in self.agent_ids
        }
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self.last_raw_observations = observations
        done = bool(terminated and all(terminated.values())) or bool(
            truncated and all(truncated.values())
        )
        scaled_rewards = {
            agent: float(
                np.nan_to_num(
                    rewards[agent], nan=-1e6, posinf=1e6, neginf=-1e6
                )
            )
            * self.reward_scale
            for agent in self.agent_ids
        }
        return self._normalize(observations), scaled_rewards, done, infos, rewards

    def global_observation(self, observations):
        return np.concatenate([observations[a] for a in self.agent_ids]).astype(
            np.float32
        )

    def copy_normalization_from(self, other):
        for agent in self.agent_ids:
            self.normalizers[agent].load_state_dict(
                other.normalizers[agent].state_dict()
            )
        self.update_normalization = False

    def normalization_state(self):
        return {
            agent: normalizer.state_dict()
            for agent, normalizer in self.normalizers.items()
        }

    def close(self):
        self.env.close()


def build_mlp(input_dim, output_dim, hidden, output_activation=None):
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    module = nn.Sequential(*layers)
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
    return module


class DeterministicActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden):
        super().__init__()
        self.net = build_mlp(observation_dim, action_dim, hidden, nn.Tanh)

    def forward(self, observation):
        return self.net(observation)


class GaussianActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden):
        super().__init__()
        if not hidden:
            raise ValueError("Gaussian actor requires at least one hidden layer")
        self.body = build_mlp(observation_dim, hidden[-1], hidden[:-1])
        self.mean = nn.Linear(hidden[-1], action_dim)
        self.log_std = nn.Linear(hidden[-1], action_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.bias)

    def distribution(self, observation):
        features = self.body(observation)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), -5.0, 2.0)
        return Normal(mean, log_std.exp())

    def sample(self, observation, deterministic=False):
        distribution = self.distribution(observation)
        raw = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        log_prob = distribution.log_prob(raw) - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        return action, log_prob.sum(-1, keepdim=True)

    def evaluate_action(self, observation, action):
        clipped = torch.clamp(action, -0.999999, 0.999999)
        raw = torch.atanh(clipped)
        distribution = self.distribution(observation)
        log_prob = distribution.log_prob(raw) - torch.log(
            1.0 - clipped.pow(2) + 1e-6
        )
        entropy = distribution.entropy().sum(-1)
        return log_prob.sum(-1), entropy


class CentralCritic(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.net = build_mlp(input_dim, 1, hidden)

    def forward(self, value):
        return self.net(value)


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_parameter, parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)


class MultiAgentReplayBuffer:
    def __init__(self, capacity, specs, agent_ids):
        self.capacity = int(capacity)
        self.agent_ids = list(agent_ids)
        self.position = 0
        self.size = 0
        self.observations = {
            a: np.zeros((capacity, specs[a].observation_dim), dtype=np.float32)
            for a in agent_ids
        }
        self.next_observations = {
            a: np.zeros((capacity, specs[a].observation_dim), dtype=np.float32)
            for a in agent_ids
        }
        self.actions = {
            a: np.zeros((capacity, specs[a].action_dim), dtype=np.float32)
            for a in agent_ids
        }
        self.rewards = {
            a: np.zeros((capacity, 1), dtype=np.float32) for a in agent_ids
        }
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, observations, actions, rewards, next_observations, done):
        index = self.position
        for agent in self.agent_ids:
            self.observations[agent][index] = observations[agent]
            self.actions[agent][index] = actions[agent]
            self.rewards[agent][index] = rewards[agent]
            self.next_observations[agent][index] = next_observations[agent]
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = np.random.randint(0, self.size, size=batch_size)
        tensor = lambda value: torch.as_tensor(value[indices], device=device)
        return {
            "observations": {a: tensor(self.observations[a]) for a in self.agent_ids},
            "actions": {a: tensor(self.actions[a]) for a in self.agent_ids},
            "rewards": {a: tensor(self.rewards[a]) for a in self.agent_ids},
            "next_observations": {
                a: tensor(self.next_observations[a]) for a in self.agent_ids
            },
            "dones": tensor(self.dones),
        }

    def __len__(self):
        return self.size


def concatenate(mapping: Mapping[str, torch.Tensor], agent_ids: Iterable[str]):
    return torch.cat([mapping[agent] for agent in agent_ids], dim=-1)
