"""Heterogeneous-agent CTDE implementations of MAPPO, MADDPG, MASAC, MATD3."""
from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core import (
    CentralCritic,
    DeterministicActor,
    GaussianActor,
    concatenate,
    parse_hidden,
    soft_update,
)


def _hidden(network_config, agent, key, default):
    config = network_config.get(agent, network_config.get("default", {}))
    return parse_hidden(config.get(key, default))


def _tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _check_loss(loss, label):
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite {label}: {float(loss.detach().cpu())}"
        )


class OffPolicyCTDE:
    def __init__(
        self,
        adapter,
        network_config,
        *,
        device,
        actor_lr,
        critic_lr,
        gamma,
        tau,
    ):
        self.agent_ids = adapter.agent_ids
        self.specs = adapter.specs
        self.global_dim = adapter.global_observation_dim
        self.joint_action_dim = adapter.joint_action_dim
        self.device = torch.device(device)
        self.gamma, self.tau = gamma, tau
        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        self.network_config = network_config
        self.update_count = 0

    def _global(self, observations):
        return concatenate(observations, self.agent_ids)

    def _joint(self, actions):
        return concatenate(actions, self.agent_ids)

    def checkpoint(self):
        raise NotImplementedError


class MADDPG(OffPolicyCTDE):
    def __init__(self, adapter, network_config, **kwargs):
        super().__init__(adapter, network_config, **kwargs)
        self.actors, self.target_actors = nn.ModuleDict(), nn.ModuleDict()
        self.critics, self.target_critics = nn.ModuleDict(), nn.ModuleDict()
        self.actor_optimizers, self.critic_optimizers = {}, {}
        critic_input = self.global_dim + self.joint_action_dim
        for agent in self.agent_ids:
            actor_hidden = _hidden(network_config, agent, "actor", (128, 128))
            critic_hidden = _hidden(network_config, agent, "critic", (256, 256))
            actor = DeterministicActor(
                self.specs[agent].observation_dim,
                self.specs[agent].action_dim,
                actor_hidden,
            ).to(self.device)
            critic = CentralCritic(critic_input, critic_hidden).to(self.device)
            self.actors[agent], self.target_actors[agent] = actor, copy.deepcopy(actor)
            self.critics[agent], self.target_critics[agent] = (
                critic,
                copy.deepcopy(critic),
            )
            self.actor_optimizers[agent] = torch.optim.Adam(
                actor.parameters(), lr=self.actor_lr
            )
            self.critic_optimizers[agent] = torch.optim.Adam(
                critic.parameters(), lr=self.critic_lr
            )

    @torch.no_grad()
    def select_actions(self, observations, deterministic=False, noise_std=0.1):
        actions = {}
        for agent in self.agent_ids:
            action = self.actors[agent](
                _tensor(observations[agent], self.device).unsqueeze(0)
            )[0]
            if not deterministic:
                action = action + torch.randn_like(action) * noise_std
            actions[agent] = action.clamp(-1, 1).cpu().numpy()
        return actions

    def update(self, batch):
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        global_obs = self._global(observations)
        global_next_obs = self._global(next_observations)
        joint_actions = self._joint(batch["actions"])
        with torch.no_grad():
            target_actions = {
                a: self.target_actors[a](next_observations[a])
                for a in self.agent_ids
            }
            target_input = torch.cat(
                [global_next_obs, self._joint(target_actions)], dim=-1
            )

        metrics = {}
        for agent in self.agent_ids:
            with torch.no_grad():
                target = batch["rewards"][agent] + self.gamma * (
                    1.0 - batch["dones"]
                ) * self.target_critics[agent](target_input)
            critic_input = torch.cat([global_obs, joint_actions], dim=-1)
            critic_loss = F.mse_loss(self.critics[agent](critic_input), target)
            _check_loss(critic_loss, f"{agent} critic loss")
            self.critic_optimizers[agent].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critics[agent].parameters(), 10.0, error_if_nonfinite=True
            )
            self.critic_optimizers[agent].step()

            policy_actions = {}
            for other in self.agent_ids:
                value = self.actors[other](observations[other])
                policy_actions[other] = value if other == agent else value.detach()
            for parameter in self.critics[agent].parameters():
                parameter.requires_grad_(False)
            actor_input = torch.cat(
                [global_obs, self._joint(policy_actions)], dim=-1
            )
            actor_loss = -self.critics[agent](actor_input).mean()
            _check_loss(actor_loss, f"{agent} actor loss")
            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actors[agent].parameters(), 10.0, error_if_nonfinite=True
            )
            self.actor_optimizers[agent].step()
            for parameter in self.critics[agent].parameters():
                parameter.requires_grad_(True)

            soft_update(self.target_actors[agent], self.actors[agent], self.tau)
            soft_update(self.target_critics[agent], self.critics[agent], self.tau)
            metrics[f"{agent}/actor_loss"] = float(actor_loss.item())
            metrics[f"{agent}/critic_loss"] = float(critic_loss.item())
        self.update_count += 1
        return metrics

    def checkpoint(self):
        return {
            "actors": self.actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "target_critics": self.target_critics.state_dict(),
        }


class MATD3(MADDPG):
    def __init__(
        self,
        adapter,
        network_config,
        *,
        policy_delay=2,
        target_noise=0.2,
        noise_clip=0.5,
        **kwargs,
    ):
        OffPolicyCTDE.__init__(self, adapter, network_config, **kwargs)
        self.policy_delay = policy_delay
        self.target_noise, self.noise_clip = target_noise, noise_clip
        self.actors, self.target_actors = nn.ModuleDict(), nn.ModuleDict()
        self.critics1, self.critics2 = nn.ModuleDict(), nn.ModuleDict()
        self.target_critics1, self.target_critics2 = nn.ModuleDict(), nn.ModuleDict()
        self.actor_optimizers, self.critic_optimizers = {}, {}
        critic_input = self.global_dim + self.joint_action_dim
        for agent in self.agent_ids:
            actor_hidden = _hidden(network_config, agent, "actor", (128, 128))
            critic_hidden = _hidden(network_config, agent, "critic", (256, 256))
            actor = DeterministicActor(
                self.specs[agent].observation_dim,
                self.specs[agent].action_dim,
                actor_hidden,
            ).to(self.device)
            critic1 = CentralCritic(critic_input, critic_hidden).to(self.device)
            critic2 = CentralCritic(critic_input, critic_hidden).to(self.device)
            self.actors[agent], self.target_actors[agent] = actor, copy.deepcopy(actor)
            self.critics1[agent], self.critics2[agent] = critic1, critic2
            self.target_critics1[agent] = copy.deepcopy(critic1)
            self.target_critics2[agent] = copy.deepcopy(critic2)
            self.actor_optimizers[agent] = torch.optim.Adam(
                actor.parameters(), lr=self.actor_lr
            )
            self.critic_optimizers[agent] = torch.optim.Adam(
                list(critic1.parameters()) + list(critic2.parameters()),
                lr=self.critic_lr,
            )

    def update(self, batch):
        observations, next_observations = (
            batch["observations"],
            batch["next_observations"],
        )
        global_obs = self._global(observations)
        global_next_obs = self._global(next_observations)
        joint_actions = self._joint(batch["actions"])
        with torch.no_grad():
            target_actions = {}
            for agent in self.agent_ids:
                noise = torch.randn_like(batch["actions"][agent]) * self.target_noise
                target_actions[agent] = (
                    self.target_actors[agent](next_observations[agent])
                    + noise.clamp(-self.noise_clip, self.noise_clip)
                ).clamp(-1, 1)
            target_input = torch.cat(
                [global_next_obs, self._joint(target_actions)], dim=-1
            )

        metrics = {}
        critic_input = torch.cat([global_obs, joint_actions], dim=-1)
        for agent in self.agent_ids:
            with torch.no_grad():
                target_q = torch.minimum(
                    self.target_critics1[agent](target_input),
                    self.target_critics2[agent](target_input),
                )
                target = batch["rewards"][agent] + self.gamma * (
                    1.0 - batch["dones"]
                ) * target_q
            q1, q2 = (
                self.critics1[agent](critic_input),
                self.critics2[agent](critic_input),
            )
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            _check_loss(critic_loss, f"{agent} twin critic loss")
            self.critic_optimizers[agent].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.critics1[agent].parameters())
                + list(self.critics2[agent].parameters()),
                10.0,
                error_if_nonfinite=True,
            )
            self.critic_optimizers[agent].step()
            metrics[f"{agent}/critic_loss"] = float(critic_loss.item())

        if self.update_count % self.policy_delay == 0:
            for agent in self.agent_ids:
                policy_actions = {}
                for other in self.agent_ids:
                    value = self.actors[other](observations[other])
                    policy_actions[other] = value if other == agent else value.detach()
                for parameter in self.critics1[agent].parameters():
                    parameter.requires_grad_(False)
                actor_input = torch.cat(
                    [global_obs, self._joint(policy_actions)], dim=-1
                )
                actor_loss = -self.critics1[agent](actor_input).mean()
                _check_loss(actor_loss, f"{agent} actor loss")
                self.actor_optimizers[agent].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actors[agent].parameters(),
                    10.0,
                    error_if_nonfinite=True,
                )
                self.actor_optimizers[agent].step()
                for parameter in self.critics1[agent].parameters():
                    parameter.requires_grad_(True)
                soft_update(self.target_actors[agent], self.actors[agent], self.tau)
                soft_update(
                    self.target_critics1[agent], self.critics1[agent], self.tau
                )
                soft_update(
                    self.target_critics2[agent], self.critics2[agent], self.tau
                )
                metrics[f"{agent}/actor_loss"] = float(actor_loss.item())
        self.update_count += 1
        return metrics

    def checkpoint(self):
        return {
            "actors": self.actors.state_dict(),
            "critics1": self.critics1.state_dict(),
            "critics2": self.critics2.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "target_critics1": self.target_critics1.state_dict(),
            "target_critics2": self.target_critics2.state_dict(),
        }


class MASAC(OffPolicyCTDE):
    def __init__(self, adapter, network_config, *, alpha_lr=3e-4, **kwargs):
        super().__init__(adapter, network_config, **kwargs)
        self.actors = nn.ModuleDict()
        self.critics1, self.critics2 = nn.ModuleDict(), nn.ModuleDict()
        self.target_critics1, self.target_critics2 = nn.ModuleDict(), nn.ModuleDict()
        self.actor_optimizers, self.critic_optimizers = {}, {}
        self.log_alpha, self.alpha_optimizers, self.target_entropy = {}, {}, {}
        critic_input = self.global_dim + self.joint_action_dim
        for agent in self.agent_ids:
            actor = GaussianActor(
                self.specs[agent].observation_dim,
                self.specs[agent].action_dim,
                _hidden(network_config, agent, "actor", (128, 128)),
            ).to(self.device)
            critic_hidden = _hidden(network_config, agent, "critic", (256, 256))
            critic1 = CentralCritic(critic_input, critic_hidden).to(self.device)
            critic2 = CentralCritic(critic_input, critic_hidden).to(self.device)
            self.actors[agent] = actor
            self.critics1[agent], self.critics2[agent] = critic1, critic2
            self.target_critics1[agent] = copy.deepcopy(critic1)
            self.target_critics2[agent] = copy.deepcopy(critic2)
            self.actor_optimizers[agent] = torch.optim.Adam(
                actor.parameters(), lr=self.actor_lr
            )
            self.critic_optimizers[agent] = torch.optim.Adam(
                list(critic1.parameters()) + list(critic2.parameters()),
                lr=self.critic_lr,
            )
            self.log_alpha[agent] = torch.tensor(
                0.0, device=self.device, requires_grad=True
            )
            self.alpha_optimizers[agent] = torch.optim.Adam(
                [self.log_alpha[agent]], lr=alpha_lr
            )
            self.target_entropy[agent] = -float(self.specs[agent].action_dim)

    @torch.no_grad()
    def select_actions(self, observations, deterministic=False, noise_std=0.0):
        return {
            agent: self.actors[agent]
            .sample(
                _tensor(observations[agent], self.device).unsqueeze(0),
                deterministic=deterministic,
            )[0][0]
            .cpu()
            .numpy()
            for agent in self.agent_ids
        }

    def update(self, batch):
        observations, next_observations = (
            batch["observations"],
            batch["next_observations"],
        )
        global_obs = self._global(observations)
        global_next_obs = self._global(next_observations)
        joint_replay_actions = self._joint(batch["actions"])
        with torch.no_grad():
            next_actions, next_log_probs = {}, {}
            for agent in self.agent_ids:
                next_actions[agent], next_log_probs[agent] = self.actors[agent].sample(
                    next_observations[agent]
                )
            target_input = torch.cat(
                [global_next_obs, self._joint(next_actions)], dim=-1
            )

        metrics = {}
        replay_input = torch.cat([global_obs, joint_replay_actions], dim=-1)
        for agent in self.agent_ids:
            alpha = self.log_alpha[agent].exp().detach()
            with torch.no_grad():
                target_q = torch.minimum(
                    self.target_critics1[agent](target_input),
                    self.target_critics2[agent](target_input),
                ) - alpha * next_log_probs[agent]
                target = batch["rewards"][agent] + self.gamma * (
                    1.0 - batch["dones"]
                ) * target_q
            q1, q2 = (
                self.critics1[agent](replay_input),
                self.critics2[agent](replay_input),
            )
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            _check_loss(critic_loss, f"{agent} twin critic loss")
            self.critic_optimizers[agent].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.critics1[agent].parameters())
                + list(self.critics2[agent].parameters()),
                10.0,
                error_if_nonfinite=True,
            )
            self.critic_optimizers[agent].step()

            sampled_actions, sampled_log_probs = {}, {}
            for other in self.agent_ids:
                action, log_prob = self.actors[other].sample(observations[other])
                sampled_actions[other] = action if other == agent else action.detach()
                sampled_log_probs[other] = log_prob
            for parameter in self.critics1[agent].parameters():
                parameter.requires_grad_(False)
            actor_input = torch.cat(
                [global_obs, self._joint(sampled_actions)], dim=-1
            )
            actor_loss = (
                alpha * sampled_log_probs[agent]
                - self.critics1[agent](actor_input)
            ).mean()
            _check_loss(actor_loss, f"{agent} actor loss")
            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actors[agent].parameters(),
                10.0,
                error_if_nonfinite=True,
            )
            self.actor_optimizers[agent].step()
            for parameter in self.critics1[agent].parameters():
                parameter.requires_grad_(True)

            alpha_loss = -(
                self.log_alpha[agent]
                * (
                    sampled_log_probs[agent].detach()
                    + self.target_entropy[agent]
                )
            ).mean()
            _check_loss(alpha_loss, f"{agent} temperature loss")
            self.alpha_optimizers[agent].zero_grad()
            alpha_loss.backward()
            self.alpha_optimizers[agent].step()
            soft_update(
                self.target_critics1[agent], self.critics1[agent], self.tau
            )
            soft_update(
                self.target_critics2[agent], self.critics2[agent], self.tau
            )
            metrics[f"{agent}/actor_loss"] = float(actor_loss.item())
            metrics[f"{agent}/critic_loss"] = float(critic_loss.item())
            metrics[f"{agent}/alpha"] = float(
                self.log_alpha[agent].exp().item()
            )
        self.update_count += 1
        return metrics

    def checkpoint(self):
        return {
            "actors": self.actors.state_dict(),
            "critics1": self.critics1.state_dict(),
            "critics2": self.critics2.state_dict(),
            "target_critics1": self.target_critics1.state_dict(),
            "target_critics2": self.target_critics2.state_dict(),
            "log_alpha": {
                agent: value.detach().cpu() for agent, value in self.log_alpha.items()
            },
        }


class MAPPO:
    def __init__(
        self,
        adapter,
        network_config,
        *,
        device,
        actor_lr,
        critic_lr,
        gamma,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        update_epochs=10,
        minibatch_size=256,
        **_,
    ):
        self.agent_ids = adapter.agent_ids
        self.specs = adapter.specs
        self.device = torch.device(device)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.actors, self.critics = nn.ModuleDict(), nn.ModuleDict()
        self.actor_optimizers, self.critic_optimizers = {}, {}
        for agent in self.agent_ids:
            actor = GaussianActor(
                self.specs[agent].observation_dim,
                self.specs[agent].action_dim,
                _hidden(network_config, agent, "actor", (128, 128)),
            ).to(self.device)
            critic = CentralCritic(
                adapter.global_observation_dim,
                _hidden(network_config, agent, "critic", (256, 256)),
            ).to(self.device)
            self.actors[agent], self.critics[agent] = actor, critic
            self.actor_optimizers[agent] = torch.optim.Adam(
                actor.parameters(), lr=actor_lr
            )
            self.critic_optimizers[agent] = torch.optim.Adam(
                critic.parameters(), lr=critic_lr
            )
        self.rollout = []

    @torch.no_grad()
    def select_actions(self, observations, deterministic=False, noise_std=0.0):
        global_obs = np.concatenate([observations[a] for a in self.agent_ids])
        global_tensor = _tensor(global_obs, self.device).unsqueeze(0)
        actions, log_probs, values = {}, {}, {}
        for agent in self.agent_ids:
            action, log_prob = self.actors[agent].sample(
                _tensor(observations[agent], self.device).unsqueeze(0),
                deterministic=deterministic,
            )
            actions[agent] = action[0].cpu().numpy()
            log_probs[agent] = float(log_prob.item())
            values[agent] = float(self.critics[agent](global_tensor).item())
        return actions, log_probs, values

    def add_transition(
        self, observations, actions, log_probs, values, rewards, done
    ):
        self.rollout.append(
            {
                "observations": {a: observations[a].copy() for a in self.agent_ids},
                "global_observation": np.concatenate(
                    [observations[a] for a in self.agent_ids]
                ),
                "actions": {a: actions[a].copy() for a in self.agent_ids},
                "log_probs": dict(log_probs),
                "values": dict(values),
                "rewards": dict(rewards),
                "done": float(done),
            }
        )

    def update(self, last_observations, last_done):
        length = len(self.rollout)
        global_last = _tensor(
            np.concatenate([last_observations[a] for a in self.agent_ids]),
            self.device,
        ).unsqueeze(0)
        metrics = defaultdict(list)
        for agent in self.agent_ids:
            with torch.no_grad():
                bootstrap = (
                    0.0
                    if last_done
                    else float(self.critics[agent](global_last).item())
                )
            advantages = np.zeros(length, dtype=np.float32)
            last_gae = 0.0
            for index in reversed(range(length)):
                next_value = (
                    bootstrap
                    if index == length - 1
                    else self.rollout[index + 1]["values"][agent]
                )
                nonterminal = 1.0 - self.rollout[index]["done"]
                delta = (
                    self.rollout[index]["rewards"][agent]
                    + self.gamma * next_value * nonterminal
                    - self.rollout[index]["values"][agent]
                )
                last_gae = (
                    delta
                    + self.gamma * self.gae_lambda * nonterminal * last_gae
                )
                advantages[index] = last_gae
            returns = advantages + np.asarray(
                [item["values"][agent] for item in self.rollout],
                dtype=np.float32,
            )
            local_obs = _tensor(
                np.asarray(
                    [item["observations"][agent] for item in self.rollout]
                ),
                self.device,
            )
            global_obs = _tensor(
                np.asarray([item["global_observation"] for item in self.rollout]),
                self.device,
            )
            actions = _tensor(
                np.asarray([item["actions"][agent] for item in self.rollout]),
                self.device,
            )
            old_log_probs = _tensor(
                np.asarray([item["log_probs"][agent] for item in self.rollout]),
                self.device,
            )
            advantages_t = _tensor(advantages, self.device)
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std(unbiased=False) + 1e-8
            )
            returns_t = _tensor(returns, self.device)
            old_values_t = _tensor(
                np.asarray(
                    [item["values"][agent] for item in self.rollout],
                    dtype=np.float32,
                ),
                self.device,
            )

            for _ in range(self.update_epochs):
                permutation = torch.randperm(length, device=self.device)
                for start in range(0, length, self.minibatch_size):
                    indices = permutation[start : start + self.minibatch_size]
                    log_prob, entropy = self.actors[agent].evaluate_action(
                        local_obs[indices], actions[indices]
                    )
                    ratio = torch.exp(log_prob - old_log_probs[indices])
                    unclipped = ratio * advantages_t[indices]
                    clipped = torch.clamp(
                        ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    ) * advantages_t[indices]
                    actor_loss = -torch.minimum(unclipped, clipped).mean()
                    actor_loss -= self.entropy_coefficient * entropy.mean()
                    _check_loss(actor_loss, f"{agent} PPO actor loss")
                    self.actor_optimizers[agent].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors[agent].parameters(),
                        0.5,
                        error_if_nonfinite=True,
                    )
                    self.actor_optimizers[agent].step()

                    value = self.critics[agent](global_obs[indices]).squeeze(-1)
                    clipped_value = old_values_t[indices] + torch.clamp(
                        value - old_values_t[indices],
                        -self.clip_ratio,
                        self.clip_ratio,
                    )
                    critic_loss = torch.maximum(
                        F.mse_loss(
                            value, returns_t[indices], reduction="none"
                        ),
                        F.mse_loss(
                            clipped_value,
                            returns_t[indices],
                            reduction="none",
                        ),
                    ).mean()
                    _check_loss(critic_loss, f"{agent} PPO critic loss")
                    self.critic_optimizers[agent].zero_grad()
                    (self.value_coefficient * critic_loss).backward()
                    nn.utils.clip_grad_norm_(
                        self.critics[agent].parameters(),
                        0.5,
                        error_if_nonfinite=True,
                    )
                    self.critic_optimizers[agent].step()
                    metrics[f"{agent}/actor_loss"].append(actor_loss.item())
                    metrics[f"{agent}/critic_loss"].append(critic_loss.item())
        self.rollout.clear()
        return {key: float(np.mean(values)) for key, values in metrics.items()}

    def checkpoint(self):
        return {
            "actors": self.actors.state_dict(),
            "critics": self.critics.state_dict(),
        }
