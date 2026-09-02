"""BenchMARL custom task for GridAges networked microgrids."""
from __future__ import annotations

import copy

from benchmarl.environments import Task
from torchrl.data import Composite
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

from env_adapter import BenchMARLMicrogrids


class GridAgesTask(Task):
    NETWORKED_MICROGRIDS = None

    def reset_factory_order(self):
        # BenchMARL 1.4 creates the evaluation factory before the training one.
        self._factory_calls = 0

    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        config = copy.deepcopy(self.config)
        factory_index = getattr(self, "_factory_calls", 0)
        self._factory_calls = factory_index + 1
        train = factory_index != 0
        agent_ids = ["MG1", "MG2", "MG3"]
        group_map = {"agents": agent_ids}

        def make_env():
            pettingzoo_env = BenchMARLMicrogrids(
                config.get("env_config", {}),
                train=train,
                seed=seed or 0,
                reward_scale=config.get("reward_scale", 0.01),
                observation_clip=config.get("observation_clip", 10.0),
            )
            return PettingZooWrapper(
                env=pettingzoo_env,
                return_state=True,
                group_map=group_map,
                use_mask=False,
                categorical_actions=False,
                seed=seed,
                done_on_any=False,
                device=device,
            )

        return make_env

    def supports_continuous_actions(self):
        return True

    def supports_discrete_actions(self):
        return False

    def max_steps(self, env):
        return int(self.config.get("episode_length", 24))

    def has_render(self, env):
        return False

    def group_map(self, env):
        return env.group_map

    def state_spec(self, env):
        if "state" in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def observation_spec(self, env):
        result = env.observation_spec.clone()
        for group in self.group_map(env):
            for key in list(result[group].keys()):
                if key != "observation":
                    del result[group][key]
        if "state" in result:
            del result["state"]
        return result

    def info_spec(self, env):
        result = env.observation_spec.clone()
        for group in self.group_map(env):
            for key in list(result[group].keys()):
                if key != "info":
                    del result[group][key]
        if "state" in result:
            del result["state"]
        return result

    def action_spec(self, env):
        return env.full_action_spec

    def action_mask_spec(self, env):
        return None

    @staticmethod
    def env_name():
        return "gridages"


def make_task(config):
    task = GridAgesTask.NETWORKED_MICROGRIDS
    task.config = copy.deepcopy(config)
    task.reset_factory_order()
    return task
