"""Configurable version of the networked microgrid PettingZoo environment."""
from __future__ import annotations

import numpy as np
import pandapower as pp

from gridages.devices import DG, ESS
from gridages.envs.multi_agent.base_env import GridEnv, NetworkedGridEnv
from gridages.envs.multi_agent.ieee34_ieee13 import read_data

from core import import_object


DEFAULT_AGENTS = {
    "MG1": {
        "connection_bus": "DSO Bus 822",
        "dg_max_p_mw": 0.66,
        "dg_sn_mva": 1.0,
        "dg_cost": [100, 72.4, 0.5011],
    },
    "MG2": {
        "connection_bus": "DSO Bus 848",
        "dg_max_p_mw": 0.60,
        "dg_sn_mva": None,
        "dg_cost": [100, 51.6, 0.4615],
    },
    "MG3": {
        "connection_bus": "DSO Bus 856",
        "dg_max_p_mw": 0.50,
        "dg_sn_mva": None,
        "dg_cost": [100, 51.6, 0.4615],
    },
}


class ConfigurableMultiAgentMicrogrids(NetworkedGridEnv):
    """Supports different network factories, device buses, and agent parameters.

    Each ``agent_configs`` entry may override:
    ``network_factory``, ``connection_bus``, ``load_scale``, ``base_power``,
    data areas, device bus names, ESS limits, DG limits/cost/capability, and
    renewable capacities.
    """

    def __init__(self, env_config):
        super().__init__(env_config)
        self.max_episode_steps = int(env_config.get("episode_length", 24))
        self.data_size = self.dso.dataset["price"].size
        self.total_days = self.data_size // self.max_episode_steps

    def _dataset(self, config):
        split = "train" if self.env_config.get("train", True) else "test"
        return read_data(
            split,
            config.get("load_area", "AVA"),
            config.get("renew_area", "NP15"),
            config.get("price_area", "0096WD_7_N001"),
        )

    def _build_net(self):
        default_load_scale = float(self.env_config.get("load_scale", 0.2))
        dso_factory = import_object(
            self.env_config.get(
                "dso_network_factory", "gridages.networks.ieee34:IEEE34Bus"
            )
        )
        net = dso_factory("DSO")
        dso = GridEnv(
            net,
            load_scale=default_load_scale,
            base_power=float(self.env_config.get("base_power", 3.0)),
        )
        dso.add_dataset(self._dataset(self.env_config.get("dso_config", {})))
        self.dso = dso

        configured = self.env_config.get("agent_configs", {})
        microgrids = []
        for name, defaults in DEFAULT_AGENTS.items():
            config = {**defaults, **configured.get(name, {})}
            network_factory = import_object(
                config.get(
                    "network_factory", "gridages.networks.ieee13:IEEE13Bus"
                )
            )
            microgrid = GridEnv(
                network_factory(name),
                load_scale=float(config.get("load_scale", default_load_scale)),
                base_power=float(config.get("base_power", 3.0)),
            )
            ess = ESS(
                "ESS1",
                bus=config.get("ess_bus", "Bus 645"),
                min_p_mw=float(config.get("ess_min_p_mw", -0.5)),
                max_p_mw=float(config.get("ess_max_p_mw", 0.5)),
                capacity=float(config.get("ess_capacity", 2.0)),
                max_e_mwh=float(config.get("ess_max_e_mwh", 2.0)),
                min_e_mwh=float(config.get("ess_min_e_mwh", 0.2)),
            )
            sn_mva = config.get("dg_sn_mva")
            dg = DG(
                "DG1",
                bus=config.get("dg_bus", "Bus 675"),
                min_p_mw=float(config.get("dg_min_p_mw", 0.0)),
                max_p_mw=float(config["dg_max_p_mw"]),
                sn_mva=np.nan if sn_mva is None else float(sn_mva),
                cost_curve_coefs=config["dg_cost"],
            )
            pv = DG(
                "PV1",
                bus=config.get("pv_bus", "Bus 652"),
                min_p_mw=0.0,
                max_p_mw=float(config.get("pv_max_p_mw", 0.1)),
                type="solar",
            )
            wind = DG(
                "WT1",
                bus=config.get("wind_bus", "Bus 645"),
                min_p_mw=0.0,
                max_p_mw=float(config.get("wind_max_p_mw", 0.1)),
                type="wind",
            )
            microgrid.add_storage(ess)
            microgrid.add_sgen([dg, pv, wind])
            microgrid.add_dataset(self._dataset(config))
            net = microgrid.add_to(net, config["connection_bus"])
            microgrids.append(microgrid)

        pp.runpp(net, numba=False)
        self.net = net
        self.possible_agents = {agent.name: agent for agent in microgrids}
        self.agents = self.possible_agents

    def _reward_and_safety(self):
        if self.net["converged"]:
            rewards = {
                name: -agent.cost for name, agent in self.agent_envs.items()
            }
            safety = {
                name: agent.safety for name, agent in self.agent_envs.items()
            }
        else:
            rewards = {name: -200.0 for name in self.agents}
            safety = {name: 20.0 for name in self.agents}
        penalty = float(self.env_config.get("penalty", 0.0))
        return (
            {
                name: rewards[name] - penalty * safety[name]
                for name in self.agents
            },
            safety,
        )
