"""One-run comparison: Uncontrolled vs Lowest-Price vs DreamerV3 vs Perfect-Foresight LP.

Outputs:
  - step-level CSVs for all four methods
  - daily summaries for all four methods
  - LP diagnostics and EV-service CSVs
  - paired four-method comparison CSV
  - summary comparison CSV
  - one comprehensive four-method comparison figure

The Perfect-Foresight LP minimizes predicted daily peak using the zero-EV
IEEE-34 replay as its base profile, then replays the optimized charging
schedule through the full IEEE-34 AC power flow. It is therefore a
perfect-foresight scheduling benchmark, not a multi-period AC-OPF optimum.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import tree
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict as SpaceDict
from scipy.optimize import linprog

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns
from ray.rllib.utils.framework import convert_to_tensor
from ray.tune.registry import register_env

from gridages.envs.single_agent.ev_charging.ieee34_evs import IEEE34Env


# ---------------------------------------------------------------------
# Shared environment setup
# ---------------------------------------------------------------------
def neutral_action(space):
    if isinstance(space, Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, Discrete):
        return 0
    if isinstance(space, MultiDiscrete):
        return np.zeros_like(space.nvec, dtype=np.int64)
    if isinstance(space, SpaceDict):
        return {k: neutral_action(v) for k, v in space.spaces.items()}
    return space.sample()


def set_test_day(env, day):
    steps_per_day = round(24.0 / env.dt)
    start_offset = round((env.start_hour % 24.0) / env.dt)
    env.data_idx = start_offset + day * steps_per_day
    env.step_idx = 0
    env._apply_dataset_scalers()


def make_env(env_config):
    """RLlib creator required when restoring the DreamerV3 checkpoint."""
    return IEEE34Env(env_config=env_config)


register_env("gridages-ieee34-ev", make_env)


def make_test_env(ev_level, control_mode):
    return IEEE34Env(env_config={
        "train": False,
        "ev_level": ev_level,
        "control_mode": control_mode,
        "episode_length": 48,
        "dt": 0.5,
        "start_hour": 8.0,
        "store_info": ["bus_voltage", "line_loading", "operating_cost"],
        "store_arrays": False,
        "store_summaries": True,
        # Keep the tested Dreamer/Uncontrolled evaluation configuration.
        "reward_scale": 100.0,
        "safety_scale": 10000.0,
        "peak_weight": 1.0,
        "shape_weight": 100.0,
        "price_weight": 0.0,
        "penalize_safety": True,
    })


def complete_test_days(env):
    T = len(env.dataset["load"])
    steps_per_day = round(24.0 / env.dt)
    start_offset = round((env.start_hour % 24.0) / env.dt)
    return (T - start_offset) // steps_per_day


def evs(env):
    return [d for d in env.devices.values() if d.__class__.__name__ == "EV"]


def snapshot_ev_requirements(env):
    """Snapshot the randomized EV realization immediately after reset."""
    rows = []
    T = env.episode_length
    for d in evs(env):
        target = getattr(d.state, "soc_target", None)
        target = float(d.max_soc if target is None else min(target, d.max_soc))
        soc = float(d.state.soc)
        cap = float(d.capacity)
        eta = float(d.ch_eff)
        pmax = float(d.max_p_mw)
        connected = np.asarray([bool(d.is_connected(t)) for t in range(T)])
        needed = max((target - soc) * cap, 0.0) / eta
        deliverable = float(connected.sum()) * pmax * env.dt
        rows.append({
            "name": d.name,
            "initial_soc": soc,
            "target_soc": target,
            "capacity_mwh": cap,
            "ch_eff": eta,
            "pmax_mw": pmax,
            "arrival_step": int(d.arrive_time),
            "departure_step": int(d.depart_time),
            "connected": connected,
            "grid_energy_needed_mwh": needed,
            "max_grid_deliverable_mwh": deliverable,
            "scheduled_grid_energy_mwh": min(needed, deliverable),
            "target_feasible": bool(needed <= deliverable + 1e-9),
        })
    return rows


def service_after_episode(env, req, day, controller):
    E = evs(env)
    if len(E) != len(req):
        raise RuntimeError("EV count changed within paired evaluation.")
    rows = []
    for d, r in zip(E, req):
        final_soc = float(d.state.soc)
        rows.append({
            "controller": controller,
            "day": day,
            "ev": d.name,
            "initial_soc": r["initial_soc"],
            "final_soc": final_soc,
            "target_soc": r["target_soc"],
            "target_feasible": r["target_feasible"],
            "target_met": bool(final_soc >= r["target_soc"] - 1e-6),
            "grid_energy_needed_mwh": r["grid_energy_needed_mwh"],
        })
    return rows


# ---------------------------------------------------------------------
# DreamerV3 inference: kept consistent with the tested evaluator
# ---------------------------------------------------------------------
class DreamerV3Inference:
    """Stateful inference adapter for current RLlib DreamerV3."""

    def __init__(self, algo, env):
        if algo is None:
            raise ValueError("DreamerV3Inference requires a restored Algorithm.")

        self.algo = algo
        self.env = env
        self.module = algo.env_runner.module

        if not isinstance(env.action_space, Box):
            raise TypeError(
                "Expected a continuous Box action space, got "
                f"{env.action_space!r}."
            )

        self.action_shape = env.action_space.shape
        self.action_low = np.asarray(env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float32)
        self.normalize_actions = bool(
            getattr(algo.config, "normalize_actions", True)
        )
        self.states = None
        self.is_first = 1.0
        self.reset()

    def reset(self):
        states = self.module.get_initial_state()
        self.states = tree.map_structure(lambda s: s.unsqueeze(0), states)
        self.is_first = 1.0

    def _to_env_action(self, action):
        a = np.asarray(action, dtype=np.float32)
        if a.shape != self.action_shape:
            raise RuntimeError(
                f"Unexpected DreamerV3 action shape {a.shape}; "
                f"expected {self.action_shape}."
            )

        if self.normalize_actions:
            a = np.clip(a, -1.0, 1.0)
            a = self.action_low + 0.5 * (a + 1.0) * (
                self.action_high - self.action_low
            )

        a = np.clip(a, self.action_low, self.action_high)
        return a.astype(self.env.action_space.dtype, copy=False)

    def action(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != self.env.observation_space.shape:
            raise RuntimeError(
                f"Observation shape {obs.shape} does not match "
                f"{self.env.observation_space.shape}."
            )

        # Non-vectorized env: [obs] -> [B=1, T=1, obs].
        obs_tensor = convert_to_tensor(obs, framework="torch")[None, None]
        batch = {
            Columns.STATE_IN: self.states,
            Columns.OBS: obs_tensor,
            "is_first": convert_to_tensor(
                self.is_first, framework="torch"
            )[None],
        }

        outs = self.module.forward_inference(batch)
        if Columns.ACTIONS not in outs or Columns.STATE_OUT not in outs:
            raise RuntimeError(
                "DreamerV3 output missing ACTIONS or STATE_OUT; "
                f"keys={list(outs.keys())}"
            )

        self.states = outs[Columns.STATE_OUT]
        self.is_first = 0.0

        actions = outs[Columns.ACTIONS]
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        else:
            actions = np.asarray(actions)

        if actions.ndim < 2 or actions.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected ACTIONS shape {actions.shape}; "
                "expected singleton time rank."
            )
        actions = actions[0]

        if actions.ndim < 1 or actions.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected ACTIONS shape after T removal: {actions.shape}; "
                "expected singleton batch rank."
            )
        return self._to_env_action(actions[0])


def refresh_obs_after_day_change(env, obs):
    if hasattr(env, "_get_obs"):
        return env._get_obs()
    if hasattr(env, "_get_observation"):
        return env._get_observation()
    return obs


# ---------------------------------------------------------------------
# DreamerV3 and uncontrolled evaluation
# ---------------------------------------------------------------------
def evaluate_controller(env, controller, seed, algo=None, max_days=None):
    n_days = complete_test_days(env)
    if max_days is not None:
        n_days = min(n_days, max_days)
    if n_days <= 0:
        raise ValueError("No complete test episodes available.")

    bus_ids = env.net.bus.index.to_numpy()
    line_ids = env.net.line.index.to_numpy()
    dummy = neutral_action(env.action_space)
    dreamer = DreamerV3Inference(algo, env) if controller == "dreamerv3" else None

    steps, days, service = [], [], []
    print(f"\n{controller}: {n_days} complete test days")

    for day in range(n_days):
        obs, _ = env.reset(seed=seed + day)
        req = snapshot_ev_requirements(env)
        set_test_day(env, day)
        obs = refresh_obs_after_day_change(env, obs)

        if dreamer is not None:
            dreamer.reset()

        gp, ep, price_used, vmin, vmax, lmax, tmax = [], [], [], [], [], [], []
        total_reward = total_safety = 0.0
        all_converged = True

        for _ in range(env.episode_length):
            data_idx, step_idx = int(env.data_idx), int(env.step_idx)
            clock_hour = float(env.clock_hour)
            dataset_price = float(env.dataset["price"][data_idx])
            action = dreamer.action(obs) if dreamer is not None else dummy
            obs, reward, terminated, truncated, info = env.step(action)
            converged = bool(info.get("converged", env.net.get("converged", False)))
            all_converged &= converged

            row = {
                "controller": controller,
                "day": day,
                "step": step_idx,
                "data_idx": data_idx,
                "clock_hour": clock_hour,
                "dataset_price": dataset_price,
                "reward": float(reward),
                "safety": float(info.get("s", 0.0)),
                "converged": converged,
            }

            if converged:
                grid_p = float(env.net.res_ext_grid.iloc[0]["p_mw"])
                ev_p = float(sum(float(d.state.P) for d in evs(env)))
                vm = env.net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(float)
                ll = env.net.res_line.loc[line_ids, "loading_percent"].to_numpy(float)
                mt = (
                    float(np.nanmax(env.net.res_trafo["loading_percent"].to_numpy(float)))
                    if len(env.net.res_trafo) else np.nan
                )
                mn, mx, ml = (
                    float(np.nanmin(vm)),
                    float(np.nanmax(vm)),
                    float(np.nanmax(ll)),
                )
                gp.append(grid_p)
                ep.append(ev_p)
                price_used.append(dataset_price)
                vmin.append(mn)
                vmax.append(mx)
                lmax.append(ml)
                tmax.append(mt)
                row.update(
                    grid_p_mw=grid_p,
                    ev_p_mw=ev_p,
                    min_vm_pu=mn,
                    max_vm_pu=mx,
                    max_line_loading_percent=ml,
                    max_trafo_loading_percent=mt,
                )
            else:
                row.update(
                    grid_p_mw=np.nan,
                    ev_p_mw=np.nan,
                    min_vm_pu=np.nan,
                    max_vm_pu=np.nan,
                    max_line_loading_percent=np.nan,
                    max_trafo_loading_percent=np.nan,
                )

            steps.append(row)
            total_reward += float(reward)
            total_safety += float(info.get("s", 0.0))
            if terminated or truncated:
                break

        service_rows = service_after_episode(env, req, day, controller)
        service.extend(service_rows)

        if gp:
            p = np.asarray(gp, float)
            e = np.asarray(ep, float)
            days.append({
                "controller": controller,
                "day": day,
                "converged": all_converged,
                "peak_grid_p_mw": float(p.max()),
                "valley_grid_p_mw": float(p.min()),
                "mean_grid_p_mw": float(p.mean()),
                "load_variance": float(np.mean((p - p.mean()) ** 2)),
                "import_energy_mwh": float(np.maximum(p, 0).sum() * env.dt),
                "peak_ev_p_mw": float(e.max()),
                "ev_energy_mwh": float(e.sum() * env.dt),
                "ev_price_weighted_cost": float(
                    np.sum(e * np.asarray(price_used, float)) * env.dt
                ),
                "min_vm_pu": float(np.min(vmin)),
                "max_vm_pu": float(np.max(vmax)),
                "max_line_loading_percent": float(np.max(lmax)),
                "max_trafo_loading_percent": float(np.nanmax(tmax)),
                "total_reward": total_reward,
                "total_safety": total_safety,
                "target_feasible_fraction": float(
                    np.mean([r["target_feasible"] for r in req])
                ),
                "target_met_fraction": float(
                    np.mean([r["target_met"] for r in service_rows])
                ),
            })

        if (day + 1) % 25 == 0 or day + 1 == n_days:
            print(f"  finished {day + 1}/{n_days}")

    return pd.DataFrame(steps), pd.DataFrame(days), pd.DataFrame(service)


# ---------------------------------------------------------------------
# Perfect-foresight LP
# ---------------------------------------------------------------------
def zero_action(env):
    if not isinstance(env.action_space, Box):
        raise TypeError(
            f"Perfect-foresight LP requires continuous Box actions, got "
            f"{env.action_space!r}"
        )
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def action_layout(env):
    """Map EV name -> index in the concatenated continuous action."""
    _, _, _, slices = env._device_action_slices()
    out = {}
    c = 0
    for dev, nc, _nd in slices:
        if nc:
            if dev.__class__.__name__ == "EV":
                if nc != 1:
                    raise ValueError(
                        f"{dev.name}: expected 1 continuous action, got {nc}"
                    )
                out[dev.name] = c
            c += nc

    if c != int(np.prod(env.action_space.shape)):
        raise RuntimeError(
            f"Action layout size {c} != action-space size "
            f"{env.action_space.shape}"
        )

    missing = [d.name for d in evs(env) if d.name not in out]
    if missing:
        raise RuntimeError(f"EVs missing from action layout: {missing}")
    return out


def zero_ev_profile(env, day, seed):
    env.reset(seed=seed)
    req = snapshot_ev_requirements(env)
    set_test_day(env, day)
    a = zero_action(env)
    vals = []

    for _ in range(env.episode_length):
        _, _, terminated, truncated, info = env.step(a)
        ok = bool(info.get("converged", env.net.get("converged", False)))
        if not ok:
            raise RuntimeError(f"Zero-EV power flow failed on day {day}")
        vals.append(float(env.net.res_ext_grid.iloc[0]["p_mw"]))
        if terminated or truncated:
            break

    if len(vals) != env.episode_length:
        raise RuntimeError(f"Incomplete zero-EV episode on day {day}")
    return np.asarray(vals, float), req


def solve_peak_lp(base, req, dt):
    """Minimize predicted peak z with EV availability and energy constraints."""
    T = len(base)
    N = len(req)
    npow = N * T
    z = npow
    nv = npow + 1

    def ix(i, t):
        return i * T + t

    c = np.zeros(nv)
    c[z] = 1.0

    A_ub = np.zeros((T, nv))
    b_ub = -np.asarray(base, float)
    for t in range(T):
        for i in range(N):
            A_ub[t, ix(i, t)] = 1.0
        A_ub[t, z] = -1.0

    A_eq = np.zeros((N, nv))
    b_eq = np.zeros(N)
    for i, r in enumerate(req):
        for t in range(T):
            A_eq[i, ix(i, t)] = dt
        b_eq[i] = r["scheduled_grid_energy_mwh"]

    bounds = []
    for r in req:
        for t in range(T):
            bounds.append(
                (0.0, r["pmax_mw"] if r["connected"][t] else 0.0)
            )
    bounds.append((None, None))

    sol = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not sol.success:
        raise RuntimeError(
            f"LP failed: status={sol.status}: {sol.message}"
        )

    sched = sol.x[:npow].reshape(N, T)
    predicted = np.asarray(base, float) + sched.sum(axis=0)

    if N:
        err = float(
            np.max(
                np.abs(
                    sched.sum(axis=1) * dt
                    - np.asarray(b_eq)
                )
            )
        )
        if err > 1e-7:
            raise RuntimeError(f"LP energy equality check failed: {err}")

    return sched, predicted, float(sol.x[z]), sol


def verify_same_ev_realization(now, req):
    if len(now) != len(req):
        raise RuntimeError("EV count changed on paired LP replay.")

    numeric = (
        "arrival_step", "departure_step", "initial_soc",
        "target_soc", "pmax_mw",
    )
    for a, b in zip(now, req):
        if a["name"] != b["name"]:
            raise RuntimeError(
                f"Paired reset EV-name mismatch: {a['name']} vs {b['name']}"
            )
        for key in numeric:
            if not np.isclose(a[key], b[key], atol=1e-10):
                raise RuntimeError(
                    f"Paired reset mismatch for {a['name']}: {key}"
                )


def replay_schedule(env, day, seed, sched, req, controller):
    env.reset(seed=seed)
    now = snapshot_ev_requirements(env)
    verify_same_ev_realization(now, req)
    set_test_day(env, day)

    E = evs(env)
    layout = action_layout(env)
    bus_ids = env.net.bus.index.to_numpy()
    line_ids = env.net.line.index.to_numpy()

    rows = []
    gp, ep, price_used, vmin, vmax, lmax, tmax = [], [], [], [], [], [], []
    total_reward = total_safety = 0.0
    all_converged = True

    for t in range(env.episode_length):
        action = zero_action(env)
        for i, d in enumerate(E):
            action[layout[d.name]] = sched[i, t]

        data_idx, step_idx = int(env.data_idx), int(env.step_idx)
        clock_hour = float(env.clock_hour)
        dataset_price = float(env.dataset["price"][data_idx])
        _, reward, terminated, truncated, info = env.step(action)

        converged = bool(
            info.get("converged", env.net.get("converged", False))
        )
        all_converged &= converged
        safety = float(info.get("s", 0.0))
        total_reward += float(reward)
        total_safety += safety

        row = {
            "controller": controller,
            "day": day,
            "step": step_idx,
            "data_idx": data_idx,
            "clock_hour": clock_hour,
            "dataset_price": dataset_price,
            "reward": float(reward),
            "safety": safety,
            "converged": converged,
        }

        if converged:
            grid_p = float(env.net.res_ext_grid.iloc[0]["p_mw"])
            ev_p = float(sum(float(d.state.P) for d in E))
            vm = env.net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(float)
            ll = env.net.res_line.loc[line_ids, "loading_percent"].to_numpy(float)
            mt = (
                float(np.nanmax(env.net.res_trafo["loading_percent"].to_numpy(float)))
                if len(env.net.res_trafo) else np.nan
            )
            mn, mx, ml = (
                float(np.nanmin(vm)),
                float(np.nanmax(vm)),
                float(np.nanmax(ll)),
            )
            gp.append(grid_p)
            ep.append(ev_p)
            price_used.append(dataset_price)
            vmin.append(mn)
            vmax.append(mx)
            lmax.append(ml)
            tmax.append(mt)
            row.update(
                grid_p_mw=grid_p,
                ev_p_mw=ev_p,
                min_vm_pu=mn,
                max_vm_pu=mx,
                max_line_loading_percent=ml,
                max_trafo_loading_percent=mt,
            )
        else:
            row.update(
                grid_p_mw=np.nan,
                ev_p_mw=np.nan,
                min_vm_pu=np.nan,
                max_vm_pu=np.nan,
                max_line_loading_percent=np.nan,
                max_trafo_loading_percent=np.nan,
            )

        rows.append(row)
        if terminated or truncated:
            break

    if not gp:
        raise RuntimeError(f"No converged {controller} replay steps on day {day}")

    service_rows = service_after_episode(
        env, req, day, controller
    )

    p = np.asarray(gp, float)
    e = np.asarray(ep, float)
    daily = {
        "controller": controller,
        "day": day,
        "converged": all_converged,
        "peak_grid_p_mw": float(p.max()),
        "valley_grid_p_mw": float(p.min()),
        "mean_grid_p_mw": float(p.mean()),
        "load_variance": float(np.mean((p - p.mean()) ** 2)),
        "import_energy_mwh": float(np.maximum(p, 0).sum() * env.dt),
        "peak_ev_p_mw": float(e.max()),
        "ev_energy_mwh": float(e.sum() * env.dt),
        "ev_price_weighted_cost": float(
            np.sum(e * np.asarray(price_used, float)) * env.dt
        ),
        "min_vm_pu": float(np.min(vmin)),
        "max_vm_pu": float(np.max(vmax)),
        "max_line_loading_percent": float(np.max(lmax)),
        "max_trafo_loading_percent": float(np.nanmax(tmax)),
        "total_reward": total_reward,
        "total_safety": total_safety,
        "target_feasible_fraction": float(
            np.mean([r["target_feasible"] for r in req])
        ),
        "target_met_fraction": float(
            np.mean([r["target_met"] for r in service_rows])
        ),
    }
    return rows, daily, service_rows



def price_vector_for_day(env, day):
    """Return the 48 price samples for the same 08:00->08:00 test episode."""
    steps_per_day = round(24.0 / env.dt)
    start_offset = round((env.start_hour % 24.0) / env.dt)
    start = start_offset + day * steps_per_day
    stop = start + env.episode_length
    price = np.asarray(env.dataset["price"][start:stop], dtype=float)
    if price.shape != (env.episode_length,):
        raise RuntimeError(
            f"Price slice for day {day} has shape {price.shape}; "
            f"expected {(env.episode_length,)}."
        )
    if not np.all(np.isfinite(price)):
        raise RuntimeError(f"Non-finite price values found on day {day}.")
    return price


def build_lowest_price_schedule(req, prices, dt):
    """Charge each EV in its cheapest connected steps.

    This is a deterministic price-forecast heuristic.  Each EV independently
    ranks only the price samples that occur while it is connected.  It charges
    at rated power in the cheapest samples, with a fractional final sample when
    needed, until the same service energy used by the PF benchmark is supplied.
    """
    prices = np.asarray(prices, dtype=float)
    T = prices.size
    sched = np.zeros((len(req), T), dtype=float)

    for i, r in enumerate(req):
        connected = np.flatnonzero(np.asarray(r["connected"], dtype=bool))
        remaining = float(r["scheduled_grid_energy_mwh"])
        if remaining <= 1e-12 or connected.size == 0:
            continue

        # Stable sorting gives a deterministic tie-break: earlier connected
        # step first when two prices are exactly equal.
        order = connected[np.argsort(prices[connected], kind="stable")]

        for t in order:
            if remaining <= 1e-12:
                break
            p = min(float(r["pmax_mw"]), remaining / dt)
            sched[i, t] = p
            remaining -= p * dt

        if remaining > 1e-8:
            raise RuntimeError(
                f"Lowest-price schedule could not deliver required energy "
                f"for {r['name']}: remaining={remaining:.12g} MWh"
            )

    # Independent numerical verification.
    target = np.asarray(
        [r["scheduled_grid_energy_mwh"] for r in req], dtype=float
    )
    delivered = sched.sum(axis=1) * dt if len(req) else np.asarray([])
    if len(req) and np.max(np.abs(delivered - target)) > 1e-7:
        raise RuntimeError("Lowest-price schedule failed energy equality check.")

    return sched


def evaluate_lowest_price(env, seed, max_days=None):
    n_days = complete_test_days(env)
    if max_days is not None:
        n_days = min(n_days, max_days)
    if n_days <= 0:
        raise ValueError("No complete test episodes available.")

    steps, days, service, diagnostics = [], [], [], []
    print(f"\nlowest_price: {n_days} complete test days")

    for day in range(n_days):
        day_seed = seed + day

        # First reset fixes exactly the same randomized EV realization used by
        # the other methods.  The schedule sees only the test-day price series
        # and each EV's own connection window/service requirement.
        env.reset(seed=day_seed)
        req = snapshot_ev_requirements(env)
        prices = price_vector_for_day(env, day)
        sched = build_lowest_price_schedule(req, prices, env.dt)

        sr, dr, er = replay_schedule(
            env, day, day_seed, sched, req, "lowest_price"
        )

        scheduled_cost = float(
            np.sum(sched.sum(axis=0) * prices * env.dt)
        )
        dr.update(
            scheduled_ev_price_cost=scheduled_cost,
            scheduled_vs_replayed_price_cost_error=(
                dr["ev_price_weighted_cost"] - scheduled_cost
            ),
            mean_episode_price=float(np.mean(prices)),
            min_episode_price=float(np.min(prices)),
            max_episode_price=float(np.max(prices)),
        )

        steps.extend(sr)
        days.append(dr)
        service.extend(er)
        diagnostics.append({
            "day": day,
            "scheduled_ev_energy_mwh": float(sched.sum() * env.dt),
            "scheduled_ev_price_cost": scheduled_cost,
            "mean_episode_price": float(np.mean(prices)),
            "min_episode_price": float(np.min(prices)),
            "max_episode_price": float(np.max(prices)),
            "target_feasible_fraction": float(
                np.mean([r["target_feasible"] for r in req])
            ),
        })

        if (day + 1) % 25 == 0 or day + 1 == n_days:
            print(
                f"  finished {day + 1}/{n_days}: "
                f"AC peak={dr['peak_grid_p_mw']:.4f} MW"
            )

    return (
        pd.DataFrame(steps),
        pd.DataFrame(days),
        pd.DataFrame(service),
        pd.DataFrame(diagnostics),
    )



def evaluate_perfect_foresight(env, seed, max_days=None):
    n_days = complete_test_days(env)
    if max_days is not None:
        n_days = min(n_days, max_days)
    if n_days <= 0:
        raise ValueError("No complete test episodes available.")

    steps, days, service, diagnostics = [], [], [], []
    print(f"\nperfect_foresight_lp: {n_days} complete test days")

    for day in range(n_days):
        day_seed = seed + day
        base, req = zero_ev_profile(env, day, day_seed)
        sched, predicted, z, sol = solve_peak_lp(base, req, env.dt)
        sr, dr, er = replay_schedule(
            env, day, day_seed, sched, req, "perfect_foresight_lp"
        )

        dr.update(
            lp_predicted_peak_mw=z,
            zero_ev_peak_mw=float(base.max()),
            lp_vs_ac_peak_error_mw=dr["peak_grid_p_mw"] - z,
        )

        steps.extend(sr)
        days.append(dr)
        service.extend(er)
        diagnostics.append({
            "day": day,
            "lp_success": bool(sol.success),
            "lp_status": int(sol.status),
            "lp_predicted_peak_mw": z,
            "zero_ev_peak_mw": float(base.max()),
            "lp_predicted_variance": float(np.var(predicted)),
            "scheduled_ev_energy_mwh": float(sched.sum() * env.dt),
            "target_feasible_fraction": float(
                np.mean([r["target_feasible"] for r in req])
            ),
            "lp_vs_ac_peak_error_mw": dr["peak_grid_p_mw"] - z,
        })

        if (day + 1) % 25 == 0 or day + 1 == n_days:
            print(
                f"  finished {day + 1}/{n_days}: "
                f"LP predicted={z:.4f} MW, "
                f"AC replay={dr['peak_grid_p_mw']:.4f} MW"
            )

    return (
        pd.DataFrame(steps),
        pd.DataFrame(days),
        pd.DataFrame(service),
        pd.DataFrame(diagnostics),
    )


# ---------------------------------------------------------------------
# Four-way comparison and outputs
# ---------------------------------------------------------------------
def paired_four_way(rd, ud, ld, pdaily):
    r = rd.set_index("day")
    u = ud.set_index("day")
    l = ld.set_index("day")
    p = pdaily.set_index("day")
    common = sorted(set(r.index) & set(u.index) & set(l.index) & set(p.index))
    rows = []

    for day in common:
        R, U, L, P = r.loc[day], u.loc[day], l.loc[day], p.loc[day]
        uc_to_rl = float(U.peak_grid_p_mw - R.peak_grid_p_mw)
        uc_to_lpprice = float(U.peak_grid_p_mw - L.peak_grid_p_mw)
        uc_to_pf = float(U.peak_grid_p_mw - P.peak_grid_p_mw)

        def pct(x):
            return 100.0 * x / U.peak_grid_p_mw if U.peak_grid_p_mw else np.nan

        rows.append({
            "day": day,
            "uncontrolled_peak_mw": U.peak_grid_p_mw,
            "lowest_price_peak_mw": L.peak_grid_p_mw,
            "dreamerv3_peak_mw": R.peak_grid_p_mw,
            "perfect_foresight_peak_mw": P.peak_grid_p_mw,
            "lowest_price_peak_reduction_mw": uc_to_lpprice,
            "dreamerv3_peak_reduction_mw": uc_to_rl,
            "perfect_foresight_peak_reduction_mw": uc_to_pf,
            "lowest_price_peak_reduction_percent": pct(uc_to_lpprice),
            "dreamerv3_peak_reduction_percent": pct(uc_to_rl),
            "perfect_foresight_peak_reduction_percent": pct(uc_to_pf),
            "lowest_price_opportunity_captured_percent": (
                100.0 * uc_to_lpprice / uc_to_pf
                if uc_to_pf > 1e-12 else np.nan
            ),
            "dreamerv3_opportunity_captured_percent": (
                100.0 * uc_to_rl / uc_to_pf
                if uc_to_pf > 1e-12 else np.nan
            ),
            "uncontrolled_load_variance": U.load_variance,
            "lowest_price_load_variance": L.load_variance,
            "dreamerv3_load_variance": R.load_variance,
            "perfect_foresight_load_variance": P.load_variance,
            "uncontrolled_ev_energy_mwh": U.ev_energy_mwh,
            "lowest_price_ev_energy_mwh": L.ev_energy_mwh,
            "dreamerv3_ev_energy_mwh": R.ev_energy_mwh,
            "perfect_foresight_ev_energy_mwh": P.ev_energy_mwh,
            "uncontrolled_ev_price_weighted_cost": U.ev_price_weighted_cost,
            "lowest_price_ev_price_weighted_cost": L.ev_price_weighted_cost,
            "dreamerv3_ev_price_weighted_cost": R.ev_price_weighted_cost,
            "perfect_foresight_ev_price_weighted_cost":
                P.ev_price_weighted_cost,
            "uncontrolled_target_met_fraction": U.target_met_fraction,
            "lowest_price_target_met_fraction": L.target_met_fraction,
            "dreamerv3_target_met_fraction": R.target_met_fraction,
            "perfect_foresight_target_met_fraction": P.target_met_fraction,
            "uncontrolled_min_vm_pu": U.min_vm_pu,
            "lowest_price_min_vm_pu": L.min_vm_pu,
            "dreamerv3_min_vm_pu": R.min_vm_pu,
            "perfect_foresight_min_vm_pu": P.min_vm_pu,
            "uncontrolled_max_line_loading_percent": U.max_line_loading_percent,
            "lowest_price_max_line_loading_percent": L.max_line_loading_percent,
            "dreamerv3_max_line_loading_percent": R.max_line_loading_percent,
            "perfect_foresight_max_line_loading_percent":
                P.max_line_loading_percent,
        })

    return pd.DataFrame(rows)


def summary_four_way(rd, ud, ld, pdaily, paired):
    datasets = {
        "uncontrolled": ud,
        "lowest_price": ld,
        "dreamerv3": rd,
        "perfect_foresight_lp": pdaily,
    }
    specs = [
        ("mean_daily_peak_grid_p_mw", "peak_grid_p_mw", "mean"),
        ("annual_peak_grid_p_mw", "peak_grid_p_mw", "max"),
        ("mean_daily_load_variance", "load_variance", "mean"),
        ("mean_daily_ev_energy_mwh", "ev_energy_mwh", "mean"),
        ("mean_daily_ev_price_weighted_cost", "ev_price_weighted_cost", "mean"),
        ("mean_target_met_fraction", "target_met_fraction", "mean"),
        ("annual_min_voltage_pu", "min_vm_pu", "min"),
        ("annual_max_voltage_pu", "max_vm_pu", "max"),
        ("annual_max_line_loading_percent", "max_line_loading_percent", "max"),
        ("annual_max_trafo_loading_percent", "max_trafo_loading_percent", "max"),
    ]

    rows = []
    for metric, col, agg in specs:
        row = {"metric": metric}
        for name, df in datasets.items():
            row[name] = getattr(df[col], agg)()
        rows.append(row)

    if len(paired):
        rows.extend([
            {
                "metric": "mean_peak_reduction_vs_uncontrolled_mw",
                "uncontrolled": 0.0,
                "lowest_price": paired.lowest_price_peak_reduction_mw.mean(),
                "dreamerv3": paired.dreamerv3_peak_reduction_mw.mean(),
                "perfect_foresight_lp":
                    paired.perfect_foresight_peak_reduction_mw.mean(),
            },
            {
                "metric": "mean_peak_reduction_vs_uncontrolled_percent",
                "uncontrolled": 0.0,
                "lowest_price":
                    paired.lowest_price_peak_reduction_percent.mean(),
                "dreamerv3":
                    paired.dreamerv3_peak_reduction_percent.mean(),
                "perfect_foresight_lp":
                    paired.perfect_foresight_peak_reduction_percent.mean(),
            },
            {
                "metric": "mean_peak_reduction_opportunity_captured_percent",
                "uncontrolled": np.nan,
                "lowest_price":
                    paired.lowest_price_opportunity_captured_percent.mean(),
                "dreamerv3":
                    paired.dreamerv3_opportunity_captured_percent.mean(),
                "perfect_foresight_lp": 100.0,
            },
        ])
    return pd.DataFrame(rows)


def make_combined_figure(
    rs, rd, us, ud, ls, ld, ps, pdaily, paired, path
):
    rv = rs[rs.converged]
    uv = us[us.converged]
    lv = ls[ls.converged]
    pv = ps[ps.converged]
    if rv.empty or uv.empty or lv.empty or pv.empty:
        print("WARNING: comparison figure skipped because a step table is empty.")
        return

    rep_day = int(
        ud.loc[
            (ud.peak_grid_p_mw - ud.peak_grid_p_mw.median()).abs().idxmin(),
            "day",
        ]
    )

    step_sets = [
        (uv, "Uncontrolled"),
        (lv, "Lowest-price"),
        (rv, "DreamerV3"),
        (pv, "Perfect-foresight LP"),
    ]
    daily_sets = [
        (ud, "Uncontrolled"),
        (ld, "Lowest-price"),
        (rd, "DreamerV3"),
        (pdaily, "Perfect-foresight LP"),
    ]

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    for df, label in step_sets:
        d = df[df.day == rep_day].sort_values("clock_hour")
        ax[0, 0].plot(
            d.clock_hour, d.grid_p_mw,
            marker="o", markersize=2.2, label=label,
        )
    ax[0, 0].set_title(f"Representative test day {rep_day}")
    ax[0, 0].set_xlabel("Clock hour")
    ax[0, 0].set_ylabel("Grid import (MW)")
    ax[0, 0].grid(alpha=.25)
    ax[0, 0].legend()

    for df, label in step_sets:
        x = df.groupby("clock_hour").grid_p_mw.mean().sort_index()
        ax[0, 1].plot(x.index, x.values, linewidth=2, label=label)
    ax[0, 1].set_title("Mean test-set load profile")
    ax[0, 1].set_xlabel("Clock hour")
    ax[0, 1].set_ylabel("Mean grid import (MW)")
    ax[0, 1].grid(alpha=.25)
    ax[0, 1].legend()

    for df, label in step_sets:
        x = df.groupby("clock_hour").ev_p_mw.mean().sort_index()
        ax[0, 2].plot(x.index, x.values, linewidth=2, label=label)
    ax[0, 2].set_title("Mean EV charging profile")
    ax[0, 2].set_xlabel("Clock hour")
    ax[0, 2].set_ylabel("EV charging power (MW)")
    ax[0, 2].grid(alpha=.25)
    ax[0, 2].legend()

    for df, label in daily_sets:
        ax[1, 0].plot(df.day, df.peak_grid_p_mw, label=label)
    ax[1, 0].set_title("Daily peak demand")
    ax[1, 0].set_xlabel("Test day")
    ax[1, 0].set_ylabel("Peak grid import (MW)")
    ax[1, 0].grid(alpha=.25)
    ax[1, 0].legend()

    for df, label in daily_sets:
        ax[1, 1].plot(df.day, df.load_variance, label=label)
    ax[1, 1].set_title("Daily load variance")
    ax[1, 1].set_xlabel("Test day")
    ax[1, 1].set_ylabel(r"Load variance (MW$^2$)")
    ax[1, 1].grid(alpha=.25)
    ax[1, 1].legend()

    q1 = paired.dropna(
        subset=["dreamerv3_opportunity_captured_percent"]
    )
    q2 = paired.dropna(
        subset=["lowest_price_opportunity_captured_percent"]
    )
    if len(q2):
        ax[1, 2].plot(
            q2.day, q2.lowest_price_opportunity_captured_percent,
            linewidth=1.5, label="Lowest-price",
        )
    if len(q1):
        ax[1, 2].plot(
            q1.day, q1.dreamerv3_opportunity_captured_percent,
            linewidth=1.5, label="DreamerV3",
        )
    ax[1, 2].axhline(
        100.0, linestyle="--", linewidth=1,
        label="Perfect-foresight reference",
    )
    ax[1, 2].set_title("Peak-reduction opportunity captured")
    ax[1, 2].set_xlabel("Test day")
    ax[1, 2].set_ylabel("Captured (%)")
    ax[1, 2].grid(alpha=.25)
    ax[1, 2].legend()

    fig.suptitle(
        "IEEE-34 EV charging comparison: Uncontrolled vs Lowest-Price "
        "vs DreamerV3 vs Perfect-Foresight LP",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=220)
    plt.close(fig)

    # Save each of the six comparison panels as an individual figure as well.
    # This does not change any calculations or the combined figure above.
    panel_dir = Path(path).parent

    fig1, a = plt.subplots(figsize=(8.5, 5.5))
    for df, label in step_sets:
        d = df[df.day == rep_day].sort_values("clock_hour")
        a.plot(d.clock_hour, d.grid_p_mw, marker="o", markersize=2.2, label=label)
    a.set_title(f"Representative test day {rep_day}")
    a.set_xlabel("Clock hour")
    a.set_ylabel("Grid import (MW)")
    a.grid(alpha=.25)
    a.legend()
    fig1.tight_layout()
    fig1.savefig(panel_dir / "comparison_representative_test_day.png", dpi=220)
    plt.close(fig1)

    fig2, a = plt.subplots(figsize=(8.5, 5.5))
    for df, label in step_sets:
        x = df.groupby("clock_hour").grid_p_mw.mean().sort_index()
        a.plot(x.index, x.values, linewidth=2, label=label)
    a.set_title("Mean test-set load profile")
    a.set_xlabel("Clock hour")
    a.set_ylabel("Mean grid import (MW)")
    a.grid(alpha=.25)
    a.legend()
    fig2.tight_layout()
    fig2.savefig(panel_dir / "comparison_mean_load_profile.png", dpi=220)
    plt.close(fig2)

    fig3, a = plt.subplots(figsize=(8.5, 5.5))
    for df, label in step_sets:
        x = df.groupby("clock_hour").ev_p_mw.mean().sort_index()
        a.plot(x.index, x.values, linewidth=2, label=label)
    a.set_title("Mean EV charging profile")
    a.set_xlabel("Clock hour")
    a.set_ylabel("EV charging power (MW)")
    a.grid(alpha=.25)
    a.legend()
    fig3.tight_layout()
    fig3.savefig(panel_dir / "comparison_mean_ev_charging_profile.png", dpi=220)
    plt.close(fig3)

    fig4, a = plt.subplots(figsize=(8.5, 5.5))
    for df, label in daily_sets:
        a.plot(df.day, df.peak_grid_p_mw, label=label)
    a.set_title("Daily peak demand")
    a.set_xlabel("Test day")
    a.set_ylabel("Peak grid import (MW)")
    a.grid(alpha=.25)
    a.legend()
    fig4.tight_layout()
    fig4.savefig(panel_dir / "comparison_daily_peak_demand.png", dpi=220)
    plt.close(fig4)

    fig5, a = plt.subplots(figsize=(8.5, 5.5))
    for df, label in daily_sets:
        a.plot(df.day, df.load_variance, label=label)
    a.set_title("Daily load variance")
    a.set_xlabel("Test day")
    a.set_ylabel(r"Load variance (MW$^2$)")
    a.grid(alpha=.25)
    a.legend()
    fig5.tight_layout()
    fig5.savefig(panel_dir / "comparison_daily_load_variance.png", dpi=220)
    plt.close(fig5)

    # Daily paired improvement relative to that SAME day's uncontrolled case.
    # Positive values mean an improvement; negative values mean worse than uncontrolled.
    daily_base = ud[["day", "peak_grid_p_mw", "load_variance"]].rename(
        columns={
            "peak_grid_p_mw": "uc_peak",
            "load_variance": "uc_variance",
        }
    )

    normalized = daily_base.copy()
    normalized_methods = [
        (ld, "lowest_price", "Lowest-price"),
        (rd, "dreamerv3", "DreamerV3"),
        (pdaily, "perfect_foresight_lp", "Perfect-foresight LP"),
    ]

    for df, key, _label in normalized_methods:
        x = df[["day", "peak_grid_p_mw", "load_variance"]].rename(
            columns={
                "peak_grid_p_mw": f"{key}_peak",
                "load_variance": f"{key}_variance",
            }
        )
        normalized = normalized.merge(x, on="day", how="inner", validate="one_to_one")
        normalized[f"{key}_peak_reduction_percent"] = np.where(
            np.abs(normalized["uc_peak"]) > 1e-12,
            100.0 * (normalized["uc_peak"] - normalized[f"{key}_peak"])
            / normalized["uc_peak"],
            np.nan,
        )
        normalized[f"{key}_variance_reduction_percent"] = np.where(
            np.abs(normalized["uc_variance"]) > 1e-12,
            100.0 * (normalized["uc_variance"] - normalized[f"{key}_variance"])
            / normalized["uc_variance"],
            np.nan,
        )

    normalized.to_csv(
        panel_dir / "comparison_daily_reduction_vs_uncontrolled.csv", index=False
    )

    fig_peak_norm, a = plt.subplots(figsize=(8.5, 5.5))
    for _df, key, label in normalized_methods:
        a.plot(
            normalized.day,
            normalized[f"{key}_peak_reduction_percent"],
            linewidth=1.4,
            label=label,
        )
    a.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    a.set_title("Daily peak-demand reduction vs. uncontrolled")
    a.set_xlabel("Test day")
    a.set_ylabel("Reduction vs. uncontrolled (%)")
    a.grid(alpha=.25)
    a.legend()
    fig_peak_norm.tight_layout()
    fig_peak_norm.savefig(
        panel_dir / "comparison_daily_peak_reduction_percent.png", dpi=220
    )
    plt.close(fig_peak_norm)

    fig_var_norm, a = plt.subplots(figsize=(8.5, 5.5))
    for _df, key, label in normalized_methods:
        a.plot(
            normalized.day,
            normalized[f"{key}_variance_reduction_percent"],
            linewidth=1.4,
            label=label,
        )
    a.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    a.set_title("Daily load-variance reduction vs. uncontrolled")
    a.set_xlabel("Test day")
    a.set_ylabel("Reduction vs. uncontrolled (%)")
    a.grid(alpha=.25)
    a.legend()
    fig_var_norm.tight_layout()
    fig_var_norm.savefig(
        panel_dir / "comparison_daily_variance_reduction_percent.png", dpi=220
    )
    plt.close(fig_var_norm)

    fig6, a = plt.subplots(figsize=(8.5, 5.5))
    if len(q2):
        a.plot(q2.day, q2.lowest_price_opportunity_captured_percent,
               linewidth=1.5, label="Lowest-price")
    if len(q1):
        a.plot(q1.day, q1.dreamerv3_opportunity_captured_percent,
               linewidth=1.5, label="Smart Charging")
    a.axhline(100.0, linestyle="--", linewidth=1,
              label="Perfect-foresight reference")
    a.set_title("Peak-reduction opportunity captured")
    a.set_xlabel("Test day")
    a.set_ylabel("Captured (%)")
    a.grid(alpha=.25)
    a.legend()
    fig6.tight_layout()
    fig6.savefig(panel_dir / "comparison_peak_reduction_opportunity_captured.png", dpi=220)
    plt.close(fig6)



# ---------------------------------------------------------------------
# Publication-oriented comparison analyses
# ---------------------------------------------------------------------
def _bootstrap_mean_ci(values, seed=2026, n_boot=10000, alpha=0.05):
    """Deterministic paired-day bootstrap CI for a mean statistic."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    # Chunk to avoid a large temporary array for long test sets.
    means = np.empty(n_boot, dtype=float)
    chunk = 1000
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, x.size, size=(stop - start, x.size))
        means[start:stop] = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def extended_comparison_statistics(rd, ud, ld, pdaily, paired):
    """Statistics chosen to answer efficacy, consistency, service, and feasibility."""
    rows = []
    reductions = {
        "lowest_price": paired.lowest_price_peak_reduction_mw,
        "dreamerv3": paired.dreamerv3_peak_reduction_mw,
        "perfect_foresight_lp": paired.perfect_foresight_peak_reduction_mw,
    }
    pf_sum = float(np.nansum(paired.perfect_foresight_peak_reduction_mw))
    for k, x0 in reductions.items():
        x = np.asarray(x0, dtype=float)
        x = x[np.isfinite(x)]
        lo, hi = _bootstrap_mean_ci(x)
        rows.append({
            "analysis": "paired_daily_peak_reduction",
            "method": k,
            "n_days": int(x.size),
            "mean": float(np.mean(x)) if x.size else np.nan,
            "median": float(np.median(x)) if x.size else np.nan,
            "std": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
            "q25": float(np.quantile(x, .25)) if x.size else np.nan,
            "q75": float(np.quantile(x, .75)) if x.size else np.nan,
            "mean_95ci_low": lo,
            "mean_95ci_high": hi,
            "fraction_days_peak_reduced": float(np.mean(x > 0.0)) if x.size else np.nan,
            "aggregate_pf_opportunity_captured_percent": (
                100.0 * float(np.nansum(x)) / pf_sum
                if k != "perfect_foresight_lp" and pf_sum > 1e-12 else
                (100.0 if k == "perfect_foresight_lp" and pf_sum > 1e-12 else np.nan)
            ),
        })

    datasets = {
        "uncontrolled": ud, "lowest_price": ld,
        "dreamerv3": rd, "perfect_foresight_lp": pdaily,
    }
    for k, df in datasets.items():
        energy = df.ev_energy_mwh.to_numpy(float)
        cost = df.ev_price_weighted_cost.to_numpy(float)
        cost_per_mwh = np.divide(
            cost, energy, out=np.full_like(cost, np.nan), where=np.abs(energy) > 1e-12
        )
        rows.append({
            "analysis": "service_cost_and_grid_feasibility",
            "method": k,
            "n_days": int(len(df)),
            "mean_target_met_fraction": float(df.target_met_fraction.mean()),
            "mean_ev_energy_mwh": float(df.ev_energy_mwh.mean()),
            "mean_ev_price_weighted_cost": float(df.ev_price_weighted_cost.mean()),
            "mean_price_weighted_cost_per_ev_mwh": float(np.nanmean(cost_per_mwh)),
            "annual_min_voltage_pu": float(df.min_vm_pu.min()),
            "annual_max_voltage_pu": float(df.max_vm_pu.max()),
            "annual_max_line_loading_percent": float(df.max_line_loading_percent.max()),
            "annual_max_trafo_loading_percent": float(df.max_trafo_loading_percent.max()),
        })
    return pd.DataFrame(rows)


def make_individual_comparison_figures(rd, ud, ld, pdaily, paired, out):
    """Save focused figures; avoids forcing all evidence into one panel."""
    out = Path(out)
    datasets = [
        (ud, "Uncontrolled"), (ld, "Lowest-price"),
        (rd, "DreamerV3"), (pdaily, "Perfect-foresight LP"),
    ]

    # 1) Paired distribution: consistency of peak reduction across test days.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    vals = [
        paired.lowest_price_peak_reduction_mw.dropna().to_numpy(),
        paired.dreamerv3_peak_reduction_mw.dropna().to_numpy(),
        paired.perfect_foresight_peak_reduction_mw.dropna().to_numpy(),
    ]
    ax.boxplot(vals, tick_labels=["Lowest-price", "DreamerV3", "Perfect-foresight LP"], showmeans=True)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Daily peak reduction relative to uncontrolled charging")
    ax.set_ylabel("Peak reduction (MW)")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "comparison_peak_reduction_distribution.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    positive = [100.0 * float(np.mean(v > 0.0)) if len(v) else np.nan for v in vals]
    ax.bar(["Lowest-price", "DreamerV3", "Perfect-foresight LP"], positive)
    ax.set_title("Fraction of test days with lower peak than uncontrolled")
    ax.set_ylabel("Test days with peak reduction (%)")
    ax.set_ylim(0, 105); ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "comparison_fraction_days_peak_reduced.png", dpi=220); plt.close(fig)

    # 2) Charging cost normalized by delivered EV energy, plus raw service success separately.
    labels, cpem, service = [], [], []
    for df, label in datasets:
        labels.append(label)
        c = df.ev_price_weighted_cost.to_numpy(float)
        e = df.ev_energy_mwh.to_numpy(float)
        ratio = np.divide(c, e, out=np.full_like(c, np.nan), where=np.abs(e) > 1e-12)
        cpem.append(float(np.nanmean(ratio)))
        service.append(100.0 * float(df.target_met_fraction.mean()))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(labels, cpem)
    ax.set_title("Mean price-weighted EV charging cost per delivered energy")
    ax.set_ylabel("Price-weighted cost per EV MWh")
    ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "comparison_ev_cost_per_mwh.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(labels, service)
    ax.set_title("EV charging-service success")
    ax.set_ylabel("EV target-SOC success (%)")
    ax.set_ylim(0, 105); ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "comparison_target_soc_success.png", dpi=220); plt.close(fig)

    # 3) Mean load variance: direct smoothing metric.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(labels, [float(df.load_variance.mean()) for df, _ in datasets])
    ax.set_title("Mean daily load variance")
    ax.set_ylabel(r"Load variance (MW$^2$)")
    ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "comparison_load_variance.png", dpi=220); plt.close(fig)

    # 4) Network feasibility: focused voltage and thermal figures.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(labels)); w = .36
    ax.bar(x - w/2, [float(df.min_vm_pu.min()) for df, _ in datasets], width=w, label="Minimum")
    ax.bar(x + w/2, [float(df.max_vm_pu.max()) for df, _ in datasets], width=w, label="Maximum")
    ax.axhline(.95, linestyle="--", linewidth=1, label="0.95 pu limit")
    ax.axhline(1.05, linestyle="--", linewidth=1, label="1.05 pu limit")
    ax.set_xticks(x, labels, rotation=15); ax.set_ylabel("Voltage (pu)")
    ax.set_title("Annual voltage extrema"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "comparison_voltage_extrema.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x - w/2, [float(df.max_line_loading_percent.max()) for df, _ in datasets], width=w, label="Line")
    ax.bar(x + w/2, [float(df.max_trafo_loading_percent.max()) for df, _ in datasets], width=w, label="Transformer")
    ax.axhline(100.0, linestyle="--", linewidth=1, label="100% loading")
    ax.set_xticks(x, labels, rotation=15); ax.set_ylabel("Maximum loading (%)")
    ax.set_title("Annual maximum network loading"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "comparison_network_loading.png", dpi=220); plt.close(fig)

    # 5) Robust aggregate opportunity capture, rather than mean of unstable daily ratios.
    pf = float(paired.perfect_foresight_peak_reduction_mw.sum())
    if pf > 1e-12:
        names = ["Lowest-price", "DreamerV3", "Perfect-foresight LP"]
        cap = [
            100.0 * float(paired.lowest_price_peak_reduction_mw.sum()) / pf,
            100.0 * float(paired.dreamerv3_peak_reduction_mw.sum()) / pf,
            100.0,
        ]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.bar(names, cap); ax.axhline(100.0, linestyle="--", linewidth=1)
        ax.set_title("Aggregate peak-reduction opportunity captured")
        ax.set_ylabel("Captured (%)"); ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(out / "comparison_aggregate_opportunity_capture.png", dpi=220); plt.close(fig)

METHOD_INFO = {
    "uncontrolled": {"label": "Uncontrolled"},
    "lowest_price": {"label": "Lowest-price"},
    "dreamerv3": {"label": "DreamerV3"},
    "perfect_foresight": {"label": "Perfect-foresight LP"},
}
METHOD_ALIASES = {
    "uncontrolled": "uncontrolled", "uc": "uncontrolled",
    "lowest_price": "lowest_price", "lowest-price": "lowest_price", "price": "lowest_price",
    "dreamerv3": "dreamerv3", "dreamer": "dreamerv3", "rl": "dreamerv3",
    "perfect_foresight": "perfect_foresight", "perfect-foresight": "perfect_foresight",
    "perfect_foresight_lp": "perfect_foresight", "pf": "perfect_foresight", "lp": "perfect_foresight",
}


def parse_methods(text):
    """Parse comma-separated method names, preserving order and rejecting duplicates."""
    raw = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not raw:
        raise ValueError("--methods must contain at least one method.")
    out = []
    for x in raw:
        if x not in METHOD_ALIASES:
            valid = ", ".join(METHOD_INFO)
            raise ValueError(f"Unknown method '{x}'. Valid methods: {valid}")
        key = METHOD_ALIASES[x]
        if key in out:
            raise ValueError(f"Method '{key}' was selected more than once.")
        out.append(key)
    return out


def selected_summary(daily_by_method):
    rows = []
    for key, df in daily_by_method.items():
        energy = df.ev_energy_mwh.to_numpy(float)
        cost = df.ev_price_weighted_cost.to_numpy(float)
        cost_per_mwh = np.divide(
            cost, energy, out=np.full_like(cost, np.nan),
            where=np.abs(energy) > 1e-12,
        )
        rows.append({
            "method": key,
            "label": METHOD_INFO[key]["label"],
            "n_days": int(len(df)),
            "mean_daily_peak_mw": float(df.peak_grid_p_mw.mean()),
            "annual_peak_mw": float(df.peak_grid_p_mw.max()),
            "mean_daily_load_variance_mw2": float(df.load_variance.mean()),
            "mean_daily_ev_energy_mwh": float(df.ev_energy_mwh.mean()),
            "mean_daily_ev_price_weighted_cost": float(df.ev_price_weighted_cost.mean()),
            "mean_price_weighted_cost_per_ev_mwh": float(np.nanmean(cost_per_mwh)),
            "mean_target_met_fraction": float(df.target_met_fraction.mean()),
            "annual_min_voltage_pu": float(df.min_vm_pu.min()),
            "annual_max_voltage_pu": float(df.max_vm_pu.max()),
            "annual_max_line_loading_percent": float(df.max_line_loading_percent.max()),
            "annual_max_trafo_loading_percent": float(df.max_trafo_loading_percent.max()),
        })
    return pd.DataFrame(rows)


def paired_reduction_table(uc, daily_by_method):
    """Daily paired peak/variance reductions, always using same-day uncontrolled as baseline."""
    base = uc[["day", "peak_grid_p_mw", "load_variance"]].rename(columns={
        "peak_grid_p_mw": "uncontrolled_peak_mw",
        "load_variance": "uncontrolled_variance_mw2",
    })
    out = base.copy()
    for key, df in daily_by_method.items():
        if key == "uncontrolled":
            continue
        x = df[["day", "peak_grid_p_mw", "load_variance"]].rename(columns={
            "peak_grid_p_mw": f"{key}_peak_mw",
            "load_variance": f"{key}_variance_mw2",
        })
        out = out.merge(x, on="day", how="inner", validate="one_to_one")
        out[f"{key}_peak_reduction_mw"] = (
            out["uncontrolled_peak_mw"] - out[f"{key}_peak_mw"]
        )
        out[f"{key}_peak_reduction_percent"] = np.where(
            np.abs(out["uncontrolled_peak_mw"]) > 1e-12,
            100.0 * out[f"{key}_peak_reduction_mw"] / out["uncontrolled_peak_mw"],
            np.nan,
        )
        out[f"{key}_variance_reduction_mw2"] = (
            out["uncontrolled_variance_mw2"] - out[f"{key}_variance_mw2"]
        )
        out[f"{key}_variance_reduction_percent"] = np.where(
            np.abs(out["uncontrolled_variance_mw2"]) > 1e-12,
            100.0 * out[f"{key}_variance_reduction_mw2"] / out["uncontrolled_variance_mw2"],
            np.nan,
        )
    return out


def make_selected_figures(step_by_method, daily_by_method, uc_daily, selected, out):
    """Figures contain ONLY user-selected methods; UC may still be used internally as baseline."""
    out = Path(out)
    step_sets = [(step_by_method[k][step_by_method[k].converged], METHOD_INFO[k]["label"])
                 for k in selected]
    daily_sets = [(daily_by_method[k], METHOD_INFO[k]["label"]) for k in selected]
    if not daily_sets:
        return

    # Representative day is based on UC when available, otherwise the first selected method.
    ref = uc_daily if uc_daily is not None else daily_sets[0][0]
    rep_day = int(ref.loc[(ref.peak_grid_p_mw - ref.peak_grid_p_mw.median()).abs().idxmin(), "day"])

    def save_lines(filename, title, ylabel, getter, daily=False):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        sets = daily_sets if daily else step_sets
        for df, label in sets:
            if daily:
                x, y = df.day, getter(df)
            else:
                d = df[df.day == rep_day].sort_values("clock_hour")
                x, y = d.clock_hour, getter(d)
            ax.plot(x, y, linewidth=1.6, label=label)
        ax.set_title(title); ax.set_xlabel("Test day" if daily else "Clock hour")
        ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(out / filename, dpi=220); plt.close(fig)

    save_lines("comparison_representative_test_day.png", f"Representative test day {rep_day}",
               "Grid import (MW)", lambda d: d.grid_p_mw)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for df, label in step_sets:
        x = df.groupby("clock_hour").grid_p_mw.mean().sort_index()
        ax.plot(x.index, x.values, linewidth=2, label=label)
    ax.set_title("Mean test-set load profile"); ax.set_xlabel("Clock hour")
    ax.set_ylabel("Mean grid import (MW)"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out / "comparison_mean_load_profile.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for df, label in step_sets:
        x = df.groupby("clock_hour").ev_p_mw.mean().sort_index()
        ax.plot(x.index, x.values, linewidth=2, label=label)
    ax.set_title("Mean EV charging profile"); ax.set_xlabel("Clock hour")
    ax.set_ylabel("EV charging power (MW)"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out / "comparison_mean_ev_charging_profile.png", dpi=220); plt.close(fig)

    save_lines("comparison_daily_peak_demand.png", "Daily peak demand", "Peak grid import (MW)",
               lambda d: d.peak_grid_p_mw, daily=True)
    save_lines("comparison_daily_load_variance.png", "Daily load variance", r"Load variance (MW$^2$)",
               lambda d: d.load_variance, daily=True)

    # Bar summaries using selected methods only.
    labels = [METHOD_INFO[k]["label"] for k in selected]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(labels, [float(daily_by_method[k].load_variance.mean()) for k in selected])
    ax.set_title("Mean daily load variance"); ax.set_ylabel(r"Load variance (MW$^2$)")
    ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "comparison_load_variance.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(labels, [100.0 * float(daily_by_method[k].target_met_fraction.mean()) for k in selected])
    ax.set_title("EV charging-service success"); ax.set_ylabel("EV target-SOC success (%)")
    ax.set_ylim(0, 105); ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "comparison_target_soc_success.png", dpi=220); plt.close(fig)

    # Same-day UC normalization. UC is a baseline, not automatically displayed.
    reductions = paired_reduction_table(uc_daily, {k: daily_by_method[k] for k in selected})
    reductions.to_csv(out / "comparison_daily_reduction_vs_uncontrolled.csv", index=False)
    non_uc = [k for k in selected if k != "uncontrolled"]
    if non_uc:
        for metric, title, filename in [
            ("peak", "Daily peak-demand reduction vs. uncontrolled", "comparison_daily_peak_reduction_percent.png"),
            ("variance", "Daily load-variance reduction vs. uncontrolled", "comparison_daily_variance_reduction_percent.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            for k in non_uc:
                ax.plot(reductions.day, reductions[f"{k}_{metric}_reduction_percent"],
                        linewidth=1.4, label=METHOD_INFO[k]["label"])
            ax.axhline(0.0, linestyle="--", linewidth=1.0)
            ax.set_title(title); ax.set_xlabel("Test day")
            ax.set_ylabel("Reduction vs. uncontrolled (%)"); ax.grid(alpha=.25); ax.legend()
            fig.tight_layout(); fig.savefig(out / filename, dpi=220); plt.close(fig)

        vals = [reductions[f"{k}_peak_reduction_mw"].dropna().to_numpy() for k in non_uc]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.boxplot(vals, tick_labels=[METHOD_INFO[k]["label"] for k in non_uc], showmeans=True)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title("Daily peak reduction relative to uncontrolled charging")
        ax.set_ylabel("Peak reduction (MW)"); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(out / "comparison_peak_reduction_distribution.png", dpi=220); plt.close(fig)

    # Compact combined 2x3 figure, again selected methods only.
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for df, label in step_sets:
        d = df[df.day == rep_day].sort_values("clock_hour")
        ax[0,0].plot(d.clock_hour, d.grid_p_mw, label=label)
        x = df.groupby("clock_hour").grid_p_mw.mean().sort_index(); ax[0,1].plot(x.index, x.values, label=label)
        x = df.groupby("clock_hour").ev_p_mw.mean().sort_index(); ax[0,2].plot(x.index, x.values, label=label)
    for df, label in daily_sets:
        ax[1,0].plot(df.day, df.peak_grid_p_mw, label=label)
        ax[1,1].plot(df.day, df.load_variance, label=label)
    if non_uc:
        for k in non_uc:
            ax[1,2].plot(reductions.day, reductions[f"{k}_peak_reduction_percent"], label=METHOD_INFO[k]["label"])
        ax[1,2].axhline(0.0, linestyle="--", linewidth=1)
    titles = [f"Representative test day {rep_day}", "Mean test-set load profile", "Mean EV charging profile",
              "Daily peak demand", "Daily load variance", "Daily peak reduction vs. uncontrolled"]
    ylabels = ["Grid import (MW)", "Mean grid import (MW)", "EV charging power (MW)",
               "Peak grid import (MW)", r"Load variance (MW$^2$)", "Reduction (%)"]
    for a, title, ylabel in zip(ax.flat, titles, ylabels):
        a.set_title(title); a.set_ylabel(ylabel); a.grid(alpha=.25)
        if a.lines: a.legend()
    for a in ax[0,:]: a.set_xlabel("Clock hour")
    for a in ax[1,:]: a.set_xlabel("Test day")
    fig.suptitle("IEEE-34 EV charging comparison: " + " vs ".join(labels), fontsize=16)
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(out / "selected_method_comparison.png", dpi=220); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", default=None,
        help="RLlib DreamerV3 checkpoint directory; required only when DreamerV3 is selected.",
    )
    p.add_argument(
        "--methods",
        default="uncontrolled,lowest_price,dreamerv3,perfect_foresight",
        help=("Comma-separated methods to DISPLAY/COMPARE. Choices: uncontrolled, "
              "lowest_price, dreamerv3, perfect_foresight. Aliases uc, price, dreamer, pf are accepted."),
    )
    p.add_argument("--ev-level", choices=["low", "medium", "high"], default="low")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", default="results_selected_methods")
    p.add_argument("--max-days", type=int, default=None,
                   help="Optional debugging limit; omit for all complete test days.")
    args = p.parse_args()

    try:
        selected = parse_methods(args.methods)
    except ValueError as e:
        p.error(str(e))
    if "dreamerv3" in selected and not args.checkpoint:
        p.error("--checkpoint is required when dreamerv3 is selected.")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print("Selected comparison methods:", ", ".join(METHOD_INFO[k]["label"] for k in selected))
    print("NOTE: Uncontrolled is evaluated internally as the same-day normalization baseline even when omitted from plots.")

    ray.init(ignore_reinit_error=True)
    algo = None
    try:
        if "dreamerv3" in selected:
            checkpoint = str(Path(args.checkpoint).expanduser().resolve())
            print(f"Restoring: {checkpoint}")
            algo = Algorithm.from_checkpoint(checkpoint)
            print("Restored algorithm:", type(algo).__name__)
            print("Ray/RLlib version:", ray.__version__)
            print("Checkpoint normalize_actions:", getattr(algo.config, "normalize_actions", None))

        # UC is always evaluated because all normalized improvement metrics use the SAME day's UC value.
        required = list(dict.fromkeys(["uncontrolled"] + selected))
        step, daily, evsvc = {}, {}, {}
        diagnostics = {}

        for key in required:
            print(f"\nEvaluating {METHOD_INFO[key]['label']} ...")
            if key == "uncontrolled":
                env = make_test_env(args.ev_level, "uncontrolled")
                step[key], daily[key], evsvc[key] = evaluate_controller(env, "uncontrolled", args.seed, None, args.max_days)
            elif key == "dreamerv3":
                env = make_test_env(args.ev_level, "rl")
                step[key], daily[key], evsvc[key] = evaluate_controller(env, "dreamerv3", args.seed, algo, args.max_days)
            elif key == "lowest_price":
                env = make_test_env(args.ev_level, "rl")
                step[key], daily[key], evsvc[key], diagnostics[key] = evaluate_lowest_price(env, args.seed, args.max_days)
            elif key == "perfect_foresight":
                env = make_test_env(args.ev_level, "rl")
                step[key], daily[key], evsvc[key], diagnostics[key] = evaluate_perfect_foresight(env, args.seed, args.max_days)

        # Pairing check across every evaluated method.
        day_sets = {k: set(df.day.tolist()) for k, df in daily.items()}
        ref_days = day_sets["uncontrolled"]
        bad = [k for k, days in day_sets.items() if days != ref_days]
        if bad:
            raise RuntimeError("Test-day mismatch relative to uncontrolled for: " + ", ".join(bad))

        # Save method-specific data only for selected methods, except hidden UC baseline gets a clearly named baseline file.
        for key in required:
            prefix = key
            hidden_baseline = key == "uncontrolled" and key not in selected
            if hidden_baseline:
                prefix = "uncontrolled_baseline"
            step[key].to_csv(out / f"{prefix}_step_metrics.csv", index=False)
            daily[key].to_csv(out / f"{prefix}_daily_summary.csv", index=False)
            evsvc[key].to_csv(out / f"{prefix}_ev_service.csv", index=False)
            if key == "lowest_price" and key in diagnostics:
                diagnostics[key].to_csv(out / "lowest_price_diagnostics.csv", index=False)
            if key == "perfect_foresight" and key in diagnostics:
                diagnostics[key].to_csv(out / "perfect_foresight_lp_diagnostics.csv", index=False)

        selected_daily = {k: daily[k] for k in selected}
        summary = selected_summary(selected_daily)
        summary.to_csv(out / "summary_selected_method_comparison.csv", index=False)
        reductions = paired_reduction_table(daily["uncontrolled"], selected_daily)
        reductions.to_csv(out / "daily_selected_method_reduction_vs_uncontrolled.csv", index=False)

        make_selected_figures(step, daily, daily["uncontrolled"], selected, out)

        print("\nSelected-method comparison")
        print(summary.to_string(index=False))
        for key in selected:
            if key == "uncontrolled":
                continue
            col = f"{key}_peak_reduction_mw"
            print(f"Mean {METHOD_INFO[key]['label']} peak reduction vs uncontrolled: "
                  f"{reductions[col].mean():.6f} MW")
        print(f"\nAll results: {out.resolve()}")
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
