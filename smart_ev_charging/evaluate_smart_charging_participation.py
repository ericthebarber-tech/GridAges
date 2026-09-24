"""Evaluate partial adoption of a trained DreamerV3 EV charging policy.

Scientific question
-------------------
With the physical EV population fixed at the LOW-EV scenario used for
training, how does feeder performance change as the fraction of EVs that
participate in DreamerV3 smart charging increases?

Participation levels default to:
    0%, 20%, 40%, 60%, 80%, 100%

Important design choices
------------------------
1. The EV population is fixed.  This measures SMART-CHARGING PARTICIPATION,
   not EV adoption.
2. A participating EV uses its component of the DreamerV3 action.
3. A non-participating EV uses the same immediate/max-rate charging rule as
   EV.automatic_action() in the uncontrolled baseline.
4. Only EV action components are overridden; any non-EV policy action
   components remain identical across participation levels. Thus 100% is
   exactly the original DreamerV3 action vector.
5. Participation is assigned once per episode and remains fixed for all 48
   steps of that episode.
6. Assignment is stratified by EV usage_type and nested across participation
   levels: within a day, the smart set at a lower participation level is a
   subset of the smart set at every higher level.
7. All participation levels use the same day, seed, EV realization, and
   exogenous dataset values.
8. DreamerV3 still receives the actual resulting observation after overridden
   actions, so its recurrent state evolves under the partially controlled grid.

Outputs
-------
- participation_step_metrics.csv
- participation_daily_summary.csv
- participation_ev_service.csv
- participation_assignment.csv
- participation_summary.csv
- participation_peak_vs_rate.png
- participation_peak_reduction_vs_rate.png
- participation_variance_vs_rate.png
- participation_target_service_vs_rate.png
- participation_daily_peak.png
- participation_mean_load_profiles.png
- participation_mean_ev_profiles.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import tree
from gymnasium.spaces import Box
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns
from ray.rllib.utils.framework import convert_to_tensor
from ray.tune.registry import register_env

from gridages.envs.single_agent.ev_charging.ieee34_evs import IEEE34Env


# ---------------------------------------------------------------------
# Environment setup -- kept aligned with the tested evaluators
# ---------------------------------------------------------------------
def make_env(env_config):
    return IEEE34Env(env_config=env_config)


register_env("gridages-ieee34-ev", make_env)


def make_test_env():
    return IEEE34Env(env_config={
        "train": False,
        "ev_level": "low",
        "control_mode": "rl",
        "episode_length": 48,
        "dt": 0.5,
        "start_hour": 8.0,
        "store_info": ["bus_voltage", "line_loading", "operating_cost"],
        "store_arrays": False,
        "store_summaries": True,
        "reward_scale": 100.0,
        "safety_scale": 10000.0,
        "peak_weight": 1.0,
        "shape_weight": 100.0,
        "price_weight": 0.0,
        "penalize_safety": True,
    })


def set_test_day(env, day):
    steps_per_day = round(24.0 / env.dt)
    start_offset = round((env.start_hour % 24.0) / env.dt)
    env.data_idx = start_offset + day * steps_per_day
    env.step_idx = 0
    env._apply_dataset_scalers()


def complete_test_days(env):
    T = len(env.dataset["load"])
    steps_per_day = round(24.0 / env.dt)
    start_offset = round((env.start_hour % 24.0) / env.dt)
    return (T - start_offset) // steps_per_day


def refresh_obs_after_day_change(env, obs):
    if hasattr(env, "_get_obs"):
        return env._get_obs()
    if hasattr(env, "_get_observation"):
        return env._get_observation()
    return obs


def evs(env):
    return [d for d in env.devices.values() if d.__class__.__name__ == "EV"]


def snapshot_ev_requirements(env):
    rows = []
    for d in evs(env):
        target = getattr(d.state, "soc_target", None)
        target = float(d.max_soc if target is None else min(target, d.max_soc))
        soc = float(d.state.soc)
        cap = float(d.capacity)
        eta = float(d.ch_eff)
        pmax = float(d.max_p_mw)
        connected = np.asarray(
            [bool(d.is_connected(t)) for t in range(env.episode_length)]
        )
        needed = max((target - soc) * cap, 0.0) / eta
        deliverable = float(connected.sum()) * pmax * env.dt
        rows.append({
            "name": d.name,
            "usage_type": str(getattr(d, "usage_type", "unknown")),
            "initial_soc": soc,
            "target_soc": target,
            "capacity_mwh": cap,
            "ch_eff": eta,
            "pmax_mw": pmax,
            "arrival_step": int(d.arrive_time),
            "departure_step": int(d.depart_time),
            "target_feasible": bool(needed <= deliverable + 1e-9),
        })
    return rows


def verify_same_ev_realization(now, ref):
    if len(now) != len(ref):
        raise RuntimeError("EV count changed across paired participation runs.")
    numeric = (
        "initial_soc", "target_soc", "pmax_mw",
        "arrival_step", "departure_step",
    )
    for a, b in zip(now, ref):
        if a["name"] != b["name"] or a["usage_type"] != b["usage_type"]:
            raise RuntimeError("EV identity/type changed across paired resets.")
        for key in numeric:
            if not np.isclose(a[key], b[key], atol=1e-10):
                raise RuntimeError(
                    f"Paired reset mismatch for {a['name']}: {key}"
                )


def action_layout(env):
    """Map each EV name to its scalar continuous-action index."""
    if not isinstance(env.action_space, Box):
        raise TypeError(
            f"Expected continuous Box action space, got {env.action_space!r}"
        )

    _, _, _, slices = env._device_action_slices()
    mapping = {}
    c = 0
    for dev, nc, _nd in slices:
        if nc:
            if dev.__class__.__name__ == "EV":
                if nc != 1:
                    raise ValueError(
                        f"{dev.name}: expected one continuous EV action, got {nc}"
                    )
                mapping[dev.name] = c
            c += nc

    if c != int(np.prod(env.action_space.shape)):
        raise RuntimeError(
            f"Continuous action layout size {c} does not match "
            f"action space {env.action_space.shape}."
        )

    missing = [d.name for d in evs(env) if d.name not in mapping]
    if missing:
        raise RuntimeError(f"EVs missing from action layout: {missing}")
    return mapping


# ---------------------------------------------------------------------
# DreamerV3 inference -- same stateful pattern as the tested evaluator
# ---------------------------------------------------------------------
class DreamerV3Inference:
    def __init__(self, algo, env):
        self.algo = algo
        self.env = env
        self.module = algo.env_runner.module

        if not isinstance(env.action_space, Box):
            raise TypeError("DreamerV3 participation test requires Box actions.")

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
                f"Dreamer action shape {a.shape}; expected {self.action_shape}."
            )
        if self.normalize_actions:
            a = np.clip(a, -1.0, 1.0)
            a = self.action_low + 0.5 * (a + 1.0) * (
                self.action_high - self.action_low
            )
        return np.clip(
            a, self.action_low, self.action_high
        ).astype(self.env.action_space.dtype, copy=False)

    def action(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != self.env.observation_space.shape:
            raise RuntimeError(
                f"Observation shape {obs.shape}; "
                f"expected {self.env.observation_space.shape}."
            )

        batch = {
            Columns.STATE_IN: self.states,
            Columns.OBS:
                convert_to_tensor(obs, framework="torch")[None, None],
            "is_first":
                convert_to_tensor(self.is_first, framework="torch")[None],
        }
        outs = self.module.forward_inference(batch)
        if Columns.ACTIONS not in outs or Columns.STATE_OUT not in outs:
            raise RuntimeError(
                "DreamerV3 inference output missing ACTIONS or STATE_OUT."
            )

        self.states = outs[Columns.STATE_OUT]
        self.is_first = 0.0

        a = outs[Columns.ACTIONS]
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        else:
            a = np.asarray(a)

        # Current DreamerV3 output for a non-vectorized env has T=1, B=1.
        if a.ndim < 2 or a.shape[0] != 1:
            raise RuntimeError(f"Unexpected Dreamer ACTIONS shape {a.shape}.")
        a = a[0]
        if a.ndim < 1 or a.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected Dreamer ACTIONS shape after T removal {a.shape}."
            )
        return self._to_env_action(a[0])


# ---------------------------------------------------------------------
# Participation assignment
# ---------------------------------------------------------------------
def parse_rates(text):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("No participation rates supplied.")
    for x in vals:
        if x < 0.0 or x > 1.0:
            raise ValueError("Participation rates must lie in [0, 1].")
    vals = sorted(set(vals))
    return vals


def stratified_nested_rankings(req, seed):
    """Create one deterministic random ranking inside each usage_type.

    Because every rate uses prefixes of the SAME ranking, smart sets are nested:
    the smart set at 20% is a subset of the smart set at 40%, etc.
    """
    rng = np.random.default_rng(seed)
    by_type = {}
    for r in req:
        by_type.setdefault(r["usage_type"], []).append(r["name"])

    rankings = {}
    for usage_type in sorted(by_type):
        names = sorted(by_type[usage_type])
        perm = rng.permutation(len(names))
        rankings[usage_type] = [names[i] for i in perm]
    return rankings


def smart_set_for_rate(rankings, rate):
    """Choose approximately rate of every usage stratum.

    round() is used independently within each usage type.  The reported
    actual participation rate is therefore used in all plots/tables.
    """
    smart = set()
    for names in rankings.values():
        k = int(np.floor(rate * len(names) + 0.5))
        k = max(0, min(k, len(names)))
        smart.update(names[:k])
    return smart


# ---------------------------------------------------------------------
# Mixed controller
# ---------------------------------------------------------------------
def uncontrolled_power_for_ev(d, step_idx, dt):
    """Exactly mirror EV.automatic_action() without mutating the device."""
    if not d.is_connected(step_idx) or float(d.state.soc) >= float(d.max_soc):
        return 0.0

    max_energy_charge = (
        (float(d.max_soc) - float(d.state.soc))
        * float(d.capacity)
        / float(d.ch_eff)
        / float(dt)
    )
    return max(0.0, min(float(d.max_p_mw), max_energy_charge))


def mixed_action(env, rl_action, smart_names, layout):
    """Override only non-participating EV components of the RL action.

    Start from the complete DreamerV3 action and replace the EV components
    belonging to non-participants with the uncontrolled immediate-charging
    rule.  This is important for a clean intervention:

      * 100% participation is exactly the original DreamerV3 action vector.
      * Any non-EV action components are identical at every participation rate.
      * Only EV smart-charging participation changes across the experiment.

    Therefore the 0% endpoint means "0% of EVs use smart charging" under the
    same background policy for any other controllable dimensions; it is not
    silently redefined by zeroing unrelated action components.
    """
    applied = np.asarray(rl_action, dtype=env.action_space.dtype).copy()
    if applied.shape != env.action_space.shape:
        raise RuntimeError(
            f"RL action shape {applied.shape} != {env.action_space.shape}"
        )

    for d in evs(env):
        if d.name not in smart_names:
            applied[layout[d.name]] = uncontrolled_power_for_ev(
                d, int(env.step_idx), env.dt
            )

    return np.clip(
        applied, env.action_space.low, env.action_space.high
    ).astype(env.action_space.dtype, copy=False)


def evaluate_one_rate(
    env, algo, rate, seed, assignment_seed, max_days=None
):
    n_days = complete_test_days(env)
    if max_days is not None:
        n_days = min(n_days, max_days)

    dreamer = DreamerV3Inference(algo, env)
    layout = action_layout(env)
    bus_ids = env.net.bus.index.to_numpy()
    line_ids = env.net.line.index.to_numpy()

    step_rows, day_rows, service_rows, assignment_rows = [], [], [], []
    print(f"\nTarget smart participation: {100*rate:.1f}%")

    for day in range(n_days):
        obs, _ = env.reset(seed=seed + day)
        req = snapshot_ev_requirements(env)
        set_test_day(env, day)
        obs = refresh_obs_after_day_change(env, obs)
        dreamer.reset()

        # Assignment is reproducible, stratified, fixed for the whole episode,
        # and shared/nested across rates through the rate-independent ranking.
        rankings = stratified_nested_rankings(
            req, assignment_seed + day
        )
        smart_names = smart_set_for_rate(rankings, rate)
        n_ev = len(req)
        actual_rate = len(smart_names) / n_ev if n_ev else np.nan

        for r in req:
            assignment_rows.append({
                "target_participation": rate,
                "actual_participation": actual_rate,
                "day": day,
                "ev": r["name"],
                "usage_type": r["usage_type"],
                "smart": r["name"] in smart_names,
            })

        gp, ep, vmin, vmax, lmax, tmax = [], [], [], [], [], []
        total_reward = total_safety = 0.0
        all_converged = True

        for _ in range(env.episode_length):
            data_idx = int(env.data_idx)
            step_idx = int(env.step_idx)
            clock_hour = float(env.clock_hour)

            # Always run the full policy.  Only the applied EV components are
            # overridden.  The next observation reflects the actual mixed action.
            rl_action = dreamer.action(obs)
            action = mixed_action(
                env, rl_action, smart_names, layout
            )

            obs, reward, terminated, truncated, info = env.step(action)
            converged = bool(
                info.get("converged", env.net.get("converged", False))
            )
            all_converged &= converged

            row = {
                "target_participation": rate,
                "actual_participation": actual_rate,
                "day": day,
                "step": step_idx,
                "data_idx": data_idx,
                "clock_hour": clock_hour,
                "reward": float(reward),
                "safety": float(info.get("s", 0.0)),
                "converged": converged,
            }

            if converged:
                grid_p = float(env.net.res_ext_grid.iloc[0]["p_mw"])
                ev_p = float(sum(float(d.state.P) for d in evs(env)))
                vm = env.net.res_bus.loc[
                    bus_ids, "vm_pu"
                ].to_numpy(float)
                ll = env.net.res_line.loc[
                    line_ids, "loading_percent"
                ].to_numpy(float)
                mt = (
                    float(np.nanmax(
                        env.net.res_trafo[
                            "loading_percent"
                        ].to_numpy(float)
                    ))
                    if len(env.net.res_trafo) else np.nan
                )
                mn = float(np.nanmin(vm))
                mx = float(np.nanmax(vm))
                ml = float(np.nanmax(ll))

                gp.append(grid_p)
                ep.append(ev_p)
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

            step_rows.append(row)
            total_reward += float(reward)
            total_safety += float(info.get("s", 0.0))

            if terminated or truncated:
                break

        # Per-EV service check.
        E = evs(env)
        if len(E) != len(req):
            raise RuntimeError("EV count changed during episode.")
        day_service = []
        for d, r in zip(E, req):
            final_soc = float(d.state.soc)
            sr = {
                "target_participation": rate,
                "actual_participation": actual_rate,
                "day": day,
                "ev": d.name,
                "usage_type": r["usage_type"],
                "smart": d.name in smart_names,
                "initial_soc": r["initial_soc"],
                "final_soc": final_soc,
                "target_soc": r["target_soc"],
                "target_feasible": r["target_feasible"],
                "target_met":
                    bool(final_soc >= r["target_soc"] - 1e-6),
            }
            service_rows.append(sr)
            day_service.append(sr)

        if gp:
            p = np.asarray(gp, dtype=float)
            e = np.asarray(ep, dtype=float)
            day_rows.append({
                "target_participation": rate,
                "actual_participation": actual_rate,
                "day": day,
                "n_evs": n_ev,
                "n_smart_evs": len(smart_names),
                "converged": all_converged,
                "peak_grid_p_mw": float(p.max()),
                "valley_grid_p_mw": float(p.min()),
                "mean_grid_p_mw": float(p.mean()),
                "load_variance":
                    float(np.mean((p - p.mean()) ** 2)),
                "import_energy_mwh":
                    float(np.maximum(p, 0).sum() * env.dt),
                "peak_ev_p_mw": float(e.max()),
                "ev_energy_mwh": float(e.sum() * env.dt),
                "min_vm_pu": float(np.min(vmin)),
                "max_vm_pu": float(np.max(vmax)),
                "max_line_loading_percent": float(np.max(lmax)),
                "max_trafo_loading_percent":
                    float(np.nanmax(tmax)),
                "total_reward": total_reward,
                "total_safety": total_safety,
                "target_feasible_fraction": float(np.mean(
                    [r["target_feasible"] for r in req]
                )),
                "target_met_fraction": float(np.mean(
                    [r["target_met"] for r in day_service]
                )),
                "smart_target_met_fraction": (
                    float(np.mean([
                        r["target_met"] for r in day_service if r["smart"]
                    ]))
                    if any(r["smart"] for r in day_service) else np.nan
                ),
                "uncontrolled_target_met_fraction": (
                    float(np.mean([
                        r["target_met"] for r in day_service if not r["smart"]
                    ]))
                    if any(not r["smart"] for r in day_service) else np.nan
                ),
            })

        if (day + 1) % 25 == 0 or day + 1 == n_days:
            print(
                f"  finished {day + 1}/{n_days}; "
                f"actual participation={100*actual_rate:.1f}%"
            )

    return (
        pd.DataFrame(step_rows),
        pd.DataFrame(day_rows),
        pd.DataFrame(service_rows),
        pd.DataFrame(assignment_rows),
    )


# ---------------------------------------------------------------------
# Checks, summaries, and figures
# ---------------------------------------------------------------------
def check_nested_assignments(assignments):
    """Verify smart sets are nested within every test day."""
    for day, gday in assignments.groupby("day"):
        rates = sorted(gday.target_participation.unique())
        previous = set()
        for rate in rates:
            current = set(
                gday[
                    (gday.target_participation == rate) & gday.smart
                ].ev
            )
            if not previous.issubset(current):
                raise RuntimeError(
                    f"Smart assignments are not nested on day {day} "
                    f"between participation levels."
                )
            previous = current


def summarize(daily):
    rows = []
    baseline = daily[
        np.isclose(daily.target_participation, 0.0)
    ].set_index("day")

    for rate, g in daily.groupby("target_participation"):
        row = {
            "target_participation": rate,
            "mean_actual_participation": g.actual_participation.mean(),
            "mean_n_smart_evs": g.n_smart_evs.mean(),
            "mean_daily_peak_grid_p_mw": g.peak_grid_p_mw.mean(),
            "annual_peak_grid_p_mw": g.peak_grid_p_mw.max(),
            "mean_daily_load_variance": g.load_variance.mean(),
            "mean_daily_ev_energy_mwh": g.ev_energy_mwh.mean(),
            "mean_target_met_fraction": g.target_met_fraction.mean(),
            "annual_min_voltage_pu": g.min_vm_pu.min(),
            "annual_max_voltage_pu": g.max_vm_pu.max(),
            "annual_max_line_loading_percent":
                g.max_line_loading_percent.max(),
            "annual_max_trafo_loading_percent":
                g.max_trafo_loading_percent.max(),
        }

        common = sorted(set(g.day) & set(baseline.index))
        if common:
            gg = g.set_index("day").loc[common]
            bb = baseline.loc[common]
            reduction = (
                bb.peak_grid_p_mw.to_numpy()
                - gg.peak_grid_p_mw.to_numpy()
            )
            row["mean_peak_reduction_vs_0pct_mw"] = float(
                np.mean(reduction)
            )
            row["mean_peak_reduction_vs_0pct_percent"] = float(
                np.mean(
                    100.0 * reduction
                    / bb.peak_grid_p_mw.to_numpy()
                )
            )
            row["days_with_lower_peak_fraction"] = float(
                np.mean(reduction > 0)
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("target_participation")


def save_figures(steps, daily, summary, out):
    # Use actual participation on x-axis because stratification/rounding means
    # the realized percentage need not equal the requested percentage exactly.
    x = 100.0 * summary.mean_actual_participation.to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, summary.mean_daily_peak_grid_p_mw, marker="o")
    ax.set_xlabel("Smart-charging participation (%)")
    ax.set_ylabel("Mean daily peak grid import (MW)")
    ax.set_title("Peak demand vs smart-charging participation")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "participation_peak_vs_rate.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        x, summary.mean_peak_reduction_vs_0pct_percent,
        marker="o",
    )
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_xlabel("Smart-charging participation (%)")
    ax.set_ylabel("Mean peak reduction vs 0% smart (%)")
    ax.set_title("Peak reduction vs smart-charging participation")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(
        out / "participation_peak_reduction_vs_rate.png", dpi=220
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, summary.mean_daily_load_variance, marker="o")
    ax.set_xlabel("Smart-charging participation (%)")
    ax.set_ylabel(r"Mean daily load variance (MW$^2$)")
    ax.set_title("Load smoothing vs smart-charging participation")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "participation_variance_vs_rate.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        x, 100.0 * summary.mean_target_met_fraction, marker="o"
    )
    ax.set_xlabel("Smart-charging participation (%)")
    ax.set_ylabel("EV target-SOC success (%)")
    ax.set_title("EV service vs smart-charging participation")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(
        out / "participation_target_service_vs_rate.png", dpi=220
    )
    plt.close(fig)

    # Daily peak: one curve per requested participation level.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for rate, g in daily.groupby("target_participation"):
        g = g.sort_values("day")
        actual = 100.0 * g.actual_participation.mean()
        ax.plot(
            g.day, g.peak_grid_p_mw,
            label=f"{actual:.1f}% smart",
        )
    ax.set_xlabel("Test day")
    ax.set_ylabel("Peak grid import (MW)")
    ax.set_title("Daily peak demand by smart-charging participation")
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "participation_daily_peak.png", dpi=220)
    plt.close(fig)

    # Mean load profile.
    valid = steps[steps.converged]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for rate, g in valid.groupby("target_participation"):
        profile = g.groupby("clock_hour").grid_p_mw.mean().sort_index()
        actual = 100.0 * g.actual_participation.mean()
        ax.plot(
            profile.index, profile.values,
            label=f"{actual:.1f}% smart",
        )
    ax.set_xlabel("Clock hour")
    ax.set_ylabel("Mean grid import (MW)")
    ax.set_title("Mean load profile by smart-charging participation")
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(
        out / "participation_mean_load_profiles.png", dpi=220
    )
    plt.close(fig)

    # Mean EV charging profile.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for rate, g in valid.groupby("target_participation"):
        profile = g.groupby("clock_hour").ev_p_mw.mean().sort_index()
        actual = 100.0 * g.actual_participation.mean()
        ax.plot(
            profile.index, profile.values,
            label=f"{actual:.1f}% smart",
        )
    ax.set_xlabel("Clock hour")
    ax.set_ylabel("Mean EV charging power (MW)")
    ax.set_title("EV charging profile by smart-charging participation")
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(
        out / "participation_mean_ev_profiles.png", dpi=220
    )
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        required=True,
        help="RLlib DreamerV3 checkpoint directory trained on low EV level.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--assignment-seed",
        type=int,
        default=2026,
        help="Seed used only for stratified smart-EV assignment.",
    )
    p.add_argument(
        "--rates",
        default="0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated smart participation fractions in [0,1].",
    )
    p.add_argument(
        "--output-dir",
        default="results_smart_participation",
    )
    p.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Debugging limit; omit for all complete test days.",
    )
    args = p.parse_args()

    rates = parse_rates(args.rates)
    if 0.0 not in rates:
        raise ValueError(
            "Include 0 in --rates so peak reduction has an uncontrolled "
            "participation reference."
        )
    if 1.0 not in rates:
        print(
            "WARNING: 100% smart participation is not included in --rates."
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True)
    algo = None
    try:
        checkpoint = str(Path(args.checkpoint).expanduser().resolve())
        print(f"Restoring: {checkpoint}")
        algo = Algorithm.from_checkpoint(checkpoint)
        print("Restored algorithm:", type(algo).__name__)
        print("Ray/RLlib version:", ray.__version__)
        print(
            "Checkpoint normalize_actions:",
            getattr(algo.config, "normalize_actions", None),
        )

        # One fresh env per rate avoids state leakage between participation
        # conditions.  All use exactly the same LOW-EV configuration.
        all_steps, all_days, all_service, all_assign = [], [], [], []

        reference_req_by_day = {}

        for rate in rates:
            env = make_test_env()
            n_days = complete_test_days(env)
            if args.max_days is not None:
                n_days = min(n_days, args.max_days)

            # Independent pre-check: seeded reset must produce exactly the same
            # EV realization at every participation level.
            for day in range(n_days):
                env.reset(seed=args.seed + day)
                now = snapshot_ev_requirements(env)
                if day not in reference_req_by_day:
                    reference_req_by_day[day] = now
                else:
                    verify_same_ev_realization(
                        now, reference_req_by_day[day]
                    )

            s, d, e, a = evaluate_one_rate(
                env=env,
                algo=algo,
                rate=rate,
                seed=args.seed,
                assignment_seed=args.assignment_seed,
                max_days=args.max_days,
            )
            all_steps.append(s)
            all_days.append(d)
            all_service.append(e)
            all_assign.append(a)

        steps = pd.concat(all_steps, ignore_index=True)
        daily = pd.concat(all_days, ignore_index=True)
        service = pd.concat(all_service, ignore_index=True)
        assignments = pd.concat(all_assign, ignore_index=True)

        # Strong post-run checks.
        check_nested_assignments(assignments)

        day_sets = [
            set(g.day.tolist())
            for _, g in daily.groupby("target_participation")
        ]
        if not all(x == day_sets[0] for x in day_sets[1:]):
            raise RuntimeError(
                "Participation levels do not contain identical test days."
            )

        summary = summarize(daily)

        steps.to_csv(
            out / "participation_step_metrics.csv", index=False
        )
        daily.to_csv(
            out / "participation_daily_summary.csv", index=False
        )
        service.to_csv(
            out / "participation_ev_service.csv", index=False
        )
        assignments.to_csv(
            out / "participation_assignment.csv", index=False
        )
        summary.to_csv(
            out / "participation_summary.csv", index=False
        )

        save_figures(steps, daily, summary, out)

        print("\nSmart-charging participation summary")
        print(summary.to_string(index=False))
        print(f"\nAll results saved to: {out.resolve()}")

    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
