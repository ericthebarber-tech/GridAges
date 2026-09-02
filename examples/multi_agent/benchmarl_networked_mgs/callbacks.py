"""CSV callbacks for physical grid metrics and BenchMARL optimization metrics."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from benchmarl.experiment import Callback


def append_rows(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


class GridMetricsCallback(Callback):
    def on_setup(self):
        self.output = Path(self.experiment.folder_name)
        self.evaluation_index = 0

    def on_train_end(self, training_td, group):
        rows = []
        for metric, value in training_td.items():
            if hasattr(value, "to"):
                value = value.to("cpu", non_blocking=False).float().mean().item()
            rows.append(
                {
                    "timestep": self.experiment.total_frames,
                    "group": group,
                    "metric": str(metric),
                    "value": float(value),
                }
            )
        append_rows(self.output / "updates.csv", rows)

    def on_evaluation_end(self, rollouts):
        rows = []
        agents = self.experiment.group_map["agents"]
        for episode, rollout in enumerate(rollouts):
            info = rollout.get(("next", "agents", "info"))
            raw_reward = info.get("raw_reward").detach().cpu().numpy()
            operating_cost = info.get("operating_cost").detach().cpu().numpy()
            safety = info.get("safety").detach().cpu().numpy()
            converged = info.get("converged").detach().cpu().numpy()
            for index, agent in enumerate(agents):
                rows.append(
                    {
                        "timestep": self.experiment.total_frames,
                        "evaluation": self.evaluation_index,
                        "episode": episode,
                        "agent": agent,
                        "reward": float(raw_reward[:, index].sum()),
                        "operating_cost": float(operating_cost[:, index].sum()),
                        "safety": float(safety[:, index].sum()),
                        "convergence_rate": float(
                            np.asarray(converged[:, index]).mean()
                        ),
                        "steps": int(rollout.shape[0]),
                    }
                )
            agent_rows = rows[-len(agents) :]
            rows.append(
                {
                    "timestep": self.experiment.total_frames,
                    "evaluation": self.evaluation_index,
                    "episode": episode,
                    "agent": "system",
                    "reward": sum(item["reward"] for item in agent_rows),
                    "operating_cost": sum(
                        item["operating_cost"] for item in agent_rows
                    ),
                    "safety": sum(item["safety"] for item in agent_rows),
                    "convergence_rate": float(
                        np.mean(
                            [item["convergence_rate"] for item in agent_rows]
                        )
                    ),
                    "steps": int(rollout.shape[0]),
                }
            )
        append_rows(self.output / "evaluation.csv", rows)
        self.evaluation_index += 1
