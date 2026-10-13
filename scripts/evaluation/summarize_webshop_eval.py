"""Summarize trajectory-level WebShop metrics from verl JSONL dumps."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No evaluation rows found in {args.input_dir}")

    episodes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        trajectory_id = str(row.get("traj_uid") or row.get("uid"))
        episodes.setdefault(trajectory_id, []).append(row)

    successes: list[float] = []
    scores: list[float] = []
    lengths: list[float] = []
    valid_flags: list[float] = []
    repeated = 0
    transitions = 0
    for trajectory_rows in episodes.values():
        trajectory_rows.sort(key=lambda row: int(row.get("turn_step", 0)))
        successes.append(max(float(row.get("trajectory_success", 0.0)) for row in trajectory_rows))
        scores.append(max(float(row.get("webshop_task_score", 0.0)) for row in trajectory_rows))
        lengths.append(max(float(row.get("episode_lengths", len(trajectory_rows))) for row in trajectory_rows))
        actions = []
        for row in trajectory_rows:
            valid_flags.append(float(bool(row.get("is_action_valid", False))))
            action = str(row.get("projected_action") or row.get("raw_model_response") or "")
            actions.append(action)
        for previous, current in zip(actions, actions[1:]):
            transitions += 1
            repeated += int(bool(current) and current == previous)

    summary = {
        "label": args.label,
        "model_path": str(args.model_path.resolve()),
        "trajectory_count": len(episodes),
        "step_count": len(rows),
        "success_rate": mean(successes),
        "mean_task_score": mean(scores),
        "valid_action_rate": mean(valid_flags),
        "mean_episode_length": mean(lengths),
        "consecutive_repeat_rate": repeated / transitions if transitions else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
