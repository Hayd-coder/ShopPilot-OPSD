#!/usr/bin/env python3
"""Build a focused, low-complexity WebShop correction curriculum.

The curriculum keeps successful trajectories that teach query reformulation or
one-candidate verification, removes long pagination, caps repeated target
products, and retains a small direct-purchase anchor to avoid forgetting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-next-actions", type=int, default=1)
    parser.add_argument("--max-trajectories-per-asin", type=int, default=20)
    parser.add_argument("--direct-anchor-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def read_trajectories(path: Path) -> list[list[dict[str, Any]]]:
    trajectories: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["turn"]) == 0 and current:
                trajectories.append(current)
                current = []
            current.append(row)
    if current:
        trajectories.append(current)
    for trajectory in trajectories:
        turns = [int(row["turn"]) for row in trajectory]
        if turns != list(range(len(trajectory))):
            raise ValueError(f"Broken turns for {trajectory[0]['trajectory_id']}: {turns}")
    return trajectories


def has_decision(trajectory: list[dict[str, Any]], decision: str) -> bool:
    return any(str(row.get("decision_type")) == decision for row in trajectory)


def actions(trajectory: list[dict[str, Any]]) -> list[str]:
    return [str(row["expert_action"]).strip().lower() for row in trajectory]


def stable_key(seed: int, trajectory: list[dict[str, Any]]) -> str:
    first = trajectory[0]
    identity = f"{first['trajectory_id']}:{first.get('goal_index')}"
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.direct_anchor_fraction < 1.0:
        raise ValueError("direct-anchor-fraction must be in [0, 1)")

    trajectories = read_trajectories(args.input_jsonl)
    eval_manifest = json.loads(args.eval_manifest.read_text(encoding="utf-8"))
    eval_asins = {str(value).upper() for value in eval_manifest["target_asins"]}

    eligible: list[list[dict[str, Any]]] = []
    rejected = Counter()
    for trajectory in trajectories:
        first = trajectory[0]
        target_asin = str(first["target_asin"]).upper()
        trajectory_actions = actions(trajectory)
        rewrite = has_decision(trajectory, "reformulate_search")
        compare = has_decision(trajectory, "reject_candidate")
        search_actions = [action for action in trajectory_actions if action.startswith("search[")]

        if target_asin in eval_asins:
            rejected["strict_eval_asin_overlap"] += 1
            continue
        if not all(bool(row.get("independent_success")) for row in trajectory):
            rejected["independent_check_failed"] += 1
            continue
        if trajectory_actions.count("click[next >]") > args.max_next_actions:
            rejected["too_many_next_actions"] += 1
            continue
        if rewrite and (len(search_actions) < 2 or len(set(search_actions)) < 2):
            rejected["rewrite_did_not_change_query"] += 1
            continue
        if compare and "click[< prev]" not in trajectory_actions:
            rejected["comparison_did_not_return"] += 1
            continue
        eligible.append(trajectory)

    asin_frequency = Counter(str(t[0]["target_asin"]).upper() for t in eligible)

    def selection_key(trajectory: list[dict[str, Any]]) -> tuple[int, str]:
        asin = str(trajectory[0]["target_asin"]).upper()
        return asin_frequency[asin], stable_key(args.seed, trajectory)

    hard = sorted(
        [
            trajectory
            for trajectory in eligible
            if has_decision(trajectory, "reformulate_search")
            or has_decision(trajectory, "reject_candidate")
        ],
        key=selection_key,
    )
    direct = sorted(
        [
            trajectory
            for trajectory in eligible
            if not has_decision(trajectory, "reformulate_search")
            and not has_decision(trajectory, "reject_candidate")
        ],
        key=selection_key,
    )

    per_asin = Counter()
    selected_hard: list[list[dict[str, Any]]] = []
    for trajectory in hard:
        asin = str(trajectory[0]["target_asin"]).upper()
        if per_asin[asin] >= args.max_trajectories_per_asin:
            continue
        selected_hard.append(trajectory)
        per_asin[asin] += 1

    desired_direct = round(
        len(selected_hard)
        * args.direct_anchor_fraction
        / (1.0 - args.direct_anchor_fraction)
    )
    selected_direct: list[list[dict[str, Any]]] = []
    for trajectory in direct:
        if len(selected_direct) >= desired_direct:
            break
        asin = str(trajectory[0]["target_asin"]).upper()
        if per_asin[asin] >= args.max_trajectories_per_asin:
            continue
        selected_direct.append(trajectory)
        per_asin[asin] += 1

    selected = selected_hard + selected_direct
    selected.sort(key=lambda trajectory: stable_key(args.seed + 1, trajectory))
    selected_rows = [row for trajectory in selected for row in trajectory]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rewrite_count = sum(
        has_decision(trajectory, "reformulate_search") for trajectory in selected
    )
    comparison_count = sum(
        has_decision(trajectory, "reject_candidate") for trajectory in selected
    )
    direct_only_count = len(selected_direct)
    manifest = {
        "source": str(args.input_jsonl.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "input_trajectories": len(trajectories),
        "eligible_trajectories": len(eligible),
        "selected_trajectories": len(selected),
        "selected_step_samples": len(selected_rows),
        "selected_unique_asins": len(per_asin),
        "max_trajectories_per_asin": max(per_asin.values(), default=0),
        "direct_only_trajectories": direct_only_count,
        "direct_only_fraction": direct_only_count / len(selected),
        "rewrite_trajectories": rewrite_count,
        "rewrite_fraction": rewrite_count / len(selected),
        "comparison_trajectories": comparison_count,
        "comparison_fraction": comparison_count / len(selected),
        "max_next_actions": args.max_next_actions,
        "strict_eval_asin_overlap": len(set(per_asin) & eval_asins),
        "trajectory_modes": dict(
            Counter(str(t[0].get("trajectory_mode", "unknown")) for t in selected)
        ),
        "rejected": dict(rejected),
        "output_sha256": file_sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
