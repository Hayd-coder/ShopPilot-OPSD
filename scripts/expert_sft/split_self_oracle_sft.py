#!/usr/bin/env python3
"""Create a train/validation split for self-oracle WebShop SFT."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--unused-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-trajectories", type=int, default=442)
    parser.add_argument("--validation-trajectories", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-asin-overlap",
        action="store_true",
        help=(
            "Split by exact trajectory/task key while allowing different tasks "
            "for the same product ASIN in train and validation."
        ),
    )
    return parser.parse_args()


def load_trajectories(path: Path) -> dict[str, list[dict[str, Any]]]:
    trajectories: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            trajectory_id = str(record["trajectory_id"])
            trajectories.setdefault(trajectory_id, []).append(record)
    for trajectory_id, records in trajectories.items():
        records.sort(key=lambda item: int(item["turn"]))
        turns = [int(record["turn"]) for record in records]
        if turns != list(range(len(records))):
            raise ValueError(f"Non-contiguous turns in {trajectory_id}: {turns}")
        if not all(record.get("independent_success") is True for record in records):
            raise ValueError(f"Unverified trajectory in input: {trajectory_id}")
        actions = [str(record["expert_action"]).lower() for record in records]
        target = str(records[0]["target_asin"]).lower()
        if f"click[{target}]" not in actions or "click[buy now]" not in actions:
            raise ValueError(f"Incomplete successful path in {trajectory_id}")
        if records[0].get("trajectory_mode") == "compare_reject_then_purchase":
            decisions = {record.get("decision_type") for record in records}
            if not {"inspect_candidate", "reject_candidate"} <= decisions:
                raise ValueError(f"Incomplete comparison path in {trajectory_id}")
    return trajectories


def write_split(
    path: Path,
    trajectory_ids: list[str],
    trajectories: dict[str, list[dict[str, Any]]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8") as handle:
        for trajectory_id in trajectory_ids:
            for record in trajectories[trajectory_id]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows += 1
    return rows


def asin_for(records: list[dict[str, Any]]) -> str:
    return str(records[0]["target_asin"]).upper()


def mode_counts(
    trajectory_ids: list[str], trajectories: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    return dict(
        Counter(
            str(trajectories[trajectory_id][0].get("trajectory_mode", "unknown"))
            for trajectory_id in trajectory_ids
        )
    )


def main() -> None:
    args = parse_args()
    trajectories = load_trajectories(args.input)
    rng = random.Random(args.seed)
    if args.allow_asin_overlap:
        trajectory_ids = list(trajectories)
        rng.shuffle(trajectory_ids)
        validation_ids = trajectory_ids[: args.validation_trajectories]
        validation_asins = {
            asin_for(trajectories[item]) for item in validation_ids
        }
        train_candidates = trajectory_ids[args.validation_trajectories :]
    else:
        by_asin: dict[str, list[str]] = defaultdict(list)
        for trajectory_id, records in trajectories.items():
            by_asin[asin_for(records)].append(trajectory_id)
        asin_groups = list(by_asin.items())
        rng.shuffle(asin_groups)

        validation_ids = []
        validation_asins = set()
        for asin, group_ids in asin_groups:
            shuffled_group = list(group_ids)
            rng.shuffle(shuffled_group)
            remaining = args.validation_trajectories - len(validation_ids)
            if remaining <= 0:
                break
            if len(shuffled_group) <= remaining:
                validation_ids.extend(shuffled_group)
                validation_asins.add(asin)
        if len(validation_ids) != args.validation_trajectories:
            raise RuntimeError(
                f"Could not make exact validation split: {len(validation_ids)} / "
                f"{args.validation_trajectories}"
            )

        train_candidates = [
            trajectory_id
            for trajectory_id, records in trajectories.items()
            if asin_for(records) not in validation_asins
        ]
        rng.shuffle(train_candidates)
    if len(train_candidates) < args.train_trajectories:
        raise RuntimeError(
            f"Only {len(train_candidates)} ASIN-clean train candidates for "
            f"{args.train_trajectories} requested"
        )
    train_ids = train_candidates[: args.train_trajectories]
    selected = set(train_ids) | set(validation_ids)
    unused_ids = [item for item in trajectories if item not in selected]

    train_rows = write_split(args.train_output, train_ids, trajectories)
    validation_rows = write_split(
        args.validation_output, validation_ids, trajectories
    )
    unused_rows = write_split(args.unused_output, unused_ids, trajectories)

    train_asins = {asin_for(trajectories[item]) for item in train_ids}
    manifest = {
        "input_trajectories": len(trajectories),
        "input_step_samples": sum(len(records) for records in trajectories.values()),
        "train_trajectories": len(train_ids),
        "train_step_samples": train_rows,
        "validation_trajectories": len(validation_ids),
        "validation_step_samples": validation_rows,
        "unused_trajectories": len(unused_ids),
        "unused_step_samples": unused_rows,
        "train_unique_asins": len(train_asins),
        "validation_unique_asins": len(validation_asins),
        "train_validation_trajectory_overlap": len(set(train_ids) & set(validation_ids)),
        "train_validation_asin_overlap": len(train_asins & validation_asins),
        "allow_asin_overlap": bool(args.allow_asin_overlap),
        "train_mode_counts": mode_counts(train_ids, trajectories),
        "validation_mode_counts": mode_counts(validation_ids, trajectories),
        "seed": args.seed,
        "train_output": str(args.train_output.resolve()),
        "validation_output": str(args.validation_output.resolve()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
