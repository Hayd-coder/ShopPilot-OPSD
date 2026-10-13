#!/usr/bin/env python3
"""Filter replayed WebShop trajectories and split them without trajectory leakage."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import OrderedDict
from pathlib import Path


PRODUCT_CLICK = re.compile(r"click\[(b[0-9a-z]{9})\]", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    trajectories: OrderedDict[str, list[dict]] = OrderedDict()
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            trajectories.setdefault(record["trajectory_id"], []).append(record)

    accepted: list[str] = []
    rejected: list[dict] = []
    for trajectory_id, records in trajectories.items():
        target_asin = trajectory_id.split("::", 1)[0].lower()
        clicked_asins = []
        for record in records:
            match = PRODUCT_CLICK.fullmatch(record["expert_action"].strip())
            if match:
                clicked_asins.append(match.group(1).lower())
        clicked_asin = clicked_asins[0] if clicked_asins else None
        if clicked_asin != target_asin:
            rejected.append(
                {
                    "trajectory_id": trajectory_id,
                    "reason": "clicked_product_asin_mismatch",
                    "target_asin": target_asin,
                    "clicked_asin": clicked_asin,
                    "task": records[0]["task"],
                }
            )
        else:
            accepted.append(trajectory_id)

    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    validation_count = max(1, round(len(accepted) * args.validation_ratio))
    validation_ids = set(accepted[:validation_count])
    train_ids = set(accepted[validation_count:])

    train_records = [
        record
        for trajectory_id, records in trajectories.items()
        if trajectory_id in train_ids
        for record in records
    ]
    validation_records = [
        record
        for trajectory_id, records in trajectories.items()
        if trajectory_id in validation_ids
        for record in records
    ]
    write_jsonl(args.train_output, train_records)
    write_jsonl(args.validation_output, validation_records)
    write_jsonl(args.rejected_output, rejected)

    print(
        json.dumps(
            {
                "input_successful_trajectories": len(trajectories),
                "accepted_trajectories": len(accepted),
                "rejected_trajectories": len(rejected),
                "train_trajectories": len(train_ids),
                "validation_trajectories": len(validation_ids),
                "train_step_samples": len(train_records),
                "validation_step_samples": len(validation_records),
                "trajectory_overlap": len(train_ids & validation_ids),
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
