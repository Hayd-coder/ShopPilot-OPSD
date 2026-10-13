"""Create a deterministic trajectory-level subset of WebShop SFT JSONL.

The input contains one JSON object per decision step, so sampling individual
rows would create broken partial trajectories.  This utility samples whole
trajectories while preserving the original trajectory-length distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-file",
        type=Path,
        help="Optional JSONL used to verify zero trajectory overlap.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "trajectory_id" not in record:
                raise ValueError(f"{path}:{line_number} has no trajectory_id")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, trajectory_id: str) -> str:
    return hashlib.sha256(f"{seed}:{trajectory_id}".encode()).hexdigest()


def asin_from_trajectory_id(trajectory_id: str) -> str:
    return trajectory_id.split("::", 1)[0].upper()


def allocate_quotas(
    counts: dict[int, int], fraction: float, target_total: int
) -> dict[int, int]:
    exact = {length: count * fraction for length, count in counts.items()}
    quotas = {length: math.floor(value) for length, value in exact.items()}
    remaining = target_total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda length: (exact[length] - quotas[length], counts[length], -length),
        reverse=True,
    )
    for length in order[:remaining]:
        quotas[length] += 1
    if sum(quotas.values()) != target_total:
        raise RuntimeError("Failed to allocate the requested number of trajectories")
    return quotas


def main() -> None:
    args = parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")

    records = load_jsonl(args.input)
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        trajectories[str(record["trajectory_id"])].append(record)

    for trajectory_id, steps in trajectories.items():
        turns = [int(step.get("turn", -1)) for step in steps]
        if sorted(turns) != list(range(len(steps))):
            raise ValueError(
                f"Trajectory {trajectory_id!r} has invalid turns: {sorted(turns)}"
            )

    by_length: dict[int, list[str]] = defaultdict(list)
    for trajectory_id, steps in trajectories.items():
        by_length[len(steps)].append(trajectory_id)

    target_total = max(1, round(len(trajectories) * args.fraction))
    length_counts = {length: len(ids) for length, ids in by_length.items()}
    quotas = allocate_quotas(length_counts, args.fraction, target_total)

    selected_ids: set[str] = set()
    for length, trajectory_ids in by_length.items():
        ranked = sorted(
            trajectory_ids,
            key=lambda trajectory_id: stable_rank(args.seed, trajectory_id),
        )
        selected_ids.update(ranked[: quotas[length]])

    selected_records = [
        record for record in records if str(record["trajectory_id"]) in selected_ids
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    validation_ids: set[str] = set()
    if args.validation_file:
        validation_ids = {
            str(record["trajectory_id"])
            for record in load_jsonl(args.validation_file)
        }
    overlap = sorted(selected_ids & validation_ids)
    if overlap:
        raise RuntimeError(
            f"Selected subset overlaps validation by {len(overlap)} trajectories"
        )

    validation_asins = {asin_from_trajectory_id(item) for item in validation_ids}
    selected_asins = {asin_from_trajectory_id(item) for item in selected_ids}
    manifest = {
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "fraction": args.fraction,
        "seed": args.seed,
        "source_sha256": file_sha256(args.input),
        "output_sha256": file_sha256(args.output),
        "source_step_count": len(records),
        "source_trajectory_count": len(trajectories),
        "selected_step_count": len(selected_records),
        "selected_trajectory_count": len(selected_ids),
        "source_length_distribution": dict(sorted(length_counts.items())),
        "selected_length_distribution": dict(
            sorted(Counter(len(trajectories[item]) for item in selected_ids).items())
        ),
        "validation_trajectory_overlap": 0,
        "validation_asin_overlap": len(selected_asins & validation_asins),
        "selected_trajectory_ids": sorted(selected_ids),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_trajectory_ids"}, indent=2))


if __name__ == "__main__":
    main()
