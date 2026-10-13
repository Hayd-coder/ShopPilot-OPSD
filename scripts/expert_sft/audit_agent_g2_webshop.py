#!/usr/bin/env python3
"""Audit Agent-G2 WebShop trajectories and build leakage-safe manifests.

This script intentionally uses only the Python standard library.  It rebuilds
the target keys of the bundled 1k-product WebShop environment without starting
Lucene or loading a language model, so it is safe to run on a laptop CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def normalize_text(value: Any) -> str:
    return " ".join(str(value).lower().split())


def normalize_option_value(value: Any) -> str:
    # Match WebShop's load_products normalization in engine.py.
    return normalize_text(str(value).strip().replace("/", " | "))


def target_key(asin: str, option_values: Iterable[Any]) -> str:
    counts = Counter(normalize_text(value) for value in option_values)
    option_text = "|".join(
        f"{value}#{count}" for value, count in sorted(counts.items())
    )
    return f"{str(asin).upper()}::{option_text}"


def trajectory_asin(trajectory_id: str) -> str:
    return trajectory_id.split("::", 1)[0].upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def balanced_subset(ids: list[str], limit: int, seed: int) -> list[str]:
    """Select complete trajectories while maximizing ASIN coverage first."""
    groups: dict[str, list[str]] = defaultdict(list)
    for trajectory_id in ids:
        groups[trajectory_asin(trajectory_id)].append(trajectory_id)

    asin_order = sorted(groups, key=lambda asin: deterministic_rank(asin, seed))
    for asin in asin_order:
        groups[asin].sort(key=lambda item: deterministic_rank(item, seed))

    selected: list[str] = []
    depth = 0
    while len(selected) < min(limit, len(ids)):
        added = False
        for asin in asin_order:
            if depth < len(groups[asin]):
                selected.append(groups[asin][depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def build_small_goal_keys(
    products: list[dict[str, Any]],
    attributes: dict[str, dict[str, Any]],
    goal_seed: int,
) -> tuple[list[str], dict[str, int]]:
    """Reproduce get_synthetic_goals target-key order, then its shuffle."""
    keys: list[str] = []
    products_without_instruction = 0
    products_without_attributes = 0

    for product in products:
        asin = str(product["asin"]).upper()
        attr = attributes.get(asin, {})
        instruction = attr.get("instruction")
        instruction_attributes = attr.get("instruction_attributes") or []
        if not instruction:
            products_without_instruction += 1
            continue
        if not instruction_attributes:
            # Upstream get_synthetic_goals asserts this invariant.  Keep the
            # audit usable and report violations instead of crashing here.
            products_without_attributes += 1
            continue

        options: dict[str, list[str]] = {}
        for option_name, option_contents in (
            product.get("customization_options") or {}
        ).items():
            if not option_contents:
                continue
            values = [
                normalize_option_value(option["value"])
                for option in option_contents
                if option.get("value") is not None
            ]
            if values:
                options[normalize_text(option_name)] = values

        option_names = sorted(options)
        combinations = itertools.product(
            *(options[name] for name in option_names)
        )
        for combination in combinations:
            keys.append(target_key(asin, combination))

    random.Random(goal_seed).shuffle(keys)
    diagnostics = {
        "products_without_instruction": products_without_instruction,
        "products_with_instruction_but_no_attributes": products_without_attributes,
    }
    return keys, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--expert-data",
        type=Path,
        default=repo_root / "data/agent_g2/webshop_sft_data.json",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=repo_root / "datasets/webshop-small/items_shuffle_1000.json",
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=repo_root / "datasets/webshop-small/items_ins_v2_1000.json",
    )
    parser.add_argument("--goal-seed", type=int, default=0)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--pilot-trajectories", type=int, default=500)
    parser.add_argument("--main-trajectories", type=int, default=2000)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts/expert_sft",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.expert_data, args.products, args.attributes):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = json.loads(args.expert_data.read_text(encoding="utf-8"))
    trajectories = payload["trajectories"]
    products = json.loads(args.products.read_text(encoding="utf-8"))
    attributes = json.loads(args.attributes.read_text(encoding="utf-8"))

    malformed = [
        item.get("id", "<missing-id>")
        for item in trajectories
        if not item.get("id")
        or not isinstance(item.get("actions"), list)
        or not isinstance(item.get("think"), list)
        or len(item["actions"]) != len(item["think"])
        or not item["actions"]
    ]
    duplicate_ids = [
        item for item, count in Counter(t["id"] for t in trajectories).items()
        if count > 1
    ]

    goal_keys, goal_diagnostics = build_small_goal_keys(
        products=products,
        attributes=attributes,
        goal_seed=args.goal_seed,
    )
    validation_size = min(args.validation_size, len(goal_keys))
    validation_keys = set(goal_keys[:validation_size])
    raw_training_goal_keys = set(goal_keys[validation_size:])
    # A few synthetically generated goals have duplicate target keys.  If the
    # same ASIN/options key occurs on both sides of the index split, treating
    # its later index as training would still leak the validation target.
    cross_split_duplicate_keys = validation_keys & raw_training_goal_keys
    training_goal_keys = raw_training_goal_keys - validation_keys
    expert_by_id = {item["id"]: item for item in trajectories}
    expert_keys = set(expert_by_id)

    leakage_ids = sorted(expert_keys & validation_keys)
    eligible_ids = sorted((expert_keys & training_goal_keys) - validation_keys)
    outside_environment_ids = sorted(expert_keys - set(goal_keys))
    missing_expert_train_keys = sorted(training_goal_keys - expert_keys)

    pilot_ids = balanced_subset(
        eligible_ids, args.pilot_trajectories, args.selection_seed
    )
    main_ids = balanced_subset(
        eligible_ids, args.main_trajectories, args.selection_seed
    )

    action_lengths = Counter(len(item["actions"]) for item in trajectories)
    report = {
        "source": {
            "expert_data": str(args.expert_data.resolve()),
            "expert_sha256": sha256_file(args.expert_data),
            "products_sha256": sha256_file(args.products),
            "attributes_sha256": sha256_file(args.attributes),
        },
        "configuration": {
            "goal_seed": args.goal_seed,
            "validation_size": validation_size,
            "selection_seed": args.selection_seed,
        },
        "counts": {
            "products": len(products),
            "environment_goals": len(goal_keys),
            "unique_environment_goal_keys": len(set(goal_keys)),
            "expert_trajectories": len(trajectories),
            "expert_action_steps": sum(len(item["actions"]) for item in trajectories),
            "expert_unique_asins": len({trajectory_asin(item["id"]) for item in trajectories}),
            "validation_goal_keys": len(validation_keys),
            "raw_training_goal_keys": len(raw_training_goal_keys),
            "cross_split_duplicate_goal_keys": len(cross_split_duplicate_keys),
            "leakage_safe_training_goal_keys": len(training_goal_keys),
            "expert_validation_leakage": len(leakage_ids),
            "eligible_train_expert_trajectories": len(eligible_ids),
            "expert_outside_environment": len(outside_environment_ids),
            "training_goals_without_expert": len(missing_expert_train_keys),
            "pilot_trajectories": len(pilot_ids),
            "pilot_action_steps": sum(len(expert_by_id[item]["actions"]) for item in pilot_ids),
            "pilot_unique_asins": len({trajectory_asin(item) for item in pilot_ids}),
            "main_trajectories": len(main_ids),
            "main_action_steps": sum(len(expert_by_id[item]["actions"]) for item in main_ids),
            "main_unique_asins": len({trajectory_asin(item) for item in main_ids}),
            "malformed_trajectories": len(malformed),
            "duplicate_trajectory_ids": len(duplicate_ids),
        },
        "action_length_distribution": dict(sorted(action_lengths.items())),
        "goal_diagnostics": goal_diagnostics,
        "examples": {
            "leakage_ids": leakage_ids[:10],
            "cross_split_duplicate_goal_keys": sorted(cross_split_duplicate_keys),
            "outside_environment_ids": outside_environment_ids[:10],
            "malformed_ids": malformed[:10],
            "duplicate_ids": duplicate_ids[:10],
        },
    }
    manifest = {
        "metadata": report,
        "validation_target_keys": sorted(validation_keys),
        "excluded_validation_trajectory_ids": leakage_ids,
        "eligible_train_trajectory_ids": eligible_ids,
        "pilot_trajectory_ids": pilot_ids,
        "main_trajectory_ids": main_ids,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "agent_g2_audit.json"
    manifest_path = args.output_dir / "agent_g2_split_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"audit_report={report_path}")
    print(f"split_manifest={manifest_path}")

    if malformed or duplicate_ids:
        raise SystemExit("Expert data integrity check failed; inspect the audit report.")
    if outside_environment_ids:
        raise SystemExit(
            "Some expert target keys are absent from the current 1k environment."
        )


if __name__ == "__main__":
    main()
