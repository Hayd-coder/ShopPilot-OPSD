#!/usr/bin/env python3
"""Replay Agent-G2 expert actions in WebShop and emit step-level SFT data.

The replay itself is CPU-only.  It requires a working WebShop installation,
including the 1k Lucene index, but it never loads Qwen or uses CUDA.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def install_sparse_pyserini_shim() -> None:
    """Load only Lucene sparse search, avoiding unused Torch dense encoders."""
    import pyserini

    package_root = Path(pyserini.__file__).resolve().parent
    search_package = types.ModuleType("pyserini.search")
    search_package.__path__ = [str(package_root / "search")]
    search_package.__package__ = "pyserini.search"
    sys.modules["pyserini.search"] = search_package

    base = importlib.import_module("pyserini.search._base")
    for name in ("JQuery", "JQueryGenerator"):
        setattr(search_package, name, getattr(base, name))

    lucene_package = types.ModuleType("pyserini.search.lucene")
    lucene_package.__path__ = [str(package_root / "search/lucene")]
    lucene_package.__package__ = "pyserini.search.lucene"
    sys.modules["pyserini.search.lucene"] = lucene_package

    searcher = importlib.import_module("pyserini.search.lucene._searcher")
    for name in (
        "JLuceneSearcherResult",
        "LuceneSimilarities",
        "LuceneFusionSearcher",
        "LuceneSearcher",
    ):
        setattr(lucene_package, name, getattr(searcher, name))


def normalize_text(value: Any) -> str:
    return " ".join(str(value).lower().split())


def goal_target_key(goal: dict[str, Any]) -> str:
    values = goal.get("goal_options") or {}
    if isinstance(values, dict):
        values = values.values()
    counts = Counter(normalize_text(value) for value in values)
    options = "|".join(
        f"{value}#{count}" for value, count in sorted(counts.items())
    )
    return f"{str(goal.get('asin', '')).upper()}::{options}"


def parse_action(action: str) -> tuple[str | None, str | None]:
    match = re.fullmatch(
        r"\s*(search|click)\[(.*)]\s*", action, flags=re.IGNORECASE | re.DOTALL
    )
    if not match:
        return None, None
    return match.group(1).lower(), normalize_text(match.group(2))


def grounded_think(
    action: str,
    available: dict[str, Any],
    history: list[dict[str, str]],
    goal: dict[str, Any],
) -> str:
    """Describe only facts verifiable from task state, legal actions, and history."""
    action_name, value = parse_action(action)
    if action_name == "search":
        return "The current page has a search box, so I should search using the task constraints."

    required_values = goal.get("goal_options") or {}
    if isinstance(required_values, dict):
        required_values = required_values.values()
    required = [normalize_text(item) for item in required_values]
    selected = []
    for record in history:
        prior_name, prior_value = parse_action(record["action"])
        if prior_name == "click" and prior_value:
            selected.append(prior_value)

    if value == "buy now":
        selected_required = [item for item in required if item in selected]
        if required and len(selected_required) == len(required):
            options = ", ".join(required)
            return (
                f"Buy Now is available, and the history records the required selectable "
                f"options ({options}); I should submit the purchase for evaluation."
            )
        return (
            "Buy Now is available on the current page; I should submit the current "
            "selection for environment evaluation without assuming hidden attributes."
        )
    if value in {"back to search", "< prev", "next >"}:
        return f"The current page offers '{value}', so I should use it to continue navigation."
    if value and re.fullmatch(r"b[0-9a-z]{9}", value):
        return (
            f"Product {value.upper()} is a clickable result on the current page; "
            "I should open it to inspect its details and options."
        )
    if value in required:
        return (
            f"The task requires option '{value}', and it is a legal action on the "
            "current page, so I should select it."
        )
    return (
        f"'{value}' is a legal action on the current page; I should open it to "
        "inspect information relevant to the task."
    )


def action_is_legal(action: str, available: dict[str, Any]) -> bool:
    action_name, value = parse_action(action)
    if action_name == "search":
        return bool(available.get("has_search_bar") and value)
    if action_name == "click":
        clickables = {normalize_text(item) for item in available.get("clickables", [])}
        return bool(value and value in clickables and value != "search")
    return False


def extract_task(raw_observation: str) -> str:
    parts = raw_observation.split(" [SEP] ")
    if len(parts) < 3 or parts[1] != "Instruction:":
        raise ValueError(f"Cannot extract WebShop task from: {raw_observation[:300]!r}")
    return parts[2]


def format_observation(raw_observation: str, task: str) -> str:
    parts = raw_observation.split(" [SEP] ")
    try:
        task_index = parts.index(task)
    except ValueError:
        return raw_observation
    return " [SEP] ".join(f"'{part}'" for part in parts[task_index + 1 :])


def format_available_actions(available: dict[str, Any]) -> str:
    unknown = set(available) - {"has_search_bar", "clickables"}
    if unknown:
        raise ValueError(f"Unknown available-action fields: {sorted(unknown)}")
    actions: list[str] = []
    if available.get("has_search_bar"):
        actions.append("search[<your query>]")
    actions.extend(f"click[{text}]" for text in available.get("clickables", []))
    return "\n".join(f"'{action}'," for action in actions)


def format_history(history: list[dict[str, str]], history_length: int) -> str:
    recent = history[-history_length:]
    start_index = len(history) - len(recent)
    lines = []
    for offset, record in enumerate(recent):
        step_number = start_index + offset + 1
        lines.append(
            f"[Observation {step_number}: '{record['observation']}', "
            f"Action {step_number}: '{record['action']}']"
        )
    return "\n".join(lines)


def build_prompt(
    *,
    task: str,
    observation: str,
    available: dict[str, Any],
    history: list[dict[str, str]],
    history_length: int,
    no_history_template: str,
    history_template: str,
) -> str:
    available_text = format_available_actions(available)
    if not history or history_length <= 0:
        return no_history_template.format(
            task_description=task,
            current_observation=observation,
            available_actions=available_text,
        )

    prompt = history_template.format(
        task_description=task,
        step_count=len(history),
        history_length=min(len(history), history_length),
        action_history=format_history(history, history_length),
        current_step=len(history) + 1,
        current_observation=observation,
        available_actions=available_text,
    )
    if len(prompt) > 13000:
        prompt = no_history_template.format(
            task_description=task,
            current_observation=observation,
            available_actions=available_text,
        )
    return prompt


def sync_bundled_data(repo_root: Path, webshop_root: Path) -> None:
    source = repo_root / "datasets/webshop-small"
    destination = webshop_root / "data"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "items_shuffle_1000.json",
        "items_ins_v2_1000.json",
        "items_human_ins.json",
    ):
        source_file = source / name
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        destination_file = destination / name
        if not destination_file.is_file() or destination_file.stat().st_size != source_file.stat().st_size:
            shutil.copy2(source_file, destination_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--expert-data",
        type=Path,
        default=repo_root / "data/agent_g2/webshop_sft_data.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "artifacts/expert_sft/agent_g2_split_manifest.json",
    )
    parser.add_argument(
        "--subset",
        choices=("pilot", "main", "eligible"),
        default="pilot",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--goal-seed", type=int, default=0)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument(
        "--think-mode",
        choices=("original", "concise", "grounded"),
        default="original",
        help="Use original expert reasoning or deterministic action-grounded reasoning.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "artifacts/expert_sft/pilot_replay_sft.jsonl",
    )
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=repo_root / "artifacts/expert_sft/pilot_replay_failures.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    webshop_root = repo_root / "agent_system/environments/env_package/webshop/webshop"
    sync_bundled_data(repo_root, webshop_root)

    index_marker = webshop_root / "search_engine/indexes_1k/segments_1"
    if not index_marker.is_file():
        raise FileNotFoundError(
            f"Missing 1k Lucene index: {index_marker}. "
            "Build it with the WebShop setup instructions in README.md on Linux/AutoDL "
            "or copy an already verified indexes_1k directory."
        )

    webshop_python_root = webshop_root
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(webshop_python_root))
    install_sparse_pyserini_shim()
    from web_agent_site.envs import WebAgentTextEnv  # noqa: PLC0415
    from agent_system.environments.prompts.webshop import (  # noqa: PLC0415
        WEBSHOP_TEMPLATE,
        WEBSHOP_TEMPLATE_NO_HIS,
    )

    expert_payload = json.loads(args.expert_data.read_text(encoding="utf-8"))
    expert_by_id = {
        trajectory["id"]: trajectory
        for trajectory in expert_payload["trajectories"]
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subset_field = {
        "pilot": "pilot_trajectory_ids",
        "main": "main_trajectory_ids",
        "eligible": "eligible_train_trajectory_ids",
    }[args.subset]
    selected_ids = manifest[subset_field]
    if args.limit > 0:
        selected_ids = selected_ids[: args.limit]

    env = WebAgentTextEnv(
        observation_mode="text",
        num_products=1000,
        human_goals=False,
        seed=args.goal_seed,
        goal_seed=args.goal_seed,
        file_path=str(webshop_root / "data/items_shuffle_1000.json"),
        attr_path=str(webshop_root / "data/items_ins_v2_1000.json"),
        session_prefix="expert-sft-",
    )
    goal_indices: dict[str, list[int]] = defaultdict(list)
    for index, goal in enumerate(env.server.goals):
        goal_indices[goal_target_key(goal)].append(index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failure_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_failures = args.failure_output.with_suffix(
        args.failure_output.suffix + ".tmp"
    )

    successful_trajectories = 0
    total_steps = 0
    failure_counts: Counter[str] = Counter()
    with temporary_output.open("w", encoding="utf-8") as output_handle, temporary_failures.open(
        "w", encoding="utf-8"
    ) as failure_handle:
        for position, trajectory_id in enumerate(selected_ids):
            trajectory = expert_by_id[trajectory_id]
            candidate_indices = goal_indices.get(trajectory_id, [])
            if not candidate_indices:
                failure = {
                    "trajectory_id": trajectory_id,
                    "reason": "target_key_not_found",
                }
                failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failure_counts[failure["reason"]] += 1
                continue

            goal_index = candidate_indices[0]
            goal = env.server.goals[goal_index]
            raw_observation, _ = env.reset(session=goal_index)
            task = extract_task(raw_observation)
            observation = format_observation(raw_observation, task)
            history: list[dict[str, str]] = []
            trajectory_records: list[dict[str, Any]] = []
            failure_reason: str | None = None
            terminal_score = 0.0

            for turn, (action, think) in enumerate(
                zip(trajectory["actions"], trajectory["think"])
            ):
                available = env.get_available_actions()
                if not action_is_legal(action, available):
                    failure_reason = f"illegal_action_at_turn_{turn}"
                    break

                prompt = build_prompt(
                    task=task,
                    observation=observation,
                    available=available,
                    history=history,
                    history_length=args.history_length,
                    no_history_template=WEBSHOP_TEMPLATE_NO_HIS,
                    history_template=WEBSHOP_TEMPLATE,
                )
                if args.think_mode in {"concise", "grounded"}:
                    response_think = grounded_think(
                        action=action,
                        available=available,
                        history=history,
                        goal=goal,
                    )
                else:
                    response_think = str(think).strip()
                response = (
                    f"<think>{response_think}</think>\n"
                    f"<action>{action}</action>"
                )
                trajectory_records.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "trajectory_id": trajectory_id,
                        "goal_index": goal_index,
                        "turn": turn,
                        "task": task,
                        "expert_action": action,
                        "generated_think": response_think,
                        "original_expert_think": str(think).strip(),
                        "think_source": (
                            "state_grounded_rules"
                            if args.think_mode in {"concise", "grounded"}
                            else "original_expert"
                        ),
                        "source_dataset": trajectory.get("source_dataset", "api_webshop"),
                    }
                )

                previous_observation = observation
                next_raw_observation, reward, done, _ = env.step(action)
                terminal_score = float(reward)
                history.append(
                    {"observation": previous_observation, "action": action}
                )

                if done:
                    if turn != len(trajectory["actions"]) - 1:
                        failure_reason = f"terminated_early_at_turn_{turn}"
                    break
                observation = format_observation(next_raw_observation, task)

            if failure_reason is None and not done:
                failure_reason = "trajectory_did_not_terminate"
            if failure_reason is None and terminal_score < 1.0 - 1e-8:
                failure_reason = "terminal_score_below_one"

            if failure_reason is not None:
                failure = {
                    "trajectory_id": trajectory_id,
                    "goal_index": goal_index,
                    "reason": failure_reason,
                    "terminal_score": terminal_score,
                    "actions": trajectory["actions"],
                }
                failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failure_counts[failure_reason] += 1
                continue

            for record in trajectory_records:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            successful_trajectories += 1
            total_steps += len(trajectory_records)
            if (position + 1) % 50 == 0 or position + 1 == len(selected_ids):
                print(
                    f"replayed={position + 1}/{len(selected_ids)} "
                    f"successful={successful_trajectories} steps={total_steps}",
                    flush=True,
                )

    temporary_output.replace(args.output)
    temporary_failures.replace(args.failure_output)
    summary = {
        "selected_trajectories": len(selected_ids),
        "successful_trajectories": successful_trajectories,
        "failed_trajectories": len(selected_ids) - successful_trajectories,
        "sft_step_samples": total_steps,
        "failure_counts": dict(failure_counts),
        "output": str(args.output.resolve()),
        "failure_output": str(args.failure_output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
