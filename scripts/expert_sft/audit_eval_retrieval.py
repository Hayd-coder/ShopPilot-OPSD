#!/usr/bin/env python3
"""Audit whether WebShop evaluation queries actually retrieve their targets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generate_oracle_webshop import (
    ASIN_RE,
    install_sparse_pyserini_shim,
    normalize_text,
    parse_action,
    sync_bundled_data,
)


TASK_RE = re.compile(r"Your task is to: (.*?)\n", re.DOTALL)


def task_key(text: str) -> str:
    normalized = normalize_text(text)
    return re.sub(r"\s+and price lower than.*$", "", normalized).strip(" ,.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=15)
    return parser.parse_args()


def locate_with_query(
    env: Any,
    goal_index: int,
    query: str,
    target_asin: str,
    max_pages: int,
) -> dict[str, Any]:
    env.reset(session=goal_index)
    _, _, done, _ = env.step(f"search[{query}]")
    if done:
        return {"found": False, "reason": "search_terminated"}
    cumulative_rank = 0
    for page in range(max_pages):
        available = env.get_available_actions()
        result_asins = [
            str(item).upper()
            for item in available.get("clickables", [])
            if ASIN_RE.fullmatch(normalize_text(item))
        ]
        if target_asin.upper() in result_asins:
            return {
                "found": True,
                "page": page,
                "rank": cumulative_rank + result_asins.index(target_asin.upper()) + 1,
            }
        cumulative_rank += len(result_asins)
        clickables = {
            normalize_text(item) for item in available.get("clickables", [])
        }
        if page + 1 >= max_pages or "next >" not in clickables:
            break
        env.step("click[next >]")
    return {"found": False, "reason": "not_in_first_pages"}


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    webshop_root = repo_root / "agent_system/environments/env_package/webshop/webshop"
    sync_bundled_data(repo_root, webshop_root)
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(webshop_root))
    install_sparse_pyserini_shim()
    from web_agent_site.envs import WebAgentTextEnv  # noqa: PLC0415

    env = WebAgentTextEnv(
        observation_mode="text",
        num_products=1000,
        human_goals=False,
        seed=0,
        goal_seed=0,
        file_path=str(webshop_root / "data/items_shuffle_1000.json"),
        attr_path=str(webshop_root / "data/items_ins_v2_1000.json"),
        session_prefix="eval-retrieval-audit-",
    )
    goal_by_task: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, goal in enumerate(env.server.goals):
        goal_by_task[task_key(goal["instruction_text"])].append((index, goal))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.evaluation.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[str(row["traj_uid"])].append(row)

    results = []
    counts: Counter[str] = Counter()
    for trajectory_id, rows in grouped.items():
        rows.sort(key=lambda item: int(item["turn_step"]))
        task_match = TASK_RE.search(rows[0]["input"])
        if not task_match:
            raise ValueError(f"Task missing in {trajectory_id}")
        task = task_match.group(1).strip()
        matches = goal_by_task.get(task_key(task), [])
        target_asins = {str(goal["asin"]).upper() for _, goal in matches}
        if not matches or len(target_asins) != 1:
            raise ValueError(
                f"Expected one target ASIN for task, got {len(matches)} goals and "
                f"{len(target_asins)} ASINs: {task[:120]}"
            )
        goal_index, goal = matches[0]
        search_action = next(
            (str(row["projected_action"]) for row in rows if str(row["projected_action"]).lower().startswith("search[")),
            "",
        )
        action_name, query = parse_action(search_action)
        if action_name != "search" or not query:
            location = {"found": False, "reason": "no_valid_search"}
        else:
            location = locate_with_query(
                env,
                goal_index,
                query,
                str(goal["asin"]),
                args.max_pages,
            )
        success = bool(float(rows[-1].get("trajectory_success", 0.0)))
        done = bool(rows[-1].get("done", False))
        if location["found"]:
            reachability = "target_retrieved"
            if int(location["page"]) >= args.max_pages - 1:
                reachability = "target_too_late"
        else:
            reachability = "target_not_retrieved"
        counts[f"{'success' if success else 'failure'}::{reachability}"] += 1
        results.append(
            {
                "trajectory_id": trajectory_id,
                "task": task,
                "target_asin": goal["asin"],
                "model_search_action": search_action,
                "success": success,
                "done": done,
                "task_score": float(rows[-1].get("webshop_task_score", 0.0)),
                "location": location,
                "actions": [row["projected_action"] for row in rows],
            }
        )

    report = {
        "trajectory_count": len(results),
        "max_pages_checked": args.max_pages,
        "counts": dict(counts),
        "trajectories": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("trajectory_count", "max_pages_checked", "counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
