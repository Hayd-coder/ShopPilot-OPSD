"""Summarize step-level JSONL produced by WebShop validation runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TASK_RE = re.compile(r"Your task is to: (.*?)\n")
STEP_RE = re.compile(r"You are now at step (\d+)")
ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
STRICT_ACTION_RE = re.compile(r"^(?:search|click)\[[^\[\]]+\]$", re.IGNORECASE | re.DOTALL)


def extract(pattern: re.Pattern[str], text: str, default: str = "unknown") -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--show", type=int, default=20, help="number of step actions to print")
    args = parser.parse_args()

    with args.jsonl.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    tasks: dict[str, list[tuple[int | None, str, float, bool]]] = defaultdict(list)
    action_types: Counter[str] = Counter()
    scores: Counter[float] = Counter()
    valid_by_projector = 0
    strict_actions = 0

    for row in rows:
        prompt = row["input"]
        response = row["output"]
        task = extract(TASK_RE, prompt)
        step_text = extract(STEP_RE, prompt, default="")
        step = int(step_text) if step_text else int(row.get("turn_step", 0)) + 1
        action_match = ACTION_RE.search(response)
        action = action_match.group(1).strip() if action_match else "<NO_ACTION_TAG>"
        score = float(row["score"])
        projector_valid = action_match is not None and CHINESE_RE.search(response) is None

        if projector_valid:
            valid_by_projector += 1
        if projector_valid and STRICT_ACTION_RE.fullmatch(action):
            strict_actions += 1
        if action.startswith("search["):
            action_types["search"] += 1
        elif action.startswith("click["):
            action_types["click"] += 1
        else:
            action_types["missing_or_other"] += 1
        scores[score] += 1
        environment_valid = bool(row.get("is_action_valid", projector_valid))
        tasks[task].append((step, action, score, environment_valid))

    print(f"rows={len(rows)}")
    print(f"unique_tasks={len(tasks)}")
    print(f"projector_valid={valid_by_projector}/{len(rows)}")
    print(f"strict_action_syntax={strict_actions}/{len(rows)}")
    print(f"action_types={dict(action_types)}")
    print(f"score_counts={dict(sorted(scores.items()))}")
    print("steps_per_task=" + str(sorted(len(steps) for steps in tasks.values())))

    shown = 0
    for task_index, (task, steps) in enumerate(tasks.items(), start=1):
        first_row = next(row for row in rows if extract(TASK_RE, row["input"]) == task)
        partial_score = first_row.get("webshop_task_score", "unknown")
        success = first_row.get("trajectory_success", "unknown")
        episode_length = first_row.get("episode_lengths", len(steps))
        print(
            f"\nTASK {task_index}: success={success} "
            f"partial_score={partial_score} length={episode_length}\n  {task}",
        )
        for step, action, score, valid in steps:
            print(f"  step={step} valid={int(valid)} score={score:g} action={action[:180]}")
            shown += 1
            if shown >= args.show:
                return


if __name__ == "__main__":
    main()
