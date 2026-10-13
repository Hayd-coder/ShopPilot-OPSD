#!/usr/bin/env python3
"""Compare fixed WebShop evaluation tasks across bad-case report checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_report(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["task"]: row for row in payload["trajectories"]}


def compact_actions(row: dict, limit: int = 8) -> str:
    actions = row.get("actions", [])
    shown = actions[:limit]
    text = " -> ".join(shown)
    if len(actions) > limit:
        text += f" -> ... ({len(actions)} turns)"
    return text.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if len(args.reports) != len(args.steps):
        raise SystemExit("--steps must have one value per report")

    tables = {step: load_report(path) for step, path in zip(args.steps, args.reports)}
    steps = args.steps
    shared = set.intersection(*(set(table) for table in tables.values()))
    if not shared:
        raise SystemExit("No shared tasks found; evaluation split is not fixed")

    lines = [
        "# Fixed-task checkpoint transition analysis",
        "",
        f"Shared evaluation tasks: **{len(shared)}**",
        "",
        "## Pairwise score changes",
        "",
        "| transition | improved | unchanged | worsened | mean score delta | successes gained | successes lost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for left, right in zip(steps, steps[1:]):
        deltas = {
            task: tables[right][task]["partial_score"] - tables[left][task]["partial_score"]
            for task in shared
        }
        improved = sum(delta > 1e-9 for delta in deltas.values())
        worsened = sum(delta < -1e-9 for delta in deltas.values())
        unchanged = len(shared) - improved - worsened
        gained = sum(
            not tables[left][task]["success"] and tables[right][task]["success"]
            for task in shared
        )
        lost = sum(
            tables[left][task]["success"] and not tables[right][task]["success"]
            for task in shared
        )
        mean_delta = sum(deltas.values()) / len(deltas)
        lines.append(
            f"| {left} -> {right} | {improved} | {unchanged} | {worsened} | "
            f"{mean_delta:+.3f} | {gained} | {lost} |"
        )

    lines += ["", "## Failure categories by checkpoint", ""]
    category_names = sorted(
        {row["category"] for table in tables.values() for row in table.values()}
    )
    lines.append("| category | " + " | ".join(f"step {step}" for step in steps) + " |")
    lines.append("|---|" + "---:|" * len(steps))
    counters = {
        step: Counter(row["category"] for row in table.values())
        for step, table in tables.items()
    }
    for category in category_names:
        lines.append(
            f"| {category} | "
            + " | ".join(str(counters[step][category]) for step in steps)
            + " |"
        )

    first, last = steps[0], steps[-1]
    transitions = Counter(
        (tables[first][task]["category"], tables[last][task]["category"])
        for task in shared
    )
    lines += [
        "",
        f"## Most common category transitions: step {first} -> {last}",
        "",
        "| from | to | tasks |",
        "|---|---|---:|",
    ]
    for (source, target), count in transitions.most_common(12):
        lines.append(f"| {source} | {target} | {count} |")

    lines += ["", "## Success transitions", ""]
    for left, right in zip(steps, steps[1:]):
        changed = [
            task
            for task in sorted(shared)
            if tables[left][task]["success"] != tables[right][task]["success"]
        ]
        lines.append(f"### Step {left} -> {right}")
        lines.append("")
        if not changed:
            lines.append("No task changed strict-success status.")
            lines.append("")
            continue
        for task in changed:
            before, after = tables[left][task], tables[right][task]
            direction = "GAINED" if after["success"] else "LOST"
            lines += [
                f"- **{direction}**: {task}",
                f"  - step {left}: score={before['partial_score']:.3f}, "
                f"category={before['category']}, actions={compact_actions(before)}",
                f"  - step {right}: score={after['partial_score']:.3f}, "
                f"category={after['category']}, actions={compact_actions(after)}",
            ]
        lines.append("")

    lines += ["## Largest regressions from first to last checkpoint", ""]
    ranked = sorted(
        shared,
        key=lambda task: tables[last][task]["partial_score"]
        - tables[first][task]["partial_score"],
    )
    for task in ranked[:5]:
        before, after = tables[first][task], tables[last][task]
        delta = after["partial_score"] - before["partial_score"]
        lines += [
            f"- delta={delta:+.3f}: {task}",
            f"  - step {first}: {before['category']}; {compact_actions(before)}",
            f"  - step {last}: {after['category']}; {compact_actions(after)}",
        ]

    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
