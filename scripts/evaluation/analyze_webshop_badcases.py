"""Classify WebShop validation trajectories and write evidence-rich bad-case reports."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK_RE = re.compile(r"Your task is to: (.*?)\n", re.DOTALL)
ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
STRICT_ACTION_RE = re.compile(r"^(search|click)\[([^\[\]]+)\]$", re.IGNORECASE | re.DOTALL)
ADMISSIBLE_RE = re.compile(
    r"Your admissible actions of the current situation are:\s*\[(.*?)\]\.?(?:\r?\n)+Now",
    re.DOTALL,
)
QUOTED_ACTION_RE = re.compile(r"'((?:search|click)\[[^']+\])'", re.IGNORECASE)
OPTION_RE = re.compile(
    r"(?:with|and)\s+([a-z][a-z ]{0,24}?):\s*([^,]+?)(?=,\s*(?:and\s+)?[a-z][a-z ]{0,24}?:|,\s*and price|,\s*price|\.$)",
    re.IGNORECASE,
)
ASIN_RE = re.compile(r"^[a-z0-9]{10}$", re.IGNORECASE)
NAVIGATION_CLICKS = {
    "buy now",
    "back to search",
    "search",
    "next >",
    "< prev",
    "description",
    "features",
    "reviews",
    "attributes",
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_task(prompt: str) -> str:
    match = TASK_RE.search(prompt)
    return match.group(1).strip() if match else "<TASK_NOT_FOUND>"


def extract_action(response: str) -> tuple[str, str, str]:
    match = ACTION_RE.search(response)
    if not match:
        return "<NO_ACTION_TAG>", "missing", ""
    action = match.group(1).strip().lower()
    strict = STRICT_ACTION_RE.fullmatch(action)
    if not strict:
        return action, "malformed", ""
    return action, strict.group(1).lower(), strict.group(2).strip().lower()


def current_admissible_actions(prompt: str) -> set[str]:
    matches = list(ADMISSIBLE_RE.finditer(prompt))
    if not matches:
        return set()
    return {action.lower() for action in QUOTED_ACTION_RE.findall(matches[-1].group(1))}


def is_environment_legal(action: str, action_type: str, admissible: set[str]) -> bool:
    if action_type == "search":
        return any(candidate.startswith("search[") for candidate in admissible)
    return action in admissible


def required_options(task: str) -> dict[str, str]:
    return {normalize(name): normalize(value) for name, value in OPTION_RE.findall(task)}


def classify(trajectory: dict[str, Any], max_steps: int) -> tuple[str, list[str]]:
    if trajectory["success"]:
        return "success", ["All WebShop constraints were satisfied."]

    reasons: list[str] = []
    bought = trajectory["bought"]
    partial = trajectory["partial_score"]
    length = trajectory["length"]

    if trajectory["missing_or_malformed"]:
        reasons.append(
            f"{trajectory['missing_or_malformed']} turns had no executable <action> block.",
        )
    if trajectory["environment_illegal"]:
        reasons.append(
            f"{trajectory['environment_illegal']} extracted actions were not in the current admissible set.",
        )
    if trajectory["loop_detected"]:
        reasons.append("Repeated actions/backtracking indicate a navigation loop.")

    if bought:
        if partial <= 1e-8:
            reasons.append("The purchased product matched none of the scored goal constraints.")
            return "wrong_product_zero_score", reasons
        if trajectory["required_options"] and not trajectory["option_clicks"]:
            reasons.append("The task required product options, but no option was selected before purchase.")
            return "missing_option_selection", reasons
        if trajectory["unmatched_options"]:
            reasons.append(
                "Required option values not observed in option clicks: "
                + ", ".join(trajectory["unmatched_options"]),
            )
            return "option_mismatch", reasons
        if length <= 3:
            reasons.append("The agent bought immediately after the first retrieved product.")
            return "premature_purchase", reasons
        reasons.append("A related product was purchased, but one or more non-option constraints failed.")
        return "partial_constraint_mismatch", reasons

    invalid_ratio = trajectory["projection_invalid"] / max(length, 1)
    illegal_ratio = trajectory["environment_illegal"] / max(length, 1)
    if invalid_ratio >= 0.4:
        reasons.append("Output-format failures consumed at least 40% of the trajectory.")
        return "output_format_failure", reasons
    if illegal_ratio >= 0.4:
        reasons.append("Illegal page actions consumed at least 40% of the trajectory.")
        return "illegal_action_failure", reasons
    if length >= max_steps:
        reasons.append(f"The trajectory exhausted the {max_steps}-step budget without purchasing.")
        return "timeout_without_purchase", reasons
    if not trajectory["product_clicks"]:
        reasons.append("No product page was opened after searching.")
        return "retrieval_or_search_failure", reasons
    reasons.append("A product page was visited, but the agent never committed to a purchase.")
    return "navigation_or_commit_failure", reasons


def load_trajectories(path: Path, max_steps: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[str(row.get("traj_uid", row.get("uid", "unknown")))].append(row)

    trajectories = []
    for traj_uid, rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("turn_step", row.get("step", 0))))
        task = extract_task(rows[0]["input"])
        actions = []
        parsed = []
        projection_invalid = 0
        environment_illegal = 0
        for row in rows:
            action, action_type, argument = extract_action(row["output"])
            actions.append(action)
            admissible = current_admissible_actions(row["input"])
            projection_valid = bool(row.get("is_action_valid", action_type not in {"missing", "malformed"}))
            if not projection_valid:
                projection_invalid += 1
            legal = (
                projection_valid
                and action_type not in {"missing", "malformed"}
                and is_environment_legal(action, action_type, admissible)
            )
            if projection_valid and not legal:
                environment_illegal += 1
            parsed.append((action_type, argument, legal))

        click_args = [argument for action_type, argument, _ in parsed if action_type == "click"]
        product_clicks = [argument for argument in click_args if ASIN_RE.fullmatch(argument)]
        option_clicks = [
            argument
            for argument in click_args
            if argument not in NAVIGATION_CLICKS and not ASIN_RE.fullmatch(argument)
        ]
        options = required_options(task)
        normalized_clicks = [normalize(value) for value in option_clicks]
        unmatched_options = [
            f"{name}={value}"
            for name, value in options.items()
            if not any(value in click or click in value for click in normalized_clicks)
        ]
        action_counts = Counter(action for action in actions if not action.startswith("<"))
        repeated = any(count >= 3 for count in action_counts.values())
        backtracks = sum(action == "click[back to search]" for action in actions)

        trajectory = {
            "traj_uid": traj_uid,
            "uid": str(rows[0].get("uid", "unknown")),
            "task": task,
            "length": int(float(rows[0].get("episode_lengths", len(rows)))),
            "success": bool(float(rows[0].get("trajectory_success", 0.0))),
            "partial_score": float(rows[0].get("webshop_task_score", 0.0)),
            "episode_reward": float(rows[0].get("episode_rewards", 0.0)),
            "actions": actions,
            "projection_invalid": projection_invalid,
            "environment_illegal": environment_illegal,
            "missing_or_malformed": sum(
                action_type in {"missing", "malformed"} for action_type, _, _ in parsed
            ),
            "searches": sum(action_type == "search" for action_type, _, _ in parsed),
            "product_clicks": product_clicks,
            "option_clicks": option_clicks,
            "required_options": options,
            "unmatched_options": unmatched_options,
            "bought": "buy now" in click_args,
            "loop_detected": repeated or backtracks >= 2,
        }
        category, reasons = classify(trajectory, max_steps=max_steps)
        trajectory["category"] = category
        trajectory["reasons"] = reasons
        trajectories.append(trajectory)
    return trajectories


def checkpoint_number(path: Path) -> int:
    return int(path.stem) if path.stem.isdigit() else -1


def summarize(path: Path, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(item["category"] for item in trajectories)
    partial_scores = [item["partial_score"] for item in trajectories]
    lengths = [item["length"] for item in trajectories]
    action_total = sum(len(item["actions"]) for item in trajectories)
    strict_successes = sum(item["success"] for item in trajectories)
    purchases = sum(item["bought"] for item in trajectories)
    return {
        "experiment": path.parent.name,
        "checkpoint": checkpoint_number(path),
        "file": str(path),
        "trajectory_count": len(trajectories),
        "strict_success_count": strict_successes,
        "strict_success_rate": strict_successes / max(len(trajectories), 1),
        "partial_score_mean": statistics.fmean(partial_scores) if partial_scores else 0.0,
        "partial_score_median": statistics.median(partial_scores) if partial_scores else 0.0,
        "purchase_rate": purchases / max(len(trajectories), 1),
        "mean_episode_length": statistics.fmean(lengths) if lengths else 0.0,
        "projection_invalid_rate": sum(item["projection_invalid"] for item in trajectories)
        / max(action_total, 1),
        "environment_illegal_rate": sum(item["environment_illegal"] for item in trajectories)
        / max(action_total, 1),
        "loop_trajectory_rate": sum(item["loop_detected"] for item in trajectories)
        / max(len(trajectories), 1),
        "categories": dict(categories),
    }


def markdown_report(
    summary: dict[str, Any],
    trajectories: list[dict[str, Any]],
    examples_per_category: int,
) -> str:
    lines = [
        f"# {summary['experiment']} step {summary['checkpoint']} bad-case report",
        "",
        "## Outcome summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| trajectories | {summary['trajectory_count']} |",
        f"| strict success rate | {summary['strict_success_rate']:.3%} |",
        f"| mean partial score | {summary['partial_score_mean']:.3f} |",
        f"| purchase rate | {summary['purchase_rate']:.3%} |",
        f"| projection-invalid action rate | {summary['projection_invalid_rate']:.3%} |",
        f"| page-illegal action rate | {summary['environment_illegal_rate']:.3%} |",
        f"| loop trajectory rate | {summary['loop_trajectory_rate']:.3%} |",
        f"| mean episode length | {summary['mean_episode_length']:.2f} |",
        "",
        "## Primary failure stage",
        "",
        "| category | trajectories |",
        "|---|---:|",
    ]
    for category, count in sorted(
        summary["categories"].items(), key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"| {category} | {count} |")

    lines.extend(["", "## Representative evidence", ""])
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in trajectories:
        if item["category"] != "success":
            by_category[item["category"]].append(item)
    for category, items in sorted(by_category.items()):
        lines.extend([f"### {category}", ""])
        items.sort(
            key=lambda item: (
                item["partial_score"],
                -item["projection_invalid"],
                -item["length"],
            ),
        )
        for item in items[:examples_per_category]:
            lines.append(
                f"- task: {item['task'].replace('|', '/')}  ",
            )
            lines.append(
                f"  score={item['partial_score']:.3f}, length={item['length']}, "
                f"projection_invalid={item['projection_invalid']}, "
                f"page_illegal={item['environment_illegal']}  ",
            )
            lines.append("  diagnosis: " + " ".join(item["reasons"]) + "  ")
            lines.append("  actions:")
            for step, action in enumerate(item["actions"], start=1):
                lines.append(f"    {step}. `{action.replace('`', '')}`")
        lines.append("")
    return "\n".join(lines)


def expand_inputs(inputs: list[Path]) -> list[Path]:
    paths = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.jsonl"), key=checkpoint_number))
        elif item.is_file():
            paths.append(item)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--examples-per-category", type=int, default=2)
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("No JSONL files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in paths:
        trajectories = load_trajectories(path, max_steps=args.max_steps)
        summary = summarize(path, trajectories)
        summaries.append(summary)
        prefix = f"{path.parent.name}_step_{path.stem}"
        (args.output_dir / f"{prefix}.json").write_text(
            json.dumps(
                {"summary": summary, "trajectories": trajectories},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (args.output_dir / f"{prefix}.md").write_text(
            markdown_report(summary, trajectories, args.examples_per_category),
            encoding="utf-8",
        )

    comparison = [
        "# WebShop bad-case checkpoint comparison",
        "",
        "| experiment | step | success | partial score | purchase | proj invalid | page illegal | loop | mean length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        comparison.append(
            f"| {item['experiment']} | {item['checkpoint']} | "
            f"{item['strict_success_rate']:.3%} | {item['partial_score_mean']:.3f} | "
            f"{item['purchase_rate']:.3%} | {item['projection_invalid_rate']:.3%} | "
            f"{item['environment_illegal_rate']:.3%} | {item['loop_trajectory_rate']:.3%} | "
            f"{item['mean_episode_length']:.2f} |",
        )
    (args.output_dir / "comparison.md").write_text("\n".join(comparison), encoding="utf-8")
    print("\n".join(comparison))


if __name__ == "__main__":
    main()
