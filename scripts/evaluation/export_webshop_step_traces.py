#!/usr/bin/env python3
"""Export every model decision and page transition from WebShop eval JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.I | re.S)
TASK_RE = re.compile(
    r"Your task is to:\s*(.*?)(?:\nPrior to this step|\nYour current observation|\nYou are now at step)",
    re.S,
)
OBS_RE = re.compile(
    r"(?:your|and your) current observation is:\s*(.*?)\nYour admissible actions",
    re.I | re.S,
)
ADMISSIBLE_RE = re.compile(
    r"Your admissible actions of the current situation are:\s*(\[.*?\])\s*\n\nNow it's your turn",
    re.S,
)


def clean(text: str) -> str:
    return text.strip().replace("\x00", "")


def extract(pattern: re.Pattern[str], text: str, default: str) -> str:
    match = pattern.search(text)
    return clean(match.group(1)) if match else default


def fence(text: str) -> str:
    return "```text\n" + text.replace("```", "` ` `") + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["traj_uid"])].append(row)

    trajectories = []
    for traj_uid, turns in grouped.items():
        turns.sort(key=lambda row: int(row["turn_step"]))
        parsed_turns = []
        for index, row in enumerate(turns):
            prompt = str(row.get("input", ""))
            response = str(row.get("output", ""))
            action_match = ACTION_RE.search(response)
            observation = row.get("observation_before") or extract(
                OBS_RE, prompt, "<OBSERVATION_PARSE_FAILED>"
            )
            next_observation = row.get("observation_after")
            if next_observation is None and index + 1 < len(turns):
                next_prompt = str(turns[index + 1].get("input", ""))
                next_observation = extract(
                    OBS_RE, next_prompt, "<OBSERVATION_PARSE_FAILED>"
                )
            parsed_action = row.get("projected_action")
            if not parsed_action:
                parsed_action = (
                    clean(action_match.group(1)) if action_match else "<NO_ACTION_TAG>"
                )
            parsed_turns.append(
                {
                    "turn": int(row["turn_step"]),
                    "observation_before": observation,
                    "admissible_actions": row.get("admissible_actions_before")
                    or extract(ADMISSIBLE_RE, prompt, "<ADMISSIBLE_ACTION_PARSE_FAILED>"),
                    "raw_model_response": row.get("raw_model_response") or response,
                    "parsed_action": parsed_action,
                    "projection_valid": bool(row.get("is_action_valid", False)),
                    "observation_after": next_observation,
                    "selected_asin": row.get("selected_asin"),
                    "selected_options": row.get("selected_options") or {},
                    "done": bool(row.get("done", False)),
                    "step_reward": float(row.get("step_reward", 0.0)),
                    "reward_breakdown": row.get("reward_breakdown") or {},
                }
            )

        first = turns[0]
        trajectories.append(
            {
                "traj_uid": traj_uid,
                "uid": first.get("uid"),
                "task": extract(TASK_RE, str(first.get("input", "")), "<TASK_PARSE_FAILED>"),
                "episode_length": int(first.get("episode_lengths", len(turns))),
                "training_reward": float(first.get("episode_rewards", 0.0)),
                "task_score": float(first.get("webshop_task_score", 0.0)),
                "success": bool(float(first.get("trajectory_success", 0.0))),
                "turns": parsed_turns,
            }
        )

    trajectories.sort(key=lambda row: (row["success"], row["task_score"], row["task"]))

    has_native_transitions = any(
        "observation_after" in row or "reward_breakdown" in row for row in rows
    )
    format_note = (
        "This JSONL contains native per-step transitions, including the final "
        "page, selected options and terminal reward decomposition."
        if has_native_transitions
        else
        "`training_reward` and `task_score` are terminal trajectory values copied "
        "to each decision row by the evaluator. They are not per-step rewards. "
        "The page after the final action is not present in this old JSONL format."
    )
    lines = [
        f"# Full WebShop step traces: {args.jsonl.name}",
        "",
        f"Trajectories: **{len(trajectories)}**; decisions: **{len(rows)}**.",
        "",
        f"> {format_note}",
        "",
    ]
    for traj_index, trajectory in enumerate(trajectories, start=1):
        lines += [
            f"## Trajectory {traj_index}: {'SUCCESS' if trajectory['success'] else 'FAIL'}",
            "",
            f"- task: {trajectory['task']}",
            f"- task_score: `{trajectory['task_score']:.3f}`",
            f"- training_reward: `{trajectory['training_reward']:.3f}`",
            f"- episode_length: `{trajectory['episode_length']}`",
            f"- traj_uid: `{trajectory['traj_uid']}`",
            "",
        ]
        for turn in trajectory["turns"]:
            lines += [
                f"### Turn {turn['turn'] + 1}",
                "",
                "Observation before action:",
                "",
                fence(turn["observation_before"]),
                "",
                "Admissible actions:",
                "",
                fence(turn["admissible_actions"]),
                "",
                "Raw Qwen response:",
                "",
                fence(turn["raw_model_response"]),
                "",
                f"Parsed action: `{turn['parsed_action'].replace('`', '')}`  ",
                f"Projection valid: `{turn['projection_valid']}`  ",
                f"Selected product: `{turn['selected_asin']}`  ",
                f"Selected options: `{json.dumps(turn['selected_options'], ensure_ascii=False)}`  ",
                f"Step reward: `{turn['step_reward']:.3f}`; done: `{turn['done']}`  ",
                f"Terminal reward breakdown: `{json.dumps(turn['reward_breakdown'], ensure_ascii=False)}`",
                "",
            ]
            if turn["observation_after"] is not None:
                lines += [
                    "Environment observation after executing this action:",
                    "",
                    fence(turn["observation_after"]),
                    "",
                ]
            else:
                lines += [
                    "Environment observation after executing this action: `<NOT_RECORDED_FOR_FINAL_TURN>`",
                    "",
                ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_output = args.output.with_suffix(".json")
    json_output.write_text(
        json.dumps(trajectories, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} and {json_output}")


if __name__ == "__main__":
    main()
