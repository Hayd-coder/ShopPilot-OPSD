"""Select an SFT checkpoint by fixed-environment behavior, not token loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    if not candidates:
        raise RuntimeError("No SFT evaluation summaries were supplied")

    # Success is primary. Partial WebShop score breaks ties, followed by valid
    # actions, fewer immediate loops, and shorter successful interaction.
    best = max(
        candidates,
        key=lambda item: (
            float(item["success_rate"]),
            float(item["mean_task_score"]),
            float(item["valid_action_rate"]),
            -float(item["consecutive_repeat_rate"]),
            -float(item["mean_episode_length"]),
        ),
    )
    result = {"selected": best, "candidates": candidates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
