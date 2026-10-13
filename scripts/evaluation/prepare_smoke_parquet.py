"""Create the placeholder parquet rows used by agent-environment rollouts.

The upstream preprocessing script downloads Geometry3K only to borrow its row
count and modality schema. WebShop supplies the real task after ``env.reset()``,
so a smoke test can create the same empty-prompt rows entirely offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


def build_rows(split: str, size: int) -> list[dict]:
    return [
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split, "index": index},
        }
        for index in range(size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="~/data/verl-agent/text")
    parser.add_argument("--train-size", type=int, default=1)
    parser.add_argument("--val-size", type=int, default=1)
    args = parser.parse_args()

    if args.train_size < 1 or args.val_size < 1:
        raise ValueError("train-size and val-size must both be positive")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    Dataset.from_list(build_rows("train", args.train_size)).to_parquet(
        output_dir / "train.parquet",
    )
    Dataset.from_list(build_rows("test", args.val_size)).to_parquet(
        output_dir / "test.parquet",
    )

    print(
        f"Prepared offline smoke data at {output_dir} "
        f"(train={args.train_size}, val={args.val_size})",
    )


if __name__ == "__main__":
    main()
