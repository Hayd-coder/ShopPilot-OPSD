#!/usr/bin/env python3
"""Stream the full WebShop product JSON into Pyserini JsonCollection docs.

The upstream converter materializes both the 5+ GiB input and every converted
document in memory.  This converter preserves the searchable text used by
WebShop while writing one document at a time, which is safer on a rented host.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ijson


def text(value: object) -> str:
    return value if isinstance(value, str) else ""


def option_text(product: dict) -> str:
    groups: list[str] = []
    options = product.get("customization_options") or {}
    if not isinstance(options, dict):
        return ""

    for raw_name, raw_contents in options.items():
        if not isinstance(raw_contents, list):
            continue
        values: list[str] = []
        for entry in raw_contents:
            if not isinstance(entry, dict):
                continue
            value = entry.get("value")
            if isinstance(value, str):
                values.append(value.strip().replace("/", " | ").lower())
        if values:
            groups.append(f"{str(raw_name).lower()}: {', '.join(values)}")
    return ", and ".join(groups)


def first_bullet(product: dict) -> str:
    value = product.get("small_description")
    if isinstance(value, list):
        return text(value[0]) if value else ""
    return text(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=25_000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    skipped = 0

    with args.input.open("rb") as source, args.output.open("w", encoding="utf-8") as target:
        for product in ijson.items(source, "item"):
            asin = product.get("asin")
            if not isinstance(asin, str) or asin == "nan" or len(asin) > 10 or asin in seen:
                skipped += 1
                continue
            seen.add(asin)

            contents = " ".join(
                [
                    text(product.get("name")),
                    text(product.get("full_description")),
                    first_bullet(product),
                    option_text(product),
                ]
            ).lower()
            target.write(json.dumps({"id": asin, "contents": contents}, ensure_ascii=False) + "\n")
            written += 1
            if written % args.progress_every == 0:
                print(f"converted={written:,} skipped={skipped:,}", flush=True)

    print(f"conversion complete: converted={written:,} skipped={skipped:,}", flush=True)


if __name__ == "__main__":
    main()
