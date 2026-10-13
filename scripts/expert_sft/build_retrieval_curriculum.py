#!/usr/bin/env python3
"""Mix hard WebShop trajectories with deterministic direct-trajectory replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct-jsonl', type=Path, required=True)
    parser.add_argument('--hard-jsonl', type=Path, required=True)
    parser.add_argument('--eval-manifest', type=Path, required=True)
    parser.add_argument('--direct-trajectories', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f'No records in {path}')
    return rows


def group_trajectories(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row['trajectory_id'])].append(row)
    for trajectory_id, steps in grouped.items():
        steps.sort(key=lambda row: int(row['turn']))
        if [int(row['turn']) for row in steps] != list(range(len(steps))):
            raise ValueError(f'Broken trajectory turns: {trajectory_id}')
    return grouped


def stable_key(seed: int, trajectory_id: str) -> str:
    return hashlib.sha256(f'{seed}:{trajectory_id}'.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    direct_rows = read_jsonl(args.direct_jsonl)
    hard_rows = read_jsonl(args.hard_jsonl)
    direct = group_trajectories(direct_rows)
    hard = group_trajectories(hard_rows)
    if args.direct_trajectories > len(direct):
        raise ValueError(
            f'Requested {args.direct_trajectories} direct trajectories, '
            f'but only {len(direct)} exist'
        )

    selected_direct_ids = sorted(
        direct,
        key=lambda trajectory_id: stable_key(args.seed, trajectory_id),
    )[: args.direct_trajectories]
    selected_direct_rows = [
        row
        for trajectory_id in selected_direct_ids
        for row in direct[trajectory_id]
    ]
    selected_hard_rows = [
        row
        for trajectory_id in sorted(hard)
        for row in hard[trajectory_id]
    ]

    direct_asins = {str(row['target_asin']).upper() for row in direct_rows}
    hard_asins = {str(row['target_asin']).upper() for row in hard_rows}
    if not hard_asins <= direct_asins:
        raise ValueError(
            f'Hard data contains {len(hard_asins - direct_asins)} ASINs '
            'outside the SFT training pool'
        )
    eval_manifest = json.loads(args.eval_manifest.read_text(encoding='utf-8'))
    eval_asins = {str(asin).upper() for asin in eval_manifest['target_asins']}
    overlap = sorted(hard_asins & eval_asins)
    if overlap:
        raise ValueError(f'Hard-data/evaluation ASIN overlap: {overlap[:10]}')
    if any(not bool(row.get('independent_success')) for row in hard_rows):
        raise ValueError('Hard data contains a trajectory that failed independent checks')

    output_rows = selected_direct_rows + selected_hard_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='\n') as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    trajectory_modes = Counter(
        str(steps[0].get('trajectory_mode', 'unknown')) for steps in hard.values()
    )
    decision_types = Counter(
        str(row.get('decision_type', 'unknown')) for row in selected_hard_rows
    )
    manifest = {
        'direct_source': str(args.direct_jsonl.resolve()),
        'hard_source': str(args.hard_jsonl.resolve()),
        'output': str(args.output.resolve()),
        'seed': args.seed,
        'direct_trajectory_count': len(selected_direct_ids),
        'hard_trajectory_count': len(hard),
        'total_trajectory_count': len(selected_direct_ids) + len(hard),
        'direct_step_count': len(selected_direct_rows),
        'hard_step_count': len(selected_hard_rows),
        'total_step_count': len(output_rows),
        'hard_unique_asins': len(hard_asins),
        'hard_eval_asin_overlap': len(overlap),
        'hard_trajectory_modes': dict(trajectory_modes),
        'hard_decision_types': dict(decision_types),
        'output_sha256': file_sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
