#!/usr/bin/env python3
"""Validate full-catalogue WebShop reset/search/click with shared sessions."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import ray

from agent_system.environments.env_package.webshop.envs import build_webshop_envs


def cgroup_memory_gib() -> float | None:
    path = "/sys/fs/cgroup/memory.current"
    try:
        with open(path, encoding="utf-8") as file:
            return int(file.read().strip()) / 1024**3
    except (OSError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-num", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--hold-seconds", type=int, default=0)
    args = parser.parse_args()

    repo = os.environ.get(
        "REPO_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
    webshop = os.path.join(
        repo,
        "agent_system/environments/env_package/webshop/webshop",
    )
    ray.init(include_dashboard=False, num_cpus=4)
    env = None
    try:
        started = time.monotonic()
        env = build_webshop_envs(
            seed=2026,
            env_num=args.env_num,
            group_n=args.group_size,
            is_train=True,
            resources_per_worker={"num_cpus": 0.1},
            env_kwargs={
                "observation_mode": "text",
                "num_products": None,
                "human_goals": False,
                "file_path": os.path.join(webshop, "data/items_shuffle.json"),
                "attr_path": os.path.join(webshop, "data/items_ins_v2.json"),
            },
        )
        observations, infos = env.reset()
        session_count = args.env_num * args.group_size
        assert len(observations) == len(infos) == session_count

        query = "black rubber sole loafers"
        observations, rewards, dones, infos = env.step(
            [f"search[{query}]"] * session_count,
        )
        assert all(not done for done in dones)

        click_actions: list[str] = []
        for info in infos:
            clickables = info["available_actions"].get("clickables", [])
            asin = next(
                (value for value in clickables if len(value) == 10 and value.isalnum()),
                None,
            )
            if asin is None:
                raise RuntimeError(f"No product ASIN found in clickables: {clickables[:10]}")
            click_actions.append(f"click[{asin}]")

        observations, rewards, dones, infos = env.step(click_actions)
        assert all("available_actions" in info for info in infos)
        elapsed = time.monotonic() - started
        memory = cgroup_memory_gib()
        print(
            "full_env_smoke=PASS "
            f"sessions={session_count} elapsed_s={elapsed:.1f} "
            f"cgroup_memory_gib={memory:.2f}" if memory is not None else
            f"full_env_smoke=PASS sessions={session_count} elapsed_s={elapsed:.1f}",
            flush=True,
        )
        if args.hold_seconds:
            time.sleep(args.hold_seconds)
    finally:
        if env is not None:
            env.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
