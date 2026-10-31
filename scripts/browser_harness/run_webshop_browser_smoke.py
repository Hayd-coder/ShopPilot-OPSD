"""针对已有WebShop服务运行浏览器动作烟测。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from browser_harness import FaultPlan, HarnessConfig, PlaywrightWebShopHarness  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShopPilot actions through a real browser")
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--channel", default=None, help="For Windows, msedge avoids a browser download")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--faults", default="", help="Example: rate_limit=1;login=2;timeout=3")
    parser.add_argument("--trace", type=Path, default=Path("results/browser_harness/smoke.jsonl"))
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--confirm-purchase", action="store_true")
    args = parser.parse_args()

    config = HarnessConfig(
        base_url=args.base_url,
        browser_channel=args.channel,
        headless=not args.headed,
        trace_path=args.trace,
    )
    with PlaywrightWebShopHarness(config, FaultPlan.from_spec(args.faults)) as harness:
        observation = harness.reset(task_index=args.task_index)
        print(json.dumps(observation.to_dict(), ensure_ascii=False, indent=2))
        for action in args.action:
            result = harness.step(action, confirm_purchase=args.confirm_purchase)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            if result.observation.done or result.infra_failure:
                break


if __name__ == "__main__":
    main()
