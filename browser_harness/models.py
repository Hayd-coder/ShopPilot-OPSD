"""Browser Harness共享的数据结构与运行配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExecutionStatus(str, Enum):
    OK = "ok"
    INVALID_ACTION = "invalid_action"
    INFRA_FAILURE = "infra_failure"
    CONFIRMATION_REQUIRED = "confirmation_required"
    DUPLICATE_BLOCKED = "duplicate_blocked"
    UNCERTAIN_SIDE_EFFECT = "uncertain_side_effect"


@dataclass(frozen=True)
class HarnessConfig:
    base_url: str = "http://127.0.0.1:3000"
    headless: bool = True
    browser_channel: str | None = None
    action_timeout_ms: int = 10_000
    max_retries: int = 2
    backoff_base_seconds: float = 0.25
    backoff_jitter_seconds: float = 0.05
    require_purchase_confirmation: bool = True
    block_external_resources: bool = True
    demo_event_pause_seconds: float = 0.0
    screenshot_dir: Path | None = None
    trace_path: Path = Path("results/browser_harness/trace.jsonl")


@dataclass(frozen=True)
class BrowserObservation:
    session_id: str
    url: str
    page_type: str
    task: str
    text: str
    available_actions: tuple[str, ...]
    selected_options: dict[str, str] = field(default_factory=dict)
    reward: float | None = None
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["available_actions"] = list(self.available_actions)
        return data


@dataclass(frozen=True)
class ActionResult:
    action_id: str
    action: str
    status: ExecutionStatus
    observation: BrowserObservation
    retries: int = 0
    http_status: int | None = None
    infra_failure: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["observation"] = self.observation.to_dict()
        return data
