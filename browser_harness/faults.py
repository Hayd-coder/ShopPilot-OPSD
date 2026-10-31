"""用于稳定复现网页异常的确定性故障注入。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FaultKind(str, Enum):
    RATE_LIMIT = "rate_limit"
    LOGIN_EXPIRED = "login_expired"
    TIMEOUT = "timeout"


@dataclass
class FaultPlan:
    """按照Agent逻辑步骤配置仅触发一次的确定性故障。"""

    rate_limit_steps: frozenset[int] = frozenset()
    login_expiry_steps: frozenset[int] = frozenset()
    timeout_steps: frozenset[int] = frozenset()
    retry_after_seconds: float = 0.1
    _fired: set[tuple[FaultKind, int]] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def from_spec(cls, spec: str | None) -> "FaultPlan":
        values: dict[str, frozenset[int]] = {
            "rate_limit": frozenset(),
            "login": frozenset(),
            "timeout": frozenset(),
        }
        if spec:
            for section in spec.split(";"):
                if not section.strip():
                    continue
                key, separator, raw_steps = section.partition("=")
                if not separator or key.strip() not in values:
                    raise ValueError(
                        "Fault spec must look like rate_limit=2;login=3;timeout=4"
                    )
                parsed = frozenset(int(item) for item in raw_steps.split(",") if item.strip())
                if any(step <= 0 for step in parsed):
                    raise ValueError("Fault step numbers start at 1")
                values[key.strip()] = parsed
        return cls(
            rate_limit_steps=values["rate_limit"],
            login_expiry_steps=values["login"],
            timeout_steps=values["timeout"],
        )

    def take(self, logical_step: int) -> FaultKind | None:
        candidates = (
            (FaultKind.LOGIN_EXPIRED, self.login_expiry_steps),
            (FaultKind.RATE_LIMIT, self.rate_limit_steps),
            (FaultKind.TIMEOUT, self.timeout_steps),
        )
        for kind, steps in candidates:
            key = (kind, logical_step)
            if logical_step in steps and key not in self._fired:
                self._fired.add(key)
                return kind
        return None
