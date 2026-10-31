"""ShopPilot动作协议的解析与合法性校验。"""

from __future__ import annotations

import re
from dataclasses import dataclass


_ACTION_RE = re.compile(r"^(search|click)\[(.+)]$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    argument: str

    @property
    def normalized(self) -> str:
        return f"{self.kind}[{self.argument}]"

    @property
    def has_side_effect(self) -> bool:
        return self.kind == "click" and self.argument.casefold() == "buy now"


def parse_action(action: str) -> ParsedAction:
    match = _ACTION_RE.fullmatch(action.strip())
    if match is None:
        raise ValueError("Action must use search[...] or click[...] syntax")
    kind, argument = match.groups()
    argument = " ".join(argument.split())
    if not argument:
        raise ValueError("Action argument cannot be empty")
    return ParsedAction(kind=kind.casefold(), argument=argument)


def match_legal_action(action: ParsedAction, available_actions: tuple[str, ...]) -> str | None:
    if action.kind == "search":
        has_search = any(
            candidate.casefold().startswith("search[") for candidate in available_actions
        )
        return action.normalized if has_search else None

    expected = action.normalized.casefold()
    for candidate in available_actions:
        if candidate.casefold() == expected:
            return candidate
    return None
