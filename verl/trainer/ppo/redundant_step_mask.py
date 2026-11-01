"""GRPO冗余步骤识别与策略优势掩码。

该模块只处理Policy Loss中的步骤信用，不删除轨迹历史，也不修改环境奖励。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class RedundantStepMaskResult:
    """每个步骤是否保留，以及被判为冗余时的原因。"""

    keep_mask: np.ndarray
    reasons: np.ndarray


def _scalar(value):
    """把NumPy标量或单元素数组还原成普通Python值。"""

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_trace_text(value) -> str:
    """规范化页面或动作文本，用于稳定比较而不改变原始轨迹。"""

    value = _scalar(value)
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def normalize_trace_value(value) -> str:
    """为结构化浏览器会话状态生成稳定表示。"""

    value = _scalar(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return normalize_trace_text(value)


def build_state_fingerprint(observation, selected_asin=None, selected_options=None):
    """构造完整决策状态指纹，而不是只使用可见页面文本。"""

    return (
        normalize_trace_text(observation),
        normalize_trace_value(selected_asin),
        normalize_trace_value(selected_options),
    )


def build_redundant_step_keep_mask(
    *,
    traj_uids: Sequence,
    turn_steps: Sequence,
    projected_actions: Sequence,
    observations_before: Sequence,
    observations_after: Sequence,
    action_valids: Sequence,
    selected_asins_before: Sequence | None = None,
    selected_options_before: Sequence | None = None,
    selected_asins_after: Sequence | None = None,
    selected_options_after: Sequence | None = None,
    mask_no_state_change: bool = False,
    mask_repeated_state_action: bool = True,
    mask_invalid_action: bool = False,
) -> RedundantStepMaskResult:
    """识别保守、可验证的冗余步骤。

    默认只识别同一轨迹中重复出现的完整状态-动作对。完整状态包括页面、
    当前商品和已选规格，不能仅因页面文本未变化就判定动作冗余。

    ``mask_no_state_change`` 是保守的可选规则，默认关闭；即使显式开启，
    也会比较完整状态而不是只比较页面文本。

    非法动作默认不Mask，因为其负奖励应继续用于纠正策略。
    """

    fields = (
        traj_uids,
        turn_steps,
        projected_actions,
        observations_before,
        observations_after,
        action_valids,
    )
    sizes = {len(field) for field in fields}
    if len(sizes) != 1:
        raise ValueError(f"Redundancy-mask fields have different lengths: {sorted(sizes)}")

    size = sizes.pop() if sizes else 0

    optional_fields = {
        "selected_asins_before": selected_asins_before,
        "selected_options_before": selected_options_before,
        "selected_asins_after": selected_asins_after,
        "selected_options_after": selected_options_after,
    }
    for name, field in optional_fields.items():
        if field is not None and len(field) != size:
            raise ValueError(
                f"Redundancy-mask field {name} has length {len(field)}, expected {size}"
            )

    def optional_value(field, index):
        return None if field is None else field[index]

    keep_mask = np.ones(size, dtype=bool)
    reasons = np.full(size, "kept", dtype=object)

    ordered_indices = sorted(
        range(size),
        key=lambda index: (
            str(_scalar(traj_uids[index])),
            int(_scalar(turn_steps[index]) or 0),
            index,
        ),
    )
    seen_state_actions: dict[str, set[tuple[tuple[str, str, str], str]]] = defaultdict(set)

    for index in ordered_indices:
        trajectory_id = str(_scalar(traj_uids[index]))
        action = normalize_trace_text(projected_actions[index])
        before = build_state_fingerprint(
            observations_before[index],
            optional_value(selected_asins_before, index),
            optional_value(selected_options_before, index),
        )
        after = build_state_fingerprint(
            observations_after[index],
            optional_value(selected_asins_after, index),
            optional_value(selected_options_after, index),
        )
        is_valid = bool(_scalar(action_valids[index]))
        state_action = (before, action)

        # 默认保留非法动作，让已有的invalid-action penalty继续生效。
        if not is_valid:
            if mask_invalid_action:
                keep_mask[index] = False
                reasons[index] = "invalid_action"
            continue

        has_observation = bool(before[0])
        if mask_no_state_change and has_observation and before == after:
            keep_mask[index] = False
            reasons[index] = "no_state_change"
        elif (
            mask_repeated_state_action
            and has_observation
            and action
            and state_action in seen_state_actions[trajectory_id]
        ):
            keep_mask[index] = False
            reasons[index] = "repeated_state_action"

        if has_observation and action:
            seen_state_actions[trajectory_id].add(state_action)

    return RedundantStepMaskResult(keep_mask=keep_mask, reasons=reasons)


def mask_redundant_advantages(
    advantages: "torch.Tensor",
    keep_mask: Sequence[bool],
    *,
    positive_advantage_only: bool = True,
) -> "torch.Tensor":
    """把冗余步骤的Policy Advantage置零。

    默认只屏蔽正优势，避免成功轨迹强化冗余步骤；负优势仍然保留，
    使失败轨迹可以继续惩罚这些动作。
    """

    import torch

    if advantages.ndim < 2:
        raise ValueError(f"Expected batched token advantages, got {advantages.shape}")
    if advantages.shape[0] != len(keep_mask):
        raise ValueError(
            "Keep-mask batch size does not match advantages: "
            f"{len(keep_mask)} != {advantages.shape[0]}"
        )

    keep = torch.as_tensor(keep_mask, dtype=torch.bool, device=advantages.device)
    redundant_rows = (~keep).view(-1, *([1] * (advantages.ndim - 1)))
    if positive_advantage_only:
        masked_positions = redundant_rows & (advantages > 0)
    else:
        masked_positions = redundant_rows.expand_as(advantages)
    return torch.where(masked_positions, torch.zeros_like(advantages), advantages)
