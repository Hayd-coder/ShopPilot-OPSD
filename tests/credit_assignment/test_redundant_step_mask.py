"""冗余步骤Mask的单元测试。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl"
    / "trainer"
    / "ppo"
    / "redundant_step_mask.py"
)
SPEC = importlib.util.spec_from_file_location("redundant_step_mask_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
build_redundant_step_keep_mask = MODULE.build_redundant_step_keep_mask
mask_redundant_advantages = MODULE.mask_redundant_advantages


def test_detects_only_repeated_full_state_action_by_default() -> None:
    result = build_redundant_step_keep_mask(
        traj_uids=np.array(["t1", "t1", "t1", "t1", "t2"], dtype=object),
        turn_steps=np.array([0, 1, 2, 3, 0], dtype=object),
        projected_actions=np.array(
            ["search[shoes]", "click[black]", "click[black]", "click[black]", "click[black]"],
            dtype=object,
        ),
        observations_before=np.array(["Home", "Item", "Item", "Item", "Item"], dtype=object),
        observations_after=np.array(["Results", "Item", "Item", "Item", "Item"], dtype=object),
        action_valids=np.array([True, True, True, True, True], dtype=object),
        selected_asins_before=np.array([None, "B001", "B001", "B001", "B001"], dtype=object),
        selected_options_before=np.array(
            [{}, {}, {"color": "black"}, {"color": "black"}, {"color": "black"}],
            dtype=object,
        ),
        selected_asins_after=np.array([None, "B001", "B001", "B001", "B001"], dtype=object),
        selected_options_after=np.array(
            [{}, {"color": "black"}, {"color": "black"}, {"color": "black"}, {"color": "black"}],
            dtype=object,
        ),
    )

    # 第一次选择颜色会改变会话状态，即使页面文本不变也必须保留；只有后续在
    # 完整状态下再次执行相同动作才会被Mask，另一条轨迹则拥有独立历史。
    assert result.keep_mask.tolist() == [True, True, True, False, True]
    assert result.reasons.tolist() == [
        "kept",
        "kept",
        "kept",
        "repeated_state_action",
        "kept",
    ]


def test_same_page_option_selection_is_not_a_noop() -> None:
    result = build_redundant_step_keep_mask(
        traj_uids=["t1"],
        turn_steps=[0],
        projected_actions=["click[black]"],
        observations_before=["Product B001"],
        observations_after=["Product B001"],
        action_valids=[True],
        selected_asins_before=["B001"],
        selected_options_before=[{}],
        selected_asins_after=["B001"],
        selected_options_after=[{"color": "black"}],
        mask_no_state_change=True,
    )

    assert result.keep_mask.tolist() == [True]
    assert result.reasons.tolist() == ["kept"]


def test_invalid_action_is_kept_by_default_for_negative_penalty() -> None:
    result = build_redundant_step_keep_mask(
        traj_uids=["t1"],
        turn_steps=[0],
        projected_actions=["click[missing]"],
        observations_before=["Search"],
        observations_after=["Search"],
        action_valids=[False],
    )

    assert result.keep_mask.tolist() == [True]
    assert result.reasons.tolist() == ["kept"]


def test_positive_only_mask_preserves_negative_advantage() -> None:
    torch = pytest.importorskip("torch")
    advantages = torch.tensor([[0.8, 0.8], [-0.5, -0.5], [0.2, 0.2]])
    masked = mask_redundant_advantages(
        advantages,
        keep_mask=[False, False, True],
        positive_advantage_only=True,
    )

    assert torch.equal(
        masked,
        torch.tensor([[0.0, 0.0], [-0.5, -0.5], [0.2, 0.2]]),
    )


def test_full_mask_zeros_both_positive_and_negative_advantage() -> None:
    torch = pytest.importorskip("torch")
    advantages = torch.tensor([[0.8], [-0.5]])
    masked = mask_redundant_advantages(
        advantages,
        keep_mask=[False, False],
        positive_advantage_only=False,
    )

    assert torch.equal(masked, torch.zeros_like(advantages))
