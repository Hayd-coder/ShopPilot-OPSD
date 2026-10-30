"""Failure-aware step credit for GiGPO-style multi-turn training.

This module is an experimental extension, not a verbatim implementation of
upstream GiGPO.  Upstream GiGPO combines an episode-relative advantage with an
anchor-state step-relative advantage.  That is already implemented in
``gigpo/core_gigpo.py``.

The extension below adds a third signal: an empirical transition-potential
advantage.  For every repeated anchor state, it estimates a leave-one-
trajectory-out value from peer rollouts.  It then compares the estimated value
before and after an action.  Consequently, an early action in a failed episode
can still receive positive credit when it moves to a state that peer rollouts
usually finish successfully from, while the later action that destroys that
progress can receive negative credit.

No critic, reward model, hidden WebShop answer, or extra rollout is required.
Singleton states safely fall back to zero transition credit.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable, Sequence

import numpy as np
import torch

from gigpo import core_gigpo


def _as_object_array(values: Sequence[Any], name: str, size: int) -> np.ndarray:
    if len(values) != size:
        raise ValueError(f"{name} must be one-dimensional with length {size}")
    # ``np.asarray(list_of_equal_length_lists, dtype=object)`` may silently
    # create a 2-D object array.  Explicit allocation preserves one state per
    # flattened rollout row for text, image, list, and dict observations.
    array = np.empty(size, dtype=object)
    array[:] = list(values)
    return array


def _state_keys(anchor_obs: np.ndarray, prompt_ids: np.ndarray) -> list[Hashable]:
    """Build deterministic task-local state keys from environment anchors."""
    return [
        (core_gigpo.to_hashable(prompt_id), core_gigpo.to_hashable(observation))
        for prompt_id, observation in zip(prompt_ids, anchor_obs)
    ]


def _normalize_within_groups(
    values: torch.Tensor,
    group_keys: Sequence[Hashable],
    valid: torch.Tensor,
    *,
    normalize_by_std: bool,
    epsilon: float,
) -> torch.Tensor:
    """Mean-center valid values inside each group; singleton groups return zero."""
    output = torch.zeros_like(values)
    members: dict[Hashable, list[int]] = defaultdict(list)
    valid_cpu = valid.detach().cpu().tolist()

    for row, (key, is_valid) in enumerate(zip(group_keys, valid_cpu)):
        if is_valid:
            members[key].append(row)

    with torch.no_grad():
        for rows in members.values():
            # Relative advantages require at least two comparable actions.
            if len(rows) < 2:
                continue
            row_index = torch.as_tensor(rows, device=values.device)
            group_values = values[row_index]
            centered = group_values - group_values.mean()
            if normalize_by_std:
                # correction=0 avoids NaN and matches a population statistic
                # over the actions observed in this rollout group.
                centered = centered / (group_values.std(correction=0) + epsilon)
            output[row_index] = centered

    return output


def _peer_state_potentials(
    step_returns: torch.Tensor,
    state_keys: Sequence[Hashable],
    trajectory_ids: np.ndarray,
    *,
    min_peer_trajectories: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate V(s) from other trajectories that visited the same state.

    Repeated visits from one trajectory are averaged first, so loops cannot
    masquerade as independent evidence.  The current trajectory is then left
    out to prevent its own terminal outcome from defining its state value.
    """
    by_state_and_traj: dict[Hashable, dict[Hashable, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    returns_cpu = step_returns.detach().cpu().tolist()

    for key, trajectory_id, value in zip(state_keys, trajectory_ids, returns_cpu):
        by_state_and_traj[key][core_gigpo.to_hashable(trajectory_id)].append(float(value))

    trajectory_means: dict[Hashable, dict[Hashable, float]] = {}
    for key, trajectory_values in by_state_and_traj.items():
        trajectory_means[key] = {
            trajectory_id: float(np.mean(values))
            for trajectory_id, values in trajectory_values.items()
        }

    potentials = torch.zeros_like(step_returns)
    available = torch.zeros_like(step_returns, dtype=torch.bool)
    for row, (key, trajectory_id) in enumerate(zip(state_keys, trajectory_ids)):
        current_id = core_gigpo.to_hashable(trajectory_id)
        peers = [
            value
            for peer_id, value in trajectory_means[key].items()
            if peer_id != current_id
        ]
        if len(peers) >= min_peer_trajectories:
            potentials[row] = float(np.mean(peers))
            available[row] = True

    return potentials, available


def _next_rows(trajectory_ids: np.ndarray, turn_steps: np.ndarray) -> np.ndarray:
    """Return the next flattened row in each trajectory, or -1 at terminal rows."""
    next_row = np.full(len(trajectory_ids), -1, dtype=np.int64)
    rows_by_trajectory: dict[Hashable, list[int]] = defaultdict(list)
    for row, trajectory_id in enumerate(trajectory_ids):
        rows_by_trajectory[core_gigpo.to_hashable(trajectory_id)].append(row)

    for rows in rows_by_trajectory.values():
        rows.sort(key=lambda row: int(turn_steps[row]))
        for current, following in zip(rows[:-1], rows[1:]):
            next_row[current] = following
    return next_row


def compute_failure_aware_gigpo_advantage(
    token_level_rewards: torch.Tensor,
    step_returns: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: Sequence[Any],
    prompt_ids: Sequence[Any],
    trajectory_ids: Sequence[Any],
    turn_steps: Sequence[int],
    *,
    gamma: float = 1.0,
    episode_weight: float = 1.0,
    anchor_weight: float = 1.0,
    transition_weight: float = 1.0,
    normalize_by_std: bool = True,
    min_peer_trajectories: int = 1,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute episode, anchor-action, and failure-aware transition credit.

    Args:
        token_level_rewards: ``[num_steps, response_length]`` episode reward
            tensor used by the existing GRPO/GiGPO path.
        step_returns: ``[num_steps]`` discounted return-to-go for each
            environment step, e.g. ``compute_step_discounted_returns`` output.
        response_mask: valid generated-token mask with the same shape as
            ``token_level_rewards``.
        anchor_obs: environment observation before each action.
        prompt_ids: task/group id for each flattened step.
        trajectory_ids: trajectory id for each flattened step.
        turn_steps: zero-based turn index inside each trajectory.
        gamma: discount used for the before/after potential difference.
        episode_weight: coefficient for ordinary GRPO episode credit.
        anchor_weight: coefficient for upstream GiGPO anchor-state credit.
        transition_weight: coefficient for the new transition-potential credit.
        normalize_by_std: standardize, rather than only mean-center, local
            step signals.
        min_peer_trajectories: minimum number of other trajectories required
            to estimate a state value.
        epsilon: numerical stabilizer.

    Returns:
        ``(advantages, returns, metrics)``.  Advantages are broadcast only to
        response tokens belonging to the corresponding environment step.
    """
    if token_level_rewards.ndim != 2:
        raise ValueError("token_level_rewards must have shape [num_steps, response_length]")
    if response_mask.shape != token_level_rewards.shape:
        raise ValueError("response_mask must match token_level_rewards")

    num_steps = token_level_rewards.shape[0]
    if step_returns.ndim != 1 or len(step_returns) != num_steps:
        raise ValueError("step_returns must have shape [num_steps]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if min_peer_trajectories < 1:
        raise ValueError("min_peer_trajectories must be at least 1")

    anchor_array = _as_object_array(anchor_obs, "anchor_obs", num_steps)
    prompt_array = _as_object_array(prompt_ids, "prompt_ids", num_steps)
    trajectory_array = _as_object_array(trajectory_ids, "trajectory_ids", num_steps)
    turn_array = np.asarray(turn_steps)
    if turn_array.ndim != 1 or len(turn_array) != num_steps:
        raise ValueError(f"turn_steps must be one-dimensional with length {num_steps}")

    step_returns = step_returns.to(device=token_level_rewards.device, dtype=token_level_rewards.dtype)
    state_keys = _state_keys(anchor_array, prompt_array)
    all_valid = torch.ones(num_steps, dtype=torch.bool, device=step_returns.device)

    # Macro credit: standard episode-relative GRPO, deduplicated by trajectory
    # so long trajectories do not receive extra weight when estimating mean/std.
    episode_advantage = core_gigpo.episode_norm_reward(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=prompt_array,
        traj_index=trajectory_array,
        epsilon=epsilon,
        remove_std=not normalize_by_std,
        compute_mean_std_cross_steps=False,
    )

    # Micro credit from upstream GiGPO: compare returns of actions taken from
    # the same task-local anchor state.
    anchor_step_scores = _normalize_within_groups(
        step_returns,
        state_keys,
        all_valid,
        normalize_by_std=normalize_by_std,
        epsilon=epsilon,
    )

    # New failure-aware signal: estimate current/next state potentials from
    # peer trajectories, then credit the transition by its potential change.
    state_potential, state_available = _peer_state_potentials(
        step_returns,
        state_keys,
        trajectory_array,
        min_peer_trajectories=min_peer_trajectories,
    )
    next_row = _next_rows(trajectory_array, turn_array)
    episode_scores = token_level_rewards.sum(dim=-1).detach()
    next_potential = torch.zeros_like(step_returns)
    transition_available = state_available.clone()

    for row, following in enumerate(next_row):
        if following >= 0:
            next_potential[row] = state_potential[following]
            transition_available[row] &= state_available[following]
        else:
            # The outcome of a terminal transition is observed exactly.
            next_potential[row] = episode_scores[row]

    raw_transition_credit = gamma * next_potential - state_potential
    transition_scores = _normalize_within_groups(
        raw_transition_credit,
        state_keys,
        transition_available,
        normalize_by_std=normalize_by_std,
        epsilon=epsilon,
    )

    anchor_advantage = anchor_step_scores.unsqueeze(-1) * response_mask
    transition_advantage = transition_scores.unsqueeze(-1) * response_mask
    advantages = (
        episode_weight * episode_advantage
        + anchor_weight * anchor_advantage
        + transition_weight * transition_advantage
    )

    group_sizes: dict[Hashable, int] = defaultdict(int)
    for key in state_keys:
        group_sizes[key] += 1
    repeated_rows = sum(size for size in group_sizes.values() if size > 1)
    failed_rows = episode_scores <= epsilon
    failed_transition_rows = failed_rows & transition_available
    failed_positive = failed_transition_rows & (transition_scores > 0)

    metrics = {
        "fa_gigpo/state_group_count": float(len(group_sizes)),
        "fa_gigpo/repeated_state_row_ratio": repeated_rows / max(num_steps, 1),
        "fa_gigpo/transition_coverage": transition_available.float().mean().item(),
        "fa_gigpo/anchor_credit_nonzero_ratio": (anchor_step_scores.abs() > epsilon).float().mean().item(),
        "fa_gigpo/transition_credit_nonzero_ratio": (transition_scores.abs() > epsilon).float().mean().item(),
        "fa_gigpo/failed_positive_step_ratio": (
            failed_positive.float().sum() / failed_transition_rows.float().sum().clamp(min=1)
        ).item(),
        "fa_gigpo/transition_credit_mean": transition_scores.mean().item(),
        "fa_gigpo/transition_credit_std": transition_scores.std(correction=0).item(),
    }

    return advantages, advantages, metrics

