"""CPU-only checks for the reversible WebShop combined experiment switches."""

import numpy as np
import torch

from agent_system.environments.env_package.webshop.envs import (
    map_webshop_training_reward,
)
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def test_reward_mapping() -> None:
    reward, won = map_webshop_training_reward(0.8, True, "graded_terminal", 10.0)
    assert reward == 8.0 and not won

    reward, won = map_webshop_training_reward(0.8, True, "sparse_success", 10.0)
    assert reward == 0.0 and not won

    reward, won = map_webshop_training_reward(1.0, True, "graded_terminal", 10.0)
    assert reward == 10.0 and won

    reward, won = map_webshop_training_reward(1.0, True, "sparse_success", 10.0)
    assert reward == 10.0 and won


def test_trajectory_level_advantage() -> None:
    # One task has two trajectories. Trajectory A has two decision steps with
    # reward 0; trajectory B has one step with reward 10. Each trajectory must
    # count once, hence the no-std advantages are [-5, -5, +5].
    rewards = torch.tensor([[0.0], [0.0], [10.0]])
    mask = torch.ones_like(rewards)
    uid = np.asarray(["task-0", "task-0", "task-0"], dtype=object)
    traj_uid = np.asarray(["traj-a", "traj-a", "traj-b"], dtype=object)

    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=uid,
        traj_index=traj_uid,
        norm_adv_by_std_in_grpo=False,
        compute_mean_std_cross_steps=False,
    )
    expected = torch.tensor([[-5.0], [-5.0], [5.0]])
    assert torch.allclose(advantages, expected), (advantages, expected)


if __name__ == "__main__":
    test_reward_mapping()
    test_trajectory_level_advantage()
    print("WebShop combined logic checks: PASS")
