# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import ray
import gym
import numpy as np


def map_webshop_training_reward(
    task_score: float,
    done: bool,
    reward_mode: str = 'sparse_success',
    reward_scale: float = 10.0,
) -> tuple[float, bool]:
    """Map WebShop's native [0, 1] score to the RL terminal reward."""
    task_score = float(task_score)
    won = bool(done and task_score >= 1.0 - 1e-8)
    if reward_mode == 'graded_terminal' and done:
        return reward_scale * task_score, won
    if reward_mode == 'sparse_success' and won:
        return reward_scale, won
    return 0.0, won

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray actor hosting many sessions that share one immutable product server.

    A full WebShop ``SimServer`` materializes roughly 1.18M products.  Creating
    one Ray actor per rollout therefore duplicates the full catalogue once per
    trajectory.  ``WebAgentTextEnv`` already accepts an existing ``server``;
    use that hook to keep independent browser/session state while sharing the
    catalogue, goals, prices, and Lucene searcher inside one actor.
    """
    
    def __init__(self, seeds, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        self.envs = []
        self.reward_mode = env_kwargs.get('reward_mode', 'sparse_success')
        self.reward_scale = float(env_kwargs.get('reward_scale', 10.0))
        if self.reward_mode not in {'sparse_success', 'graded_terminal'}:
            raise ValueError(
                'WebShop reward_mode must be sparse_success or graded_terminal, '
                f'got {self.reward_mode!r}',
            )
        shared_server = None
        for slot, seed in enumerate(seeds):
            kwargs = dict(env_kwargs)
            kwargs['seed'] = seed
            # Multiple GRPO trajectories intentionally use the same goal id.
            # Prefix session ids so those trajectories do not overwrite one
            # another in the shared SimServer.user_sessions dictionary.
            kwargs['session_prefix'] = f'slot-{slot}-'
            if shared_server is not None:
                kwargs['server'] = shared_server
            env = gym.make('WebAgentTextEnv-v0', **kwargs)
            if shared_server is None:
                shared_server = env.unwrapped.server
            self.envs.append(env)

    def _step_one(self, env, action):
        obs, reward, done, info = env.step(action)
        info = dict(info or {})
        task_score = float(reward)
        training_reward, won = map_webshop_training_reward(
            task_score=task_score,
            done=done,
            reward_mode=self.reward_mode,
            reward_scale=self.reward_scale,
        )

        info['available_actions'] = env.get_available_actions()
        info['task_score'] = task_score
        info['training_reward'] = training_reward
        info['reward_mode'] = self.reward_mode
        info['won'] = won
        return obs, training_reward, done, info
    
    def step(self, actions):
        """Execute one action for every logical environment session."""
        if len(actions) != len(self.envs):
            raise ValueError(f'Expected {len(self.envs)} actions, got {len(actions)}')
        return [self._step_one(env, action) for env, action in zip(self.envs, actions)]
    
    def reset(self, indices):
        """Reset every logical environment with its assigned goal index."""
        if len(indices) != len(self.envs):
            raise ValueError(f'Expected {len(self.envs)} indices, got {len(indices)}')
        results = []
        for env, idx in zip(self.envs, indices):
            obs, info = env.reset(session=idx)
            info = dict(info or {})
            info['available_actions'] = env.get_available_actions()
            info['selected_asin'] = None
            info['selected_options'] = {}
            info['reward_breakdown'] = {}
            info['won'] = False
            results.append((obs, info))
        return results
    
    def render(self, mode_for_render, env_idx=None):
        """Render one logical environment or all of them."""
        if env_idx is not None:
            return self.envs[env_idx].render(mode=mode_for_render)
        return [env.render(mode=mode_for_render) for env in self.envs]
    
    def get_available_actions(self):
        """Get available actions for every logical environment."""
        return [env.get_available_actions() for env in self.envs]
    
    def get_num_goals(self):
        """Return only the goal count, avoiding a large Ray object copy."""
        return len(self.envs[0].unwrapped.server.goals)
    
    def close(self):
        """Close all logical environments."""
        for env in self.envs:
            env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self.seed = seed
        self._rng = np.random.RandomState(seed)

        self._env_kwargs = dict(
            env_kwargs
            if env_kwargs is not None
            else {'observation_mode': 'text', 'num_products': None}
        )
        # Wrapper-only option: do not forward it to WebAgentTextEnv.
        validation_goal_indices_file = self._env_kwargs.pop(
            'validation_goal_indices_file', None
        )

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        worker_seeds = [seed + (i // self.group_n) for i in range(self.num_processes)]
        self._worker = env_worker.remote(worker_seeds, self._env_kwargs)

        # Only the count is needed here.  Returning the full goal list copies it
        # from the Ray actor into the trainer process; on the full catalogue that
        # list contains more than eleven million synthetic goals.
        num_goals = ray.get(self._worker.get_num_goals.remote())

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if num_goals < 2:
            raise ValueError(
                f"WebShop needs at least two goals for train/validation splits, got {num_goals}.",
            )

        # Keep the repository's original 500-goal validation split for the
        # full dataset.  The bundled 1k-product smoke dataset has far fewer
        # matched human goals, so reserve 20% (at least one) for validation
        # instead of leaving its training split empty.
        validation_size = 500 if num_goals > 500 else max(1, num_goals // 5)
        if not self.is_train:
            self.goal_idxs = range(validation_size)
        else:
            self.goal_idxs = range(validation_size, num_goals)

        if not self.is_train and validation_goal_indices_file:
            split_path = Path(validation_goal_indices_file).expanduser()
            with split_path.open('r', encoding='utf-8') as handle:
                split_payload = json.load(handle)
            if isinstance(split_payload, dict):
                split_payload = split_payload.get('goal_indices')
            if not isinstance(split_payload, list) or not split_payload:
                raise ValueError(
                    'validation_goal_indices_file must contain a non-empty '
                    'JSON list or {"goal_indices": [...]}'
                )
            explicit_indices = tuple(int(index) for index in split_payload)
            if len(set(explicit_indices)) != len(explicit_indices):
                raise ValueError('Validation goal indices must be unique')
            invalid = [
                index for index in explicit_indices
                if index < 0 or index >= num_goals
            ]
            if invalid:
                raise ValueError(
                    f'Validation goal indices outside [0, {num_goals}): '
                    f'{invalid[:10]}'
                )
            self.goal_idxs = explicit_indices

        print(
            f"WebShop {'train' if self.is_train else 'validation'} split: "
            f"{len(self.goal_idxs)} of {num_goals} goals",
        )

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Environment transitions are cheap compared with model generation.
        # Execute them in one shared-catalogue actor to avoid N catalogue copies.
        results = ray.get(self._worker.step.remote(actions))
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        results = ray.get(self._worker.reset.remote(idx))
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def reset_sampling_rng(self):
        """Restart task sampling so repeated validation uses identical goals."""
        self._rng = np.random.RandomState(self.seed)

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        return ray.get(self._worker.render.remote(mode, env_idx))

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        ray.get(self._worker.close.remote())
        ray.kill(self._worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
