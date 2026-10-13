#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_PREFIX="${WEBSHOP_AGENT_ENV_PREFIX:-/root/autodl-tmp/conda_envs/webshop-agent}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/webshop-agent-runs}"
MODEL_PATH="${MODEL_PATH:-${RUN_ROOT}/models/qwen25-3b-webshop-agent}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25-3b-webshop-grpo-s30}"

source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "${REPO_ROOT}"

python scripts/evaluation/prepare_smoke_parquet.py --train-size 8 --val-size 32
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${HOME}/data/verl-agent/text/train.parquet" \
    data.val_files="${HOME}/data/verl-agent/text/test.parquet" \
    data.train_batch_size=8 data.val_batch_size=32 \
    data.max_prompt_length=4096 data.max_response_length=512 \
    data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False algorithm.compute_mean_std_cross_steps=True \
    algorithm.norm_adv_by_std_in_grpo=True \
    env.env_name=Webshop env.seed=0 env.max_steps=15 env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 env.webshop.use_small=true \
    env.webshop.match_search_index_to_catalog=true env.webshop.fixed_goal_split=true \
    env.webshop.goal_shuffle_seed=0 env.webshop.reward_mode=sparse_success \
    env.webshop.reward_scale=10.0 env.webshop.expose_selected_options=false \
    trainer.critic_warmup=0 trainer.logger="['console']" \
    trainer.project_name=webshop_agent_rl trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.rollout_data_dir="${RUN_ROOT}/rollouts/${EXPERIMENT_NAME}" \
    trainer.validation_data_dir="${RUN_ROOT}/evaluations/${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${RUN_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
    trainer.max_actor_ckpt_to_keep=4 trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    trainer.ray_wait_register_center_timeout=600 trainer.save_freq=10 trainer.test_freq=10 \
    trainer.total_epochs=30 trainer.total_training_steps=30 \
    trainer.val_before_train=True trainer.resume_mode=disable "$@"

