#!/usr/bin/env bash
set -euo pipefail

# Evaluate one policy checkpoint on a deterministic WebShop task set.
# Example:
#   MODEL_PATH=/root/autodl-tmp/models/qwen25-3b-sft \
#   VAL_DATA_SIZE=128 SEED=0 \
#   bash scripts/evaluation/eval_fixed_tasks.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
ENV_PREFIX="${WEBSHOP_AGENT_ENV_PREFIX:-/root/autodl-tmp/conda_envs/webshop-agent}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a merged model or Hugging Face checkpoint}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-webshop_fixed_eval}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-128}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-15}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/webshop-agent-runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/evaluations/${EXPERIMENT_NAME}}"

source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
cd "${PROJECT_ROOT}"

python scripts/evaluation/prepare_smoke_parquet.py --train-size 1 --val-size "${VAL_DATA_SIZE}"
mkdir -p "${OUTPUT_DIR}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${HOME}/data/verl-agent/text/train.parquet" \
  data.val_files="${HOME}/data/verl-agent/text/test.parquet" \
  data.train_batch_size=1 \
  data.val_batch_size="${VAL_DATA_SIZE}" \
  data.max_prompt_length="${PROMPT_LENGTH}" \
  data.max_response_length="${RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=true data.truncation=error data.return_raw_chat=true \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  algorithm.use_kl_in_reward=false \
  env.env_name=Webshop env.seed="${SEED}" env.max_steps="${MAX_STEPS}" env.rollout.n=1 \
  env.resources_per_worker.num_cpus=0.1 \
  env.webshop.use_small=true \
  env.webshop.match_search_index_to_catalog=true \
  env.webshop.fixed_goal_split=true \
  env.webshop.goal_shuffle_seed="${SEED}" \
  env.webshop.reward_mode=sparse_success \
  env.webshop.reward_scale=10.0 \
  trainer.logger="['console']" trainer.project_name=webshop_agent_rl \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.validation_data_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps=1 \
  trainer.val_before_train=true trainer.val_only=true trainer.resume_mode=disable "$@"
