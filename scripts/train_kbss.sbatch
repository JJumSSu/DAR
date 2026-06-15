#!/bin/bash
#SBATCH --job-name=dar_kbss
#SBATCH --nodes=1
#SBATCH --partition=ai
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:4
#SBATCH --constraint=H100
#SBATCH --time=36:00:00
#SBATCH --output=dar_kbss.out
#SBATCH --error=dar_kbss.err

echo "START TIME: $(date)"
echo "Available GPUs:"
nvidia-smi

export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=TRUE

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/checkpoints}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}"
CONDA_ENV="${CONDA_ENV:-py310}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"

cd "$PROJECT_ROOT/trainer/verl"
module load conda
conda activate "$CONDA_ENV"
unset ROCR_VISIBLE_DEVICES

export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

export PROJECT_NAME="${PROJECT_NAME:-regression_LLM}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-FINAL_VERIFY_KBSS_qwen3_instruct_4b_DAR}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 algorithm.use_kl_in_reward=False \
 data.train_files="$DATA_DIR/KBSS_rl_train.parquet" \
 data.val_files="$DATA_DIR/KBSS_rl_valid.parquet" \
 data.train_batch_size=256 \
 data.max_prompt_length=4500 \
 data.max_response_length=32 \
 data.filter_overlong_prompts=False \
 data.truncation='error' \
 actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.rollout.n=12 \
 actor_rollout_ref.rollout.temperature=1.0 \
 actor_rollout_ref.rollout.top_p=1.0 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.ref.fsdp_config.param_offload=False  \
 reward_model.reward_manager=regression_dar \
 trainer.critic_warmup=0 \
 trainer.logger="$TRAINER_LOGGER" \
 trainer.project_name=${PROJECT_NAME} \
 trainer.experiment_name=${EXPERIMENT_NAME} \
 trainer.default_local_dir="$CHECKPOINT_DIR/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=30 \
 trainer.test_freq=10 \
 +trainer.log_freq=10 \
 trainer.total_epochs=10 \
 trainer.val_before_train=True
