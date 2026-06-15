#!/bin/bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROJECT_NAME="${PROJECT_NAME:-regression_LLM}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-FINAL_VERIFY_KBSS_qwen3_instruct_4b_DAR}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/checkpoints}"
CKPT_DIR="${CKPT_DIR:-$CHECKPOINT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME}"
DELETE_ACTOR_WEIGHTS_AFTER_MERGE="${DELETE_ACTOR_WEIGHTS_AFTER_MERGE:-false}"

cd "$PROJECT_ROOT/trainer/verl"

for actor_dir in "$CKPT_DIR"/global_step_*/actor; do
    step_dir="$(dirname "$actor_dir")"
    target_dir="$step_dir/hf"

    if [ -d "$target_dir" ] && [ -n "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        echo "Skipping (already converted): $step_dir"
        continue
    fi

    mkdir -p "$target_dir"
    echo "Converting: $actor_dir -> $target_dir"
    python -m verl.model_merger merge --backend fsdp --local_dir "$actor_dir" --target_dir "$target_dir"

    if [ "$DELETE_ACTOR_WEIGHTS_AFTER_MERGE" = "true" ]; then
        find "$actor_dir" -maxdepth 1 -type f -name '*.pt' -print -delete
    fi
done

shopt -u nullglob
