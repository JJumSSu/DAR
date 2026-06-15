#!/bin/bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
PROJECT_NAME="${PROJECT_NAME:-regression_LLM}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-FINAL_VERIFY_LIPO_qwen3_instruct_4b_DAR}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/checkpoints}"
CKPT_DIR="${CKPT_DIR:-$CHECKPOINT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME}"
PRED_OUTPUT_DIR="${PRED_OUTPUT_DIR:-$PROJECT_ROOT/evaluate/outputs}"

cd "$PROJECT_ROOT"

for actor_dir in "$CKPT_DIR"/global_step_*/actor; do
    step_dir="$(dirname "$actor_dir")"
    target_dir="$step_dir/hf"

    if [ ! -d "$target_dir" ]; then
        echo "Skipping (missing hf directory): $step_dir"
        continue
    fi

    python3 -m evaluate.run_evaluation \
        --model "$target_dir" \
        --mode rl \
        --temperature "${TEMPERATURE:-1.0}" \
        --n-decode "${N_DECODE:-32}" \
        --batch-size "${BATCH_SIZE:-4096}" \
        --task "lipo" \
        --root-dir "$DATA_DIR" \
        --pred_output_dir "$PRED_OUTPUT_DIR"
done

shopt -u nullglob
