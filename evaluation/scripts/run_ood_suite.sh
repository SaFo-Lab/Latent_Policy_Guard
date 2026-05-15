#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the full out-of-distribution evaluation suite:
#   3 models {LPG, DynaGuard-4B, Qwen3-4B} x 3 datasets
#   {WildGuardTest (prompt-only), PolicyGuardBench (test), HarmBench}
#
# Each (model, dataset) combo is dispatched to a GPU; jobs run in parallel
# in waves of NUM_GPUS.
#
# Required env vars:
#   QWEN_BASE_PATH         base Qwen3-4B HF dir (used as LPG base + Qwen baseline)
#   LPG_CKPT_DIR           LPG checkpoint dir (model.safetensors)
#   DYNAGUARD_4B_PATH      DynaGuard-4B HF dir
#
# Optional env vars:
#   NUM_GPUS               default 4
#   RESULTS_DIR            default results/ood_suite
#   HARMBENCH_PATH         default walledai/HarmBench (HF id)
#   WILDGUARD_PATH         default allenai/wildguardmix (HF id)
#   POLICYGUARDBENCH_PATH  default Rakancorle1/PolicyGuardBench (HF id)
# ---------------------------------------------------------------------------
set -euo pipefail

CDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CDIR"

: "${QWEN_BASE_PATH:?set QWEN_BASE_PATH (Qwen3-4B base model dir)}"
: "${LPG_CKPT_DIR:?set LPG_CKPT_DIR (LPG checkpoint dir)}"
: "${DYNAGUARD_4B_PATH:?set DYNAGUARD_4B_PATH (DynaGuard-4B model dir)}"

NUM_GPUS="${NUM_GPUS:-4}"
RESULTS_DIR="${RESULTS_DIR:-results/ood_suite}"
HARMBENCH_PATH="${HARMBENCH_PATH:-walledai/HarmBench}"
WILDGUARD_PATH="${WILDGUARD_PATH:-allenai/wildguardmix}"
POLICYGUARDBENCH_PATH="${POLICYGUARDBENCH_PATH:-Rakancorle1/PolicyGuardBench}"

mkdir -p "$RESULTS_DIR" logs

# (model, dataset, dataset_path) tuples to run
JOBS=(
    "latent_policy_guard wildguard         $WILDGUARD_PATH"
    "latent_policy_guard policyguardbench  $POLICYGUARDBENCH_PATH"
    "latent_policy_guard harmbench         $HARMBENCH_PATH"
    "dynaguard_hf        wildguard         $WILDGUARD_PATH"
    "dynaguard_hf        policyguardbench  $POLICYGUARDBENCH_PATH"
    "dynaguard_hf        harmbench         $HARMBENCH_PATH"
    "qwen_hf             wildguard         $WILDGUARD_PATH"
    "qwen_hf             policyguardbench  $POLICYGUARDBENCH_PATH"
    "qwen_hf             harmbench         $HARMBENCH_PATH"
)

run_job() {
    local gpu="$1" model="$2" dataset="$3" path="$4"
    local out="$RESULTS_DIR/${model}_${dataset}.json"

    case "$model" in
        latent_policy_guard)
            GPU="$gpu" MODEL="$model" DATASET="$dataset" DATASET_PATH="$path" \
            OUTPUT="$out" MODEL_PATH="$QWEN_BASE_PATH" CKPT_DIR="$LPG_CKPT_DIR" \
            bash scripts/run_single_eval.sh
            ;;
        dynaguard_hf)
            GPU="$gpu" MODEL="$model" DATASET="$dataset" DATASET_PATH="$path" \
            OUTPUT="$out" MODEL_PATH="$DYNAGUARD_4B_PATH" \
            bash scripts/run_single_eval.sh
            ;;
        qwen_hf)
            GPU="$gpu" MODEL="$model" DATASET="$dataset" DATASET_PATH="$path" \
            OUTPUT="$out" MODEL_PATH="$QWEN_BASE_PATH" \
            bash scripts/run_single_eval.sh
            ;;
    esac
}

i=0
for spec in "${JOBS[@]}"; do
    read -r model dataset path <<<"$spec"
    gpu=$((i % NUM_GPUS))
    echo "[wave $((i / NUM_GPUS))] GPU $gpu: $model x $dataset"
    run_job "$gpu" "$model" "$dataset" "$path" &
    i=$((i + 1))
    if (( i % NUM_GPUS == 0 )); then
        wait
    fi
done
wait

echo
echo "All ${#JOBS[@]} (model x dataset) runs complete. Results in $RESULTS_DIR"
ls -la "$RESULTS_DIR"
