#!/usr/bin/env bash
# Runs all experiments sequentially, each into runs/<name>/.
# A run whose results.json already exists is skipped, so the script can simply
# be relaunched after an interruption (an interrupted run restarts from scratch).
#
# Launch (from the project directory):
#   mkdir -p runs && nohup bash run_experiments.sh > runs/run_all.log 2>&1 &
#
# Extra arguments to this script are forwarded to every train.py call.

set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
mkdir -p runs

PY="${PYTHON:-python}"
EXTRA_ARGS=("$@")
MAX_EPOCHS=100
PATIENCE=20
MIN_EPOCHS=30   # early stopping inactive before this epoch
LR=1e-5         # 1e-4 with pos_weight collapsed to a constant predictor (dead ReLUs); 1e-5 as in Bien et al. 2018
MIN_LR=1e-7
WORKERS=6

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] run_experiments.sh started (pid $$) in $(pwd)"
echo "[$(ts)] python: $(command -v "$PY") | $("$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())')"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null

MODE=$("$PY" -c 'from dataset import make_splits; print(make_splits(".", log=lambda m: None)["mode"])')
if [ "$MODE" = "fallback" ]; then
    echo "################################################################################"
    echo "WARNING: dataset is in FALLBACK mode: valid/ (or val/) images not found."
    echo "WARNING: the 1130 training exams will be split 70/15/15. Results are NOT"
    echo "WARNING: comparable to the official MRNet validation set."
    echo "################################################################################"
elif [ "$MODE" = "official_test_pending" ]; then
    echo "[$(ts)] dataset mode: OFFICIAL, TEST PENDING (validation images found, label csvs missing)"
    echo "[$(ts)] training/model selection proceed normally; test evaluation runs automatically at the"
    echo "[$(ts)] end if valid-*.csv (or val-*.csv) have appeared, otherwise: python evaluate_test.py --all"
else
    echo "[$(ts)] dataset mode: OFFICIAL (official 120-exam validation set = test set)"
fi

FAILED=()

run() {   # run <name> <train.py args...>
    local name="$1"; shift
    local out="runs/$name"
    if [ -f "$out/results.json" ]; then
        echo "[$(ts)] SKIP $name ($out/results.json exists)"
        return 0
    fi
    mkdir -p "$out"
    echo "[$(ts)] START $name: $PY -u train.py --out_dir $out $* ${EXTRA_ARGS[*]}"
    local t0=$SECONDS
    "$PY" -u train.py --out_dir "$out" --workers "$WORKERS" "$@" "${EXTRA_ARGS[@]}" > "$out/stdout.log" 2>&1
    local rc=$?
    local mins=$(( (SECONDS - t0) / 60 ))
    if [ $rc -eq 0 ] && [ -f "$out/results.json" ]; then
        echo "[$(ts)] DONE  $name in ${mins} min"
        grep -E "AUC (abnormal|acl|meniscus|mean) " "$out/training.log" | tail -4
    else
        echo "[$(ts)] FAIL  $name (exit code $rc) after ${mins} min -- see $out/stdout.log"
        tail -20 "$out/stdout.log"
        FAILED+=("$name")
    fi
}

# ── short AMP memory benchmark first (a few minutes; also catches setup errors early)
run bench_no_amp --tasks abnormal,acl,meniscus --max_epochs 2 --patience 100 --limit 60 --no_amp
run bench_amp    --tasks abnormal,acl,meniscus --max_epochs 2 --patience 100 --limit 60
# benchmark checkpoints are not needed (~100 MB each; disk is nearly full)
rm -f runs/bench_no_amp/best_model.pt runs/bench_amp/best_model.pt

# ── main experiments
COMMON=(--max_epochs "$MAX_EPOCHS" --patience "$PATIENCE" --min_epochs "$MIN_EPOCHS" --lr "$LR" --min_lr "$MIN_LR")
run multitask       --tasks abnormal,acl,meniscus "${COMMON[@]}"
run single_abnormal --tasks abnormal              "${COMMON[@]}"
run single_acl      --tasks acl                   "${COMMON[@]}"
run single_meniscus --tasks meniscus              "${COMMON[@]}"

# ── deferred test evaluation (runs trained before the validation labels were on disk)
if [ "$("$PY" -c 'from dataset import official_validation_available as o; print(int(o(".")))')" = "1" ]; then
    for name in multitask single_abnormal single_acl single_meniscus; do
        if [ -f "runs/$name/results.json" ] && grep -q '"test_pending": true' "runs/$name/results.json"; then
            echo "[$(ts)] deferred TEST evaluation: $name"
            "$PY" -u evaluate_test.py "runs/$name" --workers "$WORKERS" >> "runs/$name/stdout.log" 2>&1 \
                && grep -E "AUC (abnormal|acl|meniscus|mean) " "runs/$name/training.log" | tail -4 \
                || FAILED+=("eval_$name")
        fi
    done
else
    echo "[$(ts)] validation label csvs still missing -> after adding them run: python evaluate_test.py --all"
fi

for b in bench_no_amp bench_amp; do
    if [ -f "runs/$b/results.json" ]; then
        "$PY" -c "import json; r=json.load(open('runs/$b/results.json')); m=r['memory']; print('$b: amp=%s peak allocated %.0f MB, reserved %.0f MB, mean train s/exam %.3f' % (r['amp'], m['peak_gpu_mem_mb_train_epochs_max'], m['peak_gpu_reserved_mb_max'], r['timing']['mean_train_s_per_exam']))"
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[$(ts)] finished with FAILURES: ${FAILED[*]}"
    exit 1
fi
echo "[$(ts)] all runs finished"
