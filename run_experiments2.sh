#!/usr/bin/env bash
# Round 2: final protocol = gradient accumulation (effective batch 8) with the
# learning rate scaled accordingly (8e-5). Selection between protocols was made
# on the VALIDATION split; the test set is touched once per run.
#
#   nohup setsid bash run_experiments2.sh > runs/run_all2.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY="${PYTHON:-python3}"
COMMON=(--max_epochs 100 --patience 20 --min_epochs 30 --lr 8e-5 --min_lr 8e-7 --accum_steps 8 --workers 6)
ts() { date '+%Y-%m-%d %H:%M:%S'; }
FAILED=()
run() {
    local name="$1"; shift
    local out="runs/$name"
    [ -f "$out/results.json" ] && { echo "[$(ts)] SKIP $name"; return 0; }
    mkdir -p "$out"
    echo "[$(ts)] START $name"
    local t0=$SECONDS
    "$PY" -u train.py --out_dir "$out" "$@" > "$out/stdout.log" 2>&1
    local rc=$?
    if [ -f "$out/results.json" ]; then
        echo "[$(ts)] DONE  $name in $(( (SECONDS - t0) / 60 )) min (exit $rc)"
        grep -E "AUC (abnormal|acl|meniscus|mean) " "$out/training.log" | tail -4
    else
        echo "[$(ts)] FAIL  $name (exit $rc)"; tail -15 "$out/stdout.log"; FAILED+=("$name")
    fi
}
# single-task baselines under the final protocol (needed for a fair comparison)
run single_abnormal_b8 --tasks abnormal "${COMMON[@]}"
run single_acl_b8      --tasks acl      "${COMMON[@]}"
run single_meniscus_b8 --tasks meniscus "${COMMON[@]}"
# control: same learning rate WITHOUT accumulation, to show the batch size is what makes 8e-5 usable
run multitask_b1_lr8e-5 --tasks abnormal,acl,meniscus --max_epochs 100 --patience 20 --min_epochs 30 \
    --lr 8e-5 --min_lr 8e-7 --accum_steps 1 --workers 6
[ ${#FAILED[@]} -gt 0 ] && { echo "[$(ts)] FAILURES: ${FAILED[*]}"; exit 1; }
echo "[$(ts)] round 2 finished"
