#!/bin/bash
# RQ2 online comparison: every policy rolled 12 steps through held-out task streams.
#   bash /root/cove-rl/evorl/scripts/rq2_runs.sh            # seeds 1 2 (default)
#   SEEDS="1" STEPS=12 bash .../rq2_runs.sh
#
# The Q routers are fitted on the seed-0 trees (results/main); evaluating them on other
# stream seeds (different task draws and API worlds) keeps the comparison out-of-sample.
# One experiment dir per seed (eval_s<seed>); seeds run in parallel, policies inside a seed
# run sequentially so they never expand the same cached node concurrently (shared prefixes
# such as always_H vs. a router that starts with H are reused, not recomputed).
# A policy whose curve already has all steps is skipped, so the script can be re-run to resume.
set -uo pipefail
ROOT=/root/cove-rl
PY=/opt/cove-venv/bin/python
LOG=/opt/cove-logs
SEEDS=${SEEDS:-"1 2"}
STEPS=${STEPS:-12}
CFG=$ROOT/results/main/config.json
mkdir -p "$LOG"
cd "$ROOT"

# --- Q routers (fit on the collected trees; CPU, a few minutes) ---
fit() {  # tag, extra qlearn args
  local tag=$1; shift
  [ -f "$ROOT/results/main/q_$tag.pkl" ] || $PY -m evorl.qlearn --exp main --tag "$tag" "$@"
}
fit rl     --gamma 0.9 --lam 0 --features all
fit task   --gamma 0.9 --lam 0 --features task
fit greedy --gamma 0   --lam 0 --features all
fit rlcost --gamma 0.9 --lam 0.002 --cost-model scale --features all

POLICIES=(
  "q:$ROOT/results/main/q_rl.pkl"
  "cove"
  "B" "H" "P" "N"
  "q:$ROOT/results/main/q_task.pkl"
  "q:$ROOT/results/main/q_greedy.pkl"
  "q:$ROOT/results/main/q_rlcost.pkl"
)

curve_name() {  # policy spec -> curve file stem
  case $1 in
    q:*) local f=${1##*/}; echo "${f%.pkl}" ;;
    cove) echo "cove" ;;
    *) echo "always_$1" ;;
  esac
}

run_seed() {
  local seed=$1 exp=eval_s$1
  for pol in "${POLICIES[@]}"; do
    local curve="$ROOT/results/$exp/curves/$(curve_name "$pol")_R0.jsonl"
    if [ -f "$curve" ] && [ "$(wc -l < "$curve")" -ge $((STEPS + 1)) ]; then
      echo "[s$seed] skip $pol (done)"; continue
    fi
    echo "[s$seed] $(date +%m-%d_%H:%M) start $pol"
    $PY -m evorl.run_policy --exp "$exp" --policy "$pol" --steps "$STEPS" \
        --config "$CFG" --set seed="$seed" || echo "[s$seed] FAILED $pol"
  done
  echo "[s$seed] $(date +%m-%d_%H:%M) SEED_DONE"
}

for s in $SEEDS; do
  run_seed "$s" > "$LOG/rq2_s$s.log" 2>&1 &
done
wait
echo "RQ2_DONE $(date +%m-%d_%H:%M)"
