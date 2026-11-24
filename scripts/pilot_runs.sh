#!/bin/bash
# Pilot: is there a setting where H is sometimes the best action? Fixed policies only (no router),
# seed 1, 12 steps, in two candidate settings:
#   A  (pilot_A):  API version bumps every 3 steps (drift_steps=[3,6,9]) -> parameter knowledge goes stale
#   AB (pilot_AB): A + memory retrieved for API tasks only (memory_task_types=["api"]), so memory no
#                  longer hurts MATH
# If one fixed policy wins at every step there is nothing for a router to learn; only a setting where
# the step-wise ranking of H/P/B/N changes is worth collecting new trees for.
#   bash /root/cove-rl/evorl/scripts/pilot_runs.sh
set -uo pipefail
set -f  # overrides like drift_steps=[3,6,9] must not glob
ROOT=/root/cove-rl
PY=/opt/cove-venv/bin/python
LOG=/opt/cove-logs
STEPS=${STEPS:-12}
SEED=${SEED:-1}
CFG=$ROOT/results/main/config.json
mkdir -p "$LOG"
cd "$ROOT"

run() {  # exp, overrides, policies...
  local exp=$1 ovr=$2; shift 2
  for pol in "$@"; do
    local curve="$ROOT/results/$exp/curves/always_${pol}_R0.jsonl"
    if [ -f "$curve" ] && [ "$(wc -l < "$curve")" -ge $((STEPS + 1)) ]; then
      echo "[$exp] skip $pol (done)"; continue
    fi
    echo "[$exp] $(date +%m-%d_%H:%M) start $pol"
    # shellcheck disable=SC2086
    $PY -m evorl.run_policy --exp "$exp" --policy "$pol" --steps "$STEPS" \
        --config "$CFG" --set seed="$SEED" $ovr || echo "[$exp] FAILED $pol"
  done
  echo "[$exp] $(date +%m-%d_%H:%M) LANE_DONE $*"
}

after_root() {  # exp: the second lane of a setting waits until the first has evaluated the shared root
  until [ -f "$ROOT/results/$1/nodes/R0/val.json" ]; do sleep 30; done
}

A="drift_steps=[3,6,9]"
AB="drift_steps=[3,6,9] memory_task_types=[\"api\"]"

run pilot_A  "$A"  P B > "$LOG/pilot_A1.log" 2>&1 &
(after_root pilot_A;  run pilot_A  "$A"  H N) > "$LOG/pilot_A2.log" 2>&1 &
run pilot_AB "$AB" P B > "$LOG/pilot_AB1.log" 2>&1 &
(after_root pilot_AB; run pilot_AB "$AB" H N) > "$LOG/pilot_AB2.log" 2>&1 &
wait
echo "PILOT_DONE $(date +%m-%d_%H:%M)"
