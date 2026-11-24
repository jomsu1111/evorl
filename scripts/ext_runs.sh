#!/bin/bash
# Follow-up experiments after RQ2 (rq2_runs.sh must have fitted q_rl / q_rlcost already).
#   (1) Drift-timing shift: the routers were fitted on trees whose API version bumps at step 6.
#       Re-run them on seed-1 streams whose bump is at step 4 / 8. If the switch to B tracks the
#       new bump, the router reacts to the agent's state rather than having memorised "B at t=7".
#   (2) Cost sweep: Q routers fitted with more lambdas (scale cost) on the seed-1/2 streams, to draw
#       the performance-cost frontier together with the always-X / COVE / q_rlcost points.
#   bash /root/cove-rl/evorl/scripts/ext_runs.sh
# Four lanes run in parallel (P training is serialised by the trainer lock, so GPU memory is safe);
# policies inside a lane run sequentially. Finished curves are skipped, so the script resumes.
set -uo pipefail
set -f  # overrides like drift_steps=[4] must not glob
ROOT=/root/cove-rl
PY=/opt/cove-venv/bin/python
LOG=/opt/cove-logs
STEPS=${STEPS:-12}
CFG=$ROOT/results/main/config.json
Q=$ROOT/results/main
mkdir -p "$LOG"
cd "$ROOT"

fit() {  # tag, extra qlearn args
  local tag=$1; shift
  [ -f "$Q/q_$tag.pkl" ] || $PY -m evorl.qlearn --exp main --tag "$tag" "$@"
}
for l in 0.001 0.005 0.01; do
  fit "rlcost_l$l" --gamma 0.9 --lam "$l" --cost-model scale --features all
done

curve_name() {  # policy spec -> curve file stem
  case $1 in
    q:*) local f=${1##*/}; echo "${f%.pkl}" ;;
    cove) echo "cove" ;;
    *) echo "always_$1" ;;
  esac
}

lane() {  # name, "exp seed overrides", policies...
  local name=$1 exps=$2; shift 2
  local exp seed ovr
  for spec in $exps; do
    IFS=: read -r exp seed ovr <<< "$spec"
    for pol in "$@"; do
      local curve="$ROOT/results/$exp/curves/$(curve_name "$pol")_R0.jsonl"
      if [ -f "$curve" ] && [ "$(wc -l < "$curve")" -ge $((STEPS + 1)) ]; then
        echo "[$name] skip $exp $pol (done)"; continue
      fi
      echo "[$name] $(date +%m-%d_%H:%M) start $exp $pol"
      # shellcheck disable=SC2086
      $PY -m evorl.run_policy --exp "$exp" --policy "$pol" --steps "$STEPS" \
          --config "$CFG" --set seed="$seed" $ovr || echo "[$name] FAILED $exp $pol"
    done
  done
  echo "[$name] $(date +%m-%d_%H:%M) LANE_DONE"
}

DRIFT_POL=("q:$Q/q_rlcost.pkl" "P" "q:$Q/q_rl.pkl")
LAM_POL=("q:$Q/q_rlcost_l0.001.pkl" "q:$Q/q_rlcost_l0.005.pkl" "q:$Q/q_rlcost_l0.01.pkl")

lane d4  "eval_s1_d4:1:drift_steps=[4]" "${DRIFT_POL[@]}" > "$LOG/ext_d4.log" 2>&1 &
lane d8  "eval_s1_d8:1:drift_steps=[8]" "${DRIFT_POL[@]}" > "$LOG/ext_d8.log" 2>&1 &
lane ls1 "eval_s1:1:"                   "${LAM_POL[@]}"  > "$LOG/ext_ls1.log" 2>&1 &
lane ls2 "eval_s2:2:"                   "${LAM_POL[@]}"  > "$LOG/ext_ls2.log" 2>&1 &
wait
echo "EXT_DONE $(date +%m-%d_%H:%M)"
