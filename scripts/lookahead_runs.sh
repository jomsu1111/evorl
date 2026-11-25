#!/bin/bash
# Is lookahead (gamma > 0) needed? Online follow-ups to `python -m evorl.lookahead` (see that file).
#   (C) one-step vs lookahead routers at the same cost weight (lam = 0.002, scale cost):
#       q_rlcost (FQI gamma 0.9, already fitted), q_greedycost (gamma 0), q_fh3cost (finite-horizon Q_3,
#       always looking 3 steps ahead). Seeds 1-2 reuse the cached nodes; seeds 3-5 are new streams.
#   (D) longer-horizon counterfactuals: on q_rlcost's own seed-1/2 runs, at steps t = 2, 4, 6, 8, take
#       each of H/P/B/N once and let q_rlcost continue for 5 more steps. Compares the 1-step and the
#       6-step ranking of the first action (scored by `lookahead --mode mc`).
#   bash /root/cove-rl/evorl/scripts/lookahead_runs.sh
# Lanes run in parallel; nodes are locked while computed, so lanes sharing an exp reuse each other's
# nodes instead of recomputing them. Finished curves are skipped, so the script resumes.
set -uo pipefail
set -f
ROOT=/root/cove-rl
PY=/opt/cove-venv/bin/python
LOG=/opt/cove-logs
CFG=$ROOT/results/main/config.json
Q=$ROOT/results/main
LAM=0.002
mkdir -p "$LOG"
cd "$ROOT"

[ -f "$Q/q_greedycost.pkl" ] || $PY -m evorl.qlearn --exp main --tag greedycost --gamma 0 --lam $LAM --cost-model scale --features all
[ -f "$Q/q_fh3cost.pkl" ] || $PY -m evorl.lookahead --exp main --mode fit --h 3 --lam $LAM --cost-model scale --tag fh3cost

curve_name() {  # policy spec -> curve file stem (mirrors Policy.name)
  case $1 in
    seq:*) local s=${1#seq:}; local seq=${s%%>*} then=${s#*>}; echo "seq_${seq}_then_$(curve_name "$then")" ;;
    q:*) local f=${1##*/}; echo "${f%.pkl}" ;;
    cove) echo "cove" ;;
    *) echo "always_$1" ;;
  esac
}

run() {  # lane exp seed steps policy
  local lane=$1 exp=$2 seed=$3 steps=$4 pol=$5
  local curve="$ROOT/results/$exp/curves/$(curve_name "$pol")_R0.jsonl"
  if [ -f "$curve" ] && [ "$(wc -l < "$curve")" -ge $((steps + 1)) ]; then
    echo "[$lane] skip $exp $pol (done)"; return
  fi
  echo "[$lane] $(date +%m-%d_%H:%M) start $exp $pol"
  $PY -m evorl.run_policy --exp "$exp" --policy "$pol" --steps "$steps" --config "$CFG" --set seed="$seed" \
    || echo "[$lane] FAILED $exp $pol"
}

lane_c() {  # lane seed policies...
  local lane=$1 seed=$2; shift 2
  for pol in "$@"; do run "$lane" "eval_s$seed" "$seed" 12 "$pol"; done
  echo "[$lane] $(date +%m-%d_%H:%M) LANE_DONE"
}

lane_d() {  # lane seed t...   prefix = q_rlcost's own actions on that seed before step t
  local lane=$1 seed=$2; shift 2
  local path
  path=$($PY -c "import json,sys; print(''.join(json.loads(l)['action'] for l in open(sys.argv[1]) if 'action' in json.loads(l)))" \
         "$ROOT/results/eval_s$seed/curves/q_rlcost_R0.jsonl")
  for t in "$@"; do
    for a in H P B N; do run "$lane" "eval_s$seed" "$seed" $((t + 6)) "seq:${path:0:$t}$a>q:$Q/q_rlcost.pkl"; done
  done
  echo "[$lane] $(date +%m-%d_%H:%M) LANE_DONE"
}

C_POL=("q:$Q/q_greedycost.pkl" "q:$Q/q_fh3cost.pkl")
NEW_POL=("q:$Q/q_rlcost.pkl" "q:$Q/q_greedycost.pkl" "P")

lane_d d1a 1 2 4 > "$LOG/la_d1a.log" 2>&1 &
lane_d d1b 1 6 8 > "$LOG/la_d1b.log" 2>&1 &
lane_d d2a 2 2 4 > "$LOG/la_d2a.log" 2>&1 &
lane_d d2b 2 6 8 > "$LOG/la_d2b.log" 2>&1 &
lane_c c1 1 "${C_POL[@]}" > "$LOG/la_c1.log" 2>&1 &
lane_c c2 2 "${C_POL[@]}" > "$LOG/la_c2.log" 2>&1 &
for s in 3 4 5; do lane_c "n$s" "$s" "${NEW_POL[@]}" > "$LOG/la_n$s.log" 2>&1 & done
wait
echo "LOOKAHEAD_DONE $(date +%m-%d_%H:%M)"
