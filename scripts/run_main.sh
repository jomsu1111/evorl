#!/bin/bash
# Main experiment (exp=main): behaviour/baseline runs -> 8 horizon-3 trees in two parallel queues.
#   Phase A: always-{P,H} and random(seed 0/1) for 10 steps from R0 (baseline curves + diverse roots)
#   Phase B: trees over {H,P,B,N}. Queues never expand overlapping subtrees concurrently:
#            queue 1 owns R0's tree (depth<=3) and roots at step>=4 on random0/always_H paths;
#            queue 2 owns roots at step>=3 on random1/always_P paths (outside R0's tree interior).
set -u
cd /root/cove-rl
PY=/opt/cove-venv/bin/python
LOG=/opt/cove-logs/main
mkdir -p "$LOG"
EXP=${EXP:-main}

run() { "$PY" -u -m "$@" 2>&1 | grep --line-buffered -vE "Warning|warn\("; }

node_at() {  # curve_file step -> node id at that step
  "$PY" -c "import json,sys; print([json.loads(l)['node'] for l in open(sys.argv[1]) if json.loads(l)['t']==int(sys.argv[2])][0])" "$1" "$2"
}

if [ "${SKIP_A:-0}" != "1" ]; then
  for p in P H; do run evorl.run_policy --exp $EXP --policy $p --steps 10 --root R0 > "$LOG/A_always_$p.log"; done
  for s in 0 1; do run evorl.run_policy --exp $EXP --policy random --seed $s --steps 10 --root R0 > "$LOG/A_random$s.log"; done
fi
C=/root/cove-rl/results/$EXP/curves
Q1="R0 $(node_at $C/random0_R0.jsonl 4) $(node_at $C/random0_R0.jsonl 7) $(node_at $C/always_H_R0.jsonl 4)"
Q2="$(node_at $C/random1_R0.jsonl 3) $(node_at $C/random1_R0.jsonl 6) $(node_at $C/random1_R0.jsonl 9) $(node_at $C/always_P_R0.jsonl 4)"
echo "queue1: $Q1"; echo "queue2: $Q2"
( for r in $Q1; do run evorl.collect_tree --exp $EXP --root "$r"; done > "$LOG/B_queue1.log" ) &
( for r in $Q2; do run evorl.collect_tree --exp $EXP --root "$r"; done > "$LOG/B_queue2.log" ) &
wait
echo MAIN_DONE
