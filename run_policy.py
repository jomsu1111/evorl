"""Roll a policy forward through the evolution MDP and log its learning curve.

  python -m evorl.run_policy --exp main --policy random --steps 10 --root R0
  python -m evorl.run_policy --exp main --policy q:/root/cove-rl/results/main/q_gamma0.9.pkl --steps 12

Nodes are cached, so policies that agree on a prefix share computation.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .config import EvoConfig, parse_overrides
from .env import EvolutionEnv
from .policies import make_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--root", default="R0")
    ap.add_argument("--seed", type=int, default=0, help="policy seed (random policy)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()

    cfg = EvoConfig.load(args.config, parse_overrides(args.set))
    env = EvolutionEnv(cfg, args.exp)
    pol = make_policy(args.policy, args.seed)
    node = env.load(args.root) or env.make_root(args.root)
    env.evaluate(node)
    out_dir = os.path.join(env.root, "curves")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{pol.name}_{args.root}.jsonl")
    with open(out, "w") as f:
        f.write(json.dumps({"t": 0, "node": node.node_id, **node.metrics}) + "\n")
        for t in range(args.steps):
            t0 = time.time()
            feats = env.features(node)
            a, p_tasks = pol.act(env, node, feats)
            child = env.expand(node, a, p_tasks=p_tasks, tag="c" if p_tasks is not None else "")
            env.release(node)
            node = child
            rec = {"t": t + 1, "node": node.node_id, "action": a, **node.metrics,
                   "cost": node.counters.get("cost", {}), "wall": time.time() - t0}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{pol.name}] t={t + 1} a={a} V={node.metrics['V']:.3f} api={node.metrics['acc_api']:.3f} "
                  f"math={node.metrics['acc_math']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if hasattr(pol, "log"):
        json.dump(pol.log, open(out.replace(".jsonl", "_policylog.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
