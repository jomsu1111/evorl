"""End-to-end smoke test on tiny sizes (needs both vLLM servers up).

  python -m evorl.smoke
Checks: base eval, batch rollout, H (memory write), P (RFT train + LoRA load), child eval.
"""
from __future__ import annotations

import json
import time

from .config import EvoConfig
from .env import EvolutionEnv


def main() -> None:
    cfg = EvoConfig(batch_size=8, api_val_size=8, math_val_size=8, rft_samples=2, results_dir="/root/cove-rl/results_smoke",
                    work_dir="/opt/cove-work-smoke")
    env = EvolutionEnv(cfg, "smoke")
    t0 = time.time()
    root = env.make_root("R0")
    print("root metrics", json.dumps(root.metrics), f"{time.time() - t0:.0f}s", flush=True)
    for a in ["H", "P"]:
        t0 = time.time()
        c = env.expand(root, a)
        print(a, "->", c.node_id, json.dumps(c.metrics), json.dumps(c.counters.get("cost")), f"{time.time() - t0:.0f}s", flush=True)
    feats = env.features(root)
    print("features", json.dumps({k: round(v, 3) for k, v in feats.items()}))
    eps = env.experience(root)
    e = eps[0]
    print("sample episode:", e["task_type"], e["success_t1"], e["final"]["success"])
    print(e["turns"][0]["response"][:800])
    print("bank after H:", env.bank(env.load("R0.H")).stats())
    for r in env.bank(env.load("R0.H")).mgr.load_memories()[:5]:
        print("  MEM", r.get("volatility"), r["content"][:200])
    for r in env.bank(env.load("R0.H")).mgr.load_skills()[:5]:
        print("  SKILL", r.get("volatility"), r["content"][:200])


if __name__ == "__main__":
    main()
