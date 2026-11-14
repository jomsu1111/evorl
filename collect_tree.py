"""Exhaustively expand all action sequences of length `horizon` from a root node.

  python -m evorl.collect_tree --exp main --root R0 --horizon 3
  python -m evorl.collect_tree --exp main --root R0.H.P.H   # branch from a behaviour-run state

Resumable: finished children are loaded from disk instead of recomputed.
Leaves are terminal (finite-horizon targets), so their batch features are not needed.
"""
from __future__ import annotations

import argparse
import time

from .config import EvoConfig, parse_overrides
from .env import EvolutionEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = EvoConfig.load(args.config, parse_overrides(args.set))
    env = EvolutionEnv(cfg, args.exp)
    H = args.horizon or cfg.horizon
    root = env.load(args.root) or env.make_root(args.root)
    env.evaluate(root)
    frontier, t_start = [root], time.time()
    for depth in range(H):
        nxt = []
        for n in frontier:
            for a in cfg.actions:
                t0 = time.time()
                c = env.expand(n, a)
                print(f"[tree {args.root}] d={depth} {c.node_id} V={c.metrics['V']:.3f} dV={c.metrics['V'] - n.metrics['V']:+.3f} "
                      f"({time.time() - t0:.0f}s, total {(time.time() - t_start) / 60:.1f}m)", flush=True)
                nxt.append(c)
            env.release(n)
        frontier = nxt
    for n in frontier:
        env.release(n)
    print(f"[tree {args.root}] done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
