"""Offline fitted-Q iteration over collected evolution transitions.

  python -m evorl.qlearn --exp main --gamma 0.9 --lam 0.05            # fit + save q_*.pkl
  python -m evorl.qlearn --exp main --gamma 0.0 --tag greedy          # one-step (state-aware greedy) router
  python -m evorl.qlearn --exp main --features task --tag noagent     # ablation: no agent-state features
  python -m evorl.qlearn --exp main --cv                              # leave-one-root-out tree regret

Reward: r_t = ΔV - lam * C(a_t),  C = wall-clock seconds of the action / 600.
Transitions whose child has no materialized features (tree leaves) are terminal.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .features import ALL_FEATURES, TASK_FEATURES

ACTIONS = ["H", "P", "B"]


def load(results_dir: str, exps: List[str]):
    trans, feats = [], {}
    for exp in exps:
        root = os.path.join(results_dir, exp)
        p = os.path.join(root, "transitions.jsonl")
        if not os.path.exists(p):
            continue
        seen = {}
        for l in open(p):
            r = json.loads(l)
            seen[(exp, r["child"])] = r  # dedupe re-runs: last wins
        trans += list(seen.values())
        nd = os.path.join(root, "nodes")
        for nid in os.listdir(nd):
            fp = os.path.join(nd, nid, "features.json")
            if os.path.exists(fp):
                feats[(exp, nid)] = json.load(open(fp))
    return trans, feats


def cost(r) -> float:
    c = r.get("cost", {})
    return (c.get("h_seconds", 0.0) + c.get("p_seconds", 0.0)) / 600.0


def build(trans, feats, keys, lam):
    X, A, R, Xn, done, roots = [], [], [], [], [], []
    for r in trans:
        s = feats.get((r["exp"], r["parent"]))
        if s is None:
            continue
        sn = feats.get((r["exp"], r["child"]))
        X.append([s[k] for k in keys]); A.append(r["action"]); R.append(r["dV"] - lam * cost(r))
        Xn.append([sn[k] for k in keys] if sn else [0.0] * len(keys)); done.append(sn is None)
        roots.append(f"{r['exp']}:{r['root']}")
    return np.array(X), np.array(A), np.array(R), np.array(Xn), np.array(done), np.array(roots)


def make_model():
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)


def fqi(X, A, R, Xn, done, gamma: float, iters: int = 6) -> Dict[str, object]:
    models: Dict[str, object] = {}
    y = R.copy()
    for it in range(iters):
        models = {}
        for a in ACTIONS:
            m = A == a
            if m.sum() == 0:
                continue
            mdl = make_model()
            mdl.fit(X[m], y[m])
            models[a] = mdl
        if gamma == 0:
            break
        qn = np.stack([models[a].predict(Xn) for a in models], axis=1).max(axis=1)
        y = R + gamma * np.where(done, 0.0, qn)
    return models


def tree_values(trans, lam, gamma):
    """Exact finite-horizon backups inside each collected tree: V*(node), Q*(node, a)."""
    kids = defaultdict(dict)
    for r in trans:
        kids[(r["exp"], r["parent"])][r["action"]] = r
    vstar = {}

    def V(key):
        if key in vstar:
            return vstar[key]
        if key not in kids:
            vstar[key] = 0.0
            return 0.0
        vstar[key] = max(r["dV"] - lam * cost(r) + gamma * V((key[0], r["child"])) for r in kids[key].values())
        return vstar[key]

    return kids, V


def cv_regret(trans, feats, keys, lam, gamma):
    """Leave-one-root-out: fit on other trees, follow argmax-Q inside the held-out tree.
    All policies are scored by the same undiscounted return over the tree horizon
    (= V_final - V_root - lam * total cost), whatever gamma the router was trained with."""
    X, A, R, Xn, done, roots = build(trans, feats, keys, lam)
    kids, V = tree_values(trans, lam, 1.0)
    out = {}
    for held in sorted(set(roots)):
        tr = roots != held
        models = fqi(X[tr], A[tr], R[tr], Xn[tr], done[tr], gamma)
        exp, root = held.split(":", 1)
        key, ret, path = (exp, root), 0.0, []
        while key in kids and key in feats:
            x = np.array([[feats[key][k] for k in keys]])
            qs = {a: models[a].predict(x)[0] for a in models if a in kids[key]}
            a = max(qs, key=qs.get)
            r = kids[key][a]
            ret += r["dV"] - lam * cost(r)
            path.append(a); key = (exp, r["child"])
        base = {}
        for b in ACTIONS:
            k2, rb = (exp, root), 0.0
            while k2 in kids and b in kids[k2]:
                r = kids[k2][b]; rb += r["dV"] - lam * cost(r); k2 = (exp, r["child"])
            base[f"always_{b}"] = rb
        out[held] = {"optimal": V((exp, root)), "router": ret, "path": "".join(path), **base}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--results-dir", default="/root/cove-rl/results")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--features", choices=["all", "task"], default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cv", action="store_true")
    args = ap.parse_args()

    keys = ALL_FEATURES if args.features == "all" else TASK_FEATURES
    trans, feats = load(args.results_dir, args.exp)
    print(f"transitions={len(trans)} states_with_features={len(feats)}")
    if args.cv:
        res = cv_regret(trans, feats, keys, args.lam, args.gamma)
        for k, v in res.items():
            print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}))
        if res:
            for col in ["optimal", "router", "always_H", "always_P", "always_B"]:
                print(f"mean {col}: {np.mean([v[col] for v in res.values()]):.4f}")
        return
    X, A, R, Xn, done, _ = build(trans, feats, keys, args.lam)
    models = fqi(X, A, R, Xn, done, args.gamma)
    tag = args.tag or f"g{args.gamma}_l{args.lam}_{args.features}"
    path = os.path.join(args.results_dir, args.exp[0], f"q_{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"models": models, "features": keys, "gamma": args.gamma, "lam": args.lam, "tag": tag}, f)
    print("saved", path, "n=", len(X))


if __name__ == "__main__":
    main()
