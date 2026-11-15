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

from .config import ACTIONS
from .features import ALL_FEATURES, TASK_FEATURES

# Cost models (raw costs are always stored, so this is a post-hoc choice):
#   flops : measured compute from token counts; inference 2*N, training 6*N FLOPs/token,
#           helper N=4e9, policy N=0.6e9; unit = 1e15 FLOPs (PFLOP).
#   scale : assumed large-scale ratio, C(H)=1, C(P)=10, C(B)=11, C(N)=0.
#   wall  : wall-clock seconds / 600 (distorted when collectors run concurrently).
COST = {"model": "flops"}
SCALE = {"H": 1.0, "P": 10.0, "B": 11.0, "N": 0.0}


def _depth(node_id: str) -> int:
    return node_id.count(".")


def _tree_of(parent: str, trees: List[str], horizon: int) -> str:
    """Deepest collected root whose depth-`horizon` tree contains `parent` as an interior node."""
    best = ""
    for t in trees:
        if ((parent == t or parent.startswith(t + ".")) and _depth(parent) - _depth(t) < horizon
                and len(t) > len(best)):
            best = t
    return best or "behavior"


def load(results_dir: str, exps: List[str]):
    """Transitions get r['tree'] = the collected tree root they belong to (from trees.txt) and
    r['tree_depth'] = depth of the parent below that root. Transitions outside every tree's
    horizon (behaviour runs) get 'behavior': used for fitting, never held out or scored."""
    trans, feats = [], {}
    for exp in exps:
        root = os.path.join(results_dir, exp)
        p = os.path.join(root, "transitions.jsonl")
        if not os.path.exists(p):
            continue
        cp = os.path.join(root, "config.json")
        horizon = int(json.load(open(cp)).get("horizon", 3)) if os.path.exists(cp) else 3
        tp = os.path.join(root, "trees.txt")
        trees = open(tp).read().split() if os.path.exists(tp) else []
        seen = {}
        for l in open(p):
            r = json.loads(l)
            if r["action"] not in ACTIONS:  # e.g. COVE's pool-trained "Bc" in policy runs
                continue
            r["exp"] = exp  # the results directory is authoritative (copies/renames of an exp)
            seen[(exp, r["child"])] = r  # dedupe re-runs: last wins
        for r in seen.values():
            r["tree"] = _tree_of(r["parent"], trees, horizon)
            r["horizon"] = horizon
            r["tree_depth"] = (_depth(r["parent"]) - _depth(r["tree"])) if r["tree"] != "behavior" else None
        trans += list(seen.values())
        nd = os.path.join(root, "nodes")
        for nid in os.listdir(nd):
            fp = os.path.join(nd, nid, "features.json")
            if os.path.exists(fp):
                feats[(exp, nid)] = json.load(open(fp))
    return trans, feats


def cost(r) -> float:
    c, a = r.get("cost", {}), r["action"]
    if COST["model"] == "scale":
        return SCALE[a]
    if COST["model"] == "wall":
        return (c.get("h_seconds", 0.0) + c.get("p_seconds", 0.0)) / 600.0
    return (2 * 4e9 * c.get("helper_tokens", 0) + 2 * 0.6e9 * c.get("sample_tokens", 0)
            + 6 * 0.6e9 * c.get("train_tokens", 0)) / 1e15


def build(trans, feats, keys, lam):
    X, A, R, Xn, done, roots = [], [], [], [], [], []
    for r in trans:
        s = feats.get((r["exp"], r["parent"]))
        if s is None:
            continue
        sn = feats.get((r["exp"], r["child"]))
        X.append([s[k] for k in keys]); A.append(r["action"]); R.append(r["dV"] - lam * cost(r))
        Xn.append([sn[k] for k in keys] if sn else [0.0] * len(keys)); done.append(sn is None)
        roots.append(f"{r['exp']}:{r['tree']}")
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
    """Exact finite-horizon backups: V(node, h) = best return of h more steps from node."""
    kids = defaultdict(dict)
    for r in trans:
        kids[(r["exp"], r["parent"])][r["action"]] = r
    vstar = {}

    def V(key, h: int) -> float:
        if h <= 0 or key not in kids:
            return 0.0
        if (key, h) not in vstar:
            vstar[(key, h)] = max(r["dV"] - lam * cost(r) + gamma * V((key[0], r["child"]), h - 1)
                                  for r in kids[key].values())
        return vstar[(key, h)]

    return kids, V


def cv_regret(trans, feats, keys, lam, gamma):
    """Leave-one-root-out: fit on other trees, follow argmax-Q inside the held-out tree.
    All policies are scored by the same undiscounted return over the tree horizon
    (= V_final - V_root - lam * total cost), whatever gamma the router was trained with."""
    X, A, R, Xn, done, roots = build(trans, feats, keys, lam)
    kids, V = tree_values(trans, lam, 1.0)
    out = {}
    for held in sorted(set(roots)):
        if held.endswith(":behavior"):
            continue
        tr = roots != held
        models = fqi(X[tr], A[tr], R[tr], Xn[tr], done[tr], gamma)
        exp, root = held.split(":", 1)
        H = next(r["horizon"] for r in trans if r["exp"] == exp)
        key, ret, path = (exp, root), 0.0, []
        for _ in range(H):
            if key not in kids or key not in feats:
                break
            x = np.array([[feats[key][k] for k in keys]])
            qs = {a: models[a].predict(x)[0] for a in models if a in kids[key]}
            a = max(qs, key=qs.get)
            r = kids[key][a]
            ret += r["dV"] - lam * cost(r)
            path.append(a); key = (exp, r["child"])
        base = {}
        for b in ACTIONS:
            k2, rb = (exp, root), 0.0
            for _ in range(H):
                if k2 not in kids or b not in kids[k2]:
                    break
                r = kids[k2][b]; rb += r["dV"] - lam * cost(r); k2 = (exp, r["child"])
            base[f"always_{b}"] = rb
        out[held] = {"optimal": V((exp, root), H), "router": ret, "path": "".join(path), **base}
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
    ap.add_argument("--cost-model", choices=["flops", "scale", "wall"], default="flops")
    args = ap.parse_args()
    COST["model"] = args.cost_model

    keys = ALL_FEATURES if args.features == "all" else TASK_FEATURES
    trans, feats = load(args.results_dir, args.exp)
    print(f"transitions={len(trans)} states_with_features={len(feats)}")
    if args.cv:
        res = cv_regret(trans, feats, keys, args.lam, args.gamma)
        for k, v in res.items():
            print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}))
        if res:
            for col in ["optimal", "router"] + [f"always_{a}" for a in ACTIONS]:
                print(f"mean {col}: {np.mean([v[col] for v in res.values()]):.4f}")
        return
    X, A, R, Xn, done, _ = build(trans, feats, keys, args.lam)
    models = fqi(X, A, R, Xn, done, args.gamma)
    tag = args.tag or f"g{args.gamma}_l{args.lam}_{args.cost_model}_{args.features}"
    path = os.path.join(args.results_dir, args.exp[0], f"q_{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"models": models, "features": keys, "gamma": args.gamma, "lam": args.lam,
                     "cost_model": args.cost_model, "tag": tag}, f)
    print("saved", path, "n=", len(X))


if __name__ == "__main__":
    main()
