"""COVE's routing principle as a baseline, and the new RQ1 on collected trees.

  python -m evorl.rule_analysis --exp main

rule      volatile knowledge -> harness, reasoning -> parameters, i.e. API batch -> H, MATH batch -> P
          (the principle COVE's judge is prompted with, applied mechanically).
RQ2 table every policy scored on the same interior decision nodes as judge_eval.py (scale costs,
          objectives perf: lam=0 and cost: lam=0.002): accuracy w.r.t. a*, mean regret Q*(a*)-Q*(a),
          and the return of the path the policy follows in each tree.
          rl_generic is the fitted-Q router restricted to features that do not name API/MATH
          (no batch type, no task-specific error types, no per-family accuracies).
RQ1       (a) how often the rule picks a*, and its regret, overall and by agent state;
          (b) whether the agent state predicts when the rule is right: leave-one-tree-out
          RandomForest on y = [a* == rule] with TASK vs ALL features (pooled AUC).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from .features import ALL_FEATURES, TASK_FEATURES
from .qlearn import ACTIONS, COST, build, cost, fqi, load, tree_values

DOMAIN = {"type_api", "V_api", "V_math", "mem_gain_api", "mem_gain_math", "delta_t1_same_type"} | \
    {k for k in ALL_FEATURES if k.startswith("err_")}
GENERIC = [k for k in ALL_FEATURES if k not in DOMAIN]


def rule(f) -> str:
    return "H" if f["type_api"] > 0.5 else "P"


def decision_nodes(trans, feats, lam):
    kids, V = tree_values(trans, lam, 1.0)
    nodes = []
    for key, ch in kids.items():
        if set(ch) == set(ACTIONS) and key in feats and ch["H"]["tree"] != "behavior":
            h = ch["H"]["horizon"] - ch["H"]["tree_depth"]
            q = {a: r["dV"] - lam * cost(r) + V((key[0], r["child"]), h - 1) for a, r in ch.items()}
            s = sorted(q.values())
            nodes.append({"key": key, "tree": ch["H"]["tree"], "h": h, "q": q, "f": feats[key],
                          "a_star": max(q, key=q.get), "margin": s[-1] - s[-2], "step": key[1].count(".")})
    return kids, V, nodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--results-dir", default="/root/cove-rl/results")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.002)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--jobs", type=int, default=16, help="threads per tree ensemble (results do not depend on it)")
    args = ap.parse_args()
    COST["model"] = "scale"
    from sklearn.ensemble import ExtraTreesRegressor
    from . import qlearn
    # same model as qlearn.make_model, fewer workers: with all 128 cores the per-fit pool overhead dominates
    qlearn.make_model = lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=args.jobs)
    trans, feats = load(args.results_dir, [args.exp])
    cfgp = os.path.join(args.results_dir, args.exp, "config.json")
    drift = json.load(open(cfgp)).get("drift_steps", [6]) if os.path.exists(cfgp) else [6]
    print(f"generic features ({len(GENERIC)}): {', '.join(GENERIC)}")

    for objective, lam in (("perf", 0.0), ("cost", args.lam)):
        kids, V, nodes = decision_nodes(trans, feats, lam)
        trees = sorted({n["tree"] for n in nodes})
        choice = {"rule": {n["key"]: rule(n["f"]) for n in nodes}}
        for a in ACTIONS:
            choice[f"always_{a}"] = {n["key"]: a for n in nodes}
        for variant in ("judge_task", "judge_state"):
            p = os.path.join(args.results_dir, args.exp, "judge", f"{variant}_{objective}_lam{lam:g}_n5.jsonl")
            if os.path.exists(p):
                cache = {json.loads(l)["node"]: json.loads(l)["action"] for l in open(p)}
                if all(n["key"][1] in cache for n in nodes):
                    choice[variant] = {n["key"]: cache[n["key"][1]] for n in nodes}
        for name, keys in (("rl_all", ALL_FEATURES), ("rl_task", TASK_FEATURES), ("rl_generic", GENERIC)):
            X, A, R, Xn, done, roots = build(trans, feats, keys, lam)
            choice[name] = {}
            for t in trees:
                tr = roots != f"{args.exp}:{t}"
                models = fqi(X[tr], A[tr], R[tr], Xn[tr], done[tr], args.gamma)
                for n in nodes:
                    if n["tree"] == t:
                        x = np.array([[n["f"][k] for k in keys]])
                        qs = {a: models[a].predict(x)[0] for a in models}
                        choice[name][n["key"]] = max(qs, key=qs.get)

        def path_return(tree, c) -> float:
            key = (args.exp, tree)
            H = next(n["h"] for n in nodes if n["key"] == key)
            ret = 0.0
            for _ in range(H):
                if key not in kids:
                    break
                r = kids[key][c[key]]
                ret += r["dV"] - lam * cost(r)
                key = (args.exp, r["child"])
            return ret

        print(f"\n=== RQ2 table, objective {objective} (lam={lam:g}, scale costs), "
              f"{len(nodes)} decisions in {len(trees)} trees ===")
        print(f"{'policy':12s} {'acc(a*)':>8s} {'regret':>8s} {'path return':>12s}   action mix")
        regs = {}
        order = ["rl_all", "rl_generic", "rl_task", "rule", "judge_task", "judge_state"] + [f"always_{a}" for a in ACTIONS]
        for pol in [p for p in order if p in choice]:
            c = choice[pol]
            reg = np.array([n["q"][n["a_star"]] - n["q"][c[n["key"]]] for n in nodes])
            acc = np.mean([n["q"][c[n["key"]]] >= n["q"][n["a_star"]] - 1e-12 for n in nodes])
            regs[pol] = reg
            print(f"{pol:12s} {acc:8.0%} {reg.mean():8.4f} {np.mean([path_return(t, c) for t in trees]):12.4f}   "
                  f"{dict(Counter(c.values()))}")
        rng = np.random.default_rng(0)
        for other in ("rule", "rl_task", "rl_generic"):
            d = regs[other] - regs["rl_all"]
            boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            print(f"regret({other}) - regret(rl_all) = {d.mean():+.4f}  95% CI "
                  f"[{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")

        print(f"\n--- RQ1 (a): does the rule pick the best action? objective {objective} ---")
        mem_med = np.median([n["f"]["mem_n_total"] for n in nodes])

        def report(label, sub):
            if not sub:
                return
            ok = np.mean([n["a_star"] == rule(n["f"]) for n in sub])
            reg = np.mean([n["q"][n["a_star"]] - n["q"][rule(n["f"])] for n in sub])
            mix = Counter(n["a_star"] for n in sub)
            print(f"  {label:34s} n={len(sub):3d}  rule = a* {ok:5.1%}  rule regret {reg:.4f}  a* mix {dict(mix)}")

        conf = [n for n in nodes if n["margin"] > args.noise]
        report("all", nodes)
        report(f"confident (margin > {args.noise})", conf)
        report("API batches (rule: H)", [n for n in nodes if n["f"]["type_api"] > 0.5])
        report("MATH batches (rule: P)", [n for n in nodes if n["f"]["type_api"] <= 0.5])
        report(f"memory small (<= median)", [n for n in nodes if n["f"]["mem_n_total"] <= mem_med])
        report(f"memory large (> median)", [n for n in nodes if n["f"]["mem_n_total"] > mem_med])
        report("no H so far", [n for n in nodes if n["f"]["n_H"] == 0])
        report("H done before", [n for n in nodes if n["f"]["n_H"] > 0])
        report("no P so far", [n for n in nodes if n["f"]["n_P"] == 0])
        report("P done before", [n for n in nodes if n["f"]["n_P"] > 0])
        report(f"before API change (step < {min(drift)})", [n for n in nodes if n["step"] < min(drift)])
        report(f"after API change (step >= {min(drift)})", [n for n in nodes if n["step"] >= min(drift)])

        print(f"\n--- RQ1 (b): does the agent state predict when the rule is right? objective {objective} ---")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        for name, sub in (("all", nodes), ("confident", conf)):
            y = np.array([n["a_star"] == rule(n["f"]) for n in sub], dtype=int)
            roots = np.array([n["tree"] for n in sub])
            line = f"  [{name}] n={len(sub)} rule right in {y.mean():.1%}:"
            for fname, keys in (("TASK", TASK_FEATURES), ("ALL", ALL_FEATURES), ("GENERIC", GENERIC)):
                X = np.array([[n["f"][k] for k in keys] for n in sub])
                prob = np.zeros(len(sub))
                for t in sorted(set(roots)):
                    tr, te = roots != t, roots == t
                    if len(set(y[tr])) < 2:
                        prob[te] = y[tr].mean()
                        continue
                    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0)
                    prob[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                auc = roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan")
                line += f"  {fname} AUC={auc:.3f}"
            print(line)


if __name__ == "__main__":
    main()
