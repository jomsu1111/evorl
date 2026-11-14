"""RQ1 analysis on collected trees.

  python -m evorl.analyze --exp main [--lam 0.0]

For every interior node with all actions expanded:
  Q*(s,a) = r(s,a) + V*(child)   (exact finite-horizon backup inside the tree)
  a*(s)   = argmax_a Q*(s,a),  a1(s) = argmax_a r(s,a)  (myopic)
Reports
  (1) distribution of a* by batch type,
  (2) myopic-vs-long-horizon disagreement,
  (3) matched pairs: similar task/feedback features but different agent state
      -> how often does a* differ,
  (4) leave-one-root-out predictability of a* from TASK vs ALL features,
      scored by accuracy and by regret (Q*(a*) - Q*(predicted)).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from .features import AGENT_FEATURES, ALL_FEATURES, TASK_FEATURES
from .qlearn import ACTIONS, cost, load, tree_values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--results-dir", default="/root/cove-rl/results")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--pair-eps", type=float, default=0.15)
    args = ap.parse_args()

    trans, feats = load(args.results_dir, args.exp)
    kids, V = tree_values(trans, args.lam, 1.0)
    rows = []
    for key, ch in kids.items():
        if set(ch) != set(ACTIONS) or key not in feats:
            continue
        q = {a: r["dV"] - args.lam * cost(r) + V((key[0], r["child"])) for a, r in ch.items()}
        r1 = {a: r["dV"] - args.lam * cost(r) for a, r in ch.items()}
        rows.append({"key": key, "root": f"{key[0]}:{ch['H']['root']}", "f": feats[key], "q": q,
                     "a_star": max(q, key=q.get), "a_myopic": max(r1, key=r1.get),
                     "margin": sorted(q.values())[-1] - sorted(q.values())[-2]})
    print(f"interior nodes with all actions: {len(rows)}")
    if not rows:
        return

    by_type = defaultdict(Counter)
    for r in rows:
        by_type["api" if r["f"]["type_api"] > 0.5 else "math"][r["a_star"]] += 1
    print("(1) a* by batch type:", {k: dict(v) for k, v in by_type.items()})
    dis = np.mean([r["a_star"] != r["a_myopic"] for r in rows])
    print(f"(2) myopic != long-horizon best action: {dis:.2%}")

    T = np.array([[r["f"][k] for k in TASK_FEATURES] for r in rows])
    Ag = np.array([[r["f"][k] for k in AGENT_FEATURES] for r in rows])
    Tn = (T - T.mean(0)) / (T.std(0) + 1e-8)
    An = (Ag - Ag.mean(0)) / (Ag.std(0) + 1e-8)
    same_task, diff = 0, 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i]["f"]["type_api"] != rows[j]["f"]["type_api"]:
                continue
            dt = np.linalg.norm(Tn[i] - Tn[j]) / np.sqrt(Tn.shape[1])
            da = np.linalg.norm(An[i] - An[j]) / np.sqrt(An.shape[1])
            if dt < args.pair_eps and da > 2 * args.pair_eps:
                same_task += 1
                diff += rows[i]["a_star"] != rows[j]["a_star"]
    print(f"(3) pairs with similar task/feedback but different agent state: {same_task}, "
          f"a* differs in {diff / max(1, same_task):.2%}")

    from sklearn.ensemble import RandomForestClassifier
    roots = np.array([r["root"] for r in rows])
    y = np.array([r["a_star"] for r in rows])
    for name, keys in (("TASK", TASK_FEATURES), ("ALL", ALL_FEATURES)):
        X = np.array([[r["f"][k] for k in keys] for r in rows])
        acc, reg = [], []
        for held in sorted(set(roots)):
            tr, te = roots != held, roots == held
            if len(set(y[tr])) < 2:
                pred = np.array([Counter(y[tr]).most_common(1)[0][0]] * te.sum())
            else:
                clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0).fit(X[tr], y[tr])
                pred = clf.predict(X[te])
            for r, p in zip([rows[i] for i in np.where(te)[0]], pred):
                acc.append(p == r["a_star"]); reg.append(r["q"][r["a_star"]] - r["q"][p])
        print(f"(4) {name:4s} features: LORO accuracy={np.mean(acc):.2%}  mean regret={np.mean(reg):.4f}")
    maj = Counter(y).most_common(1)[0][0]
    print(f"    majority-class ({maj}) accuracy={np.mean(y == maj):.2%} regret={np.mean([r['q'][r['a_star']] - r['q'][maj] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
