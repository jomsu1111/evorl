"""RQ1 analysis on collected trees.

  python -m evorl.analyze --exp main [--lam 0.0]

For every interior node with all actions expanded:
  Q*(s,a) = r(s,a) + V*(child)   (exact finite-horizon backup inside the tree)
  a*(s)   = argmax_a Q*(s,a),  a1(s) = argmax_a r(s,a)  (myopic)
Reports
  (1) distribution of a* by batch type,
  (2) myopic-vs-long-horizon disagreement,
  (3) controlled pairs: among same-family pairs with similar task/feedback, a* disagreement
      when the agent state is similar vs different, with a permutation p-value,
  (4) leave-one-tree-out predictability of a* from TASK, TASK+ENV (stream position, API
      version changed) and ALL features, scored by accuracy and regret (Q*(a*) - Q*(pred)),
  (5) the features that carry the prediction.
(3) and (4) are repeated on "confident" decisions whose best action beats the runner-up by
more than --noise (default 0.03, about twice the measured reproduction noise).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from .features import AGENT_FEATURES, ALL_FEATURES, TASK_FEATURES
from .qlearn import ACTIONS, COST, cost, load, tree_values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--results-dir", default="/root/cove-rl/results")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.03,
                    help="Q gap below which the best action is treated as a tie (~2x reproduction noise)")
    ap.add_argument("--cost-model", choices=["flops", "scale", "wall"], default="flops")
    args = ap.parse_args()
    COST["model"] = args.cost_model

    trans, feats = load(args.results_dir, args.exp)
    kids, V = tree_values(trans, args.lam, 1.0)
    rows = []
    for key, ch in kids.items():
        if set(ch) != set(ACTIONS) or key not in feats or ch["H"]["tree"] == "behavior":
            continue
        h = ch["H"]["horizon"] - ch["H"]["tree_depth"]  # steps left in this node's tree
        q = {a: r["dV"] - args.lam * cost(r) + V((key[0], r["child"]), h - 1) for a, r in ch.items()}
        r1 = {a: r["dV"] - args.lam * cost(r) for a, r in ch.items()}
        rows.append({"key": key, "root": f"{key[0]}:{ch['H']['tree']}", "f": feats[key], "q": q,
                     "a_star": max(q, key=q.get), "a_myopic": max(r1, key=r1.get),
                     "margin": sorted(q.values())[-1] - sorted(q.values())[-2], "h": h})
    print(f"interior nodes with all actions: {len(rows)}")
    if not rows:
        return

    by_type = defaultdict(Counter)
    for r in rows:
        by_type["api" if r["f"]["type_api"] > 0.5 else "math"][r["a_star"]] += 1
    print("(1) a* by batch type:", {k: dict(v) for k, v in by_type.items()})
    lh = [r for r in rows if r["h"] > 1]  # with one step left both criteria coincide by definition
    dis = np.mean([r["a_star"] != r["a_myopic"] for r in lh]) if lh else float("nan")
    print(f"(2) myopic != long-horizon best action (nodes with >1 step left, n={len(lh)}): {dis:.2%}")

    rng = np.random.default_rng(0)
    conf = [r for r in rows if r["margin"] > args.noise]
    print(f"    decisions whose best action beats the runner-up by more than {args.noise:.3f} "
          f"(~2x the measured reproduction noise): {len(conf)}/{len(rows)}")

    # (3) similar task/feedback: does a* agree less often when the agent state differs?
    for name, sub in (("all", rows), ("confident", conf)):
        n, d_lo, d_hi, gap, p = pair_test(sub, rng)
        print(f"(3) [{name}] {n} same-family pairs with similar task/feedback (lowest-quartile task distance): "
              f"a* differs in {d_lo:.1%} when the agent state is similar vs {d_hi:.1%} when it differs "
              f"(gap {gap:+.1%}, permutation p={p:.3f})")

    # (4) leave-one-tree-out predictability of a*; TASK+ENV checks that agent features add more than
    #     knowing the stream position / whether the API version has changed.
    drift = _drift_steps(args.results_dir, args.exp)
    for r in rows:
        step = r["key"][1].count(".")
        r["env"] = {"step": step / 10.0, "post_drift": float(any(step >= d for d in drift))}
    sets = (("TASK", TASK_FEATURES, []), ("TASK+ENV", TASK_FEATURES, ENV_FEATURES), ("ALL", ALL_FEATURES, []))
    for name, sub in (("all", rows), ("confident", conf)):
        y = np.array([r["a_star"] for r in sub])
        for fname, keys, env_keys in sets:
            acc, reg = loro(sub, keys, env_keys)
            print(f"(4) [{name}] {fname:8s} features: LORO accuracy={acc:.2%}  mean regret={reg:.4f}")
        maj = Counter(y).most_common(1)[0][0]
        print(f"    [{name}] majority-class ({maj}) accuracy={np.mean(y == maj):.2%} "
              f"regret={np.mean([r['q'][r['a_star']] - r['q'][maj] for r in sub]):.4f}")

    # (5) which state features carry the prediction
    from sklearn.ensemble import RandomForestClassifier
    X = np.array([[r["f"][k] for k in ALL_FEATURES] + [r["env"][k] for k in ENV_FEATURES] for r in rows])
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=0).fit(X, [r["a_star"] for r in rows])
    names = ALL_FEATURES + ENV_FEATURES
    top = sorted(zip(clf.feature_importances_, names), reverse=True)[:10]
    print("(5) top features for a*: " + ", ".join(f"{n}{'*' if n in AGENT_FEATURES else ''}={v:.3f}" for v, n in top)
          + "   (* = agent learning state)")


ENV_FEATURES = ["step", "post_drift"]


def _drift_steps(results_dir: str, exps) -> list:
    import json
    import os
    p = os.path.join(results_dir, exps[0], "config.json")
    return json.load(open(p)).get("drift_steps", [6]) if os.path.exists(p) else [6]


def pair_test(rows, rng, n_perm: int = 2000):
    """Same-family pairs whose task/feedback features are close (lowest quartile of distance) are
    split by agent-feature distance (below / above the median). Returns the a* disagreement rate in
    each half, their gap, and a one-sided permutation p-value (a* shuffled within family)."""
    T = np.array([[r["f"][k] for k in TASK_FEATURES] for r in rows])
    Ag = np.array([[r["f"][k] for k in AGENT_FEATURES] for r in rows])
    Tn = (T - T.mean(0)) / (T.std(0) + 1e-8)
    An = (Ag - Ag.mean(0)) / (Ag.std(0) + 1e-8)
    fam = np.array([r["f"]["type_api"] > 0.5 for r in rows])
    y = np.array([r["a_star"] for r in rows])
    I, J = np.triu_indices(len(rows), 1)
    keep = fam[I] == fam[J]
    I, J = I[keep], J[keep]
    dt = np.linalg.norm(Tn[I] - Tn[J], axis=1)
    near = dt <= np.quantile(dt, 0.25)
    I, J = I[near], J[near]
    da = np.linalg.norm(An[I] - An[J], axis=1)
    lo = da <= np.median(da)

    def stats(lbl):
        d = lbl[I] != lbl[J]
        return d[lo].mean(), d[~lo].mean()

    d_lo, d_hi = stats(y)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for f in (True, False):
            m = fam == f
            yp[m] = rng.permutation(y[m])
        a, b = stats(yp)
        null.append(b - a)
    p = (np.sum(np.array(null) >= d_hi - d_lo) + 1) / (n_perm + 1)
    return len(I), d_lo, d_hi, d_hi - d_lo, p


def loro(rows, keys, env_keys):
    """Leave-one-tree-out RandomForest prediction of a*: accuracy and mean regret."""
    from sklearn.ensemble import RandomForestClassifier
    roots = np.array([r["root"] for r in rows])
    y = np.array([r["a_star"] for r in rows])
    X = np.array([[r["f"][k] for k in keys] + [r["env"][k] for k in env_keys] for r in rows])
    acc, reg = [], []
    for held in sorted(set(roots)):
        tr, te = roots != held, roots == held
        if len(set(y[tr])) < 2:
            pred = np.array([Counter(y[tr]).most_common(1)[0][0]] * te.sum())
        else:
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0).fit(X[tr], y[tr])
            pred = clf.predict(X[te])
        for r, p in zip([rows[i] for i in np.where(te)[0]], pred):
            acc.append(p == r["a_star"])
            reg.append(r["q"][r["a_star"]] - r["q"][p])
    return float(np.mean(acc)), float(np.mean(reg))


if __name__ == "__main__":
    main()
