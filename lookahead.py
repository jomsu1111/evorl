"""Does the update decision need lookahead (RL with gamma > 0), or is a one-step router enough?

  python -m evorl.lookahead --exp main --mode oracle   # (A) value of lookahead in the trees themselves
  python -m evorl.lookahead --exp main --mode cv       # (B) can a learned router capture it?
  python -m evorl.lookahead --exp main --mode fit --h 3 --lam 0.002 --tag fh3_l0.002   # router for online runs
  python -m evorl.lookahead --exp eval_s1 eval_s2 --mode mc  # (D) longer-horizon counterfactual rollouts

(A) oracle: on the fully expanded trees, compare the best h-step path with the path that greedily
    maximises the next reward. Choosing the max of 4^h noisy paths is optimistic (winner's curse), so
    the validation tasks are split in half: paths are chosen on half A and scored on half B.
(B) cv: leave-one-tree-out. Every router is scored by the undiscounted return it collects inside the
    held-out tree (V_final - V_root - lam * cost), like qlearn.cv_regret. Routers:
      greedy   Q_1 = r                                   (one-step, gamma = 0)
      fqi      qlearn.fqi with gamma 0.9                 (current q_rl / q_rlcost recipe)
      fqi_togo fqi + "steps to go" feature               (tree leaves are terminal, so the target of a
                                                          state depends on its depth; this tells the model)
      fh_fqi   finite-horizon FQI: Q_h = r + max Q_{h-1}(s'), Q_0 = 0, one model set per h
      fh_tree  Q_h fitted to exact tree backups r + V*_tree(s', h-1) of the training trees
    Finite-horizon routers act with Q_{steps left} inside the tree. "critical" nodes are those with
    >= 2 steps left whose long-horizon best action differs from the myopic best one: only there can
    lookahead pay off.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List

import numpy as np

from .config import ACTIONS
from .features import ALL_FEATURES, TASK_FEATURES
from . import qlearn as Q


def _trees(results_dir: str, exp: str) -> List[str]:
    return open(os.path.join(results_dir, exp, "trees.txt")).read().split()


def _kids(trans) -> Dict[tuple, Dict[str, dict]]:
    kids: Dict[tuple, Dict[str, dict]] = {}
    for r in trans:
        kids.setdefault((r["exp"], r["parent"]), {})[r["action"]] = r
    return kids


# ---------------------------------------------------------------- (A) oracle, split-half
def _task_scores(results_dir: str, exp: str) -> Dict[str, Dict[str, float]]:
    out = {}
    nd = os.path.join(results_dir, exp, "nodes")
    for nid in os.listdir(nd):
        p = os.path.join(nd, nid, "val.json")
        if os.path.exists(p):
            out[nid] = {k: float(v) for k, v in json.load(open(p))["mem"].items()}
    return out


def _V(scores: Dict[str, float], keep=None) -> float:
    """V = (acc_api + acc_math) / 2 on the kept validation tasks (API task ids start with 'aval')."""
    api = [v for k, v in scores.items() if k.startswith("aval") and (keep is None or k in keep)]
    mth = [v for k, v in scores.items() if not k.startswith("aval") and (keep is None or k in keep)]
    return (np.mean(api) if api else 0.0) / 2 + (np.mean(mth) if mth else 0.0) / 2


def oracle(results_dir: str, exp: str, lam: float, reps: int, seed: int = 0) -> None:
    trans, _ = Q.load(results_dir, [exp])
    kids = {p: k for (e, p), k in _kids(trans).items()}
    H = trans[0]["horizon"]
    scores = _task_scores(results_dir, exp)
    for r in trans[:20]:  # the stored V must be what we recompute from per-task scores
        assert abs(_V(scores[r["child"]]) - r["V_child"]) < 1e-6, r["child"]
    task_ids = sorted(set().union(*[set(s) for s in scores.values()]))
    rng = np.random.default_rng(seed)
    starts = [(n, H - (n.count(".") - t.count("."))) for t in _trees(results_dir, exp)
              for n in [t] + [f"{t}.{a}" for a in ACTIONS] if n in kids]
    res: Dict[int, List[tuple]] = {}
    for _ in range(reps):
        perm = rng.permutation(task_ids)
        halves = {"A": set(perm[::2]), "B": set(perm[1::2])}
        vc: Dict[tuple, float] = {}

        def v(n, half):
            if (n, half) not in vc:
                vc[(n, half)] = _V(scores[n], halves[half])
            return vc[(n, half)]

        def rw(r, half):
            return v(r["child"], half) - v(r["parent"], half) - lam * Q.cost(r)

        memo: Dict[tuple, float] = {}

        def vstar(n, h):  # best h-step return, measured on half A
            if h == 0 or n not in kids:
                return 0.0
            if (n, h) not in memo:
                memo[(n, h)] = max(rw(r, "A") + vstar(r["child"], h - 1) for r in kids[n].values())
            return memo[(n, h)]

        def run(n, h, look):  # act with `look`-step lookahead chosen on A; score on B
            ret = 0.0
            for left in range(h, 0, -1):
                if n not in kids:
                    break
                k = min(look, left)
                r = max(kids[n].values(), key=lambda r: rw(r, "A") + vstar(r["child"], k - 1))
                ret += rw(r, "B")
                n = r["child"]
            return ret

        for n, h in starts:
            res.setdefault(h, []).append((run(n, h, h), run(n, h, 1)))
    print(f"=== (A) oracle lookahead, lam={lam}, cost={Q.COST['model']}, {reps} split-halves "
          "(paths chosen on half A of the val tasks, scored on half B) ===")
    for h in sorted(res, reverse=True):
        a = np.array(res[h]).reshape(reps, -1, 2)
        gain = (a[:, :, 0] - a[:, :, 1]).mean(axis=1)  # mean gain per split
        print(f"  steps left {h}: n={a.shape[1]:2d}  lookahead {a[:, :, 0].mean():+.4f}  "
              f"greedy {a[:, :, 1].mean():+.4f}  gain {gain.mean():+.4f}  "
              f"[2.5%, 97.5%] over splits [{np.percentile(gain, 2.5):+.4f}, {np.percentile(gain, 97.5):+.4f}]  "
              f"gain>0 in {np.mean(gain > 0):.0%} of splits")


# ---------------------------------------------------------------- (B) learned routers, LORO CV
def _togo(r) -> float:
    return float(r["horizon"] - r["tree_depth"]) if r["tree_depth"] is not None else float(r["horizon"])


def _data(trans, feats, keys, lam, togo=False):
    rows = []
    for r in trans:
        s = feats.get((r["exp"], r["parent"]))
        if s is None:
            continue
        sn = feats.get((r["exp"], r["child"]))
        x = [s[k] for k in keys] + ([_togo(r)] if togo else [])
        xn = ([sn[k] for k in keys] + ([_togo(r) - 1] if togo else [])) if sn else None
        rows.append(dict(x=x, a=r["action"], r=r["dV"] - lam * Q.cost(r), xn=xn,
                         root=f"{r['exp']}:{r['tree']}", key=(r["exp"], r["parent"]), child=(r["exp"], r["child"])))
    return rows


def _fit(X, y):
    m = Q.make_model()
    m.fit(np.array(X), np.array(y))
    return m


def fit_fh(rows, H, kids=None, V=None) -> Dict[int, Dict[str, object]]:
    """Finite-horizon Q_1..Q_H. With V (exact tree backups) the targets are r + V*(s', h-1);
    otherwise they bootstrap from the previous model set (finite-horizon FQI)."""
    models: Dict[int, Dict[str, object]] = {}
    for h in range(1, H + 1):
        models[h] = {}
        for a in ACTIONS:
            X, y = [], []
            for d in rows:
                if d["a"] != a:
                    continue
                if h == 1:
                    X.append(d["x"]); y.append(d["r"])
                elif V is not None:
                    if d["child"] in kids:  # the child was expanded, so V*(child, h-1) is exact
                        X.append(d["x"]); y.append(d["r"] + V(d["child"], h - 1))
                elif d["xn"] is not None:
                    xn = np.array([d["xn"]])
                    X.append(d["x"]); y.append(d["r"] + max(m.predict(xn)[0] for m in models[h - 1].values()))
            if X:
                models[h][a] = _fit(X, y)
    return models


def cv(results_dir: str, exps: List[str], keys, lam: float) -> None:
    trans, feats = Q.load(results_dir, exps)
    kids = _kids(trans)
    _, V = Q.tree_values(trans, lam, 1.0)
    H = trans[0]["horizon"]
    rows = _data(trans, feats, keys, lam)
    rows_t = _data(trans, feats, keys, lam, togo=True)
    held_out = sorted({d["root"] for d in rows if not d["root"].endswith(":behavior")})

    def q1(r):  # one-step value of an action at a node, from the tree
        return r["dV"] - lam * Q.cost(r)

    names = ["greedy", "fqi", "fqi_togo", "fh_fqi", "fh_tree"]
    out = {n: [] for n in names}
    crit = {n: [] for n in names}  # (picked long-horizon best?, regret) on critical nodes
    allh2 = {n: [] for n in names}
    for held in held_out:
        tr = [d for d in rows if d["root"] != held]
        trt = [d for d in rows_t if d["root"] != held]
        m_g = fit_fh(tr, 1)[1]
        A = np.array([d["a"] for d in tr])
        X = np.array([d["x"] for d in tr]); R = np.array([d["r"] for d in tr])
        Xn = np.array([d["xn"] if d["xn"] is not None else [0.0] * len(keys) for d in tr])
        done = np.array([d["xn"] is None for d in tr])
        m_f = Q.fqi(X, A, R, Xn, done, 0.9)
        At = np.array([d["a"] for d in trt]); Xt = np.array([d["x"] for d in trt])
        Xnt = np.array([d["xn"] if d["xn"] is not None else [0.0] * (len(keys) + 1) for d in trt])
        m_ft = Q.fqi(Xt, At, R, Xnt, done, 0.9)
        m_fh = fit_fh(tr, H)
        m_tree = fit_fh(tr, H, kids=kids, V=V)

        def choose(name, key, h):
            x = [feats[key][k] for k in keys]
            cand = kids[key]
            if name == "greedy":
                ms, xx = m_g, x
            elif name == "fqi":
                ms, xx = m_f, x
            elif name == "fqi_togo":
                ms, xx = m_ft, x + [float(h)]
            else:
                ms, xx = (m_fh if name == "fh_fqi" else m_tree)[h], x
            qs = {a: ms[a].predict(np.array([xx]))[0] for a in cand if a in ms}
            return max(qs, key=qs.get)

        exp, root = held.split(":", 1)
        for name in names:
            key, ret = (exp, root), 0.0
            for h in range(H, 0, -1):
                if key not in kids or key not in feats:
                    break
                a = choose(name, key, h)
                ret += q1(kids[key][a])
                key = (exp, kids[key][a]["child"])
            out[name].append((held, ret, V((exp, root), H)))
            # node-level: every interior node of the held-out tree with >= 2 steps left
            for nkey in [k for k in kids if k[0] == exp and (k[1] == root or k[1].startswith(root + "."))]:
                h = H - (nkey[1].count(".") - root.count("."))
                if h < 2 or nkey not in feats:
                    continue
                long = {a: q1(r) + V((exp, r["child"]), h - 1) for a, r in kids[nkey].items()}
                myo = {a: q1(r) for a, r in kids[nkey].items()}
                best_l, best_m = max(long, key=long.get), max(myo, key=myo.get)
                a = choose(name, nkey, h)
                reg = long[best_l] - long[a]
                allh2[name].append(reg)
                if best_l != best_m:
                    crit[name].append((a == best_l, a == best_m, reg))
    print(f"=== (B) leave-one-tree-out, lam={lam}, cost={Q.COST['model']}, features={len(keys)} ===")
    opt = np.mean([o for _, _, o in out["greedy"]])
    print(f"  optimal path return {opt:+.4f}")
    print(f"  {'router':9s} {'path return':>11s} {'regret':>7s}   {'nodes h>=2 regret':>17s}   "
          f"critical nodes (n={len(crit['greedy'])}): long-best / myopic-best / regret")
    for name in names:
        ret = np.mean([r for _, r, _ in out[name]])
        c = np.array(crit[name], dtype=float)
        print(f"  {name:9s} {ret:+11.4f} {opt - ret:7.4f}   {np.mean(allh2[name]):17.4f}   "
              f"{c[:, 0].mean():.0%} / {c[:, 1].mean():.0%} / {c[:, 2].mean():.4f}")
    base = {h: r for h, r, _ in out["greedy"]}
    for name in names[1:]:
        d = np.array([r - base[h] for h, r, _ in out[name]])
        print(f"  {name} - greedy per tree: {' '.join(f'{x:+.3f}' for x in d)}  "
              f"(better {int((d > 1e-9).sum())} / same {int((abs(d) <= 1e-9).sum())} / worse {int((d < -1e-9).sum())})")


def fit_online(results_dir: str, exps: List[str], keys, lam: float, h: int, tag: str, tree: bool) -> None:
    """Save Q_h as a q_*.pkl that QPolicy can load (receding horizon: always look h steps ahead)."""
    trans, feats = Q.load(results_dir, exps)
    rows = _data(trans, feats, keys, lam)
    if tree:
        _, V = Q.tree_values(trans, lam, 1.0)
        models = fit_fh(rows, h, kids=_kids(trans), V=V)
    else:
        models = fit_fh(rows, h)
    path = os.path.join(results_dir, exps[0], f"q_{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"models": models[h], "features": keys, "gamma": None, "horizon": h, "lam": lam,
                     "cost_model": Q.COST["model"], "tag": tag, "tree_targets": tree}, f)
    print("saved", path)


def mc(results_dir: str, exps: List[str], lam: float, reps: int, k: int = 6, seed: int = 0) -> None:
    """(D) Counterfactual rollouts from scripts/lookahead_runs.sh: at a start node of a logged run,
    each first action a is followed by the router for k-1 more steps. Compares picking the first
    action by its 1-step reward with picking it by its k-step return, choosing on half A of the val
    tasks and scoring on half B."""
    import glob
    import re
    starts = []  # (exp, t, {a: [node ids t..t+k]}, {a: [actions]})
    for exp in exps:
        runs: Dict[tuple, dict] = {}
        for f in glob.glob(os.path.join(results_dir, exp, "curves", "seq_*_then_*_R0.jsonl")):
            m = re.match(r"seq_([HPBN]+)_then_(.+)_R0\.jsonl", os.path.basename(f))
            seq, then = m.group(1), m.group(2)
            recs = [json.loads(l) for l in open(f)]
            t = len(seq) - 1
            if len(recs) < t + k + 1:
                continue  # unfinished
            runs.setdefault((then, seq[:-1]), {})[seq[-1]] = recs
        for (then, prefix), by_a in sorted(runs.items()):
            if set(by_a) == set(ACTIONS):
                t = len(prefix)
                starts.append((exp, then, t, {a: [r["node"] for r in recs[t:t + k + 1]] for a, recs in by_a.items()},
                               {a: [r["action"] for r in recs[t + 1:t + k + 1]] for a, recs in by_a.items()}))
    if not starts:
        print("no complete counterfactual starts yet")
        return
    scores = {}
    for exp, _, _, nodes, _ in starts:
        for ns in nodes.values():
            for n in ns:
                if (exp, n) not in scores:
                    scores[(exp, n)] = {kk: float(v) for kk, v in
                                        json.load(open(os.path.join(results_dir, exp, "nodes", n, "val.json")))["mem"].items()}
    cost = lambda acts: sum(Q.SCALE[a] for a in acts) if Q.COST["model"] == "scale" else 0.0
    if Q.COST["model"] != "scale":
        print("note: only the scale cost is applied here")
    task_ids = sorted(set().union(*[set(s) for s in scores.values()]))
    rng = np.random.default_rng(seed)

    def ret(exp, nodes, acts, h, keep=None):
        return _V(scores[(exp, nodes[h])], keep) - _V(scores[(exp, nodes[0])], keep) - lam * cost(acts[:h])

    print(f"=== (D) counterfactual first actions, then the router for {k - 1} steps; lam={lam} ===")
    print(f"  {'exp':10s} {'router':10s} {'t':>2s}  {'1-step return per first action':40s} {f'{k}-step return':40s} best1 best{k}")
    for exp, then, t, nodes, acts in starts:
        r1 = {a: ret(exp, nodes[a], acts[a], 1) for a in ACTIONS}
        rk = {a: ret(exp, nodes[a], acts[a], k) for a in ACTIONS}
        print(f"  {exp:10s} {then:10s} {t:2d}  " + " ".join(f"{a}{r1[a]:+.3f}" for a in ACTIONS) + "   "
              + " ".join(f"{a}{rk[a]:+.3f}" for a in ACTIONS) + f"   {max(r1, key=r1.get)}     {max(rk, key=rk.get)}")
    gains = []
    for _ in range(reps):
        perm = rng.permutation(task_ids)
        A, B = set(perm[::2]), set(perm[1::2])
        g = []
        for exp, _, t, nodes, acts in starts:
            a1 = max(ACTIONS, key=lambda a: ret(exp, nodes[a], acts[a], 1, A))
            ak = max(ACTIONS, key=lambda a: ret(exp, nodes[a], acts[a], k, A))
            g.append(ret(exp, nodes[ak], acts[ak], k, B) - ret(exp, nodes[a1], acts[a1], k, B))
        gains.append(np.mean(g))
    gains = np.array(gains)
    print(f"  {k}-step return of choosing by {k}-step vs by 1-step (chosen on half A, scored on half B): "
          f"{gains.mean():+.4f}  [2.5%, 97.5%] [{np.percentile(gains, 2.5):+.4f}, {np.percentile(gains, 97.5):+.4f}]  "
          f"> 0 in {np.mean(gains > 0):.0%} of splits; n starts = {len(starts)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--results-dir", default="/root/cove-rl/results")
    ap.add_argument("--mode", choices=["oracle", "cv", "fit", "mc"], required=True)
    ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.002])
    ap.add_argument("--cost-model", choices=["flops", "scale", "wall"], default="scale")
    ap.add_argument("--features", choices=["all", "task"], default="all")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--tree-targets", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    Q.COST["model"] = args.cost_model
    keys = ALL_FEATURES if args.features == "all" else TASK_FEATURES
    for lam in args.lam:
        if args.mode == "oracle":
            oracle(args.results_dir, args.exp[0], lam, args.reps)
        elif args.mode == "cv":
            cv(args.results_dir, args.exp, keys, lam)
        elif args.mode == "mc":
            mc(args.results_dir, args.exp, lam, args.reps)
        else:
            fit_online(args.results_dir, args.exp, keys, lam, args.h,
                       args.tag or f"fh{args.h}_l{lam}", args.tree_targets)


if __name__ == "__main__":
    main()
