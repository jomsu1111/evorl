"""Offline comparison of the RL router against LLM-judge routers on the collected trees.

  python -m evorl.judge_eval --exp main [--results-dir DIR] [--n 5] [--lam 0.002]

Every interior node of every collected tree (all four actions expanded) is one decision.
Each policy picks an action there and is scored with the tree's exact finite-horizon
values Q*(s, a) (the same backup as analyze.py), so no environment runs are needed:

  judge_task   the helper LLM sees the batch: task family, success rates, error types and
               example failures with their feedback (COVE-style routing, at batch level).
  judge_state  the same plus the agent learning state the router sees: validation accuracy,
               memory effect and size, update counts and recency, and the agent's own update
               history with the resulting change in V.
  rl_all       fitted-Q router on all features, trained leave-one-tree-out (gamma 0.9).
  rl_task      the same router on task/feedback features only.

Two objectives: performance only (lam = 0) and cost-aware (assumed scale costs, --lam).
The judge is told the objective it is scored on, including the remaining horizon.
Judge answers are cached in <results>/<exp>/judge/ so reruns only query new nodes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from .config import ACTIONS, EvoConfig
from .features import ALL_FEATURES, TASK_FEATURES
from .qlearn import COST, SCALE, build, cost, fqi, load, tree_values

INTRO = """You control how a small LLM agent (Qwen3-0.6B) learns from its own experience. The agent \
works through a stream of task batches. After each batch of 32 tasks you choose ONE update action:
- H (harness / memory): a 4B helper model distills the batch's 32 episodes into reusable memory \
notes and skills. Later, the most similar notes are retrieved into the agent's prompt. Weights do \
not change.
- P (parameters): LoRA fine-tuning of the agent on its own successful solutions to this batch's \
tasks (rejection sampling: 4 extra attempts per task, up to 2 successes kept per task). Changes \
the weights; nothing is written to memory.
- B (both): H and P.
- N (none): this batch's experience is not used.

Two task families alternate: API tasks on even steps (write Python against a library whose \
calling conventions the agent must learn; the library version can change without notice, which \
renames functions and changes conventions) and MATH competition problems (levels 1-3) on odd \
steps. Performance V is the average of API accuracy and MATH accuracy on a fixed validation set \
(80 API + 200 MATH tasks, single attempt, no feedback)."""

OBJECTIVE = {
    "perf": "Objective: maximize the total change in V over the next {h} update step(s), i.e. this \
action followed by the best possible follow-up actions.",
    "cost": "Objective: maximize (total change in V over the next {h} update step(s), in V points) \
minus (compute cost of the updates). Costs in V points: H = {cH:.1f}, P = {cP:.1f}, B = {cB:.1f}, \
N = 0. Count this action and the best possible follow-up actions.",
}

ANSWER = "Think briefly (at most 6 sentences), then give your choice as <action>H</action>, \
<action>P</action>, <action>B</action> or <action>N</action>."


def _clip(s: str, n: int, tail: bool = False) -> str:
    s = " ".join(str(s).split())
    if len(s) <= n:
        return s
    return ("..." + s[-n:]) if tail else (s[:n] + "...")


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def batch_text(eps: List[Dict[str, Any]], f: Dict[str, float]) -> str:
    api = eps[0]["task_type"] == "api"
    errs = Counter(e["turns"][0]["grade"].get("error") or "other" for e in eps if not e["success_t1"])
    out = [f"## Current batch: {len(eps)} {'API' if api else 'MATH'} tasks",
           f"- First-attempt success: {_pct(f['t1_success'])}. Of the first-attempt failures, solved "
           f"after one feedback retry: {_pct(f['t2_recovery'])}. Final success: {_pct(f['final_success'])}.",
           "- First-attempt error types: " + (", ".join(f"{k} x{v}" for k, v in errs.most_common()) or "none")]
    for i, e in enumerate([e for e in eps if not e["final"]["success"]][:3], 1):
        fb = e["final"]
        detail = fb.get("message", "") + (f" ({fb['reference']})" if fb.get("reference") else "")
        out.append(f"- Failure {i}. Task: {_clip(e['task_text'], 350)}\n"
                   f"  Agent's last answer: {_clip(e['turns'][-1]['response'], 300, tail=not api)}\n"
                   f"  Feedback: {_clip(detail, 300)}")
    for e in [e for e in eps if e["success_t1"]][:1]:
        out.append(f"- A task solved on the first attempt: {_clip(e['task_text'], 250)}")
    return "\n".join(out)


def state_text(node: Dict[str, Any], f: Dict[str, float]) -> str:
    m, hist = node["metrics"], node["history"]

    def since(acts):
        idx = [i for i, h in enumerate(hist) if h["action"] in acts]
        return str(len(hist) - 1 - idx[-1]) if idx else "never"

    out = ["## Agent learning state",
           f"- Validation accuracy now: V = {m['V']:.3f} (API {m['acc_api']:.3f}, MATH {m['acc_math']:.3f}).",
           f"- Effect of memory retrieval on validation accuracy (with minus without memory): "
           f"API {f['mem_gain_api']:+.3f}, MATH {f['mem_gain_math']:+.3f}.",
           f"- Memory bank: {round(f['mem_n_total'] * 100)} entries, {_pct(f['mem_frac_volatile'])} marked "
           f"volatile, mean usage count {f['mem_mean_usage'] * 10:.1f}, success rate of tasks that used an "
           f"entry {_pct(f['mem_outcome_success'])}, mean revision count {f['mem_mean_version'] * 5:.2f}.",
           f"- In this batch {_pct(f['ctx_rate'])} of tasks retrieved memory; first-attempt success with minus "
           f"without retrieved memory: {100 * f['ctx_success_gap']:+.0f} points.",
           f"- Updates so far: {round(f['n_P'] * 5)} parameter updates (P or B, "
           f"{round(f['train_samples_k'] * 1000)} training examples in total), "
           f"{round(f['n_H'] * 5)} harness updates (H or B).",
           f"- Steps since the last parameter update: {since(('P', 'B'))}; since the last harness update: "
           f"{since(('H', 'B'))}.",
           f"- First-attempt success compared with the previous batch of the same family: "
           f"{100 * f['delta_t1_same_type']:+.0f} points.",
           "- Update history (step: batch family, action taken, resulting change in V):"]
    out += [f"  step {h['step']}: {h['batch_type']}, {h['action']}, {h['dV']:+.3f}" for h in hist] or \
           ["  none yet (initial agent)"]
    return "\n".join(out)


def prompt(variant: str, objective: str, node: Dict[str, Any], eps, f, h: int, lam: float) -> str:
    obj = OBJECTIVE[objective].format(h=h, cH=100 * lam * SCALE["H"], cP=100 * lam * SCALE["P"],
                                      cB=100 * lam * SCALE["B"])
    parts = [INTRO, obj, batch_text(eps, f)]
    if variant == "judge_state":
        parts.append(state_text(node, f))
    parts.append(ANSWER)
    return "\n\n".join(parts)


def parse_action(text: str) -> Optional[str]:
    hits = re.findall(r"<action>\s*([HPBN])\s*</action>", text or "")
    return hits[-1] if hits else None


def ask_judge(client: OpenAI, model: str, text: str, n: int, seed: int) -> Dict[str, Any]:
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}],
                                       temperature=0.7, top_p=0.95, max_tokens=400, n=n, seed=seed)
    outs = [c.message.content or "" for c in r.choices]
    votes = [a for a in (parse_action(o) for o in outs) if a]
    if votes:
        cnt = Counter(votes)
        top = max(cnt.values())
        action = next(a for a in votes if cnt[a] == top)  # tie -> first sampled
    else:
        action = "N"
    return {"action": action, "votes": votes, "raw0": outs[0] if outs else "",
            "tokens": r.usage.prompt_tokens + r.usage.completion_tokens if r.usage else 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--results-dir", default="/root/cove-rl/results")
    ap.add_argument("--n", type=int, default=5, help="judge samples per decision (majority vote)")
    ap.add_argument("--lam", type=float, default=0.002, help="cost weight of the cost-aware objective")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--variants", nargs="+", default=["judge_task", "judge_state"])
    args = ap.parse_args()
    COST["model"] = "scale"
    cfg = EvoConfig.load()
    root = os.path.join(args.results_dir, args.exp)
    trans, feats = load(args.results_dir, [args.exp])
    jdir = os.path.join(root, "judge")
    os.makedirs(jdir, exist_ok=True)
    client = OpenAI(api_key="EMPTY", base_url=cfg.helper_url, timeout=600, max_retries=3)

    objectives = {"perf": 0.0, "cost": args.lam}
    for objective, lam in objectives.items():
        kids, V = tree_values(trans, lam, 1.0)
        nodes = []  # interior decision nodes of collected trees
        for key, ch in kids.items():
            if set(ch) == set(ACTIONS) and key in feats and ch["H"]["tree"] != "behavior":
                h = ch["H"]["horizon"] - ch["H"]["tree_depth"]
                q = {a: r["dV"] - lam * cost(r) + V((key[0], r["child"]), h - 1) for a, r in ch.items()}
                nodes.append({"key": key, "tree": ch["H"]["tree"], "h": h, "q": q,
                              "depth": ch["H"]["tree_depth"]})
        trees = sorted({n["tree"] for n in nodes})
        choice: Dict[str, Dict[Any, str]] = defaultdict(dict)

        # --- LLM judges (cached) ---
        for variant in args.variants:
            cache_p = os.path.join(jdir, f"{variant}_{objective}_lam{lam:g}_n{args.n}.jsonl")
            cache = {}
            if os.path.exists(cache_p):
                for l in open(cache_p):
                    d = json.loads(l)
                    cache[d["node"]] = d
            todo = [n for n in nodes if n["key"][1] not in cache]

            def run(n):
                nd = os.path.join(root, "nodes", n["key"][1])
                node = json.load(open(os.path.join(nd, "node.json")))
                eps = [json.loads(l) for l in open(os.path.join(nd, "batch.jsonl"))]
                text = prompt(variant, objective, node, eps, feats[n["key"]], n["h"], lam)
                d = ask_judge(client, cfg.helper_name, text, args.n, seed=zlib.crc32(n["key"][1].encode()) % 100000)
                d.update(node=n["key"][1], h=n["h"])
                return d

            if todo:
                print(f"[{variant}/{objective}] querying {len(todo)} decisions ...", flush=True)
                with ThreadPoolExecutor(args.workers) as ex, open(cache_p, "a") as fo:
                    for d in ex.map(run, todo):
                        fo.write(json.dumps(d, ensure_ascii=False) + "\n")
                        fo.flush()
                        cache[d["node"]] = d
            for n in nodes:
                choice[variant][n["key"]] = cache[n["key"][1]]["action"]

        # --- RL routers, leave-one-tree-out ---
        for name, keys in (("rl_all", ALL_FEATURES), ("rl_task", TASK_FEATURES)):
            X, A, R, Xn, done, roots = build(trans, feats, keys, lam)
            for t in trees:
                tr = roots != f"{args.exp}:{t}"
                models = fqi(X[tr], A[tr], R[tr], Xn[tr], done[tr], args.gamma)
                for n in nodes:
                    if n["tree"] != t:
                        continue
                    x = np.array([[feats[n["key"]][k] for k in keys]])
                    qs = {a: models[a].predict(x)[0] for a in models}
                    choice[name][n["key"]] = max(qs, key=qs.get)
        for a in ACTIONS:
            for n in nodes:
                choice[f"always_{a}"][n["key"]] = a

        # --- scoring ---
        def path_return(tree, pick) -> float:
            key = (args.exp, tree)
            H = next(n["h"] for n in nodes if n["key"] == key)
            ret = 0.0
            for _ in range(H):
                if key not in kids:
                    break
                a = pick(key)
                r = kids[key][a]
                ret += r["dV"] - lam * cost(r)
                key = (args.exp, r["child"])
            return ret

        def exp_random(key, h) -> float:
            if h <= 0 or key not in kids:
                return 0.0
            return float(np.mean([r["dV"] - lam * cost(r) + exp_random((key[0], r["child"]), h - 1)
                                  for r in kids[key].values()]))

        print(f"\n=== objective: {objective} (lam={lam:g}, scale costs), {len(nodes)} decisions in "
              f"{len(trees)} trees ===")
        print(f"{'policy':14s} {'acc(a*)':>8s} {'regret':>8s} {'path return':>12s}   per-tree path returns")
        opt_paths = [V((args.exp, t), next(n['h'] for n in nodes if n['key'] == (args.exp, t))) for t in trees]
        print(f"{'optimal':14s} {'100%':>8s} {0:8.4f} {np.mean(opt_paths):12.4f}   "
              + " ".join(f"{v:+.3f}" for v in opt_paths))
        regs = {}
        for pol in [*args.variants, "rl_all", "rl_task"] + [f"always_{a}" for a in ACTIONS]:
            c = choice[pol]
            reg = np.array([max(n["q"].values()) - n["q"][c[n["key"]]] for n in nodes])
            acc = np.mean([n["q"][c[n["key"]]] >= max(n["q"].values()) - 1e-12 for n in nodes])
            paths = [path_return(t, lambda k: c[k]) for t in trees]
            regs[pol] = reg
            print(f"{pol:14s} {acc:8.0%} {reg.mean():8.4f} {np.mean(paths):12.4f}   "
                  + " ".join(f"{v:+.3f}" for v in paths))
        rnd_reg = np.mean([max(n["q"].values()) - np.mean(list(n["q"].values())) for n in nodes])
        rnd_paths = [exp_random((args.exp, t), next(n['h'] for n in nodes if n['key'] == (args.exp, t)))
                     for t in trees]
        print(f"{'random (exp.)':14s} {'25%':>8s} {rnd_reg:8.4f} {np.mean(rnd_paths):12.4f}")
        # paired comparison router vs each judge, node level, bootstrap over nodes
        rng = np.random.default_rng(0)
        for v in args.variants:
            d = regs[v] - regs["rl_all"]  # > 0: router better
            boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            print(f"regret({v}) - regret(rl_all) = {d.mean():+.4f}  95% CI [{np.percentile(boots, 2.5):+.4f}, "
                  f"{np.percentile(boots, 97.5):+.4f}]  router better/equal/worse on "
                  f"{(d > 1e-12).sum()}/{(abs(d) <= 1e-12).sum()}/{(d < -1e-12).sum()} nodes")
        for v in args.variants:
            dist = Counter(choice[v].values())
            print(f"{v} action mix: {dict(dist)}")
        print(f"rl_all action mix: {dict(Counter(choice['rl_all'].values()))}")


if __name__ == "__main__":
    main()
