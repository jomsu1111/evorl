"""COVE's *original* task-level router applied to the collected trees.

  python -m evorl.cove_judge_eval --exp main [--results-dir DIR] [--workers 64]

For every interior decision node (all four actions expanded), each of the batch's
episodes is routed exactly as cove/src/evaluator.py does it: COVE's EVALUATOR_PROMPT
with the task text and the final feedback (status/score/message), temperature 0.1,
COVE's mode parser and its confidence fallback on parse failure, then COVE's success
gate (gradient/hybrid on a succeeded task -> prompt_only).

In COVE memory is always written, and gradient_candidate/hybrid only adds the task to
the parameter-training pool. So the batch-level question COVE answers is "should this
experience also go to P?". We report
  (1) task-level route mix per task family (raw LLM and after the gate),
  (2) the share of each batch sent to the P pool,
  (3) agreement with the tree optimum a* on "does the best action include P" (a* in {P, B}),
      with the batch counted as P-including when the majority of its tasks go to the pool.
Answers are cached in <results>/<exp>/judge/cove_original.jsonl (one line per episode).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from openai import OpenAI

from .config import ACTIONS, EvoConfig
from .qlearn import COST, cost, load, tree_values

MODES = ["prompt_only", "gradient_candidate", "hybrid"]


def cove_prompt(ep: Dict[str, Any]) -> str:
    from prompts.evaluator_prompt import EVALUATOR_PROMPT  # cove/src
    fb = ep["final"]
    status = "SUCCESS" if fb["success"] else "FAILED"
    # same layout as cove/src/evaluator.py::_build_prompt
    return (f"{EVALUATOR_PROMPT}\n\nTask:\n{ep['task_text']}\n\nFeedback:\n"
            f"status={status}; score={float(fb['score']):.2f}; message={fb['message']}\n")


def parse_mode(raw: str) -> str:
    """cove/src/evaluator.py::_parse_routing_mode."""
    text = str(raw or "").lower()
    cands = [(text.rfind(f"<mode>{m}</mode>"), m) for m in MODES]
    cands = [c for c in cands if c[0] >= 0]
    if cands:
        return max(cands)[1]
    cands = sorted([(text.rfind(m), m) for m in MODES if text.rfind(m) >= 0], reverse=True)
    if not cands:
        return "NULL"
    best = cands[0][0]
    if len([1 for p, _ in cands if best - p <= len("prompt_only|gradient_candidate|hybrid")]) == len(MODES):
        return "NULL"
    return cands[0][1]


def fallback_mode(ep: Dict[str, Any]) -> str:
    """cove/src/evaluator.py::_confidence_fallback."""
    fb = ep["final"]
    if fb["success"] and float(fb["score"]) >= 0.5:
        return "prompt_only"
    return "gradient_candidate" if float(fb["score"]) < 0.3 else "hybrid"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--results-dir", default="/root/cove-rl/results")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.002)
    args = ap.parse_args()
    COST["model"] = "scale"
    cfg = EvoConfig.load()
    root = os.path.join(args.results_dir, args.exp)
    trans, feats = load(args.results_dir, [args.exp])
    client = OpenAI(api_key="EMPTY", base_url=cfg.helper_url, timeout=600, max_retries=3)

    kids, _ = tree_values(trans, 0.0, 1.0)
    keys = [k for k, ch in kids.items()
            if set(ch) == set(ACTIONS) and k in feats and ch["H"]["tree"] != "behavior"]
    node_ids = sorted({k[1] for k in keys})

    cache_p = os.path.join(root, "judge", "cove_original.jsonl")
    os.makedirs(os.path.dirname(cache_p), exist_ok=True)
    cache: Dict[tuple, Dict[str, Any]] = {}
    if os.path.exists(cache_p):
        for l in open(cache_p):
            d = json.loads(l)
            cache[(d["node"], d["i"])] = d

    todo = []
    eps_of: Dict[str, list] = {}
    for nid in node_ids:
        eps = [json.loads(l) for l in open(os.path.join(root, "nodes", nid, "batch.jsonl"))]
        eps_of[nid] = eps
        todo += [(nid, i) for i in range(len(eps)) if (nid, i) not in cache]

    def run(item):
        nid, i = item
        ep = eps_of[nid][i]
        try:
            r = client.chat.completions.create(model=cfg.helper_name, temperature=0.1, max_tokens=300,
                                               messages=[{"role": "user", "content": cove_prompt(ep)}])
            raw = r.choices[0].message.content or ""
        except Exception as e:  # COVE falls back on a failed call too
            raw = f"ERROR {e}"
        llm = parse_mode(raw)
        parsed = llm != "NULL"
        mode = llm if parsed else fallback_mode(ep)
        succ = bool(ep["final"]["success"]) and float(ep["final"]["score"]) >= 0.5
        gated = "prompt_only" if (parsed and mode != "prompt_only" and succ) else mode
        return {"node": nid, "i": i, "type": ep["task_type"], "llm": llm, "mode": gated,
                "parsed": parsed, "final_success": succ, "raw": raw[-400:]}

    if todo:
        print(f"[cove_original] routing {len(todo)} episodes over {len(node_ids)} nodes ...", flush=True)
        with ThreadPoolExecutor(args.workers) as ex, open(cache_p, "a") as fo:
            for n_done, d in enumerate(ex.map(run, todo), 1):
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
                cache[(d["node"], d["i"])] = d
                if n_done % 500 == 0:
                    print(f"  {n_done}/{len(todo)}", flush=True)

    rows = [cache[(nid, i)] for nid in node_ids for i in range(len(eps_of[nid]))]

    print(f"\n=== COVE original router, {len(rows)} episodes in {len(node_ids)} decision nodes ===")
    print(f"parse failures (confidence fallback used): {sum(not r['parsed'] for r in rows)}")
    for ty in ("api", "math"):
        sub = [r for r in rows if r["type"] == ty]
        n = len(sub)
        raw_mix = Counter(r["llm"] for r in sub)
        gated_mix = Counter(r["mode"] for r in sub)
        succ = sum(r["final_success"] for r in sub)
        print(f"\n[{ty}] episodes={n}  final success={succ / n:.0%}")
        print("  LLM route (before gate): " + ", ".join(f"{m} {raw_mix[m] / n:.0%}" for m in MODES + ["NULL"] if raw_mix[m]))
        print("  route after COVE gate  : " + ", ".join(f"{m} {gated_mix[m] / n:.0%}" for m in MODES if gated_mix[m]))
        fails = [r for r in sub if not r["final_success"]]
        if fails:
            fm = Counter(r["llm"] for r in fails)
            print(f"  on failed episodes (n={len(fails)}): " + ", ".join(f"{m} {fm[m] / len(fails):.0%}" for m in MODES + ["NULL"] if fm[m]))

    share = {nid: sum(cache[(nid, i)]["mode"] != "prompt_only" for i in range(len(eps_of[nid]))) / len(eps_of[nid])
             for nid in node_ids}
    for objective, lam in (("perf", 0.0), ("cost", args.lam)):
        kids, V = tree_values(trans, lam, 1.0)
        agree = defaultdict(list)
        table = defaultdict(Counter)
        for k in keys:
            ch = kids[k]
            h = ch["H"]["horizon"] - ch["H"]["tree_depth"]
            q = {a: r["dV"] - lam * cost(r) + V((k[0], r["child"]), h - 1) for a, r in ch.items()}
            a_star = max(q, key=q.get)
            ty = "api" if feats[k]["type_api"] > 0.5 else "math"
            star_p = a_star in ("P", "B")
            cove_p = share[k[1]] > 0.5
            agree[ty].append(star_p == cove_p)
            table[ty][(f"a*{'+P' if star_p else '-P'}", f"cove{'+P' if cove_p else '-P'}")] += 1
        print(f"\n--- objective {objective} (lam={lam:g}): does the decision include P? ---")
        for ty in ("api", "math"):
            sub = [k for k in keys if (feats[k]["type_api"] > 0.5) == (ty == "api")]
            mean_share = sum(share[k[1]] for k in sub) / len(sub)
            print(f"  [{ty}] n={len(sub)}  mean P-pool share={mean_share:.0%}  "
                  f"a* includes P: {sum(n for c, n in table[ty].items() if c[0] == 'a*+P') / len(sub):.0%}  "
                  f"COVE majority includes P: {sum(n for c, n in table[ty].items() if c[1] == 'cove+P') / len(sub):.0%}  "
                  f"agreement={sum(agree[ty]) / len(agree[ty]):.0%}  {dict(table[ty])}")


if __name__ == "__main__":
    main()
