"""Measured-compute (FLOPs) view of every logged run; no GPU needed.

  python -m evorl.flops_report --exp main eval_s1 eval_s2 [--results-dir ...]

(1) per-action measured PFLOPs (helper memory curation / RFT sampling / LoRA training),
    i.e. the real C(H):C(P) ratio of this setup vs the assumed C(H)=1, C(P)=10;
(2) per online curve: mean V, final V, assumed cost and measured PFLOPs, plus J under both
    cost models. COVE's per-episode judge calls are not part of node costs (they happen in
    the policy, before the action), so they are estimated separately and added.

FLOPs use the same formula as qlearn.cost(): 2N per inference token, 6N per training token,
N = 4e9 (helper) / 0.6e9 (policy).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from .qlearn import SCALE

HELPER_N, POLICY_N = 4e9, 0.6e9
# COVE judge: EVALUATOR_PROMPT (~420 tok) + task text + feedback (<=800 chars) + ~45 completion
# tokens, one helper call per episode (32 per step). Tokens per call: (low, mid, high).
JUDGE_TOKENS = (700, 900, 1200)


def flops_parts(c: dict) -> dict:
    return {"helper": 2 * HELPER_N * c.get("helper_tokens", 0) / 1e15,
            "sample": 2 * POLICY_N * c.get("sample_tokens", 0) / 1e15,
            "train": 6 * POLICY_N * c.get("train_tokens", 0) / 1e15}


def judge_pf(tokens: int, calls: int) -> float:
    return 2 * HELPER_N * tokens * calls / 1e15


def per_action(results_dir: str, exps):
    print("(1) measured PFLOPs per action (mean over logged transitions)")
    for exp in exps:
        p = os.path.join(results_dir, exp, "transitions.jsonl")
        if not os.path.exists(p):
            continue
        rows = defaultdict(list)
        for l in open(p):
            r = json.loads(l)
            f = flops_parts(r.get("cost", {}))
            rows[r["action"]].append([f["helper"], f["sample"], f["train"], sum(f.values())])
        print(f"  [{exp}]")
        for a in ("H", "P", "B", "Bc", "N"):
            if a not in rows:
                continue
            v = np.array(rows[a])
            print(f"    {a:2s} n={len(v):4d}  helper {v[:, 0].mean():.3f}  sample {v[:, 1].mean():.3f}  "
                  f"train {v[:, 2].mean():.3f}  total {v[:, 3].mean():.3f} PF  (sd {v[:, 3].std():.3f})")
        if "H" in rows and "P" in rows:
            h, pp = np.mean([x[3] for x in rows["H"]]), np.mean([x[3] for x in rows["P"]])
            print(f"    measured C(P)/C(H) = {pp / h:.2f}   (assumed scale model: {SCALE['P'] / SCALE['H']:.0f})")


def curve_stats(path: str, batch_size: int):
    recs = [json.loads(l) for l in open(path)]
    steps = recs[1:]
    vs = [r["V"] for r in steps]
    pf = sum(sum(flops_parts(r.get("cost", {})).values()) for r in steps)
    scale = sum(SCALE.get(r["action"][0], 0.0) for r in steps)  # "Bc" (COVE pool B) costs like B
    judge = None
    if os.path.basename(path).startswith("cove_"):
        judge = [judge_pf(t, batch_size * len(steps)) for t in JUDGE_TOKENS]
    return {"acts": "".join(r["action"][0] for r in steps), "V0": recs[0]["V"], "Vf": vs[-1],
            "Vmean": float(np.mean(vs)), "scale": scale, "pf": pf, "judge": judge, "n": len(steps)}


def per_curve(results_dir: str, exps, lam_scale: float, lam_pf: float, batch_size: int):
    print(f"\n(2) online curves  J = V_final - V_0 - lam*cost   (lam_scale={lam_scale}, lam_pf={lam_pf}/PF)")
    agg = defaultdict(list)
    for exp in exps:
        files = sorted(glob.glob(os.path.join(results_dir, exp, "curves", "*_R0.jsonl")))
        files = [f for f in files if not os.path.basename(f).startswith("seq_")]
        if not files:
            continue
        print(f"  [{exp}]")
        for f in files:
            s = curve_stats(f, batch_size)
            name = os.path.basename(f)[:-len("_R0.jsonl")]
            pf_tot = s["pf"] + (s["judge"][1] if s["judge"] else 0.0)
            j_scale = s["Vf"] - s["V0"] - lam_scale * s["scale"]
            j_pf = s["Vf"] - s["V0"] - lam_pf * pf_tot
            extra = (f"  (+judge {s['judge'][1]:.2f} [{s['judge'][0]:.2f}-{s['judge'][2]:.2f}])"
                     if s["judge"] else "")
            print(f"    {name:20s} {s['acts']:13s} meanV {s['Vmean']:.3f}  Vf {s['Vf']:.3f}  "
                  f"scale {s['scale']:5.0f}  PF {s['pf']:6.2f}{extra}  J_scale {j_scale:+.3f}  J_pf {j_pf:+.3f}")
            if s["n"] == 12 and exp.startswith("eval_s") and "_d" not in exp:
                agg[name].append((exp, s["Vmean"], s["Vf"], s["scale"], pf_tot, j_scale, j_pf))
    print("\n(3) mean over seeds (12-step runs in eval_s1..s5)")
    print(f"    {'policy':20s} seeds  meanV   Vf    scale   PF    J_scale  J_pf")
    for name, v in sorted(agg.items(), key=lambda kv: -np.mean([x[1] for x in kv[1]])):
        a = np.array([x[1:] for x in v])
        print(f"    {name:20s} {len(v):5d}  {a[:, 0].mean():.3f}  {a[:, 1].mean():.3f}  {a[:, 2].mean():5.0f}  "
              f"{a[:, 3].mean():6.2f}  {a[:, 4].mean():+.3f}  {a[:, 5].mean():+.3f}   "
              f"[{','.join(x[0][-2:] for x in v)}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--results-dir", default="/root/cove-rl/results")
    ap.add_argument("--lam-scale", type=float, default=0.002)
    ap.add_argument("--lam-pf", type=float, default=0.02,
                    help="per-PFLOP penalty; 0.02 makes one B step (~1.3 PF) cost about what P costs "
                         "under the scale model at lam=0.002")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    per_action(args.results_dir, args.exp)
    per_curve(args.results_dir, args.exp, args.lam_scale, args.lam_pf, args.batch_size)


if __name__ == "__main__":
    main()
