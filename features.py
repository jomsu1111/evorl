"""State featurization s_t = (task/feedback of the current batch, agent learning state).

TASK_FEATURES  ~ what COVE's router sees (task type + execution feedback).
AGENT_FEATURES ~ the learning state the RQs are about (memory, parameters,
                 data, evolution history). The `no-agent-state` ablation
                 trains the router on TASK_FEATURES only.
"""
from __future__ import annotations

from typing import Any, Dict, List

ERRORS = ["api_error", "wrong_output", "no_code", "api_not_used", "wrong_answer", "no_answer"]
LAST_ACTIONS = ["H", "P", "B", "N"]

TASK_FEATURES = (["type_api", "t1_success", "t2_recovery", "final_success", "resp_len_k"]
                 + [f"err_{e}" for e in ERRORS])
AGENT_FEATURES = [
    "V", "V_api", "V_math", "mem_gain", "mem_gain_api", "mem_gain_math",
    "ctx_rate", "ctx_success_gap",
    "mem_n_total", "mem_frac_volatile", "mem_mean_usage", "mem_outcome_success", "mem_mean_version",
    "n_P", "n_H", "train_samples_k", "steps_since_P", "steps_since_H",
    "last_dV", "delta_t1_same_type",
] + [f"last1_{a}" for a in LAST_ACTIONS] + [f"last2_{a}" for a in LAST_ACTIONS]
ALL_FEATURES = TASK_FEATURES + AGENT_FEATURES


def batch_features(eps: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(1, len(eps))
    t1 = [e["success_t1"] for e in eps]
    fail1 = [e for e in eps if not e["success_t1"]]
    rec = [e["final"]["success"] for e in fail1]
    with_ctx = [e for e in eps if (e["context"].get("memories") or e["context"].get("skills"))]
    no_ctx = [e for e in eps if e not in with_ctx]
    f = {
        "type_api": float(eps[0]["task_type"] == "api") if eps else 0.0,
        "t1_success": sum(t1) / n,
        "t2_recovery": sum(rec) / max(1, len(rec)),
        "final_success": sum(e["final"]["success"] for e in eps) / n,
        "resp_len_k": sum(len(e["turns"][0]["response"]) for e in eps) / n / 4000.0,
        "ctx_rate": len(with_ctx) / n,
        "ctx_success_gap": ((sum(e["success_t1"] for e in with_ctx) / len(with_ctx)) -
                            (sum(e["success_t1"] for e in no_ctx) / len(no_ctx))) if with_ctx and no_ctx else 0.0,
    }
    for err in ERRORS:
        f[f"err_{err}"] = sum(1 for e in eps if e["turns"][0]["grade"].get("error") == err) / n
    return f


def agent_features(node, mem_stats: Dict[str, float]) -> Dict[str, float]:
    m = node.metrics
    hist = node.history
    f = {
        "V": m["V"], "V_api": m["acc_api"], "V_math": m["acc_math"],
        "mem_gain": m["V"] - m.get("V_nomem", m["V"]),
        "mem_gain_api": m["acc_api"] - m.get("acc_api_nomem", m["acc_api"]),
        "mem_gain_math": m["acc_math"] - m.get("acc_math_nomem", m["acc_math"]),
        "mem_n_total": mem_stats.get("n_total", 0) / 100.0,
        "mem_frac_volatile": mem_stats.get("frac_volatile", 0.0),
        "mem_mean_usage": mem_stats.get("mean_usage", 0.0) / 10.0,
        "mem_outcome_success": mem_stats.get("mean_outcome_success", 0.0),
        "mem_mean_version": mem_stats.get("mean_version", 0.0) / 5.0,
        "n_P": sum(1 for h in hist if h["action"] in ("P", "B")) / 5.0,
        "n_H": sum(1 for h in hist if h["action"] in ("H", "B")) / 5.0,
        "train_samples_k": node.counters.get("train_samples", 0) / 1000.0,
        "last_dV": hist[-1]["dV"] if hist else 0.0,
    }
    for key, acts in (("steps_since_P", ("P", "B")), ("steps_since_H", ("H", "B"))):
        idx = [i for i, h in enumerate(hist) if h["action"] in acts]
        f[key] = (len(hist) - 1 - idx[-1]) / 5.0 if idx else 1.0
    for j, tag in ((1, "last1"), (2, "last2")):
        a = hist[-j]["action"] if len(hist) >= j else None
        for x in LAST_ACTIONS:
            f[f"{tag}_{x}"] = float(a == x)
    return f


def node_features(node, eps: List[Dict[str, Any]], mem_stats: Dict[str, float]) -> Dict[str, float]:
    f = batch_features(eps)
    f.update(agent_features(node, mem_stats))
    ty = eps[0]["task_type"] if eps else ""
    prev = [h for h in node.history if h.get("batch_type") == ty]
    f["delta_t1_same_type"] = f["t1_success"] - prev[-1]["batch_t1"] if prev else 0.0
    return {k: float(f.get(k, 0.0)) for k in ALL_FEATURES}
