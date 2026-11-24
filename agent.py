"""Batched rollout + validation for the evolving policy.

A rollout over a batch is: retrieve memory (if any) -> attempt 1 -> grade ->
(if failed and max_turns >= 2) environment feedback -> attempt 2 -> grade.
Generation is concurrent against vLLM; grading happens in the main thread
(math-verify needs signal-based timeouts) or a thread pool (API subprocesses).
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from utils.memory_format import build_user_message, context_to_ctx_parts  # cove/src

from .tasks import api_world, math_tasks

SYSTEM_PROMPT = (
    "You are a careful problem-solving assistant.\n"
    "The user message may include retrieved memories and skills from your past experience. "
    "Use them when they are relevant; ignore them otherwise. Content wrapped in <volatile>...</volatile> "
    "describes an external interface that may change over time: follow it for the current task.\n"
    "For programming tasks, reply with one complete Python program in a single ```python code block. "
    "For math problems, reason concisely and put the final answer in \\boxed{}."
)
RETRY_SUFFIX = "\n\nRevise your answer using this feedback. Reply in the same format as before."

_GRADE_POOL = ThreadPoolExecutor(32)


def _seed(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _slim_ctx(ctx: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    keep = ("memory_id", "skill_id", "content", "volatility", "internalizable", "tags", "version")
    return {k: [{f: r.get(f) for f in keep if f in r} for r in (ctx or {}).get(k, [])] for k in ("memories", "skills")}


def user_message(task, ctx: Optional[Dict] = None) -> str:
    return build_user_message(task.text(), context_to_ctx_parts(ctx or {}))


def train_messages(task) -> List[Dict[str, str]]:
    """Prompt used for parametric internalization: no retrieved memory."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message(task, None)}]


def grade_many(tasks, responses: List[str], world) -> List[Dict[str, Any]]:
    out: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    api_idx = [i for i, t in enumerate(tasks) if t.task_type == "api"]
    futs = {i: _GRADE_POOL.submit(api_world.grade, world, tasks[i], responses[i]) for i in api_idx}
    for i, t in enumerate(tasks):
        if t.task_type == "math":
            out[i] = math_tasks.grade(t, responses[i])
    for i, f in futs.items():
        out[i] = f.result()
    return out  # type: ignore[return-value]


def max_tokens_for(cfg, task) -> int:
    return cfg.max_tokens_api if task.task_type == "api" else cfg.max_tokens_math


def rollout(cfg, policy, model: str, tasks, world, bank=None, *, max_turns: int = 2, temperature: float = 0.7,
            n: int = 1, seed_tag: str = "") -> List[Dict[str, Any]]:
    """Returns one episode dict per (task, sample)."""
    items = [(t, k) for t in tasks for k in range(n)]
    ctx_cache: Dict[str, Dict] = {}
    if bank is not None:
        use = [t for t in tasks if t.task_type in cfg.memory_task_types]
        for t, ctx in zip(use, bank.retrieve_many([t.text() for t in use]) if use else []):
            ctx_cache[t.task_id] = _slim_ctx(ctx)
    convs, mts, seeds = [], [], []
    for t, k in items:
        ctx = ctx_cache.get(t.task_id)
        convs.append([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message(t, ctx)}])
        mts.append(max_tokens_for(cfg, t))
        seeds.append(_seed(seed_tag, t.task_id, k, 1))
    r1 = policy.chat_many(model, convs, temperature, mts, seeds)
    g1 = grade_many([t for t, _ in items], r1, world)

    eps: List[Dict[str, Any]] = []
    for (t, k), conv, resp, g in zip(items, convs, r1, g1):
        eps.append({"task_id": t.task_id, "task_type": t.task_type, "task_text": t.text(), "sample": k,
                    "context": ctx_cache.get(t.task_id, {"memories": [], "skills": []}), "messages": conv,
                    "turns": [{"response": resp, "grade": g}], "_task": t})

    if max_turns >= 2:
        retry = [i for i, e in enumerate(eps) if not e["turns"][0]["grade"]["success"]]
        if retry:
            convs2 = []
            for i in retry:
                e = eps[i]
                convs2.append(e["messages"] + [{"role": "assistant", "content": e["turns"][0]["response"]},
                                               {"role": "user", "content": e["turns"][0]["grade"]["message"] + RETRY_SUFFIX}])
            r2 = policy.chat_many(model, convs2, temperature, [mts[i] for i in retry],
                                  [_seed(seed_tag, eps[i]["task_id"], eps[i]["sample"], 2) for i in retry])
            g2 = grade_many([eps[i]["_task"] for i in retry], r2, world)
            for i, resp, g in zip(retry, r2, g2):
                eps[i]["turns"].append({"response": resp, "grade": g})

    for e in eps:
        e["final"] = e["turns"][-1]["grade"]
        e["success_t1"] = bool(e["turns"][0]["grade"]["success"])
        parts = []
        for j, tr in enumerate(e["turns"]):
            parts.append(f"Attempt {j + 1}:\n{tr['response']}\n\nEnvironment feedback: {tr['grade']['message']}")
        e["transcript"] = "\n\n".join(parts)
    return eps


def strip_private(eps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in e.items() if not k.startswith("_")} for e in eps]


def evaluate(cfg, policy, model: str, val_tasks, world, bank=None) -> Dict[str, Any]:
    """Greedy, single-attempt accuracy on the fixed validation set."""
    eps = rollout(cfg, policy, model, val_tasks, world, bank, max_turns=1, temperature=0.0, n=1, seed_tag="val")
    res: Dict[str, Any] = {"per_task": {e["task_id"]: int(e["success_t1"]) for e in eps}}
    for ty in ("api", "math"):
        xs = [e["success_t1"] for e in eps if e["task_type"] == ty]
        res[f"acc_{ty}"] = sum(xs) / max(1, len(xs))
    res["V"] = 0.5 * (res["acc_api"] + res["acc_math"])
    return res
