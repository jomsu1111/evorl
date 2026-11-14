"""Evolution-action policies: baselines, COVE-style router, learned Q router."""
from __future__ import annotations

import pickle
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Policy:
    name = "base"

    def act(self, env, node, feats: Dict[str, float]) -> Tuple[str, Optional[list]]:
        """Return (action, p_tasks or None)."""
        raise NotImplementedError


class AlwaysPolicy(Policy):
    def __init__(self, a: str):
        self.a, self.name = a, f"always_{a}"

    def act(self, env, node, feats):
        return self.a, None


class RandomPolicy(Policy):
    def __init__(self, seed: int = 0, probs=(1 / 3, 1 / 3, 1 / 3)):
        self.rng, self.probs, self.name = random.Random(seed), probs, f"random{seed}"

    def act(self, env, node, feats):
        return self.rng.choices(["H", "P", "B"], weights=self.probs)[0], None


class CovePolicy(Policy):
    """COVE re-implemented in the step-wise env, with COVE's own prompt/thresholds.

    * Task-aware routing: helper LLM judge (cove/src/prompts/evaluator_prompt.py)
      labels each episode prompt_only / gradient_candidate / hybrid, with COVE's
      success gate (prompt channel already succeeded -> prompt_only).
    * Memory is always written (as in cove/src/agent.py): action is at least H.
    * Stage-aware scheduling: train (B) when the gradient pool reaches the sample
      threshold, on a success-rate plateau, or on cold-start failure; P trains
      on the accumulated pool, which is then cleared.
    """
    name = "cove"

    def __init__(self, sample_threshold: int = 32, plateau_window: int = 2, plateau_eps: float = 0.02,
                 cold_start_max: float = 0.1):
        self.pool: List[Any] = []
        self.succ_hist: List[float] = []
        self.thr, self.win, self.eps, self.cold = sample_threshold, plateau_window, plateau_eps, cold_start_max
        self.log: List[Dict[str, Any]] = []

    def _judge(self, env, ep) -> str:
        from prompts.evaluator_prompt import EVALUATOR_PROMPT  # cove/src
        fb = ep["final"]
        prompt = (f"{EVALUATOR_PROMPT}\n\nTask:\n{ep['task_text']}\n\nFeedback status: "
                  f"{'SUCCESS' if fb['success'] else 'FAILED'}; score: {fb['score']:.2f}; message: {fb['message'][:800]}")
        raw = env.helper.single_message_response(prompt, temperature=0.1, max_tokens=200).get("data", "")
        m = re.search(r"<mode>\s*(prompt_only|gradient_candidate|hybrid)\s*</mode>", str(raw))
        mode = m.group(1) if m else "prompt_only"
        if mode != "prompt_only" and ep["success_t1"]:
            mode = "prompt_only"  # COVE confidence gate
        return mode

    def act(self, env, node, feats):
        eps = env.experience(node)
        tasks = {t.task_id: t for t in env.stream.batch(node.step)}
        modes = list(env.helper.pool.map(lambda e: self._judge(env, e), eps))
        for e, m in zip(eps, modes):
            if m in ("gradient_candidate", "hybrid"):
                self.pool.append(tasks[e["task_id"]])
        self.succ_hist.append(feats["final_success"])
        reasons = []
        if len(self.pool) >= self.thr:
            reasons.append("sample_threshold")
        if len(self.succ_hist) >= 2 * self.win:
            recent = np.mean(self.succ_hist[-self.win:])
            prev = np.mean(self.succ_hist[-2 * self.win:-self.win])
            if recent - prev < self.eps and self.pool:
                reasons.append("plateau")
        if feats["t1_success"] <= self.cold and self.pool and node.step < 2:
            reasons.append("cold_start")
        self.log.append({"step": node.step, "modes": {m: modes.count(m) for m in set(modes)},
                         "pool": len(self.pool), "reasons": reasons})
        if reasons:
            pool, self.pool = self.pool, []
            return "B", pool
        return "H", None


class QPolicy(Policy):
    """Greedy w.r.t. a fitted Q-function (see qlearn.py)."""

    def __init__(self, path: str, name: str | None = None):
        with open(path, "rb") as f:
            self.q = pickle.load(f)
        self.name = name or f"q_{self.q['tag']}"

    def q_values(self, feats: Dict[str, float]) -> Dict[str, float]:
        x = np.array([[feats.get(k, 0.0) for k in self.q["features"]]])
        return {a: float(m.predict(x)[0]) for a, m in self.q["models"].items()}

    def act(self, env, node, feats):
        qs = self.q_values(feats)
        return max(qs, key=qs.get), None


def make_policy(spec: str, seed: int = 0) -> Policy:
    if spec in ("H", "P", "B"):
        return AlwaysPolicy(spec)
    if spec == "random":
        return RandomPolicy(seed)
    if spec == "cove":
        return CovePolicy()
    if spec.startswith("q:"):
        return QPolicy(spec[2:])
    raise ValueError(spec)
