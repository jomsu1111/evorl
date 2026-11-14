"""Reasoning task family: MATH (Hendrycks) word problems.

Stream tasks come from the MATH *train* split, validation from MATH-500 (a
subset of the *test* split), both filtered by level. Grading uses math-verify
and must run in the main thread (it relies on signal-based timeouts).
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MathTask:
    task_id: str
    problem: str
    answer: str
    level: int
    subject: str

    @property
    def task_type(self) -> str:
        return "math"

    def text(self) -> str:
        return (f"{self.problem}\n\nSolve the problem. Reason step by step concisely, "
                f"then give the final answer in \\boxed{{}}.")


def last_boxed(text: str) -> str:
    text = text or ""
    i = text.rfind("\\boxed")
    if i < 0:
        return ""
    j = text.find("{", i)
    if j < 0:
        return ""
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return ""


def _level(v: Any) -> int:
    try:
        return int(str(v).replace("Level", "").strip())
    except ValueError:
        return -1


def load_train(data_dir: str, levels: List[int]) -> List[MathTask]:
    import pandas as pd
    files = sorted(glob.glob(os.path.join(data_dir, "hendrycks_math", "**", "train-*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no MATH train parquet under {data_dir}/hendrycks_math")
    out: List[MathTask] = []
    for fp in files:
        df = pd.read_parquet(fp)
        for i, row in df.iterrows():
            lv = _level(row.get("level"))
            ans = last_boxed(str(row.get("solution", "")))
            if lv in levels and ans:
                subj = str(row.get("type", os.path.basename(os.path.dirname(fp))))
                out.append(MathTask(f"mtr-{os.path.basename(os.path.dirname(fp))}-{i}", str(row["problem"]), ans, lv, subj))
    return out


def load_val(data_dir: str, levels: List[int], n: int, seed: int) -> List[MathTask]:
    fp = os.path.join(data_dir, "MATH-500", "test.jsonl")
    rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
    tasks = [MathTask(f"mval-{r['unique_id']}", r["problem"], str(r["answer"]), int(r["level"]), r["subject"])
             for r in rows if int(r["level"]) in levels]
    random.Random(f"mathval-{seed}").shuffle(tasks)
    return tasks[:n]


def grade(task: MathTask, response: str) -> Dict[str, Any]:
    from math_verify import parse, verify
    pred_str = last_boxed(response)
    ok = False
    if pred_str:
        try:
            gold = parse(f"${task.answer}$")
            pred = parse(f"${pred_str}$")
            ok = bool(verify(gold, pred))
        except Exception:
            ok = pred_str.replace(" ", "") == task.answer.replace(" ", "")
    if ok:
        return {"success": True, "score": 1.0, "error": "", "message": "Correct.", "pred": pred_str}
    err = "no_answer" if not pred_str else "wrong_answer"
    msg = "No \\boxed{} final answer found." if not pred_str else f"Your final answer {pred_str} is incorrect."
    # `reference` is only shown to the memory curator (helper), never to the policy.
    return {"success": False, "score": 0.0, "error": err, "message": msg, "pred": pred_str,
            "reference": f"Reference answer: {task.answer}"}
