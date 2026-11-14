"""API-usage task family with *volatile* library interfaces.

Built on COVE's rule_templates operation bank (cove/src/data/rule_templates.py):
each task = one Operation (what to compute) x one fake library module (how it is
called). The calling convention of every module is hidden from the task prompt;
the agent only learns it from (a) the documentation returned with a failed
attempt, (b) its memory, or (c) its weights.

Volatility: at configured stream steps the world version bumps, which re-assigns
each module's calling convention and renames every verb. Knowledge stored in
memory can be overwritten; knowledge baked into weights goes stale.
"""
from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from data.rule_templates import OPERATIONS, compute_result, sample_input, to_expected, to_stdin  # cove/src

MODULES = ["dkit", "arrkit", "seqlib", "qtools", "nformer", "vlib", "listpro", "numkit", "datatool", "flowlib"]
FAMILIES = ["free", "chain", "config"]

# Verb names per world version (index 0 = COVE's original verb).
VERB_ALIASES: Dict[str, List[str]] = {
    "dedup_sort": ["unique_sorted", "distinct_asc", "uniq_order"],
    "running_sum": ["prefix_sums", "cumulate", "scan_add"],
    "running_max": ["prefix_maxes", "peak_track", "scan_max"],
    "top3_desc": ["three_largest", "podium", "top_three"],
    "evens_only": ["keep_even", "even_filter", "only_evens"],
    "reverse_list": ["flip_order", "backwards", "mirror"],
    "distinct_count": ["count_distinct", "n_unique", "variety"],
    "word_count": ["tally_words", "freq_table", "word_hist"],
}
OPS = {op.op_id: op for op in OPERATIONS}


def _indent(src: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + l if l.strip() else l for l in src.splitlines())


class ApiWorld:
    """Module -> calling convention, verb names; deterministic per (seed, version)."""

    def __init__(self, seed: int, version: int = 0):
        self.seed, self.version = seed, version
        rng = random.Random(f"apiworld-{seed}-{version}")
        fams = [FAMILIES[i % 3] for i in range(len(MODULES))]
        rng.shuffle(fams)
        self.family = dict(zip(MODULES, fams))
        self.verb = {op_id: names[version % len(names)] for op_id, names in VERB_ALIASES.items()}

    # -- API surface ---------------------------------------------------------
    def call_expr(self, module: str, op_id: str, arg: str = "xs") -> str:
        fam, verb = self.family[module], self.verb[op_id]
        if fam == "free":
            return f"{module}.{verb}({arg})"
        if fam == "chain":
            return f"{module}.wrap({arg}).{verb}()"
        return f"{module}.apply({arg}, mode='{verb}')"

    def required_tokens(self, module: str, op_id: str) -> List[str]:
        fam, verb = self.family[module], self.verb[op_id]
        if fam == "free":
            return [f"{module}.{verb}"]
        if fam == "chain":
            return [f"{module}.wrap", verb]
        return [f"{module}.apply", verb]

    def doc(self, module: str) -> str:
        fam = self.family[module]
        conv = {
            "free": f"call `{module}.<verb>(seq)`; it returns the result list",
            "chain": f"wrap the sequence with `{module}.wrap(seq)` then call `.<verb>()` on it; it returns the result list",
            "config": f"call `{module}.apply(seq, mode='<verb>')`; it returns the result list",
        }[fam]
        verbs = "; ".join(f"{self.verb[o.op_id]} = {o.semantics}" for o in OPERATIONS)
        return f"Library `{module}`: {conv}. Available verbs: {verbs}."

    def mock_src(self, module: str) -> str:
        fam = self.family[module]
        parts = [f'"""{self.doc(module)}"""\n']
        if fam == "free":
            for op in OPERATIONS:
                parts.append(f"def {self.verb[op.op_id]}(seq):\n    seq = list(seq)\n{_indent(op.body, 4)}\n    return result\n")
        elif fam == "chain":
            parts.append("class _Box:\n    def __init__(self, data):\n        self._data = list(data)\n")
            for op in OPERATIONS:
                parts.append(f"    def {self.verb[op.op_id]}(self):\n        seq = list(self._data)\n{_indent(op.body, 8)}\n        return result\n")
            parts.append("\ndef wrap(data):\n    return _Box(data)\n")
        else:
            lines = ["def apply(seq, mode=None):", "    seq = list(seq)"]
            for op in OPERATIONS:
                lines.append(f"    if mode == {self.verb[op.op_id]!r}:")
                lines.append(_indent(op.body, 8))
                lines.append("        return result")
            lines.append("    raise ValueError('unsupported mode: %r' % (mode,))")
            parts.append("\n".join(lines) + "\n")
        return "\n".join(parts)


@dataclass
class ApiTask:
    task_id: str
    op_id: str
    module: str
    inputs: List[List[Any]]

    @property
    def task_type(self) -> str:
        return "api"

    def text(self) -> str:
        op = OPS[self.op_id]
        return (
            f"{op.narrative}\n{op.input_format}\n{op.output_format}\n\n"
            f"Requirement: the core computation MUST be done by calling the `{self.module}` library "
            f"(already installed; `import {self.module}`). Write a complete Python program that reads "
            f"from stdin and prints the answer. Reply with the program in a single ```python code block."
        )


def make_task(rng: random.Random, task_id: str, op_id: str | None = None, module: str | None = None, n_tests: int = 3) -> ApiTask:
    op_id = op_id or rng.choice(list(OPS))
    module = module or rng.choice(MODULES)
    op = OPS[op_id]
    return ApiTask(task_id, op_id, module, [sample_input(op, rng) for _ in range(n_tests)])


_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    blocks = _CODE_RE.findall(text or "")
    if blocks:
        return blocks[-1]
    if "import " in (text or "") and "print" in (text or ""):
        return text
    return ""


def grade(world: ApiWorld, task: ApiTask, response: str, timeout: float = 3.0) -> Dict[str, Any]:
    """Run the program against the module mock. Returns score/success/message."""
    op = OPS[task.op_id]
    code = extract_code(response)
    doc = world.doc(task.module)
    if not code:
        return {"success": False, "score": 0.0, "error": "no_code",
                "message": f"No Python code block found.\n\nLibrary documentation:\n"
                           f"Correct call for this task: `{world.call_expr(task.module, task.op_id)}`\n{doc}"}
    used = all(tok in code for tok in world.required_tokens(task.module, task.op_id))
    passed, first_err = 0, ""
    with tempfile.TemporaryDirectory(prefix="apiw_") as td:
        with open(os.path.join(td, f"{task.module}.py"), "w") as f:
            f.write(world.mock_src(task.module))
        prog = os.path.join(td, "main.py")
        with open(prog, "w") as f:
            f.write(code)
        env = {"PYTHONPATH": td, "PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        for inp in task.inputs:
            exp = to_expected(compute_result(op, inp))
            try:
                r = subprocess.run([sys.executable, prog],
                                   input=to_stdin(op, inp), capture_output=True, text=True, timeout=timeout, cwd=td, env=env)
                out = " ".join(r.stdout.split())
                if r.returncode == 0 and out == exp:
                    passed += 1
                elif not first_err:
                    tail = "\n".join((r.stderr or "").strip().splitlines()[-3:])
                    first_err = tail if r.returncode != 0 else f"Wrong output. input={inp} expected `{exp}` got `{out[:200]}`"
            except subprocess.TimeoutExpired:
                first_err = first_err or "Timeout."
    total = len(task.inputs)
    success = used and passed == total
    if success:
        return {"success": True, "score": 1.0, "error": "", "message": "All tests passed."}
    doc = f"Correct call for this task: `{world.call_expr(task.module, task.op_id)}`\n{doc}"
    if passed == total and not used:
        err, msg = "api_not_used", f"Tests pass, but the `{task.module}` library was not used as required."
    else:
        low = first_err.lower()
        err = "api_error" if ("attributeerror" in low or "unsupported mode" in low or "modulenotfound" in low
                              or "typeerror" in low or "has no attribute" in low) else "wrong_output"
        msg = f"Failed {total - passed}/{total} tests. First error:\n{first_err}"
    return {"success": False, "score": (passed / total) * (1.0 if used else 0.5), "error": err,
            "message": f"{msg}\n\nLibrary documentation:\n{doc}"}
