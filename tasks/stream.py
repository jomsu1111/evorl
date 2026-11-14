"""Exogenous task stream: identical for every branch of a tree / every policy.

step t -> (task type, batch of tasks, API world version). Deterministic in
(seed, t) so that sibling branches see exactly the same experiences.
"""
from __future__ import annotations

import itertools
import random
from typing import List

from .api_world import MODULES, OPS, ApiTask, ApiWorld, make_task
from .math_tasks import MathTask, load_train, load_val


class TaskStream:
    def __init__(self, cfg):
        self.cfg = cfg
        self.math_train: List[MathTask] = load_train(cfg.data_dir, cfg.math_levels)
        self.math_val: List[MathTask] = load_val(cfg.data_dir, cfg.math_levels, cfg.math_val_size, cfg.seed)
        rng = random.Random(f"apival-{cfg.seed}")
        combos = list(itertools.product(sorted(OPS), MODULES))
        rng.shuffle(combos)
        self.api_val: List[ApiTask] = [make_task(rng, f"aval-{i}", op, mod)
                                       for i, (op, mod) in enumerate(combos[: cfg.api_val_size])]

    def version(self, step: int) -> int:
        return sum(1 for d in self.cfg.drift_steps if step >= d)

    def world(self, step: int) -> ApiWorld:
        return ApiWorld(self.cfg.seed, self.version(step))

    def batch_type(self, step: int) -> str:
        if self.cfg.type_schedule == "alternate":
            return "api" if step % 2 == 0 else "math"
        return random.Random(f"type-{self.cfg.seed}-{step}").choice(["api", "math"])

    def batch(self, step: int):
        k = self.cfg.batch_size
        rng = random.Random(f"batch-{self.cfg.seed}-{step}")
        if self.batch_type(step) == "math":
            return rng.sample(self.math_train, k)
        return [make_task(rng, f"api-s{step}-{i}") for i in range(k)]

    def val_tasks(self):
        return list(self.api_val) + list(self.math_val)
