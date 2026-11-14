"""Experiment configuration for the RL-routed self-evolution testbed.

Everything that changes between runs lives here so a run directory can store
its exact config (config.json) and be reproduced.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List

ACTIONS = ["H", "P", "B", "N"]  # Harness(memory) / Parameter(RFT) / Both / No update


@dataclass
class EvoConfig:
    # ---- paths (server defaults) ----
    cove_src: str = "/root/cove-rl/cove/src"
    policy_model: str = "/root/cove-rl/models/Qwen3-0.6B"
    helper_model: str = "/opt/cove-models/Qwen3-4B-Instruct-2507"
    embed_model: str = "/root/cove-rl/models/all-MiniLM-L6-v2"
    data_dir: str = "/root/cove-rl/data"
    work_dir: str = "/opt/cove-work"          # adapters + node scratch (ephemeral, large)
    results_dir: str = "/root/cove-rl/results"  # transitions, curves (persistent, small)

    # ---- servers ----
    policy_url: str = "http://127.0.0.1:8001/v1"
    policy_name: str = "policy"               # served-model-name of the base policy
    helper_url: str = "http://127.0.0.1:8002/v1"
    helper_name: str = "helper"
    train_gpu: str = "1"
    max_concurrency: int = 64

    # ---- task stream ----
    seed: int = 0
    batch_size: int = 32                      # tasks per evolution step
    type_schedule: str = "alternate"          # alternate | random
    drift_steps: List[int] = field(default_factory=lambda: [6])  # API world version bumps
    api_val_size: int = 80                    # all (op, module) combos
    math_val_size: int = 200
    math_levels: List[int] = field(default_factory=lambda: [1, 2, 3])

    # ---- rollout ----
    max_turns: int = 2                        # experience collection (attempt + 1 retry)
    rollout_temperature: float = 0.7
    max_tokens_api: int = 768
    max_tokens_math: int = 1536
    memory_top_k: int = 3
    memory_min_sim: float = 0.35
    skill_min_sim: float = 0.30

    # ---- P: rejection-sampling fine-tuning (RFT) ----
    rft_samples: int = 4                      # extra samples per task for data
    rft_keep_per_task: int = 2
    lora_rank: int = 16
    lora_alpha: int = 32
    train_epochs: int = 4
    train_lr: float = 3e-4
    train_batch_size: int = 8

    # ---- tree collection ----
    horizon: int = 3
    actions: List[str] = field(default_factory=lambda: list(ACTIONS))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | None = None, overrides: Dict | None = None) -> "EvoConfig":
        data: Dict = {}
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.update(overrides or {})
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def parse_overrides(pairs: List[str]) -> Dict:
    """`key=value` CLI overrides; values parsed as JSON when possible."""
    out: Dict = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out
