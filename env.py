"""Evolution MDP environment.

A node = full agent state (LoRA adapter + memory bank + evolution history)
at stream step t. `expand(node, a)` applies one evolution action to the
experiences gathered on batch t and returns the child at step t+1, logging
the transition (s_t, a_t, ΔV, cost, s_{t+1}).

Layout:
  results_dir/<exp>/nodes/<node_id>/{node.json, mem/, batch.jsonl, features.json, val.json, rft.jsonl}
  results_dir/<exp>/transitions.jsonl
  work_dir/<exp>/adapters/<node_id>/           (LoRA weights; large, ephemeral)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import agent
from .config import EvoConfig
from .features import node_features
from .llm_clients import HelperLLM, PolicyClient, meter_delta
from .memory import MemoryBank
from .tasks.stream import TaskStream

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Node:
    node_id: str
    parent_id: Optional[str]
    root_id: str
    depth: int
    step: int
    adapter: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    counters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


class EvolutionEnv:
    def __init__(self, cfg: EvoConfig, exp: str):
        self.cfg, self.exp = cfg, exp
        self.root = os.path.join(cfg.results_dir, exp)
        self.adapter_root = os.path.join(cfg.work_dir, exp, "adapters")
        os.makedirs(os.path.join(self.root, "nodes"), exist_ok=True)
        os.makedirs(self.adapter_root, exist_ok=True)
        # Record the config actually used by every invocation (the latest one is config.json).
        cfg.save(os.path.join(self.root, "config.json"))
        cfg.save(os.path.join(self.root, "configs", f"{time.strftime('%Y%m%d-%H%M%S')}.json"))
        self.stream = TaskStream(cfg)
        self.policy = PolicyClient(cfg.policy_url, cfg.policy_name, cfg.max_concurrency)
        self.helper = HelperLLM(cfg.helper_url, cfg.helper_name)
        self.trans_path = os.path.join(self.root, "transitions.jsonl")

    # ------------------------------------------------------------------ nodes
    def node_dir(self, node_id: str) -> str:
        return os.path.join(self.root, "nodes", node_id)

    def save(self, n: Node) -> None:
        d = self.node_dir(n.node_id)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "node.json.tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(n), f, indent=1)
        os.replace(tmp, os.path.join(d, "node.json"))

    def load(self, node_id: str) -> Optional[Node]:
        p = os.path.join(self.node_dir(node_id), "node.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return Node(**json.load(f))

    def bank(self, n: Node) -> MemoryBank:
        return MemoryBank(os.path.join(self.node_dir(n.node_id), "mem"), self.helper, self.cfg)

    def make_root(self, root_id: str = "R0") -> Node:
        n = self.load(root_id)
        if n is None:
            n = Node(root_id, None, root_id, 0, 0, counters={"train_samples": 0})
            self.save(n)
        self.evaluate(n)
        return n

    @staticmethod
    def adapter_name(path: str) -> str:
        """Unique per trained weights: vLLM keeps serving the *old* weights if a name is
        re-registered, so the name includes the weight file's mtime+size."""
        st = os.stat(os.path.join(path, "adapter_model.safetensors"))
        tag = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:10]
        return "a_" + tag

    def model_name(self, n: Node) -> str:
        if not n.adapter:
            return self.cfg.policy_name
        name = self.adapter_name(n.adapter)
        self.policy.load_adapter(name, n.adapter)
        return name

    def release(self, n: Node) -> None:
        if n.adapter and os.path.exists(n.adapter):
            self.policy.unload_adapter(self.adapter_name(n.adapter))

    # ------------------------------------------------------------ measurement
    def evaluate(self, n: Node) -> Dict[str, Any]:
        if "V" in n.metrics:
            return n.metrics
        t0 = time.time()
        world, val, model = self.stream.world(n.step), self.stream.val_tasks(), self.model_name(n)
        bank = self.bank(n)
        has_mem = bank.stats()["n_total"] > 0
        res = agent.evaluate(self.cfg, self.policy, model, val, world, bank if has_mem else None)
        res_nm = agent.evaluate(self.cfg, self.policy, model, val, world, None) if has_mem else res
        n.metrics = {"V": res["V"], "acc_api": res["acc_api"], "acc_math": res["acc_math"],
                     "V_nomem": res_nm["V"], "acc_api_nomem": res_nm["acc_api"], "acc_math_nomem": res_nm["acc_math"],
                     "eval_seconds": time.time() - t0}
        with open(os.path.join(self.node_dir(n.node_id), "val.json"), "w") as f:
            json.dump({"mem": res["per_task"], "nomem": res_nm["per_task"]}, f)
        self.save(n)
        return n.metrics

    def experience(self, n: Node) -> List[Dict[str, Any]]:
        p = os.path.join(self.node_dir(n.node_id), "batch.jsonl")
        if os.path.exists(p):
            return [json.loads(l) for l in open(p)]
        eps = agent.rollout(self.cfg, self.policy, self.model_name(n), self.stream.batch(n.step), self.stream.world(n.step),
                            self.bank(n), max_turns=self.cfg.max_turns, temperature=self.cfg.rollout_temperature,
                            seed_tag=f"exp-{n.step}")
        eps = agent.strip_private(eps)
        with open(p + ".tmp", "w") as f:
            for e in eps:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
        return eps

    def features(self, n: Node) -> Dict[str, float]:
        p = os.path.join(self.node_dir(n.node_id), "features.json")
        if os.path.exists(p):
            return json.load(open(p))
        self.evaluate(n)
        feats = node_features(n, self.experience(n), self.bank(n).stats())
        json.dump(feats, open(p, "w"), indent=1)
        return feats

    # ---------------------------------------------------------------- actions
    def expand(self, n: Node, action: str, p_tasks: Optional[list] = None, tag: str = "") -> Node:
        """Apply action to node n. `p_tasks` overrides the tasks used by P (COVE's pool)."""
        assert action in ("H", "P", "B"), action
        cid = f"{n.node_id}.{action}{tag}"
        done = self.load(cid)
        if done is not None and "V" in done.metrics:
            return done
        self.features(n)  # make sure s_t is materialized before the branch mutates anything
        eps = self.experience(n)
        child = Node(cid, n.node_id, n.root_id, n.depth + 1, n.step + 1, n.adapter,
                     copy.deepcopy(n.history), copy.deepcopy(n.counters), {})
        self.bank(n).copy_to(os.path.join(self.node_dir(cid), "mem"))
        cost: Dict[str, Any] = {}

        if action in ("H", "B"):
            t0, m0 = time.time(), self.helper.meter.snapshot()
            cost["mem_ops"] = self.bank(child).apply_experiences(eps)
            d = meter_delta(self.helper.meter.snapshot(), m0)
            cost.update(h_seconds=time.time() - t0, helper_tokens=d["prompt"] + d["completion"], helper_calls=d["calls"])

        if action in ("P", "B"):
            cost.update(self._train_P(n, child, eps, p_tasks))

        self.evaluate(child)
        dV = child.metrics["V"] - n.metrics["V"]
        child.history.append({"action": action, "step": n.step, "dV": dV, "batch_type": self.stream.batch_type(n.step),
                              "batch_t1": sum(e["success_t1"] for e in eps) / max(1, len(eps))})
        child.counters["train_samples"] = child.counters.get("train_samples", 0) + cost.get("n_train", 0)
        child.counters["cost"] = cost
        self.save(child)
        rec = {"exp": self.exp, "root": n.root_id, "parent": n.node_id, "child": cid, "depth": n.depth, "step": n.step,
               "action": action, "dV": dV, "V_parent": n.metrics["V"], "V_child": child.metrics["V"],
               "parent_metrics": n.metrics, "child_metrics": child.metrics, "cost": cost, "ts": time.time()}
        with open(self.trans_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return child

    def _train_P(self, parent: Node, child: Node, eps: List[Dict[str, Any]], p_tasks: Optional[list]) -> Dict[str, Any]:
        t0, m0 = time.time(), self.policy.meter.snapshot()
        world = self.stream.world(parent.step)
        tasks = p_tasks if p_tasks is not None else self.stream.batch(parent.step)
        by_id = {t.task_id: t for t in tasks}
        succ: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
        if p_tasks is None:  # reuse the batch experience itself
            for e in eps:
                for tr in e["turns"]:
                    if tr["grade"]["success"]:
                        succ[e["task_id"]].append(tr["response"])
        extra = agent.rollout(self.cfg, self.policy, self.model_name(parent), tasks, world, self.bank(child),
                              max_turns=self.cfg.max_turns, temperature=0.8, n=self.cfg.rft_samples,
                              seed_tag=f"rft-{child.node_id}")
        for e in extra:
            for tr in e["turns"]:
                if tr["grade"]["success"]:
                    succ[e["task_id"]].append(tr["response"])
        rows = []
        for tid, resps in succ.items():
            seen: List[str] = []
            for r in resps:
                if r.strip() and r not in seen:
                    seen.append(r)
            for r in seen[: self.cfg.rft_keep_per_task]:
                rows.append({"messages": agent.train_messages(by_id[tid]), "response": r})
        d = meter_delta(self.policy.meter.snapshot(), m0)
        cost = {"n_train": len(rows), "sample_tokens": d["prompt"] + d["completion"],
                "rft_yield": sum(1 for v in succ.values() if v) / max(1, len(succ))}
        if not rows:
            cost["p_seconds"] = time.time() - t0
            return cost
        data_path = os.path.join(self.node_dir(child.node_id), "rft.jsonl")
        with open(data_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out = os.path.join(self.adapter_root, child.node_id)
        if os.path.exists(out):
            shutil.rmtree(out)
        c = self.cfg
        cmd = [sys.executable, "-m", "evorl.trainer", "--base", c.policy_model, "--data", data_path, "--out", out,
               "--epochs", str(c.train_epochs), "--lr", str(c.train_lr), "--rank", str(c.lora_rank),
               "--alpha", str(c.lora_alpha), "--bs", str(c.train_batch_size)]
        if parent.adapter:
            cmd += ["--init-adapter", parent.adapter]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=c.train_gpu, TOKENIZERS_PARALLELISM="false")
        log = os.path.join(self.node_dir(child.node_id), "train.log")
        with open(log, "w") as lf:
            subprocess.run(cmd, cwd=PKG_PARENT, env=env, stdout=lf, stderr=subprocess.STDOUT, check=True)
        stats = json.load(open(os.path.join(out, "train_stats.json")))
        child.adapter = out
        cost.update(train_tokens=stats["train_tokens"], train_seconds=stats["seconds"],
                    loss_first=stats["loss_first"], loss_last=stats["loss_last"], p_seconds=time.time() - t0)
        return cost
