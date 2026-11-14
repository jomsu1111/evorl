"""Harness channel (H): a per-node copy of COVE's memory/skill bank.

Reuses cove/src/memory_manager.MemorySkillManager unchanged (summarize ->
add/merge/replace/discard via LLM curator), driven by the fixed helper LLM.
Each tree node owns its own bank directory; children copy the parent's.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Any, Dict, List

import numpy as np


def _install_cove_patches(embed_model: str) -> None:
    """(1) Embedding cache keyed by *content* (COVE keys by id+version, which
    collides across sibling branches that edit the same record differently).
    (2) Load the embedder on CPU so it never competes with vLLM on GPU."""
    import utils.embedder as emb

    def _key(record: Dict[str, Any], field: str) -> str:
        return "h=" + hashlib.sha1(str(record.get(field, "")).encode("utf-8", "replace")).hexdigest()

    emb._record_cache_key = _key
    if emb._embedder_instance is None:
        from sentence_transformers import SentenceTransformer
        emb._embedder_instance = SentenceTransformer(embed_model, device="cpu")


class MemoryBank:
    _patched = False

    def __init__(self, mem_dir: str, helper_llm, cfg):
        if not MemoryBank._patched:
            _install_cove_patches(cfg.embed_model)
            MemoryBank._patched = True
        from memory_manager import MemorySkillManager
        from storage import ArtifactStore
        self.dir = mem_dir
        os.makedirs(mem_dir, exist_ok=True)
        self.cfg = cfg
        self.store = ArtifactStore(root=mem_dir, run_root=mem_dir)
        self.mgr = MemorySkillManager(self.store, helper_llm)
        self.helper = helper_llm

    # -- read -------------------------------------------------------------------
    # COVE's HyDE query expansion depends only on the task text, so expansions
    # are cached process-wide (val tasks are expanded once for the whole run).
    _HYDE: Dict[str, List[str]] = {}

    def _queries(self, texts: List[str]) -> List[List[str]]:
        todo = [t for t in dict.fromkeys(texts) if t not in MemoryBank._HYDE]
        for t, q in zip(todo, self.helper.pool.map(self.mgr._build_retrieval_queries, todo)):
            MemoryBank._HYDE[t] = q
        return [MemoryBank._HYDE[t] for t in texts]

    def retrieve_many(self, texts: List[str]) -> List[Dict[str, List[Dict[str, Any]]]]:
        mems, skills = self.mgr.load_memories(), self.mgr.load_skills()
        if not mems and not skills:
            return [{"memories": [], "skills": []} for _ in texts]
        out = []
        for qs in self._queries(texts):
            out.append({
                "memories": self.mgr._topk_by_multi_query(qs, mems, field="content", top_k=self.cfg.memory_top_k,
                                                          min_score=self.cfg.memory_min_sim),
                "skills": self.mgr._topk_by_multi_query(qs, skills, field="content", top_k=self.cfg.memory_top_k,
                                                        min_score=self.cfg.skill_min_sim),
            })
        return out

    def retrieve(self, task_text: str) -> Dict[str, List[Dict[str, Any]]]:
        return self.retrieve_many([task_text])[0]

    def stats(self) -> Dict[str, float]:
        mems, skills = self.mgr.load_memories(), self.mgr.load_skills()
        recs = mems + skills
        vol = [r for r in recs if r.get("volatility") == "volatile"]
        usage = [int(r.get("usage_count", 0) or 0) for r in recs]
        succ = [float(r.get("outcome_success_rate", 0.0)) for r in recs if int(r.get("outcome_count", 0) or 0) > 0]
        return {
            "n_memories": len(mems), "n_skills": len(skills), "n_total": len(recs),
            "frac_volatile": len(vol) / max(1, len(recs)),
            "mean_usage": float(np.mean(usage)) if usage else 0.0,
            "mean_outcome_success": float(np.mean(succ)) if succ else 0.0,
            "mean_version": float(np.mean([int(r.get("version", 1) or 1) for r in recs])) if recs else 0.0,
        }

    # -- write (the H action) ------------------------------------------------------
    def apply_experiences(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        from models import FeedbackRecord, TaskRecord
        ops = {"add": 0, "merge": 0, "replace": 0, "discard": 0, "other": 0}

        # 1) COVE bookkeeping on the retrieved context (usage + outcome telemetry).
        for ep in episodes:
            ctx = ep.get("context") or {}
            self.mgr._bump_usage_counts(self.store.memory_file, [dict(m) for m in ctx.get("memories", [])], "memory_id")
            self.mgr._bump_usage_counts(self.store.skill_file, [dict(s) for s in ctx.get("skills", [])], "skill_id")
            fb = ep["final"]
            self.mgr.record_context_outcome(ctx, TaskRecord(ep["task_id"], ep["task_text"], ep["task_type"]),
                                            FeedbackRecord("SUCCESS" if fb["success"] else "FAILED", fb["score"], fb["message"]))

        # 2) Summaries are stateless -> parallel; bank edits are sequential.
        def _summ(ep):
            fb = ep["final"]
            msg = fb["message"] + (("\n" + fb["reference"]) if fb.get("reference") else "")
            return self.mgr.summarize_candidates(
                TaskRecord(ep["task_id"], ep["task_text"], ep["task_type"]), ep["transcript"],
                FeedbackRecord("SUCCESS" if fb["success"] else "FAILED", fb["score"], msg))

        cands = list(self.helper.pool.map(_summ, episodes))
        for cand in cands:
            for bank in ("memory", "skill"):
                r = self.mgr._apply_candidate_to_bank(
                    bank_type=bank, candidate_text=cand.get(bank, ""), tags=cand.get("tags", []),
                    confidence=float(cand.get("confidence", 0.5)), volatility=str(cand.get("volatility", "unknown")))
                a = str((r or {}).get("action", "other"))
                ops[a if a in ops else "other"] += 1
        return ops

    def copy_to(self, new_dir: str) -> str:
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        shutil.copytree(self.dir, new_dir)
        return new_dir
