"""Rebuild LoRA adapters lost with the ephemeral /opt after a container restart.

  python -m evorl.rebuild_adapters --exp main NODE [NODE ...] [--verify]

A node uses the adapter recorded in its node.json; that adapter was trained by its *owner*
(the adapter directory's basename). The owner's adapter is retrained from the owner's saved
RFT data (nodes/<owner>/rft.jsonl) on top of the owner's parent adapter (rebuilt first,
recursively), with the same trainer, hyperparameters and seed as the original run, into the
same path. Nothing in the results directory is modified except a train_rebuild.log per owner.

--verify compares each rebuilt owner's final training loss with the one recorded at collection
time, and re-evaluates each requested node with its rebuilt adapter (not saved) against the V
recorded in node.json.
"""
from __future__ import annotations

import argparse
import os

from . import agent
from .config import EvoConfig, parse_overrides
from .env import EvolutionEnv


def rebuild(env: EvolutionEnv, owner_id: str, done: dict) -> None:
    if owner_id in done:
        return
    owner = env.load(owner_id)
    if owner is None or not owner.adapter:
        raise RuntimeError(f"{owner_id}: not an adapter owner")
    if os.path.exists(os.path.join(owner.adapter, "adapter_model.safetensors")):
        done[owner_id] = None  # still on disk
        return
    parent = env.load(owner.parent_id)
    if parent.adapter:
        rebuild(env, os.path.basename(parent.adapter), done)
    data = os.path.join(env.node_dir(owner_id), "rft.jsonl")
    stats, _ = env.run_trainer(data, owner.adapter, parent.adapter, os.path.join(env.node_dir(owner_id), "train_rebuild.log"))
    rec = owner.counters.get("cost", {})
    done[owner_id] = (rec.get("loss_last"), stats["loss_last"], rec.get("train_tokens"), stats["train_tokens"])
    print(f"[rebuild] {owner_id}: loss_last recorded={rec.get('loss_last')} rebuilt={stats['loss_last']:.4f} "
          f"train_tokens recorded={rec.get('train_tokens')} rebuilt={stats['train_tokens']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("nodes", nargs="+")
    args = ap.parse_args()
    cfg = EvoConfig.load(args.config, parse_overrides(args.set))
    env = EvolutionEnv(cfg, args.exp)
    done: dict = {}
    for nid in args.nodes:
        n = env.load(nid)
        if n is None:
            raise SystemExit(f"unknown node {nid}")
        if n.adapter:
            rebuild(env, os.path.basename(n.adapter), done)
        else:
            print(f"[rebuild] {nid}: uses the base model, nothing to rebuild")
    if not args.verify:
        return
    for nid in args.nodes:
        n = env.load(nid)
        bank = env.bank(n)
        has_mem = bank.stats()["n_total"] > 0
        res = agent.evaluate(cfg, env.policy, env.model_name(n), env.stream.val_tasks(), env.stream.world(n.step),
                             bank if has_mem else None)
        print(f"[verify] {nid}: V recorded={n.metrics['V']:.4f} rebuilt={res['V']:.4f} "
              f"(api {n.metrics['acc_api']:.3f}->{res['acc_api']:.3f}, math {n.metrics['acc_math']:.3f}->{res['acc_math']:.3f})",
              flush=True)


if __name__ == "__main__":
    main()
