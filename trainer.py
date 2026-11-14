"""Parameter channel (P) worker: LoRA SFT on rejection-sampled trajectories.

Run as a subprocess on the training GPU:
  CUDA_VISIBLE_DEVICES=1 python -m evorl.trainer --base M --data d.jsonl --out dir [--init-adapter parent]

Each data row: {"messages": [system, user], "response": "..."}.
Loss is on the response tokens only; the prompt is rendered exactly as vLLM
renders it at inference (Qwen3 chat template, enable_thinking=False).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-adapter", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).cuda()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    model.train()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    examples = []
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(r["response"].strip() + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        ids = (p_ids + r_ids)[: args.max_len]
        labels = ([-100] * len(p_ids) + r_ids)[: args.max_len]
        examples.append((ids, labels))

    # Memory: long math traces x 152k vocab logits OOM'd at bs=8, so each optimizer
    # step of `bs` examples is split into micro-batches of `micro_bs` (token-normalized
    # gradient accumulation) with gradient checkpointing.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_total = max(1, args.epochs * math.ceil(len(examples) / args.bs))
    warm = max(1, steps_total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.1, 1 - max(0, s - warm) / max(1, steps_total - warm)))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    losses, n_tok = [], 0
    for ep in range(args.epochs):
        random.shuffle(examples)
        for b in range(0, len(examples), args.bs):
            batch = examples[b:b + args.bs]
            n_lab = sum(sum(1 for t in y if t != -100) for _, y in batch)
            step_loss = 0.0
            for m in range(0, len(batch), args.micro_bs):
                mb = batch[m:m + args.micro_bs]
                L = max(len(x) for x, _ in mb)
                ids = torch.full((len(mb), L), pad, dtype=torch.long)
                lab = torch.full((len(mb), L), -100, dtype=torch.long)
                att = torch.zeros((len(mb), L), dtype=torch.long)
                for i, (x, y) in enumerate(mb):
                    ids[i, :len(x)] = torch.tensor(x)
                    lab[i, :len(y)] = torch.tensor(y)
                    att[i, :len(x)] = 1
                n_tok += int(att.sum())
                logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, :-1]
                tgt = lab[:, 1:].cuda()
                loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                         ignore_index=-100, reduction="sum") / max(1, n_lab)
                loss.backward()
                step_loss += float(loss)
                del logits, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(step_loss)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    k = max(1, len(losses) // 10)
    stats = {"n_examples": len(examples), "train_tokens": n_tok, "epochs": args.epochs, "steps": len(losses),
             "loss_first": sum(losses[:k]) / k if losses else None, "loss_last": sum(losses[-k:]) / k if losses else None,
             "seconds": time.time() - t0}
    with open(os.path.join(args.out, "train_stats.json"), "w") as f:
        json.dump(stats, f)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
