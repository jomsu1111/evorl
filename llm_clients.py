"""OpenAI-compatible clients for the two vLLM servers.

* PolicyClient — the evolving agent (Qwen3-0.6B base + runtime-loaded LoRA).
* HelperLLM    — fixed curator/judge model; drop-in for COVE's `utils.llm.LLM`
                 (same get_response / single_message_response interface) with
                 thread-safe token accounting, used as the cost of H.
"""
from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import requests
from openai import OpenAI


class TokenMeter:
    def __init__(self):
        self._lock = threading.Lock()
        self.prompt = 0
        self.completion = 0
        self.calls = 0

    def add(self, usage) -> None:
        if usage is None:
            return
        with self._lock:
            self.prompt += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion += int(getattr(usage, "completion_tokens", 0) or 0)
            self.calls += 1

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {"prompt": self.prompt, "completion": self.completion, "calls": self.calls}


def meter_delta(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
    return {k: after[k] - before.get(k, 0) for k in after}


class PolicyClient:
    def __init__(self, base_url: str, base_name: str, max_concurrency: int = 64):
        self.base_url = base_url.rstrip("/")
        self.base_name = base_name
        self.client = OpenAI(api_key="EMPTY", base_url=self.base_url, timeout=600, max_retries=3)
        self.pool = ThreadPoolExecutor(max_concurrency)
        self.meter = TokenMeter()
        self._loaded: set[str] = set()
        self._lock = threading.Lock()

    # -- LoRA management (needs VLLM_ALLOW_RUNTIME_LORA_UPDATING=True) --------
    def load_adapter(self, name: str, path: str) -> None:
        with self._lock:
            if name in self._loaded:
                return
            root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            r = requests.post(f"{root}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path}, timeout=300)
            if r.status_code != 200 and "already" not in r.text.lower():
                raise RuntimeError(f"load_lora_adapter({name}) failed: {r.status_code} {r.text}")
            self._loaded.add(name)

    def unload_adapter(self, name: str) -> None:
        with self._lock:
            if name not in self._loaded:
                return
            root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            requests.post(f"{root}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=120)
            self._loaded.discard(name)

    # -- generation -------------------------------------------------------------
    def chat(self, model: str, messages: List[Dict], temperature: float, max_tokens: int, seed: Optional[int] = None) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                    top_p=1.0 if temperature == 0 else 0.95, seed=seed,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                self.meter.add(resp.usage)
                return resp.choices[0].message.content or ""
            except Exception:
                if attempt == 2:
                    traceback.print_exc()
                    return ""
                time.sleep(2 * (attempt + 1))
        return ""

    def chat_many(self, model: str, convs: Sequence[List[Dict]], temperature: float, max_tokens: Sequence[int] | int,
                  seeds: Optional[Sequence[int]] = None) -> List[str]:
        if isinstance(max_tokens, int):
            max_tokens = [max_tokens] * len(convs)
        seeds = seeds or [None] * len(convs)
        futs = [self.pool.submit(self.chat, model, c, temperature, mt, s) for c, mt, s in zip(convs, max_tokens, seeds)]
        return [f.result() for f in futs]


class HelperLLM:
    """COVE-compatible LLM wrapper (see cove/src/utils/llm.py)."""

    def __init__(self, base_url: str, model: str, max_concurrency: int = 32):
        self.client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=600, max_retries=3)
        self.model = model
        self.meter = TokenMeter()
        self.pool = ThreadPoolExecutor(max_concurrency)
        self.infer_engine = None
        self.enable_thinking = False

    def set_infer_engine(self, infer_engine) -> None:  # COVE interface no-op
        pass

    def get_response(self, messages, temperature=1, max_tokens=2048, request_only=False, enable_thinking=None):
        if request_only:
            return {"status": "SUCCESS", "data": messages}
        try:
            c = self.client.chat.completions.create(model=self.model, messages=messages,
                                                    max_tokens=max_tokens, temperature=temperature)
            self.meter.add(c.usage)
            return {"status": "SUCCESS", "data": c.choices[0].message.content or ""}
        except Exception:
            return {"status": "FAILED", "data": traceback.format_exc()}

    def single_message_response(self, message, temperature=1, max_tokens=2048, request_only=False, enable_thinking=None):
        return self.get_response([{"role": "user", "content": message}], temperature, max_tokens, request_only)
