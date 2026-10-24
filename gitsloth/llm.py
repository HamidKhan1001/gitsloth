"""Minimal Ollama client. Refuses any host that is not loopback: your code never leaves this machine."""
from __future__ import annotations

import json
import time
import urllib.request
from urllib.parse import urlparse

LOOPBACK = {"localhost", "127.0.0.1", "::1"}


class LLMError(Exception):
    pass


class OllamaClient:
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://127.0.0.1:11434", timeout: float = 600):
        host = urlparse(base_url).hostname
        if host not in LOOPBACK:
            raise LLMError(f"refusing non-local LLM host {host!r}: gitsloth never sends code off this machine")
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.calls = self.prompt_tokens = self.output_tokens = 0
        self.seconds = 0.0

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "model": self.model, "stream": False, "format": schema, "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 7, "num_ctx": 4096},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        req = urllib.request.Request(f"{self.base_url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.load(r)
            out = json.loads(resp["message"]["content"])
        except (OSError, KeyError, json.JSONDecodeError) as e:
            raise LLMError(f"Ollama call failed ({e}). Is `ollama serve` running and `{self.model}` pulled?") from e
        self.calls += 1
        self.seconds += time.time() - t0
        self.prompt_tokens += resp.get("prompt_eval_count", 0)
        self.output_tokens += resp.get("eval_count", 0)
        return out
