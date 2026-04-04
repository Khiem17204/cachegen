"""Live vLLM serving benchmark.

Hits baseline and CacheGen-enabled vLLM OpenAI-compatible endpoints and logs
TTFT and end-to-end latency. Assumes you run two servers (or the same server
with/without CacheGen flag) and point to them via environment variables.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import requests

PROMPTS = {
    "short": "What is the capital of France?",
    "medium": "Explain what attention does in transformers and give an example.",
    "long": "Summarize the role of reinforcement learning from human feedback in aligning large language models across multilingual deployments, including reward modeling and safety mitigations.",
}

DEFAULT_MODELS = ["facebook/opt-1.3b"]


def _stream_chat(endpoint: str, model: str, prompt: str) -> Dict:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "stream": True,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    with requests.post(endpoint, json=payload, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        first = None
        tokens = 0
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                data = line[len(b"data: ") :]
                if data.strip() == b"[DONE]":
                    break
                if first is None:
                    first = time.perf_counter()
                tokens += 1
        end = time.perf_counter()
    return {
        "ttft": (first - start) if first else None,
        "latency": end - start,
        "tokens": tokens,
    }


def run(endpoint: str, label: str, models: List[str]) -> List[Dict]:
    results = []
    for model in models:
        for name, prompt in PROMPTS.items():
            metrics = _stream_chat(endpoint, model, prompt)
            results.append({
                "endpoint": label,
                "model": model,
                "prompt_name": name,
                **metrics,
            })
    return results


def main() -> None:
    baseline_ep = os.environ.get("BASELINE_ENDPOINT", "http://localhost:8000/v1/chat/completions")
    cachegen_ep = os.environ.get("CACHEGEN_ENDPOINT")
    models = os.environ.get("VLLM_MODELS")
    models = [m.strip() for m in models.split(",")] if models else DEFAULT_MODELS

    all_results = []
    all_results += run(baseline_ep, "baseline", models)
    if cachegen_ep:
        all_results += run(cachegen_ep, "cachegen", models)

    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "live_results.json"
    out_path.write_text(json.dumps({"runs": all_results, "generated_at": time.time()}, indent=2))
    print(f"Saved live results to {out_path}")


if __name__ == "__main__":
    main()
