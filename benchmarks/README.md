# Benchmark Results (True End-to-End CacheGen in vLLM)

`run_benchmarks.py` now runs true end-to-end KV transfer via an in-tree vLLM
`CacheGenConnector` (vendored under `third_party/vllm`) with two modes:

- `baseline`: raw KV transfer (`cachegen_enabled=false`)
- `cachegen`: CacheGen-compressed KV transfer (`cachegen_enabled=true`)

Each prompt is executed twice per mode/bandwidth:
1. **seed pass** stores KV blocks to external storage
2. **measured pass** reloads KV blocks and records transfer-aware metrics

## Colab A100 Smoke Checklist (single A100, true end-to-end)

```bash
# 0) Runtime sanity
!nvidia-smi

# 1) Clone + submodule (vLLM pinned by superproject pointer)
git clone <YOUR_CACHEGEN_REPO_URL>
cd cachegen
git submodule update --init --recursive third_party/vllm
# Submodule commit is hosted in this repo on branch:
# vendor/vllm-cachegen-e2e-v0.19.0 (based on v0.19.0).

# 2) Python env
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ./third_party/vllm

# 3) Fast smoke (both modes, one bandwidth)
python run_benchmarks.py \
  --model "facebook/opt-125m|2048|float16" \
  --modes baseline cachegen \
  --bandwidth-mbps 500 \
  --max-tokens 32

# 4) Validate the measured pass actually used external KV
python - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("benchmarks/results.json").read_text())
runs = data["runs"]
assert runs, "No runs found in benchmarks/results.json"
assert {"baseline", "cachegen"} <= {r["mode"] for r in runs}, "Missing mode(s)"
assert any(r.get("cached_tokens", 0) > 0 for r in runs), "No external KV hits"
for r in runs:
    for field in ["raw_bytes", "compressed_bytes", "compression_ratio", "network_time", "decode_time"]:
        assert field in r, f"Missing field {field}"
print("Smoke validation passed.")
PY

# 5) Render plots
python visualize_results.py
```

## Run Benchmarks

```bash
python run_benchmarks.py \
  --model "facebook/opt-125m|2048|float16" \
  --model "mistralai/Mistral-7B-Instruct-v0.3|8192|bfloat16" \
  --modes baseline cachegen \
  --bandwidth-mbps 100 500 1000 \
  --max-tokens 64

python visualize_results.py
```

## Bandwidth Sweep Validation

```bash
python run_benchmarks.py \
  --model "facebook/opt-125m|2048|float16" \
  --modes baseline cachegen \
  --bandwidth-mbps 100 500 1000 \
  --max-tokens 64
```

Expected trend: `network_time` (and typically TTFT) should increase as
bandwidth decreases.

## Artifacts

- `benchmarks/results.json`
- `benchmarks/ttft_by_model_mode.png`
- `benchmarks/throughput_by_model_mode.png`
- `benchmarks/compression_ratio_by_mode.png`
- `benchmarks/network_time_by_mode_bandwidth.png`

## Per-Run Metrics in `results.json`

- `ttft_seconds`
- `end_to_end_seconds`
- `tokens_per_second`
- `raw_bytes`
- `compressed_bytes`
- `compression_ratio`
- `network_time`
- `decode_time`
- `bandwidth_mbps`
- `cached_tokens`

## Notes

- `--bandwidth-mbps` applies **real throttling delay** in connector load path, so
  TTFT/end-to-end include link-throttling effects in measured wall-clock time.
- First implementation targets standard attention KV layouts used by
  OPT/Mistral/Llama-class models.

## Submodule Workflow for Clean PRs

When this repo changes files under `third_party/vllm`, make sure the vLLM
submodule has its own commit and that commit is pushable from the submodule URL
in `.gitmodules` before opening a PR from this superproject.

```bash
# In superproject root:
git -C third_party/vllm checkout -b cachegen-e2e
git -C third_party/vllm add \
  vllm/distributed/kv_transfer/kv_connector/factory.py \
  vllm/distributed/kv_transfer/kv_connector/v1/__init__.py \
  vllm/distributed/kv_transfer/kv_connector/v1/cachegen_connector.py
git -C third_party/vllm commit -m "Add CacheGenConnector with raw/compressed KV transfer toggle"

# Push submodule commit to your vLLM fork, then update URL if needed:
git -C third_party/vllm remote add fork <YOUR_VLLM_FORK_URL>  # if not already set
git -C third_party/vllm push fork cachegen-e2e

# Back in superproject:
git add third_party/vllm .gitmodules
git commit -m "Pin vLLM submodule to CacheGenConnector-integrated commit"
```

If your submodule commit only exists in a fork, `.gitmodules` should point to
that fork URL so collaborators/CI can fetch the referenced commit.
