# Live Benchmark Summary: OPT-1.3B

Real endpoint measurements captured from a Google Colab Pro A100 (40GB) using sequential single-server runs:

- `baseline_live`: standard vLLM server
- `cachegen_live`: CacheGen-configured run measured separately on the same host

Headline numbers:

- Average latency: baseline `0.342s`, CacheGen `0.404s`
- Average delta: CacheGen `+0.063s` (`+18.4%`) for this small-model setup
- Prompt-level deltas:
  - `short`: `+0.173s`
  - `medium`: `+0.005s`
  - `long`: `+0.010s`
- Token counts and output previews matched across both runs for all three prompts

Artifacts:

- `live_results_opt13b.json`: raw run data and summary
- `live_opt13b_latency.png`: latency comparison bar chart
