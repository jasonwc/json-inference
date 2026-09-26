# json-inference

Run, benchmark and compare models on the two-node DGX Spark cluster. Each model
setup is one file in `models/`; the `inference` CLI stages it, serves it behind
an OpenAI-compatible endpoint on `json-spark-1.json.lab`, and benchmarks it.

The Sparks' OS, network and monitoring live in
[json-lab/sparks](https://github.com/jasonwc/json-lab) (private). This repo only
decides what runs on them.

## Use

```bash
direnv allow                        # devshell puts bin/ on PATH

inference list                      # model definitions
inference pull gpt-oss-120b         # stage image + weights (resumable, runs on the Sparks)
inference up gpt-oss-120b           # stop anything running, serve, wait until ready
inference bench gpt-oss-120b        # smoke + speed benchmark, saved to results/
inference results                   # latest result per model, side by side
inference status | logs <model> [-f] | down
```

Clients point at `http://json-spark-1.json.lab:<port>/v1` with the model's
name from `inference list` (the `port` in its definition).

One model runs at a time; `up` stops whatever else is running, because large
models take nearly all of both nodes' unified memory.

## Model definitions

```toml
# models/<name>.toml
description = "..."
engine = "vllm"          # or "launcher"
nodes = 1                # 1 = json-spark-1 only, 2 = tensor parallel across both
port = 8000

[vllm]
image = "nvcr.io/nvidia/vllm:26.05-py3"
model = "openai/gpt-oss-120b"      # Hugging Face repo, optional `revision`
args = ["--max-model-len", "131072"]

[bench]
prefill_tokens = [2048, 8192, 32768]
```

- **vllm**: containers following NVIDIA's `dgx-spark-playbooks/playbook-vllm`,
  either one container on the head, or a Ray cluster over the QSFP link with
  `--tensor-parallel-size 2`. Weights live in each node's
  `~/.cache/huggingface`.
- **launcher**: a third-party recipe with its own scripts, checked out at a
  pinned `rev` on the head under `~/json-inference/recipes/<name>`.
  `[launcher.env]` is appended to the recipe's `.env.example`, with
  `{head.cx7_ip}`-style placeholders filled from `cluster.toml`.
  `deepseek-v4.1-flash-exl3` wraps
  [MiaAI-Lab's two-Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) this way.

## Benchmarks

`inference bench` measures, over streaming chat completions from wherever it
runs:

- **smoke**: 17 × 19 = 323
- **decode**: time to first token and tokens/s for one 512-token stream
- **prefill**: prompt tokens/s at each `prefill_tokens` size, using random
  prompts so the prefix cache doesn't help
- **parallel**: aggregate tokens/s at 1, 2 and 4 concurrent streams

Each run is saved as `results/<model>/<timestamp>.json`, together with the
definition and repo revision it ran with, and committed so models can be
compared over time.

## json-mini

`json-mini/` holds the older Ollama + Open WebUI setup for json-mini's iGPU
(`json-mini/setup.sh`, then `ollama-services start|stop|status`).
