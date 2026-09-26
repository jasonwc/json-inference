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
inference pull gpt-oss-120b         # stage image + weights on the Sparks (resumable; --no-wait to return early)
inference up gpt-oss-120b           # stop anything running, serve, wait until ready
inference bench gpt-oss-120b        # smoke + speed benchmark, saved to results/
inference results                   # latest result per model, side by side
inference status | logs <model> [-f] | down
inference cancel <model>            # stop its background downloads/boots on the Sparks
```

The CLI runs on a LAN machine and drives the Sparks over SSH; long steps run
detached on the Sparks with their logs in `~/json-inference/logs/`.

Every model serves on `http://json-spark-1.json.lab:8000/v1`, which json-lab
also exposes as `http://inference.json.lab/v1`. The model id is the definition's
file name unless it sets `served_name`; `GET /v1/models` shows what is up.

One model runs at a time; `up` stops whatever else is running, because large
models take nearly all of both nodes' unified memory.

## Layout

```
cluster.toml       # how to reach the Sparks: SSH names, QSFP IPs, NCCL netdev/RoCE device
models/<name>.toml # one serving setup per file
json_inference/    # the CLI (stdlib-only Python 3.11+): config, engines, bench
bin/inference      # runs the CLI from the checkout
results/<model>/   # committed benchmark runs
json-mini/         # older Ollama setup for json-mini
```

## Model definitions

```toml
# models/<name>.toml
description = "..."
engine = "vllm"          # or "launcher"
nodes = 1                # 1 = json-spark-1 only, 2 = tensor parallel across both
port = 8000              # every model uses 8000, so inference.json.lab has one backend

[vllm]
image = "nvcr.io/nvidia/vllm:26.05-py3"
model = "openai/gpt-oss-120b"      # Hugging Face repo, optional `revision`
download_exclude = ["original/*", "metal/*"]
args = ["--max-model-len", "131072"]

[bench]
prefill_tokens = [2048, 8192, 32768]
```

- **vllm**: containers following NVIDIA's `dgx-spark-playbooks/playbook-vllm`,
  either one container on the head, or a Ray cluster over the QSFP link with
  `--tensor-parallel-size 2`. Weights live in each node's
  `~/.cache/huggingface`; serving runs offline, so `pull` must come first.
- **launcher**: a third-party recipe with its own scripts (`[launcher]` sets
  `repo`, a pinned `rev`, and its `pull`/`start`/`stop`/`logs` commands),
  checked out on the head under `~/json-inference/recipes/<name>`.
  `[launcher.env]` is appended to the recipe's `.env.example`, with
  `{head.cx7_ip}`-style placeholders filled from `cluster.toml`. The two-Spark
  DeepSeek, GLM, MiMo and Qwen definitions wrap
  [MiaAI-Lab](https://github.com/MiaAI-Lab)'s recipes this way.

## Benchmarks

`inference bench` measures, over streaming chat completions from wherever it
runs:

- **smoke**: 17 × 19 = 323
- **decode** (prose): time to first token and tokens/s for one 512-token stream
- **prefill**: prompt tokens/s at each `prefill_tokens` size, using random
  prompts so the prefix cache doesn't help
- **parallel**: aggregate tokens/s at 1, 2 and 4 concurrent streams
- **code**: single-stream and 4-stream decode on a coding prompt at temperature
  0, where speculative decoding helps most (prose at 0.7 is near its worst case)

`inference up` also records how long the switch took (`results/<model>/up-<timestamp>.json`,
shown as "start s" in `inference results`). Each benchmark run is saved as
`results/<model>/<timestamp>.json`, together with the
definition and repo revision it ran with, and committed so models can be
compared over time. A new model's definition and its first results are
committed together.

## Clients

Anything OpenAI-compatible can use the endpoint. json-lab runs Open WebUI at
`chat.json.lab` against it, and json-workstation configures opencode, pi and
DeepSeek's `dsh` with the Sparks as an extra provider alongside their hosted
ones.

## json-mini

`json-mini/` holds the older Ollama + Open WebUI setup for json-mini's iGPU
(`json-mini/setup.sh`, then `ollama-services start|stop|status`).
