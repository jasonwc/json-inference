# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

json-inference: run, benchmark and compare models on the two-node DGX Spark
cluster (json-spark-1 = head / API, json-spark-2 = worker, 200G QSFP link
between them). The Sparks' OS, network and monitoring belong to json-lab
(`sparks/`), not here. json-mini's small-model Ollama setup lives in
`json-mini/`.

## Layout

- `cluster.toml`: how to reach the Sparks (SSH names, QSFP IPs, NCCL netdev/RoCE device/GID)
- `models/<name>.toml`: one serving setup per file; the file name is the model's API id
- `json_inference/`: the CLI, stdlib-only Python 3.11+
  - `config.py`: loading definitions
  - `engines.py`: `vllm` and `launcher` engines, detached remote jobs
  - `bench.py`: benchmark and results table
- `bin/inference`: runs the CLI from the checkout (the devshell puts `bin/` on PATH)
- `results/<model>/<timestamp>.json`: committed benchmark runs
- `docs/`: write-ups of benchmark sweeps and findings

## Key Commands

- `inference list | pull <m> | up <m> | bench <m> | results | status | logs <m> [-f] | cancel <m> | sweep [m...] | down`
- `python3 -m py_compile json_inference/*.py`: quick syntax check
- `nix flake check`: validate flake

## Conventions

- One model at a time: `up` always runs `down` first (big models use nearly all
  of both nodes' 128 GB unified memory).
- The CLI runs on a LAN machine and drives the Sparks over SSH (`bash -lc`, so
  system-manager's PATH is present). Long steps run under nohup on the Spark
  and are followed via their log in `~/json-inference/logs/`.
- vLLM containers follow NVIDIA's dgx-spark-playbooks (NGC image, Ray for
  TP=2, NCCL pinned to the QSFP netdev). Serve-time downloads are off
  (`HF_HUB_OFFLINE=1`); `pull` stages weights.
- Third-party recipes are wrapped with `engine = "launcher"` at a pinned
  commit, never vendored.
- New models: add a definition, `pull`, `up`, `bench`, then commit the
  definition and its results together.
- Devshell activated via direnv (`use flake`).
