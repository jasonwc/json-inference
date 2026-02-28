# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

json-clankers — local AI model experimentation and agent patterns, running on json-mini.

## Hardware (json-mini)

- **CPU**: AMD Ryzen AI 9 HX 370 (12 cores / 24 threads, 5.1 GHz boost, AVX-512)
- **RAM**: 28 GB
- **GPU**: Radeon 890M (integrated RDNA 3.5, shared memory)
- **NPU**: AMD XDNA (Linux support maturing)
- **OS**: Pop!_OS (x86_64-linux)

## Stack

- **Ollama**: Local model serving (systemd user service)
- **Open WebUI**: Browser-based chat interface (systemd user service, port 8080)
- **Python + uv**: Agent scripting and experimentation
- **Nix flake**: Devshell provides all dependencies

## Key Commands

- `ollama list` — show downloaded models
- `ollama pull <model>` — download a model (e.g. `llama3.1:8b`, `mistral`, `phi3`)
- `ollama run <model>` — interactive chat
- `systemctl --user status ollama` — check Ollama service
- `systemctl --user status open-webui` — check Open WebUI service
- `nix flake check` — validate flake

## Conventions

- Devshell activated via direnv (`use flake`)
- Services managed via systemd user units (installed by `setup.sh`)
- Model files are not committed (`.gitignore`)
