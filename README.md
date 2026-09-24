# json-inference

Local model serving and evaluation: which models to run, how to serve them, and
how fast and how good they are. The primary target is the two-node DGX Spark
cluster, with json-mini as a small-model secondary. Today it holds the Ollama +
Open WebUI setup on json-mini.

## Quick Start

```bash
git clone https://github.com/jasonwc/json-inference.git
cd json-inference
direnv allow

# Install and enable systemd services (Ollama + Open WebUI)
bash setup.sh

# Pull a model
ollama pull llama3.1:8b

# Open WebUI is available at http://json-mini.local:8080
```

## Services

| Service  | Port | Description             |
| -------- | ---- | ----------------------- |
| Ollama   | 11434 | Model serving API      |
| Open WebUI | 8080 | Browser chat interface |

Both run as systemd user services. Start and stop them with `inference start|stop|status`.
