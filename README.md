# json-clankers

Local AI model experimentation and agent patterns on json-mini.

## Quick Start

```bash
git clone https://github.com/jasonwc/json-clankers.git
cd json-clankers
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

Both run as systemd user services and start automatically on login.
