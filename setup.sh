#!/usr/bin/env bash
set -euo pipefail

# Setup script for json-inference services on json-mini.
# Installs systemd user services for Ollama and Open WebUI.
#
# Prerequisites: Run from within the devshell (direnv allow).

# ---------- helpers ----------

info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- checks ----------

command -v ollama &>/dev/null || error "ollama not found — are you in the devshell? Run 'direnv allow' first."
command -v open-webui &>/dev/null || error "open-webui not found — are you in the devshell? Run 'direnv allow' first."

OLLAMA_PATH=$(which ollama)
OPEN_WEBUI_PATH=$(which open-webui)

# ---------- systemd user services ----------

mkdir -p "$HOME/.config/systemd/user"

info "Installing Ollama systemd user service..."
cat > "$HOME/.config/systemd/user/ollama.service" <<EOF
[Unit]
Description=Ollama Model Server
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=${OLLAMA_PATH} serve
Restart=on-failure
RestartSec=5
Environment=OLLAMA_HOST=0.0.0.0:11434

[Install]
WantedBy=default.target
EOF

info "Installing Open WebUI systemd user service..."
cat > "$HOME/.config/systemd/user/open-webui.service" <<EOF
[Unit]
Description=Open WebUI
After=ollama.service
Wants=ollama.service

[Service]
ExecStart=${OPEN_WEBUI_PATH} serve --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

info "Installing inference CLI..."
install "$SCRIPT_DIR/inference" "$HOME/.local/bin/inference"

systemctl --user daemon-reload
systemctl --user enable ollama.service open-webui.service

echo
info "Setup complete. Services are enabled but not started."
info "Use 'inference start' to spin them up, 'inference stop' to conserve resources."
echo
info "Pull a model to get started:"
info "  inference start"
info "  ollama pull llama3.1:8b"
