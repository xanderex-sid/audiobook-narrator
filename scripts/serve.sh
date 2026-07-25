#!/usr/bin/env bash
# Start the local LLM server with its model store INSIDE this project.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OLLAMA_MODELS="$ROOT/models/ollama"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
exec "$HOME/.local/bin/ollama" serve
