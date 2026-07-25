#!/usr/bin/env bash
# Launch the text narrator (Milestone A). Starts the LLM server if it's down.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
if ! curl -fsS -m 2 "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
  echo "starting ollama serve (model store: $ROOT/models/ollama) ..."
  OLLAMA_MODELS="$ROOT/models/ollama" nohup "$HOME/.local/bin/ollama" serve \
    > "$ROOT/data/ollama-serve.log" 2>&1 &
  for _ in $(seq 1 20); do
    curl -fsS -m 2 "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1 && break; sleep 1
  done
fi
cd "$ROOT/src"
exec python3 -m narrator.cli "$@"
