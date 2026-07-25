#!/usr/bin/env bash
# Full audiobook + "Hey Narrator" loop with speech. Add --voice for mic, else type.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_PY="${NARRATOR_VOICE_PYTHON:-/home/x0zby2/projects/audiobook-gen/.venv/bin/python}"
source "$ROOT/scripts/_ensure_ollama.sh"; ensure_ollama "$ROOT"
cd "$ROOT/src"; exec "$VOICE_PY" -m narrator.app "$@"
