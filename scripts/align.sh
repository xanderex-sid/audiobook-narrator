#!/usr/bin/env bash
# Build forced alignment (exact spoiler cutoff + phrase-restart) from chapter audio.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_PY="${NARRATOR_VOICE_PYTHON:-/home/x0zby2/projects/audiobook-gen/.venv/bin/python}"
cd "$ROOT/src"; exec "$VOICE_PY" -m narrator.align "$@"
