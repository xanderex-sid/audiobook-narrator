#!/usr/bin/env bash
# ZERO-KEY hands-free mode — talk to the audiobook entirely by voice.
#
#   🎧 Wear headphones (so the mic never hears the book or the narrator).
#   Say  “Friday”              -> the book pauses and it listens
#   ...just speak your question -> it answers, then keeps listening
#   Say  “Ok continue story”   -> the book resumes (or just fall silent a while)
#   Press  q ENTER             -> quit
#
# Wake/resume words are configurable: NARRATOR_WAKE_WORD / NARRATOR_RESUME_WORD.
# Cloud backend only (Deepgram streaming). Keys load from ./.env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_PY="${NARRATOR_VOICE_PYTHON:-/home/x0zby2/projects/audiobook-gen/.venv/bin/python}"
export NARRATOR_BACKEND="${NARRATOR_BACKEND:-cloud}"

if [ ! -f "$ROOT/data/book_manifest.json" ]; then
  echo "building book manifest ..."
  (cd "$ROOT/src" && "$VOICE_PY" -m narrator.corpus)
fi

# Ollama is best-effort here (only for semantic-memory embeddings).
source "$ROOT/scripts/_ensure_ollama.sh"; ensure_ollama "$ROOT" || true
cd "$ROOT/src"
exec "$VOICE_PY" -m narrator.app --wake "$@"
