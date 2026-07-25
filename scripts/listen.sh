#!/usr/bin/env bash
# THE main experience — start the audiobook from the beginning and talk to it by voice.
#
#   ENTER  = pause the book and start talking (then just SPEAK — no keys between turns)
#   ENTER  = stop talking and resume the book
#   q ENTER= quit
#
# Exactly two ENTERs per talking session, voice-only in between. Use headphones.
#
# Options are passed through, e.g.:
#   scripts/listen.sh --resume        # resume where you left off instead of the start
#   scripts/listen.sh --chapter 2     # start at a specific chapter
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_PY="${NARRATOR_VOICE_PYTHON:-/home/x0zby2/projects/audiobook-gen/.venv/bin/python}"
# Backend defaults to CLOUD: OpenAI gpt-5.4-mini (brain) + Deepgram (voice).
# API keys load automatically from ./.env. To run fully local instead:
#   NARRATOR_BACKEND=local ./scripts/listen.sh
BACKEND="${NARRATOR_BACKEND:-cloud}"

# Ensure the manifest exists (auto-build if missing).
if [ ! -f "$ROOT/data/book_manifest.json" ]; then
  echo "building book manifest ..."
  (cd "$ROOT/src" && "$VOICE_PY" -m narrator.corpus)
fi

# Ollama is required only for LOCAL backend; for CLOUD it's used (best-effort) just
# for semantic-memory embeddings, so never block the demo on it.
if [ "$BACKEND" = "local" ]; then
  source "$ROOT/scripts/_ensure_ollama.sh"; ensure_ollama "$ROOT"
else
  source "$ROOT/scripts/_ensure_ollama.sh"; ensure_ollama "$ROOT" || true
fi
cd "$ROOT/src"
exec "$VOICE_PY" -m narrator.app --listen "$@"
