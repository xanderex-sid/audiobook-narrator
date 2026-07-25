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
# Brain defaults to 7B (fast). For the 14B upgrade, run:
#   NARRATOR_LLM=qwen2.5:14b-instruct ./scripts/listen.sh

# Ensure the manifest exists (auto-build if missing).
if [ ! -f "$ROOT/data/book_manifest.json" ]; then
  echo "building book manifest ..."
  (cd "$ROOT/src" && "$VOICE_PY" -m narrator.corpus)
fi

source "$ROOT/scripts/_ensure_ollama.sh"; ensure_ollama "$ROOT"
cd "$ROOT/src"
exec "$VOICE_PY" -m narrator.app --listen "$@"
