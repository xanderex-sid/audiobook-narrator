#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/_ensure_ollama.sh"; ensure_ollama "$ROOT"
cd "$ROOT/src"; exec python3 -m narrator.app --demo
