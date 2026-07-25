"""Central configuration and paths for the audiobook narrator.

Everything is resolved relative to the project root so the whole system is
self-contained inside this directory (models included).
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env into the environment (no override).

    Keeps API keys out of the code and out of every launch script — the app
    reads them straight from PROJECT_ROOT/.env. Existing env vars win, so an
    explicit `KEY=... ./scripts/listen.sh` still overrides the file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k:
            os.environ.setdefault(k, v)


_load_dotenv(PROJECT_ROOT / ".env")

BOOK_TXT = PROJECT_ROOT / "book.txt"
CHAPTERS_DIR = PROJECT_ROOT / "audiobook_chapters" / "chapters"

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "book_manifest.json"
SESSIONS_DIR = DATA_DIR / "sessions"
ALIGNMENT_DIR = DATA_DIR / "alignment"       # per-chapter word<->time maps (B4)
PREFERENCES_DIR = DATA_DIR / "preferences"   # listener preference JSON (WS-9)

MODELS_DIR = PROJECT_ROOT / "models"
OLLAMA_MODELS_DIR = MODELS_DIR / "ollama"       # OLLAMA_MODELS points here
SPEECH_MODELS_DIR = MODELS_DIR / "speech"        # copied Kokoro / faster-whisper

# ── Backend selection: cloud (OpenAI + Deepgram) or local (Ollama + whisper/Kokoro)
# `cloud` is the default now — it removes local inference latency. `local` keeps the
# fully-offline stack as a fallback. Toggle with $NARRATOR_BACKEND. The two layers
# (brain, voice) are swapped behind identical interfaces, so nothing else changes.
BACKEND = os.environ.get("NARRATOR_BACKEND", "cloud").lower()

# ── LLM brain ────────────────────────────────────────────────────────────────
# Cloud: OpenAI gpt-5.4-mini (fast, strong tool-calling). Local: Qwen2.5-7B via
# Ollama. Override the model with $NARRATOR_LLM either way.
_DEFAULT_LLM = "gpt-5.4-mini" if BACKEND == "cloud" else "qwen2.5:7b-instruct"
LLM_MODEL = os.environ.get("NARRATOR_LLM", _DEFAULT_LLM)
LLM_TEMPERATURE = 0.2          # low = reliable, consistent answers
LLM_TIMEOUT_S = 120

# OpenAI (cloud brain). Key comes from .env.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Deepgram (cloud voice: Nova STT + Aura TTS). Key comes from .env.
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_STT_MODEL = os.environ.get("NARRATOR_DG_STT", "nova-3")
DEEPGRAM_TTS_MODEL = os.environ.get("NARRATOR_DG_TTS", "aura-2-thalia-en")

# ── Wake-mode trigger words (--wake) ──────────────────────────────────────────
# Matched on the DISTINCTIVE keyword, not the full phrase, because STT reliably
# hears the content word ("narrator") but often drops the leading "hey" (-> "a").
# Pick single, uncommon words for best reliability; comma-separate to allow a few.
# These are also sent to Deepgram as `keyterm` boosts. Override via env.
WAKE_WORDS = [w.strip().lower() for w in
              os.environ.get("NARRATOR_WAKE_WORD", "friday").split(",") if w.strip()]
RESUME_WORDS = [w.strip().lower() for w in
                os.environ.get("NARRATOR_RESUME_WORD", "continue,resume").split(",") if w.strip()]
# In wake mode, finalize a spoken turn only after this many seconds of silence, so a
# mid-thought pause doesn't cut you off (any speech resets it). Override via env.
WAKE_SILENCE_SEC = float(os.environ.get("NARRATOR_WAKE_SILENCE", "1.5"))

# Fresh session: ignore all prior memory/history and don't persist this run. Set by
# `--fresh` (or $NARRATOR_FRESH). Non-destructive — saved data on disk is untouched.
FRESH = bool(os.environ.get("NARRATOR_FRESH", ""))

# ── Local Ollama (used for `local` backend, and always for memory embeddings) ──
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}"

# Embedding model for semantic cross-session memory (B7) stays LOCAL (nomic via
# Ollama) in both backends. If Ollama isn't up, memory falls back to recent summaries.
EMBED_MODEL = os.environ.get("NARRATOR_EMBED", "nomic-embed-text")

# ── Speech models (local, copied into the project) ───────────────────────────
# Point HuggingFace loaders at the in-project cache so STT/TTS load offline and
# nothing depends on the global ~/.cache. voice.py sets these into os.environ.
SPEECH_HF_HOME = SPEECH_MODELS_DIR / "hf"


def _default_whisper() -> str:
    """Prefer the more accurate small.en if it's been downloaded, else base.en (WS-6).

    small.en reduces name/word errors; base.en is the always-present fallback so a
    fresh clone still works. Override either with $NARRATOR_WHISPER.
    """
    env = os.environ.get("NARRATOR_WHISPER")
    if env:
        return env
    small = SPEECH_HF_HOME / "hub" / "models--Systran--faster-whisper-small.en"
    return "Systran/faster-whisper-small.en" if small.exists() else "Systran/faster-whisper-base.en"


WHISPER_REPO = _default_whisper()
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_VOICE = os.environ.get("NARRATOR_VOICE", "af_heart")
KOKORO_LANG = "a"  # American English

# Devices: STT defaults to CPU int8 (base.en is tiny -> fast and rock-solid,
# avoids cuDNN/GPU fragility). TTS tries CUDA then falls back to CPU.
STT_DEVICE = os.environ.get("NARRATOR_STT_DEVICE", "cpu")
STT_COMPUTE = os.environ.get("NARRATOR_STT_COMPUTE", "int8")
TTS_DEVICE = os.environ.get("NARRATOR_TTS_DEVICE", "auto")  # auto|cuda|cpu

# ── Book structure ───────────────────────────────────────────────────────────
# Title is NOT hardcoded to one story. Priority: $NARRATOR_TITLE env >
# data/book_meta.json "title" > first non-empty line of book.txt > "the book".
# (corpus.resolve_title implements this.) Kept here only as the final fallback.
BOOK_TITLE_FALLBACK = "the book"

# Single-file audiobook support (WS-3): if audiobook_chapters/chapters/ has a
# single wav (or $NARRATOR_AUDIO points at one file) and the text has no chapter
# markers, the whole book is treated as one chapter. An optional chapters.json
# ([{title, start_sec}]) overrides boundary detection.
SINGLE_AUDIO_ENV = os.environ.get("NARRATOR_AUDIO")  # optional explicit wav path
CHAPTERS_JSON = CHAPTERS_DIR / "chapters.json"        # optional boundary map
