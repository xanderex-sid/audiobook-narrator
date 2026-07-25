"""Central configuration and paths for the audiobook narrator.

Everything is resolved relative to the project root so the whole system is
self-contained inside this directory (models included).
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

# ── Local LLM (Ollama, OpenAI-compatible) ────────────────────────────────────
# The brain. Qwen2.5-7B-Instruct: reliable instruction-following, fits 12GB VRAM
# with room for the speech models. `ollama pull qwen2.5:14b-instruct` is a
# drop-in quality upgrade — just change LLM_MODEL.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}"
LLM_MODEL = os.environ.get("NARRATOR_LLM", "qwen2.5:7b-instruct")
LLM_TEMPERATURE = 0.2          # low = reliable, consistent answers
LLM_TIMEOUT_S = 120

# Local embedding model for semantic cross-session memory (B7). Pulled via
# `ollama pull nomic-embed-text`. If absent, memory falls back to recent summaries.
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
