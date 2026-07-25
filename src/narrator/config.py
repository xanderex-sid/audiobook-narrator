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

# ── Speech models (local, copied into the project) ───────────────────────────
# Point HuggingFace loaders at the in-project cache so STT/TTS load offline and
# nothing depends on the global ~/.cache. voice.py sets these into os.environ.
SPEECH_HF_HOME = SPEECH_MODELS_DIR / "hf"
WHISPER_REPO = "Systran/faster-whisper-base.en"
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_VOICE = os.environ.get("NARRATOR_VOICE", "af_heart")
KOKORO_LANG = "a"  # American English

# Devices: STT defaults to CPU int8 (base.en is tiny -> fast and rock-solid,
# avoids cuDNN/GPU fragility). TTS tries CUDA then falls back to CPU.
STT_DEVICE = os.environ.get("NARRATOR_STT_DEVICE", "cpu")
STT_COMPUTE = os.environ.get("NARRATOR_STT_COMPUTE", "int8")
TTS_DEVICE = os.environ.get("NARRATOR_TTS_DEVICE", "auto")  # auto|cuda|cpu

# ── Book structure ───────────────────────────────────────────────────────────
BOOK_TITLE = "The Monkey's Paw"
