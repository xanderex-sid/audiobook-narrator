# Audiobook Narrator 🎧🗣️

A **fully local, free** voice companion for an audiobook. Listen to the book; say
**"Hey Narrator"** to pause and ask anything — characters, recaps, meanings, "where
are we", spoilers on demand — then **"Ok Continue Story"** to resume from the right
spot. Answers are **spoiler-gated** to how far you've actually listened.

No cloud. Everything (LLM, speech models, data) runs on this machine and lives inside
this project.

## Stack (all local)
| Part | What |
|---|---|
| Brain | **Qwen2.5-7B-Instruct** via **Ollama** (model store: `models/ollama/`) |
| Speech-to-text | **faster-whisper** `base.en` (`models/speech/`) |
| Text-to-speech | **Kokoro-82M** (`models/speech/`) |
| GPU | NVIDIA RTX 5070 Ti (CUDA) — brain runs on GPU; STT on CPU by default |

> **Not in this repo:** the model weights (`models/`, ~4.9 GB) and the audiobook
> `.wav` files (`audiobook_chapters/chapters/`, ~112 MB) are **git-ignored**. Clone
> the repo, then fetch them with the steps below. Everything still lives *inside* the
> project directory — nothing depends on a global cache.

## Requirements
- **Linux / WSL2** (audio path here targets WSLg PulseAudio — see Notes).
- **Ollama** installed userspace (`~/.local/bin/ollama`), for the brain.
- **Python 3** for the text brain (stdlib only — no packages needed).
- A **Python < 3.13 environment** for the voice components (STT/TTS), because
  `kokoro` 0.9.4 declares `requires-python < 3.13`. Install into it:
  `pip install faster-whisper kokoro soundfile numpy torch` (see `requirements.txt`).
  Point the voice scripts at it via `NARRATOR_VOICE_PYTHON=/path/to/venv/bin/python`
  (the default in `scripts/voice.sh` is a machine-specific path — change it).
- **ffmpeg** + PulseAudio client tools (`paplay`/`pacat`/`parec`/`parecord`) for live
  audio; e.g. `conda install -c conda-forge pulseaudio` and a static `ffmpeg`.

## Getting the models
All weights live under `models/` (git-ignored). From the project root:

**1. Brain — Qwen2.5-7B-Instruct (Ollama), stored in `models/ollama/`:**
```bash
# install Ollama userspace if you don't have it:
curl -fsSL https://ollama.com/install.sh | sh      # installs to ~/.local

# pull the model INTO the in-project store (note OLLAMA_MODELS):
./scripts/serve.sh &                               # starts serve with the right store
OLLAMA_MODELS="$PWD/models/ollama" ~/.local/bin/ollama pull qwen2.5:7b-instruct
# quality upgrade (optional): ... pull qwen2.5:14b-instruct  (fits 12 GB VRAM)
```

**2. Speech — faster-whisper + Kokoro, stored in `models/speech/hf/`:**
```bash
pip install "huggingface_hub[cli]"
export HF_HOME="$PWD/models/speech/hf"             # in-project HF cache
huggingface-cli download Systran/faster-whisper-base.en
huggingface-cli download hexgrad/Kokoro-82M
```
`voice.py` loads these **offline** from `models/speech/hf` (it sets `HF_HOME` +
`HF_HUB_OFFLINE=1`), so this download is the only time the network is used.

## Getting the audiobook audio
The demo book is **W. W. Jacobs' _The Monkey's Paw_** (public domain); its text is in
`book.txt`, split into 3 sections. Put one narration `.wav` per section here:
```
audiobook_chapters/chapters/chapter_01.wav
audiobook_chapters/chapters/chapter_02.wav
audiobook_chapters/chapters/chapter_03.wav
```
The originals were rendered with Kokoro TTS from `book.txt`, but any clear narration of
the three sections works. After placing new files, **rebuild the manifest** so durations
match your audio (the committed `data/book_manifest.json` matches the original renders):
```bash
./scripts/narrate.sh --rebuild --chapter 1        # or: python -m narrator.corpus
```

## Quick start — the one command

```bash
./scripts/listen.sh          # 🎧 start the audiobook from the beginning and talk to it
```
Use **headphones**. Then:
- **ENTER** → pauses the book and starts a talking session — now just **speak**; the
  narrator listens automatically between answers (no keys per turn).
- **ENTER** again → stops talking and resumes the audiobook where you left off.
- **q** then ENTER → quit.

So exactly **two ENTERs per talking session**, voice-only in between. It auto-starts the
local LLM and builds the manifest if needed. `./scripts/listen.sh --resume` continues
where you last stopped instead of the start.

### Other entry points
```bash
./scripts/serve.sh                              # start the local LLM only (auto-started otherwise)
./scripts/narrate.sh --chapter 2 --offset 50%   # text brain — type questions, no audio
./scripts/align.sh                              # build exact spoiler cutoff + phrase-restart (one-time)
./scripts/preferences.sh --from-chapter 3       # write the generator preference JSON
./scripts/voice-selftest.sh --selftest          # TTS → STT round-trip check
./scripts/demo.sh                               # scripted full-loop demo (no mic)
```

## What it does (verified)
- **Hands-free talk** — press ENTER once, then converse entirely by voice (auto
  endpointing); ENTER again resumes. Two ENTERs per session, nothing else.
- **Tool-calling brain** — every capability (answer, recap, summarize, where-am-I,
  goto/skip/restart-from-a-quoted-line, set-resume) is an explicit LLM tool.
- **Spoiler gate with warn→confirm** — a spoiler question is never answered outright;
  the narrator warns ("that's a spoiler, you'll find it in Chapter N — sure?") and only
  reveals after you confirm. Consent is deterministic; three independent, leak-proof
  signals decide what's a spoiler. "I don't care, tell me the ending" reveals directly.
- **Exact position** — forced alignment (faster-whisper word timestamps) maps time↔text,
  so the cutoff lands on real sentence boundaries and "restart from '<line>'" is exact.
- **Any book** — auto-detects chapter markers (Roman/`Chapter N`/`N.`/markdown) and
  supports a single combined audiobook wav, not just this story.
- **Semantic memory** — recalls relevant things you asked in earlier sessions (local
  embeddings), and resumes where you left off.
- **Preference handoff** — infers your vocabulary/pacing preferences and writes JSON for
  a downstream audiobook generator (`data/preferences/`).
- **Fully local** — LLM, STT, TTS, embeddings all on-device.

## Architecture
Independent blocks in `src/narrator/`: `corpus`(B0) · `player`/`playback`(B1) ·
`control`(B2) · `voice`(B3) · `position`(B4, the spoiler gate) · `knowledge`(B5) ·
`agent`(B6, the brain) · `memory`(B7) · `orchestrator`+`app`(B8). See `goal.md` for the
full design.

## Notes
- **Live audio (WSL2):** PortAudio/ALSA don't see a device under WSLg, so audio goes
  through **WSLg's PulseAudio** — `paplay`/`parec`/`pacat` (from conda) + `ffmpeg` for
  seekable playback (`src/narrator/paudio.py`). One-time setup already done on this
  machine: `conda install -c conda-forge pulseaudio` and `~/.config/pulse/client.conf`
  with `enable-shm = no` (the WSLg fix). If audio ever errors, check
  `PULSE_SERVER=unix:/mnt/wslg/PulseServer` and that client.conf line.
- Voice components run on the GPU-proven interpreter at
  `NARRATOR_VOICE_PYTHON` (defaults to the sibling project's venv); the text brain runs
  on any Python 3 (stdlib only).
- Quality upgrade: `ollama pull qwen2.5:14b-instruct` and set `NARRATOR_LLM=qwen2.5:14b-instruct`.
