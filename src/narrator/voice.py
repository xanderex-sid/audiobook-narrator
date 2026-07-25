"""B3 — Voice I/O: STT (faster-whisper) + TTS (Kokoro), all local.

Models are loaded from the in-project copies under models/speech (HF_HOME is
pointed there and offline mode is forced), so nothing depends on the global
cache or the network.

Run this module directly to prove the pipeline:

    # full speech round-trip (TTS -> STT), no mic needed:
    python -m narrator.voice --selftest

    # speak the answer to a question end-to-end (question is synthesized as
    # speech, transcribed, answered, and the answer is synthesized):
    python -m narrator.voice --demo "Who is Sergeant-Major Morris?" --chapter 1

    # transcribe any wav:
    python -m narrator.voice --transcribe some.wav

Live mic capture / speaker playback use `sounddevice` if it's installed
(needs PortAudio); otherwise the file-based paths above work fully and the
CLI tells you how to enable live audio.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import config

# Point HF loaders at the in-project model copies, offline.
os.environ.setdefault("HF_HOME", str(config.SPEECH_HF_HOME))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_TTS_SR = 24000
_STT_SR = 16000


# ── model path resolution (offline, in-project) ──────────────────────────────
def _hf_snapshot(repo: str) -> Path:
    """Resolve a local HF snapshot dir (the one containing the model files)."""
    slug = "models--" + repo.replace("/", "--")
    snaps = config.SPEECH_HF_HOME / "hub" / slug / "snapshots"
    for d in sorted(snaps.glob("*")):
        if d.is_dir() and any(d.iterdir()):
            return d
    raise FileNotFoundError(f"No local snapshot for {repo} under {snaps}")


# Whisper's classic empty-audio hallucinations — rejected when they arrive alone
# on near-silent input (WS-7).
_HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching.", "thanks for watching",
    "you", "you.", ".", "bye.", "bye", "so", "okay.", "please subscribe.",
    "thank you for watching.", "i'm sorry.",
}


# ── STT ──────────────────────────────────────────────────────────────────────
class STT:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            path = str(_hf_snapshot(config.WHISPER_REPO))
            self._model = WhisperModel(
                path, device=config.STT_DEVICE, compute_type=config.STT_COMPUTE
            )
        return self._model

    def transcribe(self, audio) -> str:
        """audio: a wav path (str/Path) or a float32 mono 16k numpy array."""
        text, _ok = self.transcribe_checked(audio)
        return text

    def transcribe_checked(self, audio) -> tuple[str, bool]:
        """Transcribe and judge whether it's a real utterance (WS-7).

        Returns (text, ok). ok=False for near-silent input or a lone classic
        hallucination (e.g. "Thank you." on empty audio), so the caller can drop
        phantom questions instead of answering them.
        """
        import numpy as np

        model = self._load()
        src = str(audio) if isinstance(audio, (str, Path)) else audio

        # Energy gate for in-memory arrays: reject near-silence outright.
        if not isinstance(audio, (str, Path)):
            arr = np.asarray(audio, dtype="float32")
            if arr.size < 1600 or float(np.sqrt(np.mean(arr ** 2))) < 0.006:
                return "", False

        # vad_filter drops non-speech regions -> far fewer silence hallucinations.
        segments, _info = model.transcribe(
            src, language="en", beam_size=1,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
        )
        segs = list(segments)
        text = "".join(s.text for s in segs).strip()
        if not text:
            return "", False

        # Reject a lone classic hallucination or low-confidence single token.
        low = text.lower().strip()
        avg_logprob = sum(getattr(s, "avg_logprob", 0.0) for s in segs) / max(1, len(segs))
        no_speech = max((getattr(s, "no_speech_prob", 0.0) for s in segs), default=0.0)
        if low in _HALLUCINATIONS and (no_speech > 0.5 or avg_logprob < -0.8):
            return "", False
        if no_speech > 0.85:
            return "", False
        return text, True

    def transcribe_words(self, wav_path) -> list[tuple[str, float, float]]:
        """Word-level timestamps for forced alignment (WS-4): [(word, start, end), ...]."""
        model = self._load()
        segments, _info = model.transcribe(
            str(wav_path), language="en", beam_size=1, word_timestamps=True
        )
        words: list[tuple[str, float, float]] = []
        for seg in segments:
            for w in (seg.words or []):
                words.append((w.word, float(w.start), float(w.end)))
        return words


# ── TTS ──────────────────────────────────────────────────────────────────────
class TTS:
    def __init__(self):
        self._pipe = None
        self.sr = _TTS_SR

    def _device(self) -> str:
        if config.TTS_DEVICE in ("cuda", "cpu"):
            return config.TTS_DEVICE
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self):
        if self._pipe is None:
            from kokoro import KPipeline

            self._pipe = KPipeline(
                lang_code=config.KOKORO_LANG,
                repo_id=config.KOKORO_REPO,
                device=self._device(),
            )
        return self._pipe

    def synth(self, text: str):
        """Return (float32 mono numpy audio, sample_rate)."""
        import numpy as np

        pipe = self._load()
        chunks = []
        for _gs, _ps, audio in pipe(text, voice=config.KOKORO_VOICE):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(arr.astype(np.float32))
        if not chunks:
            return np.zeros(1, dtype="float32"), self.sr
        return np.concatenate(chunks), self.sr

    def to_wav(self, text: str, path: str | Path) -> Path:
        import soundfile as sf

        audio, sr = self.synth(text)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, sr)
        return Path(path)


# ── live audio (optional; needs sounddevice + PortAudio) ─────────────────────
def _sd():
    try:
        import sounddevice as sd  # noqa

        return sd
    except Exception:
        return None


def play(audio, sr: int) -> bool:
    """Play audio to the speaker. Returns True if it actually played."""
    from . import paudio

    if paudio.available() and paudio.play_array(audio, sr):
        return True
    sd = _sd()
    if sd is None:
        return False
    try:
        sd.play(audio, sr)
        sd.wait()
        return True
    except Exception:
        return False


def record_until_silence(max_sec: float = 15.0, silence_sec: float = 1.2):
    """Capture mic audio until a pause; returns float32 mono 16k numpy array."""
    import numpy as np

    sd = _sd()
    if sd is None:
        raise RuntimeError(
            "Live mic needs `sounddevice` + PortAudio. Install with "
            "`conda install -c conda-forge python-sounddevice portaudio`, or use "
            "--from-wav / --text instead."
        )
    frames, silent = [], 0.0
    block = 0.1
    with sd.InputStream(samplerate=_STT_SR, channels=1, dtype="float32") as stream:
        for _ in range(int(max_sec / block)):
            data, _ = stream.read(int(_STT_SR * block))
            frames.append(data.copy())
            rms = float(np.sqrt(np.mean(data**2)))
            silent = silent + block if rms < 0.01 else 0.0
            if silent >= silence_sec and len(frames) * block > 0.6:
                break
    return np.concatenate(frames).flatten() if frames else np.zeros(1, "float32")


# ── CLI: selftest / demo / transcribe ────────────────────────────────────────
def _selftest() -> int:
    import tempfile

    phrase = "Who is Sergeant Major Morris and what did he bring?"
    print(f"TTS  ->  synthesizing: {phrase!r}")
    tts = TTS()
    wav = Path(tempfile.gettempdir()) / "narrator_selftest.wav"
    tts.to_wav(phrase, wav)
    import soundfile as sf

    info = sf.info(str(wav))
    print(f"     wrote {wav.name}  ({info.duration:.1f}s @ {info.samplerate}Hz)")
    print("STT  ->  transcribing it back...")
    text = STT().transcribe(wav)
    print(f"     heard: {text!r}")
    ok = "morris" in text.lower()
    print("ROUND-TRIP:", "OK ✓" if ok else "MISMATCH ✗")
    return 0 if ok else 1


def _demo(text: str, chapter: int, offset_pct: float) -> int:
    from . import corpus
    from .orchestrator import NarratorSession
    from .position import PlaybackPosition

    manifest = corpus.load_manifest()
    dur = next(c["duration_sec"] for c in manifest["chapters"] if c["index"] == chapter)
    tts, stt = TTS(), STT()

    q_wav = config.DATA_DIR / "voice_demo_question.wav"
    a_wav = config.DATA_DIR / "voice_demo_answer.wav"

    print(f'[mic-sim] synthesizing the spoken question: "{text}"')
    tts.to_wav(text, q_wav)
    heard = stt.transcribe(q_wav)
    print(f"[STT]     transcribed question: {heard!r}")

    session = NarratorSession(manifest, PlaybackPosition(chapter, dur * offset_pct))
    print("[position]", session.position_line())
    resp, note = session.handle(heard or text)
    print(f"[narrator] {resp.speech_text}"
          + (" [spoiler revealed]" if resp.spoiler_used else ""))
    if note:
        print("[nav]     ", note)

    print("[TTS]     synthesizing the spoken answer...")
    tts.to_wav(resp.speech_text, a_wav)
    played = play_saved(a_wav)
    print(f"[audio]    answer saved to {a_wav}"
          + ("  (also played to speaker)" if played else "  (no speaker device; play the wav to hear it)"))
    session.end()
    return 0


def play_saved(wav: Path) -> bool:
    try:
        import soundfile as sf

        audio, sr = sf.read(str(wav), dtype="float32")
        return play(audio, sr)
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Voice I/O (STT + TTS), local")
    ap.add_argument("--selftest", action="store_true", help="TTS->STT round-trip")
    ap.add_argument("--demo", type=str, help="end-to-end: speak the answer to this question")
    ap.add_argument("--transcribe", type=str, help="transcribe a wav file")
    ap.add_argument("--chapter", type=int, default=1)
    ap.add_argument("--offset", type=float, default=1.0, help="fraction 0..1 into the chapter")
    args = ap.parse_args(argv)

    if args.transcribe:
        print(STT().transcribe(args.transcribe))
        return 0
    if args.demo:
        return _demo(args.demo, args.chapter, args.offset)
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
