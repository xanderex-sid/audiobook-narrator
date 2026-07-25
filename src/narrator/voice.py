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

    def listen(self, should_stop=None, on_start=None) -> tuple[str, bool]:
        """Capture one utterance from the mic and transcribe it (local VAD).

        Backend-agnostic entry point used by the hands-free loop. Returns
        (text, ok); ok=False for silence / a rejected hallucination (WS-7).
        """
        from . import paudio

        audio = paudio.record_utterance(should_stop=should_stop, on_start=on_start)
        if should_stop is not None and should_stop():
            return "", False
        return self.transcribe_checked(audio)


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


# ── Deepgram cloud providers (streaming STT + streaming TTS) ──────────────────
_DG_STT_WS = "wss://api.deepgram.com/v1/listen"
_DG_STT_HTTP = "https://api.deepgram.com/v1/listen"
_DG_TTS_HTTP = "https://api.deepgram.com/v1/speak"


def _keyterms() -> str:
    """`&keyterm=` query fragment boosting the wake/resume words (Nova-3 feature)."""
    import urllib.parse

    terms = list(dict.fromkeys(config.WAKE_WORDS + config.RESUME_WORDS))
    return "".join(f"&keyterm={urllib.parse.quote(t)}" for t in terms if t)


def _pcm16_wav_bytes(audio, sr: int) -> bytes:
    """Pack a float32 mono array into an in-memory 16-bit PCM wav."""
    import io
    import wave

    import numpy as np

    pcm = (np.clip(np.asarray(audio, dtype="float32"), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class DeepgramSTT:
    """Deepgram Nova STT. `listen()` streams live over a websocket; the buffered
    `transcribe*` paths use the prerecorded API (for --voice and alignment)."""

    _SR = 16000

    def __init__(self):
        self._auth = {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}

    # -- prerecorded (buffered) --------------------------------------------------
    def _listen_bytes(self, data: bytes, content_type: str, words: bool = False) -> dict:
        import requests

        params = {
            "model": config.DEEPGRAM_STT_MODEL,
            "smart_format": "true",
            "punctuate": "true",
            "language": "en",
        }
        r = requests.post(
            _DG_STT_HTTP, params=params, data=data,
            headers={**self._auth, "Content-Type": content_type}, timeout=60,
        )
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _best(j: dict) -> dict:
        try:
            return j["results"]["channels"][0]["alternatives"][0]
        except (KeyError, IndexError):
            return {}

    def transcribe(self, audio) -> str:
        text, _ok = self.transcribe_checked(audio)
        return text

    def transcribe_checked(self, audio) -> tuple[str, bool]:
        import numpy as np

        if isinstance(audio, (str, Path)):
            data, ct = Path(audio).read_bytes(), "audio/wav"
        else:
            arr = np.asarray(audio, dtype="float32")
            if arr.size < 1600 or float(np.sqrt(np.mean(arr ** 2))) < 0.006:
                return "", False  # near-silence (WS-7)
            data, ct = _pcm16_wav_bytes(arr, self._SR), "audio/wav"
        try:
            alt = self._best(self._listen_bytes(data, ct))
        except Exception:
            return "", False
        text = (alt.get("transcript") or "").strip()
        return self._filter(text, alt.get("confidence"))

    def transcribe_words(self, wav_path) -> list[tuple[str, float, float]]:
        """Word-level timestamps for forced alignment (WS-4)."""
        try:
            alt = self._best(self._listen_bytes(Path(wav_path).read_bytes(), "audio/wav", words=True))
        except Exception:
            return []
        return [(w.get("word", ""), float(w.get("start", 0.0)), float(w.get("end", 0.0)))
                for w in (alt.get("words") or [])]

    @staticmethod
    def _filter(text: str, confidence) -> tuple[str, bool]:
        if not text:
            return "", False
        low = text.lower().strip()
        if low in _HALLUCINATIONS and (confidence is not None and confidence < 0.5):
            return "", False
        return text, True

    # -- streaming (live websocket) ---------------------------------------------
    def listen(self, should_stop=None, on_start=None) -> tuple[str, bool]:
        """Stream mic audio to Deepgram live and return the final transcript.

        Deepgram's own endpointing/VAD marks end-of-utterance, so this returns as
        soon as you stop talking. A keypress (should_stop) aborts immediately and
        falls through to whatever was heard. Falls back to the buffered path if the
        socket can't be opened.
        """
        import json as _json
        import threading

        from . import paudio

        try:
            from websockets.sync.client import connect
        except Exception:
            audio = paudio.record_utterance(should_stop=should_stop, on_start=on_start)
            return self.transcribe_checked(audio)

        # interim_results MUST be true for utterance_end_ms/vad endpointing. We still
        # only accumulate final (is_final) transcripts below, so interims are ignored.
        params = (
            f"model={config.DEEPGRAM_STT_MODEL}&encoding=linear16&sample_rate={self._SR}"
            "&channels=1&language=en&punctuate=true&smart_format=true"
            "&interim_results=true&vad_events=true&endpointing=300&utterance_end_ms=1000"
        )
        try:
            ws = connect(f"{_DG_STT_WS}?{params}", additional_headers=self._auth,
                         open_timeout=8, close_timeout=2)
        except Exception:
            audio = paudio.record_utterance(should_stop=should_stop, on_start=on_start)
            return self.transcribe_checked(audio)

        parts: list[str] = []
        done = threading.Event()
        started = threading.Event()

        def _stop() -> bool:
            return done.is_set() or (should_stop is not None and should_stop())

        def sender():
            for frame in paudio.mic_frames(should_stop=_stop):
                try:
                    ws.send(frame)
                except Exception:
                    break
            try:
                ws.send(_json.dumps({"type": "CloseStream"}))
            except Exception:
                pass

        t = threading.Thread(target=sender, daemon=True)
        t.start()
        try:
            while not (should_stop is not None and should_stop()):
                try:
                    msg = ws.recv(timeout=0.3)
                except TimeoutError:
                    if done.is_set():
                        break
                    continue
                except Exception:
                    break
                try:
                    d = _json.loads(msg)
                except Exception:
                    continue
                typ = d.get("type")
                if typ == "SpeechStarted":
                    if on_start is not None and not started.is_set():
                        started.set()
                        on_start()
                    continue
                if typ == "UtteranceEnd":
                    done.set()
                    break
                alt = (d.get("channel", {}).get("alternatives") or [{}])[0]
                tr = (alt.get("transcript") or "").strip()
                if tr and d.get("is_final"):
                    parts.append(tr)
                    if d.get("speech_final"):
                        done.set()
                        break
        finally:
            done.set()
            try:
                ws.close()
            except Exception:
                pass

        if should_stop is not None and should_stop():
            return "", False
        return self._filter(" ".join(p for p in parts if p).strip(), None)


class DeepgramTTS:
    """Deepgram Aura TTS. `stream_speak()` plays audio progressively as it streams."""

    def __init__(self):
        self.sr = 24000
        self._auth = {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}

    def _url(self) -> str:
        return (f"{_DG_TTS_HTTP}?model={config.DEEPGRAM_TTS_MODEL}"
                f"&encoding=linear16&sample_rate={self.sr}")

    def synth(self, text: str):
        """Return (float32 mono numpy audio, sample_rate)."""
        import numpy as np
        import requests

        r = requests.post(self._url(), json={"text": text},
                          headers={**self._auth, "Content-Type": "application/json"}, timeout=60)
        r.raise_for_status()
        pcm = np.frombuffer(r.content, dtype="<i2").astype("float32") / 32768.0
        return pcm, self.sr

    def to_wav(self, text: str, path: str | Path) -> Path:
        import soundfile as sf

        audio, sr = self.synth(text)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, sr)
        return Path(path)

    def stream_speak(self, text: str) -> bool:
        """Synthesize and play progressively — first audio without waiting for the
        whole clip. Returns True if it played."""
        import requests

        from . import paudio

        if not paudio.available():
            return False
        sink = paudio.PcmSink(self.sr, 1)
        try:
            with requests.post(self._url(), json={"text": text},
                               headers={**self._auth, "Content-Type": "application/json"},
                               stream=True, timeout=60) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=4096):
                    if chunk:
                        sink.write(chunk)
        except Exception:
            sink.close()
            return False
        sink.close()
        return True


class DeepgramStream:
    """A persistent Deepgram live-STT stream for hands-free WAKE mode.

    Unlike `DeepgramSTT.listen()` (one utterance), this keeps a single socket open
    for the whole session, continuously pumping the mic, and exposes an event queue:

        {"type": "transcript",   "text": <final text>, "final": <speech_final?>}
        {"type": "utterance_end"}
        {"type": "speech_started"}

    The caller runs the PLAYING<->CONVERSING state machine off these events —
    matching the wake phrase while the book plays, then the question + continue
    phrase while conversing. Headphones assumed (mic never hears the book/TTS).
    """

    _SR = 16000

    def __init__(self):
        self._auth = {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}
        self._ws = None
        self._q = None
        self._stop = None
        self._dead = None

    def start(self) -> bool:
        import queue
        import threading

        from . import paudio

        try:
            from websockets.sync.client import connect
        except Exception:
            return False

        params = (
            f"model={config.DEEPGRAM_STT_MODEL}&encoding=linear16&sample_rate={self._SR}"
            "&channels=1&language=en&punctuate=true&smart_format=true"
            "&interim_results=true&vad_events=true&endpointing=300&utterance_end_ms=1000"
        )
        params += _keyterms()  # boost the wake/resume words so STT hears them reliably
        try:
            self._ws = connect(f"{_DG_STT_WS}?{params}", additional_headers=self._auth,
                               open_timeout=8, close_timeout=2)
        except Exception:
            self._ws = None
            return False

        self._q = queue.Queue()
        self._stop = threading.Event()
        self._dead = threading.Event()

        def sender():
            for frame in paudio.mic_frames(should_stop=self._stop.is_set):
                if self._stop.is_set():
                    break
                try:
                    self._ws.send(frame)
                except Exception:
                    break

        def reader():
            import json as _json

            while not self._stop.is_set():
                try:
                    msg = self._ws.recv(timeout=0.3)
                except TimeoutError:
                    continue
                except Exception:
                    break
                try:
                    d = _json.loads(msg)
                except Exception:
                    continue
                typ = d.get("type")
                if typ == "UtteranceEnd":
                    self._q.put({"type": "utterance_end"})
                    continue
                if typ == "SpeechStarted":
                    self._q.put({"type": "speech_started"})
                    continue
                alt = (d.get("channel", {}).get("alternatives") or [{}])[0]
                tr = (alt.get("transcript") or "").strip()
                if tr and d.get("is_final"):
                    self._q.put({"type": "transcript", "text": tr, "final": bool(d.get("speech_final"))})
            self._dead.set()

        for fn in (sender, reader):
            threading.Thread(target=fn, daemon=True).start()
        return True

    def get(self, timeout: float = 0.2):
        import queue

        if self._q is None:
            return None
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def alive(self) -> bool:
        return self._ws is not None and self._dead is not None and not self._dead.is_set()

    def stop(self):
        if self._stop:
            self._stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass
        self._ws = None


# ── provider factories (backend-selected) ─────────────────────────────────────
def make_stt():
    """STT provider for the active backend (Deepgram cloud or faster-whisper local)."""
    return DeepgramSTT() if config.BACKEND == "cloud" else STT()


def make_tts():
    """TTS provider for the active backend (Deepgram Aura cloud or Kokoro local)."""
    return DeepgramTTS() if config.BACKEND == "cloud" else TTS()


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
