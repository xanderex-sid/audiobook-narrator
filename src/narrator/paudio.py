"""PulseAudio (WSLg) audio backend — reliable mic + speaker in WSL2.

PortAudio/ALSA don't see a device under WSLg, but WSLg's PulseAudio does
(RDPSink = speaker, RDPSource = mic). This module drives playback and capture
through the pulse client tools + ffmpeg, which is the path that actually works
here. All process-spawning stays in one place.

Requires (installed userspace): paplay/pacat/parec (conda), ffmpeg (~/.local/bin),
and ~/.config/pulse/client.conf with `enable-shm = no`.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

_CONDA_BIN = Path.home() / "miniconda3" / "bin"


def _tool(name: str) -> str:
    return shutil.which(name) or str(_CONDA_BIN / name)


PAPLAY = _tool("paplay")
PACAT = _tool("pacat")
PAREC = _tool("parec")
PARECORD = _tool("parecord")
FFMPEG = shutil.which("ffmpeg") or str(Path.home() / ".local" / "bin" / "ffmpeg")

# WSLg PulseAudio socket.
if not os.environ.get("PULSE_SERVER") and Path("/mnt/wslg/PulseServer").exists():
    os.environ["PULSE_SERVER"] = "unix:/mnt/wslg/PulseServer"


def available() -> bool:
    return Path(PAPLAY).exists() and os.environ.get("PULSE_SERVER") is not None


# ── playback ─────────────────────────────────────────────────────────────────
def play_wav(path: str | Path, block: bool = True) -> bool:
    try:
        p = subprocess.Popen([PAPLAY, str(path)])
        if block:
            p.wait()
        return True
    except Exception:
        return False


def play_array(audio, sr: int) -> bool:
    """Play a float32 numpy array by writing a temp wav and paplay-ing it."""
    try:
        import soundfile as sf

        tmp = Path(tempfile.gettempdir()) / "narrator_tts.wav"
        sf.write(str(tmp), audio, sr)
        return play_wav(tmp)
    except Exception:
        return False


# ── book playback with seek (ffmpeg -ss | pacat) ─────────────────────────────
class BookStream:
    """Plays a wav from an arbitrary offset; stop() halts it."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None

    def start(self, wav_path: str, offset_sec: float) -> bool:
        self.stop()
        # -nostdin: ffmpeg must NOT read the terminal (it would steal keystrokes
        # from the app's input() prompts). Also detach the pipeline's stdin.
        cmd = (
            f'{FFMPEG} -nostdin -v error -ss {max(0.0, offset_sec):.3f} -i "{wav_path}" '
            f"-f s16le -ar 44100 -ac 2 - | "
            f"{PACAT} --format=s16le --rate=44100 --channels=2"
        )
        try:
            self._proc = subprocess.Popen(
                cmd, shell=True, start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            self._proc = None
            return False

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        self._proc = None


# ── streaming sink: progressive linear16 playback (cloud TTS) ────────────────
class PcmSink:
    """Play raw little-endian PCM as it arrives by piping into pacat.

    Lets streamed TTS start speaking before the whole clip is synthesized:
    write() bytes as they come off the wire, close() to drain and finish.
    """

    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self._proc = subprocess.Popen(
            [PACAT, "--format=s16le", f"--rate={sample_rate}", f"--channels={channels}"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def write(self, data: bytes) -> None:
        if not data or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(data)
        except (BrokenPipeError, ValueError):
            pass

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=15)
        except Exception:
            try:
                self._proc.terminate()
            except Exception:
                pass


# ── streaming mic frames (cloud STT) ──────────────────────────────────────────
def mic_frames(should_stop=None, frame_ms: int = 50):
    """Yield raw s16le 16k mono frames from the mic until should_stop() is True.

    Feeds Deepgram's live STT socket; endpointing/VAD there decides when the
    utterance ends, so this just pumps audio and bails promptly on a keypress.
    """
    sr = 16000
    chunk = int(sr * frame_ms / 1000)
    bytes_per = chunk * 2
    proc = subprocess.Popen(
        [PAREC, "--format=s16le", f"--rate={sr}", "--channels=1", "--latency-msec=30"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        while True:
            if should_stop is not None and should_stop():
                break
            buf = proc.stdout.read(bytes_per)
            if not buf or len(buf) < bytes_per:
                break
            yield buf
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


# ── press-to-stop recording (reliable; no VAD guessing) ──────────────────────
def start_recording(path: str | Path) -> "subprocess.Popen":
    return subprocess.Popen(
        [PARECORD, "--channels=1", "--rate=16000", "--file-format=wav", str(path)],
        stderr=subprocess.DEVNULL,
    )


def stop_recording(proc: "subprocess.Popen") -> None:
    try:
        proc.send_signal(signal.SIGINT)  # parecord finalizes the wav trailer on SIGINT
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            pass


# ── microphone capture with silence detection (hands-free option) ────────────
def record_utterance(
    max_sec: float = 15.0,
    silence_sec: float = 1.0,
    start_timeout_sec: float = 8.0,
    threshold: float = 0.012,
    tail_pad_sec: float = 0.4,
    should_stop=None,
    on_start=None,
):
    """Capture one spoken utterance from the mic. Returns float32 mono 16k array.

    Hands-free: waits (up to start_timeout_sec) for speech, then records until
    `silence_sec` of quiet, keeping `tail_pad_sec` of trailing audio so the last
    word isn't clipped (WS-7).

    `should_stop()` is polled each 50ms frame; if it returns True the capture
    aborts and returns whatever was heard (used so a keypress can interrupt
    listening). `on_start()` fires once when speech is first detected.

    Returns an empty-ish array if nothing was said (or aborted before speech).
    """
    import numpy as np

    sr = 16000
    chunk = int(sr * 0.05)  # 50ms frames
    pad_frames = int(tail_pad_sec / 0.05)
    cmd = [PAREC, "--format=s16le", f"--rate={sr}", "--channels=1", "--latency-msec=30"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames: list = []
    speaking = False
    silent = 0.0
    waited = 0.0
    bytes_per = chunk * 2
    try:
        while True:
            if should_stop is not None and should_stop():
                break
            buf = proc.stdout.read(bytes_per)
            if not buf or len(buf) < bytes_per:
                break
            arr = np.frombuffer(buf, dtype="<i2").astype("float32") / 32768.0
            rms = float(np.sqrt(np.mean(arr**2)))
            if not speaking:
                waited += 0.05
                if rms >= threshold:
                    speaking = True
                    if on_start is not None:
                        on_start()
                    frames.append(arr)
                elif waited >= start_timeout_sec:
                    break
                continue
            frames.append(arr)
            silent = silent + 0.05 if rms < threshold else 0.0
            total = len(frames) * 0.05
            if silent >= silence_sec and total > 0.4:
                # keep a little trailing audio, then stop
                for _ in range(pad_frames):
                    extra = proc.stdout.read(bytes_per)
                    if not extra or len(extra) < bytes_per:
                        break
                    frames.append(np.frombuffer(extra, dtype="<i2").astype("float32") / 32768.0)
                break
            if total >= max_sec:
                break
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
    if not frames:
        return np.zeros(1, dtype="float32")
    return np.concatenate(frames)
