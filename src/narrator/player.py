"""B1 (audio backend) — AudioPlayer.

Plays chapter audio and, crucially, tracks the *live playback position* by wall
clock so the spoiler gate and resume logic work whether or not a speaker device
is present. If `sounddevice` is available the audio is actually played; if not,
the player runs "silent" (clock only) so the whole control loop is still
exercisable headless.

Kept separate from playback.py so the pure position/navigation helpers there stay
dependency-free for the text CLI.
"""
from __future__ import annotations

import time

from . import playback
from .position import PlaybackPosition


class AudioPlayer:
    def __init__(self, manifest: dict, start: PlaybackPosition, audio: bool = True):
        self.manifest = manifest
        self.pos = playback.clamp(manifest, start)
        self.playing = False
        self._base = self.pos.position_sec     # position when the current play span started
        self._t0: float | None = None          # wall clock when it started
        self._audio = False
        self._book = None
        if audio:
            from . import paudio

            if paudio.available():
                self._audio = True
                self._book = paudio.BookStream()

    @property
    def audio_available(self) -> bool:
        return bool(self._audio)

    def _dur(self, idx: int) -> float:
        for ch in self.manifest["chapters"]:
            if ch["index"] == idx:
                return float(ch["duration_sec"])
        return 0.0

    def _audio_path(self, idx: int) -> str:
        for ch in self.manifest["chapters"]:
            if ch["index"] == idx:
                return ch["audio_path"]
        return ""

    def _audio_start(self, idx: int) -> float:
        """Offset of this chapter inside its wav (nonzero for single-file books)."""
        for ch in self.manifest["chapters"]:
            if ch["index"] == idx:
                return float(ch.get("audio_start_sec", 0.0))
        return 0.0

    def _start_audio(self):
        if not self._audio or self._book is None:
            return
        idx = self.pos.chapter_index
        self._book.start(self._audio_path(idx), self._audio_start(idx) + self._base)

    def _stop_audio(self):
        if self._book is not None:
            self._book.stop()

    # ── position / transport ──────────────────────────────────────────────────
    def position(self) -> PlaybackPosition:
        """Current position, advanced by wall clock while playing (auto-rolls chapters)."""
        if not self.playing or self._t0 is None:
            return PlaybackPosition(self.pos.chapter_index, self._base)
        elapsed = time.monotonic() - self._t0
        sec = self._base + elapsed
        idx = self.pos.chapter_index
        # roll forward across chapters if we ran past the end
        while sec > self._dur(idx) and idx < self.manifest["n_chapters"]:
            sec -= self._dur(idx)
            idx += 1
        sec = min(sec, self._dur(idx))
        self.pos = PlaybackPosition(idx, sec)
        return self.pos

    def play(self):
        self.playing = True
        self._t0 = time.monotonic()
        self._base = self.pos.position_sec
        self._start_audio()

    def pause(self) -> PlaybackPosition:
        cur = self.position()
        self.playing = False
        self._base = cur.position_sec
        self._t0 = None
        self._stop_audio()
        return cur

    def resume(self):
        self.play()

    def seek_to(self, pos: PlaybackPosition):
        self.pos = playback.clamp(self.manifest, pos)
        self._base = self.pos.position_sec
        if self.playing:
            self._t0 = time.monotonic()
            self._start_audio()
