"""B2 — Control / Wake + trigger sources.

Emits control events that drive the PLAYING <-> CONVERSING state machine:
  - WAKE      ("Hey Narrator")      -> pause the book, start talking
  - CONTINUE  ("Ok Continue Story") -> stop talking, resume the book
  - QUESTION  (any other utterance) -> a thing to answer while conversing
  - QUIT

Three interchangeable sources implement the same `events()` iterator:
  - ScriptedController : predefined events (used for the verifiable demo)
  - TextController     : typed lines (reliable live driver; phrases mirror voice)
  - VoiceController    : microphone + STT phrase-spotting (best-effort; needs mic)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

WAKE = "wake"
CONTINUE = "continue"
QUESTION = "question"
QUIT = "quit"

_WAKE_PHRASES = ("hey narrator", "hey, narrator", "hi narrator", "ok narrator")
_CONT_PHRASES = ("ok continue", "okay continue", "ok, continue", "continue story",
                 "continue the story", "resume the story", "continue playing")


def classify_phrase(text: str) -> str | None:
    t = " ".join(text.lower().split())
    if any(p in t for p in _WAKE_PHRASES):
        return WAKE
    if any(p in t for p in _CONT_PHRASES):
        return CONTINUE
    return None


# ── keyword-based wake detection (for --wake streaming) ────────────────────────
# STT reliably transcribes the distinctive content word but drops short leading
# words ("hey" -> "a"), so we match the KEYWORD (with a 1-edit fuzz), not the phrase.
def _norm(text: str) -> str:
    return "".join(c if c.isalnum() or c == " " else " " for c in text.lower()).split()


def _lev1(a: str, b: str) -> bool:
    """True if edit distance(a, b) <= 1 (cheap, tolerates one STT slip)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:  # one substitution
        return sum(x != y for x, y in zip(a, b)) == 1
    # one insertion/deletion
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
        elif skipped:
            return False
        else:
            skipped = True; j += 1
    return True


def matches_keyword(text: str, words) -> bool:
    """True if any trigger word appears in `text` (substring or a 1-edit token match)."""
    toks = _norm(text)
    joined = " ".join(toks)
    for w in words:
        if not w:
            continue
        if w in joined:                       # multi-word or exact substring
            return True
        if any(_lev1(tok, w) for tok in toks):  # single dropped/slipped word
            return True
    return False


@dataclass
class Event:
    kind: str
    text: str = ""
    at_sec: float | None = None   # scripted: simulate reaching this playback time first


class ScriptedController:
    """Deterministic event source for demos/tests. `speed` scales simulated waits."""

    def __init__(self, events: list[Event], speed: float = 0.0):
        self.events = events
        self.speed = speed

    def events_iter(self) -> Iterator[Event]:
        for ev in self.events:
            if self.speed and ev.at_sec:
                time.sleep(min(ev.at_sec * self.speed, 2.0))
            yield ev


class TextController:
    """Live, reliable driver: type to control. 'hey narrator' / 'ok continue story'."""

    def events_iter(self) -> Iterator[Event]:
        while True:
            try:
                line = input().strip()
            except (EOFError, KeyboardInterrupt):
                yield Event(QUIT)
                return
            if not line:
                continue
            if line in (":quit", ":q", "exit"):
                yield Event(QUIT)
                return
            kind = classify_phrase(line)
            if kind == WAKE:
                yield Event(WAKE)
            elif kind == CONTINUE:
                yield Event(CONTINUE)
            else:
                yield Event(QUESTION, text=line)


class VoiceController:
    """Microphone driver (best-effort; needs sounddevice + PortAudio).

    While PLAYING it listens in short windows for the wake phrase. While
    CONVERSING the app calls `listen_question()` to capture a full utterance and
    checks it for the continue phrase.
    """

    def __init__(self, stt):
        self.stt = stt

    def wait_for_wake(self) -> None:
        from . import voice

        while True:
            audio = voice.record_until_silence(max_sec=4.0, silence_sec=0.8)
            if classify_phrase(self.stt.transcribe(audio)) == WAKE:
                return

    def listen_question(self) -> Event:
        from . import voice

        audio = voice.record_until_silence(max_sec=15.0, silence_sec=1.2)
        text = self.stt.transcribe(audio)
        kind = classify_phrase(text)
        if kind == CONTINUE:
            return Event(CONTINUE)
        return Event(QUESTION, text=text)
