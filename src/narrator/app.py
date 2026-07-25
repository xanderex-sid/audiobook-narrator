"""B8 (full loop) — the complete narrator: audiobook + "Hey Narrator" + resume.

State machine:
    PLAYING  --("Hey Narrator")-->  CONVERSING  --("Ok Continue Story")-->  PLAYING

- On WAKE: pause the book, snapshot the exact timestamp, start a conversation
  whose spoiler gate is tied to that position.
- While CONVERSING: answer questions (spoiler-safe), and navigation commands
  update where playback will resume.
- On CONTINUE: brief "Okay", a 2-second pause, then resume the audiobook at the
  (possibly updated) position.

Modes:
    python -m narrator.app --demo               # scripted, fully self-contained
    python -m narrator.app --text --chapter 2   # type to drive it live
    python -m narrator.app --voice --chapter 2  # microphone (needs PortAudio)
"""
from __future__ import annotations

import time

from . import control, corpus, llm
from .control import CONTINUE, QUESTION, QUIT, WAKE
from .orchestrator import NarratorSession
from .player import AudioPlayer
from .position import PlaybackPosition

_RESUME_PAUSE_SEC = 2.0


class NarratorApp:
    def __init__(self, manifest, start_pos, speak=None, audio=True):
        self.manifest = manifest
        self.player = AudioPlayer(manifest, start_pos, audio=audio)
        self.speak = speak or (lambda t: print(f"  narrator> {t}"))
        self.session: NarratorSession | None = None

    def _pos_str(self, pos: PlaybackPosition) -> str:
        title = next(c["title"] for c in self.manifest["chapters"] if c["index"] == pos.chapter_index)
        return f"{title} @ {pos.position_sec:.0f}s"

    # ── PLAYING <-> CONVERSING core, driven by a push event iterator ──────────
    def drive(self, event_iter):
        self.player.play()
        print(f"▶  playing — {self._pos_str(self.player.position())}")
        state = "PLAYING"

        for ev in event_iter:
            if ev.kind == QUIT:
                break

            if state == "PLAYING":
                if ev.kind == WAKE:
                    pos = self.player.pause()
                    self.session = NarratorSession(self.manifest, pos)
                    print(f'⏸  “Hey Narrator” — paused at {self._pos_str(pos)}')
                    print(f"   {self.session.position_line()}")

                    state = "CONVERSING"
                continue

            # CONVERSING
            if ev.kind == CONTINUE:
                self.speak("Okay.")
                time.sleep(_RESUME_PAUSE_SEC)
                resume_pos = self.session.pos
                self.session.end()
                self.session = None
                self.player.seek_to(resume_pos)
                self.player.resume()
                print(f'▶  “Ok Continue Story” — resumed at {self._pos_str(resume_pos)}')
                state = "PLAYING"
            elif ev.kind == QUESTION:
                print(f"  you> {ev.text}")
                resp, note = self.session.handle(ev.text)
                tag = "  [spoiler revealed]" if resp.spoiler_used else ""
                self.speak(resp.speech_text + tag)
                if note:
                    print(f"   {note}  →  resume now set to {self._pos_str(self.session.pos)}")
            # WAKE while conversing: ignore

        if self.session:
            self.session.end()
        self.player.pause()
        print("■  stopped.")

    # ── live loop: push-to-talk (Enter → speak → Enter), spoken answers ───────
    def run_voice(self, stt):
        import pathlib
        import tempfile

        from . import paudio

        self.player.play()
        print(f"\n▶  playing the audiobook — {self._pos_str(self.player.position())}")
        print("   CONTROLS while playing:  [Enter] talk to the narrator   ·   q [Enter] quit")
        print("   (you can also type a question instead of pressing Enter)\n")
        qn = 0

        while True:
            try:
                cmd = input("🎧 [Enter]=talk  q=quit > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd.lower() == "q":
                break

            pos = self.player.pause()
            self.session = NarratorSession(self.manifest, pos)
            print(f"⏸  “Hey Narrator” — paused at {self._pos_str(pos)}")
            first_typed = cmd if cmd else None  # allow typing a question directly
            if not first_typed:
                self.speak("Yes?")

            while True:
                if first_typed:
                    text, first_typed = first_typed, None
                else:
                    try:
                        sub = input("🎙️  [Enter]=record  ·  type a question  ·  c=continue book  ·  q=quit > ").strip()
                    except (EOFError, KeyboardInterrupt):
                        sub = "c"
                    if sub.lower() == "c":
                        self.speak("Okay.")
                        time.sleep(_RESUME_PAUSE_SEC)
                        self._resume()
                        break
                    if sub.lower() == "q":
                        self._resume()
                        if self.session:
                            self.session.end()
                        self.player.pause()
                        print("■  stopped.")
                        return
                    if sub:
                        text = sub  # typed question
                    else:
                        wav = pathlib.Path(tempfile.gettempdir()) / f"narrator_q_{qn}.wav"
                        qn += 1
                        rec = paudio.start_recording(wav)
                        try:
                            input("   🔴 recording… speak, then press <Enter> to send > ")
                        except (EOFError, KeyboardInterrupt):
                            pass
                        paudio.stop_recording(rec)
                        text = stt.transcribe(wav).strip()

                if not text:
                    print("   (didn't catch that — try again, or 'c' to continue)")
                    continue
                print(f"  you> {text}")
                if control.classify_phrase(text) == CONTINUE:
                    self.speak("Okay.")
                    time.sleep(_RESUME_PAUSE_SEC)
                    self._resume()
                    break
                resp, note = self.session.handle(text)
                tag = "  [spoiler revealed]" if resp.spoiler_used else ""
                self.speak(resp.speech_text + tag)
                if note:
                    print(f"   {note}  →  resume now set to {self._pos_str(self.session.pos)}")

        if self.session:
            self.session.end()
        self.player.pause()
        print("■  stopped.")

    def _resume(self):
        resume_pos = self.session.pos
        self.session.end()
        self.session = None
        self.player.seek_to(resume_pos)
        self.player.resume()
        print(f'▶  resumed at {self._pos_str(self.player.position())} — press <Enter> to talk again')


# ── scripted demo (verifiable, no hardware) ──────────────────────────────────
def _demo(manifest) -> int:
    # Start ~1.5s before a meaningful moment in Chapter II, so the paused
    # timestamp reflects live playback tracking.
    ch2_dur = next(c["duration_sec"] for c in manifest["chapters"] if c["index"] == 2)
    start = PlaybackPosition(2, min(60.0, ch2_dur * 0.4))
    events = [
        control.Event(WAKE, at_sec=1.5),
        control.Event(QUESTION, "Who is Herbert, and remind me what the first wish was?"),
        control.Event(QUESTION, "What is the third wish and how does it all end?"),  # spoiler guard
        control.Event(QUESTION, "Okay, skip ahead to chapter 3."),                  # nav -> resume moves
        control.Event(CONTINUE),
        control.Event(QUIT),
    ]
    app = NarratorApp(manifest, start, audio=True)
    print(f"(audio device available: {app.player.audio_available})\n")
    app.drive(control.ScriptedController(events, speed=1.0).events_iter())
    return 0


def _make_voice_speak():
    from . import voice

    tts = voice.TTS()

    def speak(text: str):
        print(f"  narrator> {text}")
        audio, sr = tts.synth(text)
        voice.play(audio, sr)

    return speak


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Audiobook Narrator — full loop")
    ap.add_argument("--demo", action="store_true", help="run the scripted demo")
    ap.add_argument("--text", action="store_true", help="drive live by typing")
    ap.add_argument("--voice", action="store_true", help="drive by microphone (needs PortAudio)")
    ap.add_argument("--chapter", type=int, default=2)
    ap.add_argument("--offset", type=float, default=0.4, help="fraction 0..1 into the chapter")
    ap.add_argument("--no-audio", action="store_true", help="don't open a speaker device")
    args = ap.parse_args(argv)

    manifest = corpus.load_manifest()
    if not llm.is_up() or not llm.model_available():
        print("! Local LLM not ready — run scripts/serve.sh and pull the model.")
        return 2

    if args.demo:
        return _demo(manifest)

    dur = next(c["duration_sec"] for c in manifest["chapters"] if c["index"] == args.chapter)
    start = PlaybackPosition(args.chapter, dur * args.offset)

    if args.voice:
        from . import voice

        app = NarratorApp(manifest, start, speak=_make_voice_speak(), audio=not args.no_audio)
        app.run_voice(voice.STT())
        return 0

    # default: text-driven live loop
    speak = _make_voice_speak() if False else None
    app = NarratorApp(manifest, start, speak=speak, audio=not args.no_audio)
    print('Type “hey narrator” to interrupt, ask questions, then “ok continue story”. :quit to exit.')
    app.drive(control.TextController().events_iter())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
