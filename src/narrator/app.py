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

    # ── hands-free loop: 2 ENTERs per talk session, voice-only in between ──────
    def run_handsfree(self, stt, tts):
        """The full voice experience with minimal keys (WS-5).

        PLAYING: the book plays aloud.
          - press ENTER  -> pause and start a hands-free talking session
          - press q ENTER -> quit
        CONVERSING: talk by VOICE only — ask, hear the answer, ask again; the
        narrator auto-listens between turns (no keys).
          - press ENTER  -> stop talking and resume the book

        So exactly two ENTERs per talking session (one in, one out), plus q to quit.
        """
        import threading
        from . import paudio

        if not paudio.available():
            print("! No PulseAudio device — falling back to typed mode.")
            self.drive(control.TextController().events_iter())
            return

        enter_evt = threading.Event()
        quit_evt = threading.Event()

        def watch_stdin():
            while not quit_evt.is_set():
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    quit_evt.set()
                    return
                if line.strip().lower() in ("q", "quit", "exit"):
                    quit_evt.set()
                    return
                enter_evt.set()  # any other ENTER = toggle talk/resume

        threading.Thread(target=watch_stdin, daemon=True).start()

        self.player.play()
        print(f"\n▶  playing — {self._pos_str(self.player.position())}")
        print("   Press ENTER to talk (book pauses) · ENTER again to resume · q ENTER to quit")
        print("   While talking, just SPEAK — the narrator listens automatically between answers.\n")

        state = "PLAYING"
        while not quit_evt.is_set():
            if state == "PLAYING":
                if enter_evt.wait(timeout=0.2):
                    enter_evt.clear()
                    pos = self.player.pause()
                    self.session = NarratorSession(self.manifest, pos)
                    print(f"\n⏸  paused at {self._pos_str(pos)} — talking mode (speak; ENTER to resume)")
                    if self.session.memory_context:
                        print("   (remembering earlier sessions)")
                    self.speak("Yes?")
                    state = "CONVERSING"
                continue

            # CONVERSING — hands-free voice turns until ENTER or quit
            if enter_evt.is_set() or quit_evt.is_set():
                enter_evt.clear()
                self.speak("Okay.")
                time.sleep(_RESUME_PAUSE_SEC)
                self._resume_handsfree()
                state = "PLAYING"
                continue

            print("   🎙️  listening…", flush=True)
            audio = paudio.record_utterance(
                should_stop=lambda: enter_evt.is_set() or quit_evt.is_set(),
                on_start=lambda: print("   …hearing you", flush=True),
            )
            if enter_evt.is_set() or quit_evt.is_set():
                continue  # ENTER pressed mid-listen -> loop back, will resume
            text, ok = stt.transcribe_checked(audio)
            if not ok or not text:
                continue  # silence / hallucination -> just keep listening (WS-7)
            print(f"  you> {text}")
            resp, note = self.session.handle(text)
            tag = "  [spoiler revealed]" if resp.spoiler_used else ""
            self.speak(resp.speech_text + tag)
            if note:
                print(f"   {note}  →  resume now {self._pos_str(self.session.pos)}")

        if self.session:
            self.session.end()
        self.player.pause()
        print("\n■  stopped. Progress and notes saved.")

    def _resume_handsfree(self):
        resume_pos = self.session.pos
        self.session.end()
        self.session = None
        self.player.seek_to(resume_pos)
        self.player.resume()
        print(f"▶  resumed at {self._pos_str(self.player.position())} — ENTER to talk again")


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
    ap.add_argument("--listen", action="store_true",
                    help="hands-free voice: ENTER to talk, ENTER to resume, q to quit")
    ap.add_argument("--demo", action="store_true", help="run the scripted demo")
    ap.add_argument("--text", action="store_true", help="drive live by typing")
    ap.add_argument("--voice", action="store_true", help="push-to-talk voice (legacy)")
    ap.add_argument("--chapter", type=int, default=None, help="start chapter (default: from the beginning / resume)")
    ap.add_argument("--offset", type=float, default=None, help="fraction 0..1 into the chapter")
    ap.add_argument("--resume", action="store_true", help="resume where you left off instead of the start")
    ap.add_argument("--no-audio", action="store_true", help="don't open a speaker device")
    args = ap.parse_args(argv)

    manifest = corpus.load_manifest()
    if not llm.is_up() or not llm.model_available():
        print("! Local LLM not ready — run scripts/serve.sh and pull the model.")
        return 2

    if args.demo:
        return _demo(manifest)

    # Starting position: explicit args win; else resume (if asked); else the very start.
    if args.chapter is not None or args.offset is not None:
        ch = args.chapter or 1
        dur = next(c["duration_sec"] for c in manifest["chapters"] if c["index"] == ch)
        start = PlaybackPosition(ch, dur * (args.offset if args.offset is not None else 0.0))
    elif args.resume:
        from . import memory
        start = memory.get_resume_position() or PlaybackPosition(1, 0.0)
    else:
        start = PlaybackPosition(1, 0.0)   # from the beginning

    if args.listen:
        from . import voice

        app = NarratorApp(manifest, start, speak=_make_voice_speak(), audio=not args.no_audio)
        app.run_handsfree(voice.STT(), voice.TTS())
        return 0

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
