#!/usr/bin/env bash
# Live wake-word debug: opens the SAME Deepgram stream wake mode uses and prints
# everything it transcribes from your mic (plus whether it matches a wake/continue
# phrase). No audiobook playing — just talk. Ctrl-C to stop.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_PY="${NARRATOR_VOICE_PYTHON:-/home/x0zby2/projects/audiobook-gen/.venv/bin/python}"
export NARRATOR_BACKEND=cloud
cd "$ROOT/src"
exec "$VOICE_PY" - "$@" <<'PY'
import time, sys
from narrator import voice, control, config, paudio

SECS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
print(f"wake words: {config.WAKE_WORDS}  |  resume words: {config.RESUME_WORDS}")
print(f"pulse mic available: {paudio.available()}")
stream = voice.DeepgramStream()
if not stream.start():
    print("!! Deepgram stream FAILED to open (check DEEPGRAM_API_KEY / network).")
    raise SystemExit(1)
print(f"stream open — TALK NOW for {SECS}s. Say your wake word ({', '.join(config.WAKE_WORDS)}), "
      f"then a sentence, then 'Ok continue story'.\n")

t0 = time.time()
got_any = False
try:
    while time.time() - t0 < SECS:
        ev = stream.get(timeout=0.3)
        if not ev:
            continue
        if ev["type"] == "transcript":
            got_any = True
            is_wake = control.matches_keyword(ev["text"], config.WAKE_WORDS)
            is_resume = control.matches_keyword(ev["text"], config.RESUME_WORDS)
            mark = "  <<< WAKE!" if is_wake else ("  <<< CONTINUE" if is_resume else "")
            print(f"  [{time.time()-t0:5.1f}s] final={ev['final']}  text={ev['text']!r}{mark}")
        elif ev["type"] == "speech_started":
            print(f"  [{time.time()-t0:5.1f}s] (speech detected)")
        elif ev["type"] == "utterance_end":
            print(f"  [{time.time()-t0:5.1f}s] (end of utterance)")
except KeyboardInterrupt:
    pass
finally:
    stream.stop()

print("\n----")
if not got_any:
    print("NO transcripts arrived → audio isn't reaching Deepgram (mic device / sending), "
          "or Deepgram rejected the stream. Report this.")
else:
    print("Transcripts arrived. If your 'Hey Narrator' didn't show '<<< WAKE!', paste what it "
          "transcribed instead so I can widen the phrase matching.")
PY
