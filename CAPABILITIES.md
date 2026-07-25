# Audiobook Narrator — Capabilities & Status

_Last updated: 2026-07-25. A plain-language snapshot of what the system can and can't do
today, where it will fail, what's left to build against `goal.md`, and how to test it._

---

## One-line summary
The **brain, spoiler-gating, memory, navigation, and full voice in/out all work locally**.
The main gap versus the goal is **hands-free control**: today you interrupt with the
**Enter key + push-to-talk**, not by *saying* "Hey Narrator" while the book plays.

---

## The stack (all local, free, in-project)
| Part | What | Where |
|---|---|---|
| Brain (LLM) | Qwen2.5-7B-Instruct via **Ollama** (GPU) | `models/ollama/` |
| Speech-to-text | **faster-whisper** `base.en` | `models/speech/` |
| Text-to-speech | **Kokoro-82M** (voice `af_heart`) | `models/speech/` |
| Live audio (WSL2) | WSLg **PulseAudio** (`paplay`/`parec`/`pacat`) + `ffmpeg` | `src/narrator/paudio.py` |

No cloud. Book context fits in one LLM prompt, so no vector DB is used yet.

---

## ✅ What works (mapped to the goal)
| Goal | Status | Notes |
|---|---|---|
| Talk to it; it replies **in voice** | ✅ | your speech → Whisper → brain → Kokoro → speaker |
| Full **context/memory of all chapters** | ✅ | whole book reasoned over |
| **Character** questions / history / recap of their intro | ✅ | grounded in what you've heard |
| **Recaps, summaries, meanings** of words/sentences/scenes | ✅ | Q&A over heard text |
| **Current setting / "where are we"** | ✅ | from heard portion |
| **Spoiler-gating by how far you've listened** | ✅ (approx.) | refuses future events; reveals only on explicit consent |
| **"Ok Continue Story"** resume at the right timestamp | ✅ | spoken phrase, or `c`, or typed |
| Navigation: **skip / go to chapter**, updates resume point | ✅ | "skip to chapter 3", "go to chapter 1" |
| **Timestamp** captured on interrupt; updated by skip/goto | ✅ | shown as "paused at Chapter II @ 130s" |
| **Cross-session memory** + resume where you left off | ✅ (basic) | JSON summaries + resume position in `data/sessions/` |
| Runs **fully local & free** | ✅ | Qwen2.5-7B + Whisper + Kokoro, all in-project |

---

## ❌ What does NOT work / where it will fail
| Gap | Impact | Why |
|---|---|---|
| **No hands-free "Hey Narrator" wake word** | you press **Enter** to interrupt, not speak the phrase while the book plays | always-listening wake word over playing audio isn't built yet (needs openWakeWord + echo handling). Biggest deviation from "never touch the keyboard." |
| **Push-to-talk, not barge-in** | press Enter to start/stop each utterance; no mid-sentence interruption | reliable in WSL2, but not the fluid turn-taking in the goal |
| **Within-chapter spoiler cutoff is approximate** | near your exact position it may withhold a line you just heard, or rarely mention a sentence seconds ahead | position→text is `elapsed/duration` (proportional), not word-level forced alignment. Chapter-level is exact; sub-chapter is estimated |
| **7B model factual wobble** | occasional slips ("as of Chapter I" when you're in II; minor who-did-what) | small model. Spoiler logic is solid; fine-grained accuracy isn't perfect — 14B fixes most |
| **Whisper hallucination on silence** | pressing "send" without speaking may invent "Thank you." → a bogus question | base.en artifact on empty audio |
| **End-of-speech clipping** | last ~0.3–0.5 s cut if you send the instant you stop talking | `parecord` finalizes on signal; pause a beat before sending |
| **Echo if using speakers** | mic may catch book/TTS audio | book is paused before recording, so mostly fine — **use headphones** to be safe |
| **mem0 not integrated** | memory is last-6 session summaries, not semantic recall | planned refinement; JSON is v1 |
| **Latency** | ~2–4 s per answer (first is slower while model loads); long answers take a few seconds to synthesize | local GPU inference; acceptable, not instant |
| **WSLg-specific audio** | breaks if `PULSE_SERVER` or pulse `client.conf` changes | see README setup notes |

---

## 🔧 Gaps vs the goal — priority for the next stage
1. **Hands-free wake word** "Hey Narrator" (openWakeWord or continuous STT spotting) + echo handling — the #1 goal gap.
2. **Word-level forced alignment** (WhisperX/aeneas) → exact within-chapter spoiler cutoff.
3. **Answer accuracy** → try `qwen2.5:14b-instruct` (fits 12 GB) and/or a larger Whisper (`small`/`distil`) for names.
4. **Barge-in / smoother turn-taking** (trustworthy VAD end-pointing, or a realtime pipeline).
5. **mem0** for real cross-session semantic memory.
6. Robustness: reject silent/hallucinated captures; pad the recording tail.

---

## 🧪 How to test

**Setup:** use **headphones**, then:
```bash
cd ~/projects/audiobook-narrator
./scripts/voice.sh --voice --chapter 2 --offset 0.4     # starts ~2 min into Ch. II
```
Controls: **Enter** = pause & talk → **Enter** = start recording, *speak*, **Enter** = send
→ answer plays → repeat, or say **"ok continue story"** / type `c` to resume, `q` to quit.

> To judge the **brain** without mic variance, run `./scripts/voice.sh --text` and *type*
> the same lines first; then use `--voice` to judge speech.

**Say (or type) these and check the result:**

| # | Say this | PASS looks like |
|---|---|---|
| 1 | *(press Enter)* | "⏸ paused at Chapter II @ ~130s" |
| 2 | "Who is Herbert?" | the Whites' son; from heard content only |
| 3 | "Give me a recap so far." | covers Ch. I–II up to your point, **nothing from Ch. III** |
| 4 | "What did Morris mean about the fakir's spell?" | explains the spell / three-wishes idea |
| 5 | "What is the third wish and how does it end?" | **refuses**, offers to reveal — *no ending leaked* |
| 6 | "I don't care about spoilers, tell me the ending." | **reveals** the ending |
| 7 | "Skip to chapter 3." | "jumping to Chapter 3" → resume point becomes Ch. III @ 0s |
| 8 | "Where are we in the story right now?" | describes your current point, spoiler-safe |
| 9 | "ok continue story" | "Okay" → 2 s pause → book resumes at the (updated) position |
| 10 | `q`, then rerun `./scripts/voice.sh --voice` (no args) | resumes where you left off + "(remembering earlier sessions)"; ask "what did I ask last time?" → it recalls |

**Note for each:** did STT hear you right (watch the `you>` line)? · any spoiler leak on
#3/#5/#8? · answer accuracy · TTS clarity · latency · resume timestamp after #7/#9 · mic
clipping or phantom "Thank you." on empty sends.

---

## All run commands
```bash
./scripts/serve.sh                              # start the local LLM (auto-started otherwise)
./scripts/narrate.sh --chapter 2 --offset 50%   # text brain only (type questions)
./scripts/demo.sh                               # scripted full-loop demo (no mic)
./scripts/voice-selftest.sh --selftest          # TTS → STT round-trip check
./scripts/voice.sh --voice                      # SPEAK to it (mic + speaker)
./scripts/voice.sh --text                       # full loop, type to drive
```
Quality upgrade: `ollama pull qwen2.5:14b-instruct` then `export NARRATOR_LLM=qwen2.5:14b-instruct`.

See `goal.md` for the full design and `README.md` for setup details.
