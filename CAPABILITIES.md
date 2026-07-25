# Audiobook Narrator — Capabilities & Status

_Last updated: 2026-07-25. A plain-language snapshot of what the system can and can't do
today, where it will fail, what's left to build against `goal.md`, and how to test it._

---

## One-line summary
The **brain (LLM tool-calling), warn-then-confirm spoiler gating, forced-alignment
position, semantic memory, navigation, preference handoff, and hands-free voice all work
locally.** You press **ENTER once** to start talking, converse **entirely by voice**
(auto endpointing — no keys per turn), and **ENTER again** to resume: two ENTERs per
session. The remaining goal gap is a true **always-on "Hey Narrator" wake word over
playing audio** (deferred as unreliable in WSL2; the 2-ENTER flow is the reliable
substitute).

**Run it:** `./scripts/listen.sh` (headphones). See §How to test.

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
| Talk to it; it replies **in voice**, hands-free between turns | ✅ | ENTER→ talk by voice → ENTER resumes (2 ENTERs/session) |
| Every capability is a real **LLM tool call** | ✅ | 9 tools (answer/recap/summarize/where-am-I/goto/skip/restart-from-phrase/set-resume/smalltalk) via Ollama native tool-calling |
| **Character** questions / history / recap of their intro | ✅ | grounded in what you've heard |
| **Recaps, summaries, meanings** of words/sentences/scenes | ✅ | Q&A over heard text |
| **Current setting / "where are we"** | ✅ | from heard portion |
| **Spoiler-gating** with **warn → confirm** | ✅ | never answered outright; warns "spoiler — you'll find it in Ch N, sure?"; reveals only on confirm |
| **Never reveal without confirmation** | ✅ | 3 independent leak-proof signals; deterministic consent; 0 false-positives, only a late-final-chapter edge miss on the test corpus |
| **Restart from an exact quoted line** | ✅ | `restart_from_phrase` → forced-alignment timestamp |
| **Exact within-chapter position** | ✅ | forced alignment (faster-whisper word timestamps) → real sentence boundaries |
| Resume at the right timestamp on ENTER | ✅ | 2s pause then resume at the (possibly updated) position |
| Navigation: **skip / go to chapter**, updates resume point | ✅ | "skip this chapter", "go to chapter 1" |
| **Cross-session semantic memory** + resume | ✅ | local embeddings (nomic) recall relevant past asks; resume position persisted |
| **Works for any book** (not just this one) | ✅ | auto marker detection + single combined-wav support; no hardcoded title |
| **Preference handoff** to the audiobook generator | ✅ | infers vocabulary/pacing → `data/preferences/<book>.json` |
| Runs **fully local & free** | ✅ | Qwen2.5 + faster-whisper(small.en) + Kokoro + nomic-embed, all in-project |

---

## ❌ What does NOT work / where it will fail
| Gap | Impact | Why |
|---|---|---|
| **No always-on "Hey Narrator" wake word** | you press **ENTER once** to start a talk session (then it's voice-only); you don't speak the phrase while the book plays | always-listening over playing audio needs openWakeWord + acoustic echo cancellation, unreliable in WSL2 and un-testable here. The 2-ENTER flow is the deliberate, reliable substitute. |
| **No barge-in mid-answer** | you wait for the narrator to finish, then it auto-listens | fluid, but not interrupt-the-TTS turn-taking |
| **14B is opt-in, not default** | slightly less accurate answers on 7B | 14B co-residency with STT/TTS in 12 GB VRAM is unverified; enable with `NARRATOR_LLM=qwen2.5:14b-instruct` once pulled |
| **Single-file books use approximate boundaries** | chapter splits proportional unless `chapters.json` given | per-chapter forced alignment is built; single-wav alignment is a later refinement |
| **Rare late-book spoiler edge** | asking a character's fate *while already inside the final chapter* may answer | all "ending/future while early" spoilers are caught; this residual case reveals what you're about to hear anyway |
| **Within-chapter spoiler cutoff is approximate** | near your exact position it may withhold a line you just heard, or rarely mention a sentence seconds ahead | position→text is `elapsed/duration` (proportional), not word-level forced alignment. Chapter-level is exact; sub-chapter is estimated |
| **7B model factual wobble** | occasional slips ("as of Chapter I" when you're in II; minor who-did-what) | small model. Spoiler logic is solid; fine-grained accuracy isn't perfect — 14B fixes most |
| **Echo if using speakers** | mic may catch book/TTS audio | book is paused while talking and TTS/mic are sequential, so mostly fine — **use headphones** to be safe |
| **Latency** | ~3–6 s per spoiler-checked answer (multiple guarded LLM calls); first is slower while models load | local inference; the price of leak-proof gating |
| **WSLg-specific audio** | breaks if `PULSE_SERVER` or pulse `client.conf` changes | see README setup notes |

---

## 🔧 Gaps vs the goal — status after this stage
1. ✅ **Forced alignment** (faster-whisper word timestamps) → exact within-chapter cutoff + exact phrase-restart. *(WhisperX avoided to protect the working env.)*
2. ✅ **Better STT** — `small.en` auto-default; `vad_filter` + energy/no-speech gating kill silence hallucinations; tail padding stops clipping.
3. ✅ **Semantic cross-session memory** — local `nomic-embed-text` + cosine recall (in place of the mem0 package, to stay light and env-safe).
4. ✅ **Tool-calling**, **warn-then-confirm spoilers**, **single-file / any-book**, **restart-from-phrase**, **preference handoff**.
5. ◑ **14B** — pulled/opt-in (`NARRATOR_LLM=qwen2.5:14b-instruct`); not default until VRAM co-residency with STT/TTS is confirmed.
6. ✗ **Always-on "Hey Narrator" wake word** over playing audio + **barge-in** — still open (WSL2 echo cancellation risk); 2-ENTER is the reliable substitute.

---

## 🧪 How to test

**Setup:** use **headphones**, then start mid-book so there's something to spoil:
```bash
cd ~/projects/audiobook-narrator
./scripts/listen.sh --chapter 2        # start in Chapter II
```
Press **ENTER** to start talking, then **speak** each line (it auto-listens between
answers); watch the `you>` line to see what it heard. Press **ENTER** to resume, **q** to
quit.

> To judge the **brain** without mic variance first, type the same lines via
> `./scripts/narrate.sh --chapter 2 --offset 50%`, then use `./scripts/listen.sh`.

**Say these and check the result:**

| # | Say this | PASS looks like |
|---|---|---|
| 1 | *(press ENTER)* | book pauses, narrator says "Yes?", "🎙️ listening…" |
| 2 | "Who is Herbert?" | the Whites' son; heard content only (tool: answer_about_story) |
| 3 | "Give me a recap so far." | covers up to your point, **nothing from Ch. III** (tool: recap) |
| 4 | "What did Morris say about the paw?" | explains the spell / three-wishes idea |
| 5 | "What is the third wish and how does it end?" | **warns**: "that's a spoiler… sure?" — *no ending leaked* |
| 6 | "no" | "Okay, I'll keep it a secret" — still no leak |
| 7 | "How does it end?" then "yes" | warns, then on **yes** reveals the ending |
| 8 | "Skip this chapter." | "skipping to Chapter 3" → resume becomes Ch. III @ 0s |
| 9 | "Restart from 'two hundred pounds'." | finds the line → resume set there (exact via alignment) |
| 10 | "Where are we right now?" | current point, spoiler-safe |
| 11 | *(ENTER to resume)* | "Okay" → 2 s pause → book resumes at the updated position |
| 12 | `q`, then rerun `./scripts/listen.sh --resume` | resumes where you left off + "(remembering earlier sessions)"; ask "what did I ask last time?" → it recalls |

**Also check:** `data/preferences/*.json` after a session heavy on "what does X mean?" +
skips → `vocabulary: simplified`, `pacing: condensed`. And confirm no phantom "Thank you."
on silence, and the last word of your speech isn't clipped.

---

## All run commands
```bash
./scripts/listen.sh                             # ⭐ hands-free voice (the main experience)
./scripts/listen.sh --resume                    # resume where you left off
./scripts/serve.sh                              # start the local LLM only
./scripts/narrate.sh --chapter 2 --offset 50%   # text brain (type questions, no audio)
./scripts/align.sh                              # build forced alignment (one-time)
./scripts/preferences.sh --from-chapter 3       # write the generator preference JSON
./scripts/voice-selftest.sh --selftest          # TTS → STT round-trip check
./scripts/demo.sh                               # scripted full-loop demo (no mic)
```
Quality upgrade: `export NARRATOR_LLM=qwen2.5:14b-instruct` (once pulled).

See `goal.md` for the full design and `README.md` for setup details.
