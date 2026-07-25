So here is the problem, In the start I have audiobook chapter wise
here is the path of audiobook chapters: /home/x0zby2/projects/audiobook-narrator/audiobook_chapters/chapters
I have also provided .txt textbook of it, path: /home/x0zby2/projects/audiobook-narrator/book.txt

and I want to create a talking agent system with whom I can actually talk and ask everything related to audiobook.
1. Things that are very clear:
- I will talk with this chat agent and agent will also reply in voice back to me, I will not write the prompts, every input from human and output to the human is in voice.
- Agent system must always have the full context and memory of all the chapters of the audiobook we are talking about.
2. Now, I should be able to do following things with my talking agent:
- I can ask about any characters related to that story like what was their history, how are they involved in the story with the current plot, I forgot when he/she was introduced so please provide a recap of the portion of the story where they were introduced first time, what these two characters are right now talking about, etc.
- I must be able to summarize a chapter or skip a chapter or ask for spoilers, end of the story, go to a particular chapter of the story, what is the current setting of the story, what is the surrounding, etc.
- I must be able to ask meanings and explanations of part of the story or a particular word or sentence or certain conversation between two characters, etc.
3. Now, How to detect when I have asked for a query or general question and how to detect when to stop talking with me and again start the audiobook from the exact timestamp where I stopped it:
- when I will say "Hey Narrator" system must stop audiobook and note the timestamp where it stopped and start my conversation with my chatting agent (narrator basically)
- when I will say "Ok Continue Story" system must stop the conversation and also stop talking to me and again start the audiobook at the timestamp where I want to continue listening (by default, timestamp will be where we stopped audiobook to talk and timestamp can change depending upon the conversations I had with the talking agent).
4. What inputs will my talking agentic system will get the moment I say "Hey Narrator":
- System will get "Hey Narrator" voice itself to stop the audiobook and start the talking session.
- simultaneously system will also get the timestamp, that is "How far I have covered the story", accordingly he will set every answer, what portion of story will be answered as covered part and what portion of the story will be answered as spoiler, etc.
- System will get my question, query, doubt, whatever (this can be in the continuation with my "Hey Narrator" voice or could be after the system says something to me.)
- System will also keep track of everything user has asked so far in previous sessions that will go as a history, as an input each session also and also every k sessions we will summarize what user has discussed so far, to not increase to much context length (we want to keep each session in certain token limits to make everything cost effective.)
5. What will happen the moment I say "Ok Continue Story":
- So, I can say "Ok Continue Story" at the end of my last voice prompt before agent is able to say its last response, in such a case: agentic system must be able to say its last response and then stop the talking session and continue the audiobook.
- If I say it at the very end of our talking session, then simply say "Ok" and with a 2 second pause, re start the audiobook at the updated timestamp of a chapter.
6. What things I expect from this agentic talking system:
- It must keep track of previous sessions (to a certain limit) of my conversation with it, so it do not lose the context in the current session. you can store our previous conversations in text format or whatever format is more suitable in a database or json or whatever. you can use mean zero for persistent memory.
- you can use livekit for orchestration or something else which is more better for this system.
- If during a conversation, starting of audiobook changes after session ends maybe due to skipping a chapter or on demand of me, I expect system to update the timestamp from where audiobook has to be restarted.
- It must keep the context of full story so it can response efficiently to my all voice prompts whether they are the part of covered story or spoilers or any language related question, we are sticking to english stories for now.

---

# System Design (v1 — fully local, free, native)

_Added after discussion. This is the working design doc for the build._

## Context
Voice-driven conversational agent layered over an audiobook. You listen; say **"Hey Narrator"**
to pause and ask anything (characters, recaps, summaries, meanings, navigation, spoilers-on-demand);
say **"Ok Continue Story"** to resume from the right spot. The agent respects *how far you've
listened* (spoiler-gating), remembers past sessions, and updates the resume point on skip/jump.

**Test corpus:** `book.txt` = *The Monkey's Paw* (W.W. Jacobs), 3 sections (markers `I./II./III.`
at lines 3/137/199), ~3,940 words (~5k tokens). `chapter_01/02/03.wav` map 1:1 to those sections.
The whole book fits in one LLM context → **v1 needs no vector DB / RAG**; feed heard text directly.

## Locked decisions
- **Fully local & free. No cloud.** Everything runs natively on this machine.
- **Runtime:** local desktop app; local mic/speaker/player. Hotkey backstop for "Hey Narrator" so
  triggering never fails on a demo.
- **Position precision (simplest):** proportional-by-time — heard text within the current chapter
  ≈ `elapsed_sec / chapter_duration_sec`. No forced alignment yet.
- **Build order:** text-first to prove logic → voice → playback+wake word → refine the 20%.
- **Language:** Python 3.13 (conda `base`).

## Machine (measured)
RTX 5070 Laptop **12 GB VRAM** (CUDA 13.1) · Core Ultra 9 275HX 24c · 15 GB RAM · 914 GB free ·
Ubuntu 24.04 / WSL2. No passwordless sudo → **userspace installs only**.

## Local model stack
| Block | Choice | Notes |
|---|---|---|
| LLM (brain) | **Qwen2.5-7B-Instruct** (Q4) via **Ollama**, OpenAI-compatible API `localhost:11434/v1` | ~5 GB VRAM; tool-calling capable; 14B is the upgrade path |
| STT | faster-whisper (`distil-small.en`/`small`) on GPU | ~1–2 GB |
| TTS | Kokoro (quality) or Piper (simplest) | small, GPU/CPU |
| Wake word | hotkey v1 → openWakeWord (train "Hey Narrator") later | — |
| Memory | JSON transcript + naive summary v1 → mem0 later | the "mean zero" in the notes above = **mem0** |

## Architecture — independent blocks (I/O)
- **B0 Corpus Prep** (offline): `book.txt` + wavs → `data/book_manifest.json`
  `[{index,title,text,char_count,audio_path,duration_sec}]` (durations via stdlib `wave`).
- **B1 Playback Controller**: play/pause/resume/seek/goto/skip → emits `PlaybackPosition{chapter_index,position_sec}`.
- **B2 Control / Wake + State machine**: mic "Hey Narrator" / "Ok Continue Story" + hotkey → `ENTER/END_CONVERSATION`; owns `PLAYING⇄CONVERSING`.
- **B3 Voice I/O**: STT (mic→text, VAD) and TTS (text→speaker); pluggable providers.
- **B4 Position→Text / Spoiler Gate** *(hidden core problem, isolated)*: `PlaybackPosition`+manifest → `HeardCutoff{chapter_index,char_offset,heard_text,unheard_exists}`. v1 = proportional-by-time; upgrade = forced alignment, no caller changes.
- **B5 Knowledge Provider**: query+cutoff → context. v1 = full heard text; RAG slots in later.
- **B6 Agent Brain (local LLM)**: transcript+cutoff+context+memory+tools → `AgentResponse{speech_text,actions[],updated_position?}`. Intents: Q&A / navigation / spoiler-request / control. **Spoiler policy:** default = heard text only; unheard reachable only via explicit `reveal_spoiler`; answers framed "as of chapter N…". Tools: `goto_chapter`,`skip_chapter`,`summarize_chapter`,`reveal_spoiler`,`set_resume_position`.
- **B7 Memory Store**: turns/session-end → last-session summary + relevant facts; summarize every k turns to cap tokens.
- **B8 Session Orchestrator**: on ENTER → pause(B1)→capture pos→cutoff(B4)→load mem(B7)→loop[STT→B6→TTS]→apply nav; on "Ok Continue Story" → finish reply, 2s pause, persist pos+mem, resume(B1) at updated pos.

## Build order (each milestone independently demoable) — STATUS
- **A — Brains (text-only):** ✅ DONE & verified. B0,B4,B5,B6 + CLI. Spoiler-gating,
  recaps, summaries, navigation, resume-position. Run: `scripts/narrate.sh`.
- **B — Memory:** ✅ DONE & verified. B7 JSON (session summaries + resume position) + B8
  orchestrator. Resumes where you left off; recalls earlier sessions.
- **C — Voice:** ✅ DONE & verified. B3 faster-whisper (STT) + Kokoro (TTS), local/offline,
  in-project models. TTS→STT round-trip exact; full speech Q&A. Run: `scripts/voice-selftest.sh`.
- **D — Playback + Wake word:** ✅ DONE & verified. B1 AudioPlayer (clock-tracked position),
  B2 control (scripted/text/voice; "Hey Narrator"/"Ok Continue Story"), B8 `app` full loop.
  Run: `scripts/demo.sh` (scripted) / `scripts/voice.sh --text|--voice`.
- **Refinements (the 20%):** forced alignment (exact within-chapter cutoff), mem0,
  openWakeWord for hands-free wake, live audio via PortAudio, RAG for long books, 14B model.

### Reliability decisions worth keeping
- LLM leaks famous-story endings from pretraining → **classify/answer split** + "ignore
  outside knowledge; heard text is the only source of truth"; unheard text reaches the
  model only on consent.
- **Spoiler consent is deterministic** (keyword gate), not the LLM's judgment — a plain
  "how does it end?" is guarded; only explicit opt-in ("spoil it", "I don't care…") reveals.
- Answers are plain-text (not JSON) with `num_predict` caps and `num_ctx=8192` to avoid
  hallucinated JSON keys, retelling, and truncation.

## Project layout
```
audiobook-narrator/
  book.txt · audiobook_chapters/chapters/*.wav · goal.md
  data/            book_manifest.json · sessions/
  src/narrator/    corpus.py(B0) playback.py(B1) control.py(B2) voice.py(B3)
                   position.py(B4) knowledge.py(B5) agent.py(B6) memory.py(B7)
                   orchestrator.py(B8) config.py cli.py
  requirements.txt
```
