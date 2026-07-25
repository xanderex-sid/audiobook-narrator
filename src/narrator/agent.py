"""B6 — Agent Brain / Dialogue Manager (local LLM via Ollama).

Turns a listener utterance + current position into an AgentResponse:
- classifies intent + explicit-spoiler-consent + navigation action
- answers in a spoiler-safe way

Why classification is separated from answering
----------------------------------------------
The LLM often *already knows* famous public-domain stories (e.g. "The Monkey's
Paw") from pretraining, so it will happily reveal the ending from parametric
memory even if we only hand it the heard text. Two defenses:

1. The answering prompts forbid using any outside knowledge — the heard text is
   the ONLY source of truth.
2. Answering is split from classifying. When a question needs unheard content and
   the listener did NOT consent to spoilers, we never ask the model to "answer" —
   we return a fixed, leak-proof deflection. Unheard text is sent to the model
   ONLY on explicit spoiler consent (Pass 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import knowledge, llm
from .position import HeardCutoff, PlaybackPosition

_NAV_TYPES = {"goto_chapter", "skip_chapter", "set_resume_position"}

# Explicit spoiler-consent is decided DETERMINISTICALLY, not by the LLM — a small
# 7B classifier will occasionally mistake a plain "how does it end?" for consent
# and leak. Reveal only when the listener clearly opts in.
_CONSENT_PHRASES = (
    "spoil", "spoiler",
    "don't care", "dont care", "do not care", "don't mind", "dont mind",
    "tell me anyway", "just tell me", "go ahead", "reveal", "ruin it",
    "even if", "regardless", "i already know", "yes please", "please tell",
    "i don't want to wait", "dont want to wait",
)


def wants_spoiler(text: str) -> bool:
    t = " ".join(text.lower().split())
    return any(p in t for p in _CONSENT_PHRASES)


@dataclass
class AgentResponse:
    speech_text: str
    action: dict = field(default_factory=lambda: {"type": "none"})
    intent: str = "qa"
    spoiler_used: bool = False


# ── Prompts ──────────────────────────────────────────────────────────────────
_CLASSIFY_SYSTEM = """\
You classify what a listener of the audiobook "{title}" just said. You do NOT \
answer their question. Be strict and literal.

Return ONLY a JSON object:
{{
  "intent": "qa" | "navigate" | "control" | "smalltalk",
  "user_wants_spoilers": true | false,
  "action": {{"type": "none"|"goto_chapter"|"skip_chapter"|"set_resume_position", "chapter": <int|null>, "position_sec": <number|null>}},
  "speech_text": "<short spoken reply, ONLY for navigate/control/smalltalk; empty string for qa>"
}}

Rules:
- user_wants_spoilers is true ONLY if the listener EXPLICITLY accepts spoilers or \
demands to know ahead regardless — e.g. "spoil it", "tell me the ending anyway", \
"I don't care about spoilers", "just tell me what happens", "skip ahead and tell me". \
A plain question like "what is the third wish?", "how does it end?", or "what happens \
next?" is NOT consent -> user_wants_spoilers=false.
- navigate/control actions (never spoilers): "go to chapter N"/"jump to chapter N" -> \
goto_chapter; "skip this chapter"/"skip to the next" -> skip_chapter; "restart chapter N"/\
"go back to chapter N" -> goto_chapter; "resume at 90 seconds" -> set_resume_position. \
Otherwise action.type="none".
- For qa, leave speech_text as an empty string (someone else will answer).
"""

_ANSWER_HEARD_SYSTEM = """\
You are "Narrator", a warm, spoiler-safe voice companion for someone LISTENING to \
the audiobook "{title}". You speak your answers out loud.

The text under "HEARD SO FAR" is EVERYTHING the listener has heard, and it is your \
ONLY source of truth.

ABSOLUTE RULES:
- Use only facts stated in HEARD SO FAR. Do not add, infer, or foreshadow later events.
- You may recognize this story from your own training. IGNORE all of that outside \
knowledge completely. If something is not in HEARD SO FAR, it has not happened yet as \
far as you and the listener are concerned.
- If the honest answer (or any part of it) is not present in HEARD SO FAR, DO NOT \
reveal or guess it. Briefly say it hasn't come up yet / you can't say without spoiling, \
and offer to reveal it if they'd like. Include NO details about later events.
- Answer the question DIRECTLY in your own words. Do NOT retell the story, continue the \
narrative, or quote long passages. Ground answers in the timeline when useful \
("As of Chapter II, ...").
- Be concise: at most about 5 sentences (a bit more only for an explicitly requested recap).

Reply with ONLY your spoken answer — plain text, no preamble, no labels, no JSON.
"""

_ANSWER_FULL_SYSTEM = """\
You are "Narrator", a voice companion for the audiobook "{title}". The listener has \
EXPLICITLY asked to be told beyond where they've listened, so spoilers are permitted \
for THIS answer. Use the FULL STORY below as the source of truth and answer their \
question directly.

Reply with ONLY your spoken answer as plain prose: at most about 5 natural spoken \
sentences. Do NOT add author names, attributions, meta commentary, headings, bullet \
points, or numbered lists — just answer the question conversationally.
"""


def _classify(title: str, position_line: str, query: str) -> dict:
    system = _CLASSIFY_SYSTEM.format(title=title)
    user = f"{position_line}\nListener said: \"{query}\""
    out = llm.chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    action = out.get("action") or {"type": "none"}
    if action.get("type") not in (_NAV_TYPES | {"none"}):
        action = {"type": "none"}
    return {
        "intent": out.get("intent", "qa"),
        "user_wants_spoilers": bool(out.get("user_wants_spoilers")),
        "action": action,
        "speech_text": (out.get("speech_text") or "").strip(),
    }


def _mem_block(memory_context: str) -> str:
    return f"\n\nPRIOR CONTEXT (from earlier sessions, NOT part of the book):\n{memory_context}" if memory_context else ""


def _answer_heard(
    title: str, position_line: str, query: str, heard: str, memory_context: str = ""
) -> str:
    system = _ANSWER_HEARD_SYSTEM.format(title=title)
    user = (
        f"{position_line}{_mem_block(memory_context)}\n\n"
        f"Listener asked: \"{query}\"\n\nHEARD SO FAR:\n{heard}"
    )
    return llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        num_predict=320,
    ).strip()


def _answer_full(title: str, query: str, full_text: str) -> str:
    system = _ANSWER_FULL_SYSTEM.format(title=title)
    user = f"Listener asked: \"{query}\"\n\nFULL STORY:\n{full_text}"
    return llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        num_predict=320,
    ).strip()


def _nav_speech(manifest: dict, pos: PlaybackPosition, action: dict, fallback: str) -> str:
    """Deterministic, reliable spoken confirmation for navigation."""
    n = manifest["n_chapters"]
    kind = action.get("type")
    if kind == "goto_chapter":
        ch = int(action.get("chapter") or pos.chapter_index)
        return f"Sure — jumping to Chapter {max(1, min(ch, n))}."
    if kind == "skip_chapter":
        nxt = min(pos.chapter_index + 1, n)
        if nxt == pos.chapter_index:
            return "You're already on the last chapter."
        return f"Okay, skipping ahead to Chapter {nxt}."
    if kind == "set_resume_position":
        ch = int(action.get("chapter") or pos.chapter_index)
        sec = float(action.get("position_sec") or 0.0)
        return f"Got it — I'll resume at Chapter {ch}, {sec:.0f} seconds in."
    return fallback or "Okay."


def respond(
    manifest: dict,
    query: str,
    pos: PlaybackPosition,
    cutoff: HeardCutoff,
    *,
    position_line: str | None = None,
    memory_context: str = "",
) -> AgentResponse:
    title = manifest.get("title", "the book")
    if position_line is None:
        position_line = (
            f"The listener is currently in Chapter {cutoff.chapter_index} "
            f"of {manifest['n_chapters']}."
        )

    cls = _classify(title, position_line, query)
    action = cls["action"]
    intent = cls["intent"]

    # Pure navigation / control / smalltalk: no story content needed.
    if action["type"] in _NAV_TYPES or intent in ("navigate", "control"):
        speech = _nav_speech(manifest, pos, action, cls["speech_text"])
        return AgentResponse(speech, action, intent, spoiler_used=False)

    # Explicit spoiler consent (deterministic) + unheard content -> reveal.
    if wants_spoiler(query) and cutoff.unheard_exists:
        speech = _answer_full(title, query, knowledge.full_context(manifest))
        return AgentResponse(
            speech or "Here's what happens...", action, intent, spoiler_used=True
        )

    # Default: spoiler-safe answer grounded ONLY in heard text.
    speech = _answer_heard(
        title, position_line, query, knowledge.heard_context(cutoff), memory_context
    )
    return AgentResponse(
        speech or "I'm not sure yet from what you've heard.",
        action,
        intent,
        spoiler_used=False,
    )
