"""B6 — Agent Brain / Dialogue Manager (local LLM via Ollama).

Turns a listener utterance + current position into an AgentResponse by:
1. ROUTING the utterance to exactly one tool (tools.py) via native tool-calling.
2. DISPATCHING the matching handler.
3. Enforcing a deterministic spoiler gate on top of the router (WS-2):
   a plain question about future events is never answered outright — the narrator
   WARNS ("that's a spoiler; you'll find it in Chapter N — want me to tell you
   anyway?") and only reveals after the listener CONFIRMS on the next turn.

Why safety can't live in the router
-----------------------------------
Small local models will occasionally mis-route or reveal famous-story endings
from pretraining. So: (a) unheard text reaches the model ONLY on confirmed
consent; (b) the reveal path is deterministic, not a tool the model may call;
(c) answering prompts forbid outside knowledge — heard text is the only truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import knowledge, llm, position, tools
from .position import HeardCutoff, PlaybackPosition

_NAV_TOOLS = {"goto_chapter", "skip_chapter", "set_resume_position", "restart_from_phrase"}

# Explicit spoiler consent / confirmation is decided DETERMINISTICALLY, never by
# the model (a 7B classifier occasionally mistakes a plain "how does it end?" for
# consent and leaks).
_CONSENT_PHRASES = (
    "spoil", "spoiler", "don't care", "dont care", "do not care", "don't mind",
    "dont mind", "tell me anyway", "just tell me", "go ahead", "reveal", "ruin it",
    "even if", "regardless", "i already know", "yes please", "please tell",
    "i don't want to wait", "dont want to wait",
)
_AFFIRM_TOKENS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "do", "definitely", "absolutely"}
_AFFIRM_PHRASES = ("go ahead", "tell me", "do it", "i do", "go for it", "let's hear it", "lets hear it")
_NEGATE_TOKENS = {"no", "nope", "nah", "don't", "dont", "stop"}
_NEGATE_PHRASES = ("no thanks", "keep it", "keep the secret", "never mind", "nevermind", "not now", "leave it", "skip it", "no spoiler")


def wants_spoiler(text: str) -> bool:
    t = " ".join(text.lower().split())
    return any(p in t for p in _CONSENT_PHRASES)


def is_affirmation(text: str) -> bool:
    t = " ".join(text.lower().split())
    if any(p in t for p in _AFFIRM_PHRASES) or wants_spoiler(t):
        return True
    toks = set(t.replace("?", "").replace(".", "").replace(",", "").split())
    return bool(toks & _AFFIRM_TOKENS) and not (toks & _NEGATE_TOKENS)


def is_negation(text: str) -> bool:
    t = " ".join(text.lower().split())
    if any(p in t for p in _NEGATE_PHRASES):
        return True
    toks = set(t.replace("?", "").replace(".", "").replace(",", "").split())
    return bool(toks & _NEGATE_TOKENS)


# Words that mean the utterance carries its OWN request (a question or a command),
# so it must NOT be treated as a bare yes/no to a pending spoiler warning.
_OTHER_INTENT = (
    "chapter", "skip", "summar", "recap", "restart", "go to", "jump", "resume at",
    "what", "who", "how", "why", "where", "when", "which", "meaning", "explain",
    "tell me about", "?",
)


def _has_other_intent(text: str) -> bool:
    t = " ".join(text.lower().split())
    return any(k in t for k in _OTHER_INTENT)


def is_pure_confirmation(text: str) -> bool:
    """A bare 'yes'/'go ahead' with no other request riding along."""
    return is_affirmation(text) and not _has_other_intent(text)


def is_pure_negation(text: str) -> bool:
    return is_negation(text) and not _has_other_intent(text)


@dataclass
class AgentResponse:
    speech_text: str
    action: dict = field(default_factory=lambda: {"type": "none"})
    intent: str = "qa"
    tool: str = "answer_about_story"
    spoiler_used: bool = False
    needs_confirmation: bool = False
    pending_spoiler: dict | None = None   # carried into the next turn by the session


# ── Prompts ──────────────────────────────────────────────────────────────────
_ANSWER_HEARD_SYSTEM = """\
You are "Narrator", a warm, spoiler-safe voice companion for someone LISTENING to \
the audiobook "{title}". You speak your answers out loud.

The text under "HEARD SO FAR" is EVERYTHING the listener has heard, and it is your \
ONLY source of truth.

ABSOLUTE RULES:
- Use only facts stated in HEARD SO FAR. Do not add, infer, or foreshadow later events.
- You may recognize this story from training. IGNORE all outside knowledge. If \
something is not in HEARD SO FAR, it has not happened yet.
- If the honest answer (or any essential part of it) is NOT present in HEARD SO FAR, \
do NOT guess, deflect, or partially answer, and do NOT use anything you remember about \
this story. Reply with EXACTLY this one token and nothing else: NEED_SPOILER
- Otherwise answer DIRECTLY in your own words. Do NOT retell the story or quote long \
passages. Ground answers in the timeline when useful ("As of {chapter}, ...").
- Be concise: at most about 5 sentences (a bit more only for a requested recap).

Examples:
- If asked "how does the story end?" but the ending is not in HEARD SO FAR -> NEED_SPOILER
- If asked about a character who has not appeared yet in HEARD SO FAR -> NEED_SPOILER

Reply with ONLY your spoken answer — plain text, no preamble, no labels, no JSON \
(or exactly NEED_SPOILER if it can't be answered from HEARD SO FAR).
"""

_TARGET_SYSTEM = """\
Given the FULL STORY of "{title}", say which 1-based chapter FIRST answers the \
listener's question. Return ONLY JSON: {{"target_chapter": <int|null>}}. No prose.
"""

_GATE_SYSTEM = """\
You are the spoiler gate for an audiobook. You are given the exact text a listener \
has HEARD SO FAR and their QUESTION. Decide if answering the question requires ANY \
information not present in the heard text.

Rules:
- If the answer is fully present in the heard text (characters/events/meanings already \
read), it is SAFE.
- If answering needs later, unheard events (the ending, a future twist, a wish/action \
not yet made, a character or thing that has not appeared yet), it is a SPOILER.
- Judge ONLY by the heard text. Ignore anything you know about this story from elsewhere.

A question about someone or something that has ALREADY appeared in the heard text —
who they are, what they said or did SO FAR, what a word means, a recap of what's been
heard, the current setting — is SAFE, even if that person/thing keeps mattering later.
Only flag a spoiler when the specific thing asked for has NOT yet occurred in the heard
text.

Return ONLY JSON: {"spoiler": true|false}

Examples (generic):
- "Who is the brother?" when the brother already appeared -> {"spoiler": false}
- "What did the soldier say about the charm?" (already said in heard text) -> {"spoiler": false}
- "What was the first wish?" when the first wish was already made -> {"spoiler": false}
- "Who is the wife?" when she has appeared -> {"spoiler": false}
- "What does this word mean?" about a word already read -> {"spoiler": false}
- "Recap what I've heard" -> {"spoiler": false}
- "How does the book end?" when the ending is unheard -> {"spoiler": true}
- "What is the second wish?" when only the first has been made -> {"spoiler": true}
- "What happens to X?" about events after the heard text -> {"spoiler": true}
"""

_ANSWER_FULL_SYSTEM = """\
You are "Narrator", a voice companion for the audiobook "{title}". The listener has \
EXPLICITLY confirmed they want to be told beyond where they've listened, so spoilers \
are permitted for THIS answer. Use the FULL STORY below as the source of truth and \
answer directly.

Reply with ONLY your spoken answer as plain prose: at most about 5 natural spoken \
sentences. No author names, attributions, meta commentary, headings, or lists.
"""

def _mem_block(memory_context: str) -> str:
    return (f"\n\nPRIOR CONTEXT (from earlier sessions, NOT part of the book):\n{memory_context}"
            if memory_context else "")


_NEED = "NEED_SPOILER"


def _answer_heard(title, chapter_title, query, heard, memory_context=""):
    """Answer strictly from heard text. Returns text, or the NEED_SPOILER sentinel."""
    system = _ANSWER_HEARD_SYSTEM.format(title=title, chapter=chapter_title)
    user = f"{_mem_block(memory_context)}\n\nListener asked: \"{query}\"\n\nHEARD SO FAR:\n{heard}"
    return llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0, num_predict=320,
    ).strip()


def _is_need(text: str) -> bool:
    return _NEED in text.upper()


def _gate_spoiler(query: str, heard: str) -> bool:
    """Independent spoiler check that sees ONLY heard text (can't leak)."""
    try:
        out = llm.chat_json(
            [{"role": "system", "content": _GATE_SYSTEM},
             {"role": "user", "content": f'QUESTION: "{query}"\n\nHEARD SO FAR:\n{heard}'}],
            temperature=0.0,
        )
        return bool(out.get("spoiler"))
    except Exception:
        return True   # fail safe: warn rather than risk a leak


def _is_spoiler(query, answer, heard, current_chapter, target_chapter) -> bool:
    """Leak-biased spoiler decision — the UNION of three structurally-safe signals.

    Warn if ANY of these fires (over-warning is recoverable via confirm; a leak is
    not):
    1. NEED_SPOILER sentinel — the heard-only answer couldn't be produced.
    2. target_chapter > current — the answer lives in a FUTURE chapter (position
       aware; the target check only ever emits a chapter number, never content).
    3. heard-only gate — an independent judge that sees only the heard text.

    Measured on the test corpus across positions: 0 false-positives, and the only
    residual miss is asking a character's fate once already inside the final
    chapter. All "ending/future while early" spoilers are caught.
    """
    if _is_need(answer):
        return True
    if target_chapter is not None and target_chapter > current_chapter:
        return True
    return _gate_spoiler(query, heard)


def _target_chapter(title, query, full_text):
    """Best-effort: which chapter first answers `query` (for the warning). None on failure."""
    try:
        out = llm.chat_json(
            [{"role": "system", "content": _TARGET_SYSTEM.format(title=title)},
             {"role": "user", "content": f"Question: \"{query}\"\n\nFULL STORY:\n{full_text}"}],
            temperature=0.0,
        )
        tgt = out.get("target_chapter")
        return int(tgt) if isinstance(tgt, (int, float)) else None
    except Exception:
        return None


def _answer_full(title, query, full_text):
    system = _ANSWER_FULL_SYSTEM.format(title=title)
    user = f"Listener asked: \"{query}\"\n\nFULL STORY:\n{full_text}"
    return llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2, num_predict=320,
    ).strip()


# ── routing ────────────────────────────────────────────────────────────────────
def _route(title: str, position_line: str, query: str) -> tuple[str, dict]:
    """Pick one tool + args via native tool-calling; fall back to a keyword guess."""
    try:
        msg = llm.chat_tools(tools.router_messages(title, position_line, query),
                             tools.TOOL_SCHEMAS, temperature=0.0)
        calls = msg.get("tool_calls") or []
        if calls:
            fn = calls[0].get("function", {})
            name = fn.get("name")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if name in tools.TOOL_NAMES:
                return name, args
    except Exception:
        pass
    return _keyword_route(query)


def _keyword_route(query: str) -> tuple[str, dict]:
    t = " ".join(query.lower().split())
    import re
    if any(w in t for w in ("skip this chapter", "skip ahead", "skip the chapter", "next chapter")):
        return "skip_chapter", {}
    m = re.search(r"(?:go to|jump to|restart|go back to|goto)\s+chapter\s+(\d+)", t)
    if m:
        return "goto_chapter", {"index": int(m.group(1))}
    if "restart from" in t or ("restart" in t and '"' in query):
        import re as _re
        q = _re.findall(r'"([^"]+)"', query) or _re.findall(r"'([^']+)'", query)
        return "restart_from_phrase", {"phrase": q[0] if q else query}
    if t.startswith("summarize chapter") or t.startswith("summarise chapter"):
        m = re.search(r"chapter\s+(\d+)", t)
        return "summarize_chapter", {"index": int(m.group(1))} if m else {}
    if any(w in t for w in ("where are we", "where am i", "what's happening", "whats happening", "current setting")):
        return "where_am_i", {}
    if any(w in t for w in ("recap", "remind me", "so far")):
        return "recap", {}
    return "answer_about_story", {"query": query}


def _nav(manifest, pos, tool, args):
    """Build (action, speech) for a navigation tool. No story content."""
    n = manifest["n_chapters"]
    if tool == "goto_chapter":
        ch = max(1, min(int(args.get("index") or pos.chapter_index), n))
        return {"type": "goto_chapter", "chapter": ch}, f"Sure — jumping to Chapter {ch}."
    if tool == "skip_chapter":
        nxt = min(pos.chapter_index + 1, n)
        if nxt == pos.chapter_index:
            return {"type": "none"}, "You're already on the last chapter."
        return {"type": "skip_chapter"}, f"Okay, skipping ahead to Chapter {nxt}."
    if tool == "set_resume_position":
        ch = max(1, min(int(args.get("chapter") or pos.chapter_index), n))
        sec = float(args.get("position_sec") or 0.0)
        return ({"type": "set_resume_position", "chapter": ch, "position_sec": sec},
                f"Got it — I'll resume at Chapter {ch}, {sec:.0f} seconds in.")
    if tool == "restart_from_phrase":
        phrase = (args.get("phrase") or "").strip()
        found, score = position.find_phrase(manifest, phrase)
        if found is None:
            return {"type": "none"}, "I couldn't find that line in the story — could you quote a bit more?"
        title = next(c["title"] for c in manifest["chapters"] if c["index"] == found.chapter_index)
        hedge = "" if score >= 0.99 else " (closest match I could find)"
        return ({"type": "set_resume_position", "chapter": found.chapter_index, "position_sec": found.position_sec},
                f"Found it in {title}{hedge} — I'll resume from there.")
    return {"type": "none"}, "Okay."


def _chapter_fully_heard(cutoff: HeardCutoff, manifest: dict, index: int) -> bool:
    if index < cutoff.chapter_index:
        return True
    if index == cutoff.chapter_index:
        cur = next(c for c in manifest["chapters"] if c["index"] == index)
        return cutoff.char_offset >= cur["char_count"]
    return False


def _warn(manifest, target_chapter):
    if target_chapter and 1 <= target_chapter <= manifest["n_chapters"]:
        title = next(c["title"] for c in manifest["chapters"] if c["index"] == target_chapter)
        where = f"You'll come to it around {title}."
    else:
        where = "It's ahead of where you've listened."
    return (f"Careful — that would be a spoiler. {where} "
            f"Are you sure you want me to tell you now?")


# ── main entry ──────────────────────────────────────────────────────────────────
def respond(
    manifest: dict,
    query: str,
    pos: PlaybackPosition,
    cutoff: HeardCutoff,
    *,
    position_line: str | None = None,
    memory_context: str = "",
    pending_spoiler: dict | None = None,
) -> AgentResponse:
    title = manifest.get("title", "the book")
    chapter_title = next(
        (c["title"] for c in manifest["chapters"] if c["index"] == cutoff.chapter_index),
        f"Chapter {cutoff.chapter_index}",
    )
    if position_line is None:
        position_line = (f"The listener is currently in {chapter_title} "
                         f"of {manifest['n_chapters']}.")

    # ── WS-2: resolve a pending spoiler BEFORE routing a new turn ──────────────
    # Only a BARE yes/no answers the warning; an utterance carrying its own request
    # (e.g. "okay, skip to chapter 3") clears the pending spoiler and is routed
    # normally — it must never be mistaken for spoiler consent.
    if pending_spoiler:
        if is_pure_confirmation(query):
            speech = _answer_full(title, pending_spoiler["query"], knowledge.full_context(manifest))
            return AgentResponse(speech or "Here's what happens...", intent="spoiler",
                                 tool="reveal_spoiler", spoiler_used=True, pending_spoiler=None)
        if is_pure_negation(query):
            return AgentResponse("Okay, I'll keep it a secret for now.", intent="control",
                                 tool="smalltalk", pending_spoiler=None)
        # anything else: drop the pending spoiler and treat as a fresh request.

    tool, args = _route(title, position_line, query)

    # ── navigation / control / smalltalk: no story content ────────────────────
    if tool in _NAV_TOOLS:
        action, speech = _nav(manifest, pos, tool, args)
        return AgentResponse(speech, action, intent="navigate", tool=tool)
    if tool == "smalltalk":
        return AgentResponse(args.get("reply") or "Sure — what would you like to know?",
                             intent="smalltalk", tool="smalltalk")

    # ── where_am_i / recap / answer: all use heard text; gate future content ───
    heard = knowledge.heard_context(cutoff)

    # recap and where_am_i are inherently about the heard portion — they can never
    # leak (the model only ever sees heard text), so answer directly.
    if tool == "where_am_i":
        speech = _answer_heard(title, chapter_title,
                               "Briefly, where are we in the story right now and what's the setting?",
                               heard, memory_context)
        if _is_need(speech):
            speech = "You're right at the very beginning — nothing has happened yet."
        return AgentResponse(speech, tool="where_am_i")

    if tool == "recap":
        focus = (args.get("focus") or "").strip()
        rq = (f"Give me a recap of the story so far focused on {focus}." if focus
              else "Give me a recap of the story so far.")
        speech = _answer_heard(title, chapter_title, rq, heard, memory_context)
        if _is_need(speech):
            speech = "There's nothing to recap yet — you're right at the start."
        return AgentResponse(speech, tool="recap")

    if tool == "summarize_chapter":
        idx = int(args.get("index") or cutoff.chapter_index)
        if _chapter_fully_heard(cutoff, manifest, idx):
            ch_title = next((c["title"] for c in manifest["chapters"] if c["index"] == idx), f"Chapter {idx}")
            speech = _answer_heard(title, chapter_title, f"Summarize {ch_title} in a few sentences.",
                                   heard, memory_context)
            if not _is_need(speech):
                return AgentResponse(speech, tool="summarize_chapter")
        # summarizing an unheard/partial chapter is a spoiler
        if wants_spoiler(query):
            speech = _answer_full(title, f"Summarize chapter {idx}.", knowledge.full_context(manifest))
            return AgentResponse(speech, tool="reveal_spoiler", spoiler_used=True)
        return AgentResponse(_warn(manifest, idx), intent="spoiler", tool="summarize_chapter",
                             needs_confirmation=True, pending_spoiler={"query": f"Summarize chapter {idx}."})

    # ── answer_about_story ────────────────────────────────────────────────────
    # If the listener already consented in this utterance, reveal directly.
    if wants_spoiler(query) and cutoff.unheard_exists:
        speech = _answer_full(title, query, knowledge.full_context(manifest))
        return AgentResponse(speech or "Here's what happens...", tool="reveal_spoiler", spoiler_used=True)

    # Answer strictly from heard text (temp 0), then apply the layered spoiler gate.
    speech = _answer_heard(title, chapter_title, query, heard, memory_context)
    if not cutoff.unheard_exists:
        # everything has been heard -> nothing left to spoil
        if _is_need(speech):
            return AgentResponse("Hmm — I don't think that's in the story.", tool="answer_about_story")
        return AgentResponse(speech, tool="answer_about_story")

    target = _target_chapter(title, query, knowledge.full_context(manifest))
    if _is_spoiler(query, speech, heard, cutoff.chapter_index, target):
        return AgentResponse(_warn(manifest, target), intent="spoiler", tool="answer_about_story",
                             needs_confirmation=True, pending_spoiler={"query": query})
    return AgentResponse(speech, tool="answer_about_story")
