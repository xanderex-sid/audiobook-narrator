"""B6 (tools) — the capability registry the LLM routes to.

Every listener capability is an explicit, documented tool. The brain (agent.py)
asks the local model, via Ollama native tool-calling, to pick exactly ONE tool
and its arguments; agent.py then runs the matching handler. Spoiler safety does
NOT depend on the router — `reveal_spoiler` is deliberately NOT exposed to the
model; unheard content is reachable only through agent.py's deterministic
warn-then-confirm gate (WS-2).

Schemas here are the single source of truth documented in ARCHITECTURE.md.
"""
from __future__ import annotations

# Tool schemas offered to the model (Ollama /api/chat `tools`). One is chosen per
# turn. `reveal_spoiler` is intentionally absent — see module docstring.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "answer_about_story",
            "description": (
                "Answer a listener's question about the story: characters and their "
                "history, what is happening, the meaning of a word/sentence/scene, the "
                "current setting. Use for any question that is NOT a recap, summary, or "
                "navigation command."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the listener's question, verbatim"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recap",
            "description": (
                "Give a recap/'remind me' of the story so far, optionally focused on a "
                "character or thread (e.g. 'remind me who that character is', 'recap so far')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {"type": "string", "description": "optional character/topic to focus on; empty for a general recap"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_chapter",
            "description": "Summarize a specific chapter by number (or the current one if omitted).",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "1-based chapter number; omit for current"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "where_am_i",
            "description": "Describe the listener's current place/setting in the story, spoiler-safe.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "goto_chapter",
            "description": "Jump playback to the start of a chapter. Use for 'go to chapter N', 'restart chapter N', 'go back to chapter N'.",
            "parameters": {
                "type": "object",
                "properties": {"index": {"type": "integer", "description": "1-based chapter number"}},
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skip_chapter",
            "description": "Skip the current chapter and move to the next one.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_from_phrase",
            "description": (
                "Resume the audiobook from a specific spot the listener names — either a "
                "quoted line (e.g. 'restart from \"be careful what you wish for\"') OR a "
                "described moment (e.g. 'play from where the soldier arrives at the door'). "
                "Finds that spot and moves the resume point to it."
            ),
            "parameters": {
                "type": "object",
                "properties": {"phrase": {"type": "string", "description": "the quoted words OR a description of the moment to jump to"}},
                "required": ["phrase"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_resume_position",
            "description": "Set exactly where playback resumes: a chapter and seconds offset (e.g. 'resume at chapter 2, 90 seconds').",
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter": {"type": "integer"},
                    "position_sec": {"type": "number"},
                },
                "required": ["chapter"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "smalltalk",
            "description": "Greetings, thanks, or chit-chat not about the story. Reply briefly.",
            "parameters": {
                "type": "object",
                "properties": {"reply": {"type": "string", "description": "a short, friendly spoken reply"}},
            },
        },
    },
]

# Handler names, so agent.py can validate a routed tool.
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}

_ROUTER_SYSTEM = """\
You route what a listener of the audiobook "{title}" just said to tool calls. Do \
not answer the question yourself; do not write prose. Pick the best tool(s) and \
fill their arguments.

Usually the listener wants ONE thing -> make ONE tool call. But if they clearly \
ask for MORE THAN ONE thing in the same breath (e.g. "recap chapter 1 AND move to \
chapter 2"), make ONE tool call for EACH distinct request, in the order they should \
happen (information/recap first, navigation last). Do not invent extra calls.

Guidance:
- Questions about characters, plot, meanings, or the setting -> answer_about_story.
- "recap", "remind me", "who is X again" -> recap.
- "summarize chapter N" -> summarize_chapter.
- "where are we", "what's happening now" -> where_am_i.
- "go to / jump to / restart chapter N", "go back to chapter N", "move on to chapter N" -> goto_chapter.
- "skip this chapter", "skip ahead" -> skip_chapter.
- "restart from '<line>'", or "play/continue/start from where <something happens>" \
(a quoted line OR a described moment) -> restart_from_phrase, with the quote or the \
description as `phrase`.
- "resume at chapter N, M seconds" -> set_resume_position.
- greetings/thanks -> smalltalk.
Never refuse. Never reveal plot beyond the question. Always call at least one tool.
"""


def router_messages(title: str, position_line: str, query: str) -> list[dict]:
    return [
        {"role": "system", "content": _ROUTER_SYSTEM.format(title=title)},
        {"role": "user", "content": f"{position_line}\nListener said: \"{query}\""},
    ]
