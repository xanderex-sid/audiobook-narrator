"""B5 — Knowledge Provider.

Turns a HeardCutoff into the text context the brain reasons over.

v1 uses full-context (the whole book is ~5k tokens, so no retrieval needed):
- `heard_context`  -> everything the listener has heard (spoiler-safe default)
- `full_context`   -> the entire book (only used on an explicit spoiler request)

The interface is deliberately small so a RAG implementation can slot in later
for long books without changing the agent.
"""
from __future__ import annotations

from .position import HeardCutoff


def heard_context(cutoff: HeardCutoff) -> str:
    return cutoff.heard_text or "(the listener has not heard anything yet)"


def full_context(manifest: dict) -> str:
    parts = [f"[{ch['title']}]\n{ch['text']}" for ch in manifest["chapters"]]
    return "\n\n".join(parts).strip()


def chapter_text(manifest: dict, index: int) -> str:
    for ch in manifest["chapters"]:
        if ch["index"] == index:
            return ch["text"]
    raise IndexError(f"chapter {index} not in manifest")
