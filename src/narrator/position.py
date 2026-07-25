"""B4 — Position -> Text resolver / Spoiler Gate  (the hidden core problem).

Playback gives a *time* (chapter + seconds). Spoiler-gating needs a *text*
position. v1 bridges the two with the simplest method: proportional-by-time
within the current chapter (char_offset ~= chars * elapsed / duration).

This block is isolated so the precision can later be upgraded to forced
alignment (aeneas / WhisperX) without changing any caller.

Data contracts
--------------
PlaybackPosition { chapter_index (1-based), position_sec }
HeardCutoff      { chapter_index, char_offset, heard_text, unheard_exists }
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlaybackPosition:
    chapter_index: int      # 1-based
    position_sec: float


@dataclass
class HeardCutoff:
    chapter_index: int
    char_offset: int              # chars heard within the current chapter
    heard_text: str               # everything heard so far (with chapter headers)
    unheard_exists: bool


def _chapter_by_index(manifest: dict, index: int) -> dict:
    for ch in manifest["chapters"]:
        if ch["index"] == index:
            return ch
    raise IndexError(f"chapter {index} not in manifest")


def resolve(manifest: dict, pos: PlaybackPosition) -> HeardCutoff:
    chapters = manifest["chapters"]
    n = len(chapters)
    idx = max(1, min(pos.chapter_index, n))

    cur = _chapter_by_index(manifest, idx)
    duration = cur["duration_sec"] or 1.0
    frac = max(0.0, min(1.0, pos.position_sec / duration))
    char_offset = int(round(cur["char_count"] * frac))

    parts: list[str] = []
    for ch in chapters:
        if ch["index"] < idx:
            parts.append(f"[{ch['title']}]\n{ch['text']}")
        elif ch["index"] == idx:
            heard_slice = ch["text"][:char_offset]
            if heard_slice.strip():
                parts.append(f"[{ch['title']} (in progress)]\n{heard_slice}")
    heard_text = "\n\n".join(parts).strip()

    # Is there anything the listener has NOT heard yet?
    remainder_in_chapter = char_offset < cur["char_count"]
    later_chapters = idx < n
    unheard_exists = remainder_in_chapter or later_chapters

    return HeardCutoff(
        chapter_index=idx,
        char_offset=char_offset,
        heard_text=heard_text,
        unheard_exists=unheard_exists,
    )


def describe(manifest: dict, pos: PlaybackPosition, cutoff: HeardCutoff) -> str:
    """Human-readable one-liner for the CLI / logs."""
    cur = _chapter_by_index(manifest, cutoff.chapter_index)
    pct = 0 if not cur["char_count"] else round(100 * cutoff.char_offset / cur["char_count"])
    return (
        f"position: {cur['title']} @ {pos.position_sec:.0f}s "
        f"(~{pct}% into it) · heard {len(cutoff.heard_text)} chars · "
        f"unheard ahead: {cutoff.unheard_exists}"
    )
