"""B1 — Playback Controller (headless core for Milestone A).

Milestone A has no audio yet, but the *position model* and the navigation
semantics live here so the CLI (and later the real audio player) share one
source of truth. The audio backend (sounddevice/ffplay) plugs in at Milestone D
behind the same command surface.
"""
from __future__ import annotations

from .position import PlaybackPosition


def clamp(manifest: dict, pos: PlaybackPosition) -> PlaybackPosition:
    n = len(manifest["chapters"])
    idx = max(1, min(pos.chapter_index, n))
    dur = _duration(manifest, idx)
    sec = max(0.0, min(pos.position_sec, dur))
    return PlaybackPosition(idx, sec)


def _duration(manifest: dict, index: int) -> float:
    for ch in manifest["chapters"]:
        if ch["index"] == index:
            return float(ch["duration_sec"])
    return 0.0


def apply_action(
    manifest: dict, pos: PlaybackPosition, action: dict | None
) -> tuple[PlaybackPosition, str]:
    """Apply a navigation action to the current position.

    Returns (new_position, human_note). Non-navigation actions leave the
    position unchanged.
    """
    if not action:
        return pos, ""
    n = len(manifest["chapters"])
    kind = (action.get("type") or "none").lower()

    if kind == "goto_chapter":
        ch = int(action.get("chapter") or pos.chapter_index)
        new = clamp(manifest, PlaybackPosition(ch, 0.0))
        return new, f"↦ jumped to Chapter {new.chapter_index} (start)"

    if kind == "skip_chapter":
        new = clamp(manifest, PlaybackPosition(pos.chapter_index + 1, 0.0))
        if new.chapter_index == pos.chapter_index and pos.chapter_index == n:
            return pos, "↦ already at the last chapter; nothing to skip"
        return new, f"↦ skipped to Chapter {new.chapter_index} (start)"

    if kind == "set_resume_position":
        ch = int(action.get("chapter") or pos.chapter_index)
        sec = float(action.get("position_sec") or 0.0)
        new = clamp(manifest, PlaybackPosition(ch, sec))
        return new, f"↦ resume set to Chapter {new.chapter_index} @ {new.position_sec:.0f}s"

    # summarize_chapter / none: no position change
    return pos, ""
