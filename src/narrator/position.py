"""B4 — Position <-> Text resolver / Spoiler Gate  (the hidden core problem).

Playback gives a *time* (chapter + seconds). Spoiler-gating needs a *text*
position. Two methods, chosen automatically per chapter:

- **Forced alignment** (preferred): if `data/alignment/chapter_<i>.json` exists
  (built by `align.py` from faster-whisper word timestamps), time<->char is
  looked up from a real word/time map -> exact within-chapter cutoff and exact
  "restart from this line".
- **Proportional-by-time** (fallback): char_offset ~= chars * elapsed / duration.

Callers are unchanged whichever method is active — that is the whole point of
isolating this block.

Data contracts
--------------
PlaybackPosition { chapter_index (1-based), position_sec }
HeardCutoff      { chapter_index, char_offset, heard_text, unheard_exists }
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from . import config


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


# ── forced-alignment cache (optional) ─────────────────────────────────────────
def _alignment(index: int) -> list[list[float]] | None:
    """Return monotonic [[t_sec, char_offset], ...] checkpoints, or None."""
    path = config.ALIGNMENT_DIR / f"chapter_{index}.json"
    if not path.exists():
        return None
    try:
        pts = json.loads(path.read_text(encoding="utf-8")).get("points")
        return pts if pts and len(pts) >= 2 else None
    except Exception:
        return None


def _interp(points: list[list[float]], x: float, xi: int, yi: int) -> float:
    """Piecewise-linear lookup of column yi given column xi == x (monotonic)."""
    if x <= points[0][xi]:
        return points[0][yi]
    if x >= points[-1][xi]:
        return points[-1][yi]
    for a, b in zip(points, points[1:]):
        if a[xi] <= x <= b[xi]:
            span = (b[xi] - a[xi]) or 1.0
            frac = (x - a[xi]) / span
            return a[yi] + frac * (b[yi] - a[yi])
    return points[-1][yi]


def time_to_char(chapter: dict, position_sec: float) -> int:
    """Seconds within a chapter -> char offset (alignment if available)."""
    pts = _alignment(chapter["index"])
    if pts:
        return int(round(_interp(pts, position_sec, 0, 1)))
    duration = chapter["duration_sec"] or 1.0
    frac = max(0.0, min(1.0, position_sec / duration))
    return int(round(chapter["char_count"] * frac))


def char_to_time(chapter: dict, char_offset: int) -> float:
    """Char offset within a chapter -> seconds (alignment if available)."""
    pts = _alignment(chapter["index"])
    if pts:
        return round(float(_interp(pts, char_offset, 1, 0)), 3)
    chars = chapter["char_count"] or 1
    frac = max(0.0, min(1.0, char_offset / chars))
    return round(chapter["duration_sec"] * frac, 3)


def resolve(manifest: dict, pos: PlaybackPosition) -> HeardCutoff:
    chapters = manifest["chapters"]
    n = len(chapters)
    idx = max(1, min(pos.chapter_index, n))

    cur = _chapter_by_index(manifest, idx)
    char_offset = max(0, min(cur["char_count"], time_to_char(cur, pos.position_sec)))

    parts: list[str] = []
    for ch in chapters:
        if ch["index"] < idx:
            parts.append(f"[{ch['title']}]\n{ch['text']}")
        elif ch["index"] == idx:
            heard_slice = ch["text"][:char_offset]
            if heard_slice.strip():
                parts.append(f"[{ch['title']} (in progress)]\n{heard_slice}")
    heard_text = "\n\n".join(parts).strip()

    remainder_in_chapter = char_offset < cur["char_count"]
    later_chapters = idx < n
    unheard_exists = remainder_in_chapter or later_chapters

    return HeardCutoff(
        chapter_index=idx,
        char_offset=char_offset,
        heard_text=heard_text,
        unheard_exists=unheard_exists,
    )


# ── phrase search -> playback position (WS-3: restart_from_phrase) ─────────────
def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def find_phrase(manifest: dict, phrase: str) -> tuple[PlaybackPosition | None, float]:
    """Locate a quoted line in the book. Returns (position, confidence 0..1).

    1) EXACT word-sequence match over the RAW text, tolerant of punctuation, extra
       whitespace, and case (so "paw dried" matches "paw, dried"). This returns the
       TRUE character offset -> an accurate timestamp (no proportional drift).
    2) Otherwise a fuzzy sliding-window match. The char offset is converted to a
       timestamp via alignment when available, else proportionally.
    """
    toks = _tokens(phrase)
    if not toks:
        return None, 0.0

    # 1) exact: allow any non-word chars (spaces, commas, quotes, newlines) between words
    pattern = re.compile(r"\W+".join(re.escape(t) for t in toks), re.IGNORECASE)
    for ch in manifest["chapters"]:
        m = pattern.search(ch["text"])
        if m:
            return PlaybackPosition(ch["index"], char_to_time(ch, m.start())), 1.0

    # 2) fuzzy: slide a window the size of the needle across each chapter (raw offsets)
    needle = _norm(phrase)
    best_ch, best_off, best_score = None, 0, 0.0
    for ch in manifest["chapters"]:
        hay = ch["text"]
        w = max(len(needle), 12)
        step = max(1, w // 2)
        for i in range(0, max(1, len(hay) - w + 1), step):
            score = SequenceMatcher(None, needle, _norm(hay[i:i + w])).ratio()
            if score > best_score:
                best_score, best_ch, best_off = score, ch["index"], i

    if best_ch is not None and best_score >= 0.6:
        chapter = _chapter_by_index(manifest, best_ch)
        return PlaybackPosition(best_ch, char_to_time(chapter, best_off)), best_score
    return None, best_score


def describe(manifest: dict, pos: PlaybackPosition, cutoff: HeardCutoff) -> str:
    """Human-readable one-liner for the CLI / logs."""
    cur = _chapter_by_index(manifest, cutoff.chapter_index)
    pct = 0 if not cur["char_count"] else round(100 * cutoff.char_offset / cur["char_count"])
    aligned = "aligned" if _alignment(cutoff.chapter_index) else "proportional"
    return (
        f"position: {cur['title']} @ {pos.position_sec:.0f}s "
        f"(~{pct}% into it, {aligned}) · heard {len(cutoff.heard_text)} chars · "
        f"unheard ahead: {cutoff.unheard_exists}"
    )
