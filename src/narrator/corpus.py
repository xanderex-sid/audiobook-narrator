"""B0 — Corpus Prep (offline, one-time).

Input : book.txt  +  audiobook audio (either per-chapter wavs OR one combined wav)
Output: data/book_manifest.json
        {title, n_chapters, single_file, chapters:[{index, title, text,
         char_count, audio_path, audio_start_sec, duration_sec}, ...]}

Nothing here is specific to one story (WS-3):
- Chapter markers are auto-detected across several conventions (Roman numerals,
  "Chapter N", plain "N.", markdown headings). If none are found the whole book
  is a single chapter.
- Audio may be per-chapter wavs (chapter_XX.wav) OR a single combined wav. With a
  single wav, chapter time-boundaries come from (in order) chapters.json →
  forced-alignment cache (B4) → proportional-by-characters. `audio_start_sec` is
  the chapter's offset inside the shared wav (0.0 for per-chapter wavs).

WAV durations are read from the header via the stdlib `wave` module (no ffmpeg).
"""
from __future__ import annotations

import json
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config


@dataclass
class ChapterMeta:
    index: int              # 1-based chapter number
    title: str              # e.g. "Chapter I" or a detected heading
    text: str               # full text of this section
    char_count: int
    audio_path: str         # absolute path to the wav (shared in single-file mode)
    audio_start_sec: float  # offset of this chapter inside audio_path (0.0 per-chapter)
    duration_sec: float     # this chapter's own length in seconds


# ── title resolution (not hardcoded) ──────────────────────────────────────────
def resolve_title() -> str:
    import os

    env = os.environ.get("NARRATOR_TITLE")
    if env:
        return env.strip()
    meta = config.CHAPTERS_DIR.parent.parent / "data" / "book_meta.json"
    if meta.exists():
        try:
            t = json.loads(meta.read_text(encoding="utf-8")).get("title")
            if t:
                return str(t).strip()
        except Exception:
            pass
    # first short non-empty line of book.txt that isn't itself a chapter marker
    if config.BOOK_TXT.exists():
        for line in config.BOOK_TXT.read_text(encoding="utf-8").splitlines():
            s = line.strip().lstrip("#").strip()
            if s and not _detect_marker(s) and len(s) <= 80:
                return s
    return config.BOOK_TITLE_FALLBACK


# ── marker detection (multiple conventions) ───────────────────────────────────
_ROMAN = re.compile(r"^\s*([IVXLCDM]+)\.?\s*$", re.IGNORECASE)
_CHAPTER_WORD = re.compile(r"^\s*(?:chapter|part|book)\s+([0-9]+|[IVXLCDM]+)\b\.?\s*[:\-—]?\s*(.*)$", re.IGNORECASE)
_ARABIC = re.compile(r"^\s*([0-9]{1,3})\.\s*$")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,3}\s+(.+?)\s*#*\s*$")


def _detect_marker(line: str) -> str | None:
    """Return a chapter title if `line` is a chapter marker, else None."""
    m = _CHAPTER_WORD.match(line)
    if m:
        tail = m.group(2).strip()
        head = f"Chapter {m.group(1).upper() if not m.group(1).isdigit() else m.group(1)}"
        return f"{head}: {tail}" if tail else head
    m = _ROMAN.match(line)
    if m:
        return f"Chapter {m.group(1).upper()}"
    m = _ARABIC.match(line)
    if m:
        return f"Chapter {m.group(1)}"
    m = _MD_HEADING.match(line)
    if m:
        return m.group(1).strip()
    return None


def split_sections(book_text: str) -> list[tuple[str, str]]:
    """Split into (title, text) sections on the first marker style that yields >=2.

    Each style is tried independently so a stray "1." inside prose doesn't split a
    Roman-numeral book. If nothing yields >=2 sections, the whole text is one
    chapter titled "Chapter 1".
    """
    lines = book_text.splitlines()

    def split_with(matcher) -> list[tuple[str, str]]:
        sections: list[tuple[str, list[str]]] = []
        current: list[str] | None = None
        for line in lines:
            title = matcher(line)
            if title is not None:
                current = []
                sections.append((title, current))
                continue
            if current is not None:
                current.append(line)
        return [(t, "\n".join(body).strip()) for t, body in sections]

    matchers = [
        lambda ln: (_CHAPTER_WORD.match(ln) and _detect_marker(ln)) or None,
        lambda ln: (_ROMAN.match(ln) and _detect_marker(ln)) or None,
        lambda ln: (_MD_HEADING.match(ln) and _detect_marker(ln)) or None,
        lambda ln: (_ARABIC.match(ln) and _detect_marker(ln)) or None,
    ]
    best: list[tuple[str, str]] = []
    for matcher in matchers:
        secs = [(t, b) for t, b in split_with(matcher) if b.strip()]
        if len(secs) > len(best):
            best = secs
        if len(best) >= 2:
            break
    if len(best) >= 2:
        return best
    # no reliable markers -> single chapter (whole book)
    return [("Chapter 1", book_text.strip())]


# ── audio discovery ───────────────────────────────────────────────────────────
def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return round(frames / float(rate), 3) if rate else 0.0


def _per_chapter_wavs() -> list[Path]:
    return sorted(config.CHAPTERS_DIR.glob("chapter_*.wav"))


def _single_wav() -> Path | None:
    if config.SINGLE_AUDIO_ENV:
        p = Path(config.SINGLE_AUDIO_ENV).expanduser().resolve()
        return p if p.exists() else None
    wavs = sorted(config.CHAPTERS_DIR.glob("*.wav"))
    # "single-file" only when there's exactly one wav and it's not the chapter_XX set
    if len(wavs) == 1:
        return wavs[0]
    return None


def _boundaries_from_json(total_sec: float, n: int) -> list[float] | None:
    """Return chapter start offsets from chapters.json, or None if absent/invalid."""
    if not config.CHAPTERS_JSON.exists():
        return None
    try:
        data = json.loads(config.CHAPTERS_JSON.read_text(encoding="utf-8"))
        starts = [float(c["start_sec"]) for c in data]
    except Exception:
        return None
    if len(starts) != n:
        return None
    return starts


def _boundaries_proportional(sections: list[tuple[str, str]], total_sec: float) -> list[float]:
    """Approximate chapter start offsets by cumulative character share."""
    counts = [max(1, len(b)) for _, b in sections]
    total_chars = sum(counts)
    starts, acc = [], 0
    for c in counts:
        starts.append(round(total_sec * acc / total_chars, 3))
        acc += c
    return starts


# ── manifest build ────────────────────────────────────────────────────────────
def build_manifest() -> tuple[list[ChapterMeta], bool]:
    book_text = config.BOOK_TXT.read_text(encoding="utf-8")
    sections = split_sections(book_text)
    if not sections:
        raise RuntimeError(f"No text found in {config.BOOK_TXT}")

    per_chapter = _per_chapter_wavs()
    single = _single_wav()

    chapters: list[ChapterMeta] = []

    # An explicit NARRATOR_AUDIO always forces single-file mode.
    forced_single = bool(config.SINGLE_AUDIO_ENV) and single is not None

    # Mode A: per-chapter wavs that match the section count (classic path).
    if per_chapter and len(per_chapter) == len(sections) and not forced_single:
        for i, ((title, text), wav) in enumerate(zip(sections, per_chapter), start=1):
            chapters.append(ChapterMeta(
                index=i, title=title, text=text, char_count=len(text),
                audio_path=str(wav.resolve()), audio_start_sec=0.0,
                duration_sec=_wav_duration_sec(wav),
            ))
        return chapters, False

    # Mode B: single combined wav, split by time boundaries.
    if single is not None:
        total = _wav_duration_sec(single)
        n = len(sections)
        starts = _boundaries_from_json(total, n) or _boundaries_proportional(sections, total)
        for i, (title, text) in enumerate(sections, start=1):
            start = starts[i - 1]
            end = starts[i] if i < n else total
            chapters.append(ChapterMeta(
                index=i, title=title, text=text, char_count=len(text),
                audio_path=str(single.resolve()), audio_start_sec=round(start, 3),
                duration_sec=round(max(0.0, end - start), 3),
            ))
        return chapters, True

    # Mode C: per-chapter wavs but count mismatch — clear, actionable error.
    if per_chapter:
        raise RuntimeError(
            f"Section/chapter mismatch: {len(sections)} text sections but "
            f"{len(per_chapter)} chapter_*.wav in {config.CHAPTERS_DIR}. Either fix "
            f"the split, provide one combined wav, or add chapters.json."
        )
    raise RuntimeError(
        f"No audio found in {config.CHAPTERS_DIR} (need chapter_*.wav or a single wav)."
    )


def save_manifest(chapters: list[ChapterMeta], single_file: bool = False) -> Path:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": resolve_title(),
        "n_chapters": len(chapters),
        "single_file": single_file,
        "chapters": [asdict(c) for c in chapters],
    }
    config.MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config.MANIFEST_PATH


def load_manifest() -> dict:
    if not config.MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{config.MANIFEST_PATH} not found — run `python -m narrator.corpus` first."
        )
    return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))


def main() -> None:
    chapters, single = build_manifest()
    path = save_manifest(chapters, single_file=single)
    print(f"Wrote {path}  (title: {resolve_title()!r}, single_file={single})")
    total = sum(c.duration_sec for c in chapters)
    for c in chapters:
        mins, secs = int(c.duration_sec // 60), int(c.duration_sec % 60)
        loc = f"@{c.audio_start_sec:.0f}s" if single else Path(c.audio_path).name
        print(f"  {c.title[:28]:<28} {c.char_count:>6} chars  {mins:>2}m{secs:02d}s  {loc}")
    print(f"  total audio: {total/60:.1f} min")


if __name__ == "__main__":
    main()
