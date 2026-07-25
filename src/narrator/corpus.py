"""B0 — Corpus Prep (offline, one-time).

Input : book.txt  +  audiobook_chapters/chapters/chapter_XX.wav
Output: data/book_manifest.json
        [{index, title, text, char_count, audio_path, duration_sec}, ...]

Chapter audio durations are read from the WAV header via the stdlib `wave`
module, so this step needs no ffmpeg and no third-party packages.
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
    title: str              # e.g. "Chapter I"
    text: str               # full text of this section
    char_count: int
    audio_path: str         # absolute path to the chapter wav
    duration_sec: float


# A section marker is a line that is just a Roman numeral + period, e.g. "II."
_SECTION_RE = re.compile(r"^\s*([IVXLCDM]+)\.\s*$", re.IGNORECASE)


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return round(frames / float(rate), 3) if rate else 0.0


def split_sections(book_text: str) -> list[str]:
    """Split the book into section texts on Roman-numeral marker lines.

    The text before the first marker (title / front matter) is dropped from the
    per-chapter text but preserved by the caller if needed.
    """
    lines = book_text.splitlines()
    sections: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if _SECTION_RE.match(line):
            current = []
            sections.append(current)
            continue
        if current is not None:
            current.append(line)
    return ["\n".join(s).strip() for s in sections]


def _chapter_wavs() -> list[Path]:
    return sorted(config.CHAPTERS_DIR.glob("chapter_*.wav"))


def build_manifest() -> list[ChapterMeta]:
    book_text = config.BOOK_TXT.read_text(encoding="utf-8")
    sections = split_sections(book_text)
    wavs = _chapter_wavs()

    if not sections:
        raise RuntimeError(f"No section markers found in {config.BOOK_TXT}")
    if len(sections) != len(wavs):
        raise RuntimeError(
            f"Section/chapter mismatch: {len(sections)} text sections but "
            f"{len(wavs)} wav files in {config.CHAPTERS_DIR}"
        )

    chapters: list[ChapterMeta] = []
    for i, (text, wav) in enumerate(zip(sections, wavs), start=1):
        chapters.append(
            ChapterMeta(
                index=i,
                title=f"Chapter {_to_roman(i)}",
                text=text,
                char_count=len(text),
                audio_path=str(wav.resolve()),
                duration_sec=_wav_duration_sec(wav),
            )
        )
    return chapters


def save_manifest(chapters: list[ChapterMeta]) -> Path:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": config.BOOK_TITLE,
        "n_chapters": len(chapters),
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


def _to_roman(n: int) -> str:
    numerals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, sym in numerals:
        while n >= value:
            out += sym
            n -= value
    return out


def main() -> None:
    chapters = build_manifest()
    path = save_manifest(chapters)
    print(f"Wrote {path}")
    total = sum(c.duration_sec for c in chapters)
    for c in chapters:
        mins = int(c.duration_sec // 60)
        secs = int(c.duration_sec % 60)
        print(
            f"  {c.title:<12} {c.char_count:>6} chars  "
            f"{mins:>2}m{secs:02d}s  {Path(c.audio_path).name}"
        )
    print(f"  total audio: {total/60:.1f} min")


if __name__ == "__main__":
    main()
