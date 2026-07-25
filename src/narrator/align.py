"""B4 (offline) — Forced alignment: word timestamps -> time<->char map.

Transcribes each chapter's audio with faster-whisper word timestamps, then aligns
those spoken words to the chapter TEXT to produce monotonic [time_sec, char_offset]
checkpoints. `position.py` reads these so the within-chapter spoiler cutoff and
"restart from this line" become exact instead of proportional.

Runs in the voice Python (needs faster-whisper). Writes one file per chapter:
    data/alignment/chapter_<i>.json  = {chapter_index, duration_sec, points:[[t,c],...]}

Usage:
    $NARRATOR_VOICE_PYTHON -m narrator.align            # align all chapters
    $NARRATOR_VOICE_PYTHON -m narrator.align --chapter 2

Per-chapter wavs align directly. Single-file audiobooks (shared wav) currently
fall back to proportional/ chapters.json in position.py — alignment there is a
later refinement.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from . import config, corpus

_TOKEN = re.compile(r"[A-Za-z0-9']+")


def _text_tokens(text: str) -> list[tuple[str, int]]:
    """[(lowercased_token, char_index_in_text), ...]."""
    return [(m.group(0).lower(), m.start()) for m in _TOKEN.finditer(text)]


def _clean(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalnum() or ch == "'")


def align_chapter(chapter: dict, stt) -> dict | None:
    """Build the checkpoint map for one chapter. Returns the payload or None.

    Aligns the SPOKEN transcript (word + timestamp) to the book TEXT with a proper
    subsequence alignment (difflib), not a greedy forward scan — the greedy version
    raced ahead on common words and compressed the whole map into the first third of
    the audio. Each matched run contributes monotonic [time_sec, char_offset] points.
    """
    words = stt.transcribe_words(chapter["audio_path"])
    if not words:
        return None

    toks = _text_tokens(chapter["text"])          # [(token, char_offset), ...]
    if not toks:
        return None

    # spoken tokens with their start times (drop empties)
    spoken = [(_clean(raw), float(start)) for raw, start, _end in words if _clean(raw)]
    if not spoken:
        return None
    dg_seq = [w for w, _ in spoken]
    bk_seq = [t for t, _ in toks]

    # autojunk=False so frequent words ("the", "a") still anchor the alignment.
    sm = SequenceMatcher(None, dg_seq, bk_seq, autojunk=False)
    points: list[list[float]] = [[0.0, 0]]

    def add(t: float, c: int) -> None:
        if t >= points[-1][0] and c >= points[-1][1]:
            points.append([round(t, 3), int(c)])

    for a, b, size in sm.get_matching_blocks():
        if size == 0:
            continue
        add(spoken[a][1], toks[b][1])                       # start of the matched run
        if size > 8:                                        # and the end of a long run
            add(spoken[a + size - 1][1], toks[b + size - 1][1])

    points.append([float(chapter["duration_sec"]), int(chapter["char_count"])])

    # Downsample: keep points at least ~1.5s apart to bound file size.
    thinned = [points[0]]
    for p in points[1:]:
        if p[0] - thinned[-1][0] >= 1.5 or p is points[-1]:
            thinned.append(p)
    if thinned[-1] != points[-1]:
        thinned.append(points[-1])

    return {
        "chapter_index": chapter["index"],
        "duration_sec": chapter["duration_sec"],
        "points": thinned,
    }


def run(chapter_index: int | None = None) -> int:
    from . import voice

    manifest = corpus.load_manifest()
    if manifest.get("single_file"):
        print("Single-file audiobook: alignment falls back to proportional/chapters.json "
              "(per-chapter alignment not built).")
        return 0

    config.ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
    # Backend-aware: cloud uses Deepgram word timestamps (accurate), local uses
    # faster-whisper. The map is what makes find_phrase / spoiler cutoff land exactly.
    stt = voice.make_stt()
    print(f"[align backend: {config.BACKEND}]")
    done = 0
    for ch in manifest["chapters"]:
        if chapter_index and ch["index"] != chapter_index:
            continue
        print(f"aligning {ch['title']} ...", flush=True)
        payload = align_chapter(ch, stt)
        if not payload:
            print(f"  (skipped {ch['title']} — no words)")
            continue
        out = config.ALIGNMENT_DIR / f"chapter_{ch['index']}.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        print(f"  wrote {out.name}  ({len(payload['points'])} checkpoints)")
        done += 1
    print(f"done: {done} chapter(s) aligned -> {config.ALIGNMENT_DIR}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Forced alignment (word timestamps -> char map)")
    ap.add_argument("--chapter", type=int, default=None, help="align only this chapter")
    args = ap.parse_args(argv)
    return run(args.chapter)


if __name__ == "__main__":
    raise SystemExit(main())
