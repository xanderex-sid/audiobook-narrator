"""Text narrator CLI (Milestones A + B).

Set where you are in the audiobook, then type questions. Proves the whole brain:
spoiler-gated Q&A, recaps, summaries, navigation, resume-position updates, and
cross-session memory — with no audio infrastructure.

Examples
--------
    python -m narrator.cli --chapter 2 --offset 50%
    python -m narrator.cli                      # resumes where you left off

REPL:
    who is Sergeant-Major Morris?
    :pos          show position     :goto N   jump chapter     :quit
"""
from __future__ import annotations

import argparse
import sys

from . import corpus, llm, memory
from .orchestrator import NarratorSession
from .position import PlaybackPosition


def _parse_offset(offset: str, duration_sec: float) -> float:
    offset = offset.strip().lower()
    if offset.endswith("%"):
        return max(0.0, min(1.0, float(offset[:-1]) / 100.0)) * duration_sec
    if offset.endswith("s"):
        offset = offset[:-1]
    return float(offset)


def _duration(manifest: dict, index: int) -> float:
    for ch in manifest["chapters"]:
        if ch["index"] == index:
            return float(ch["duration_sec"])
    return 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audiobook Narrator — text mode")
    ap.add_argument("--chapter", type=int, default=None, help="current chapter (1-based)")
    ap.add_argument("--offset", type=str, default=None, help="seconds (90/90s) or percent (50%%)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the manifest first")
    args = ap.parse_args(argv)

    if args.rebuild or not corpus.config.MANIFEST_PATH.exists():
        corpus.save_manifest(corpus.build_manifest())
    manifest = corpus.load_manifest()

    if not llm.is_up():
        print(f"! Ollama not reachable at {llm.config.OLLAMA_URL} — run scripts/serve.sh",
              file=sys.stderr)
        return 2
    if not llm.model_available():
        print(f"! Model '{llm.config.LLM_MODEL}' not pulled — run: ollama pull {llm.config.LLM_MODEL}",
              file=sys.stderr)
        return 2

    # Starting position: explicit args override; otherwise resume from memory.
    if args.chapter is not None or args.offset is not None:
        ch = args.chapter or 1
        dur = _duration(manifest, ch)
        start = PlaybackPosition(ch, _parse_offset(args.offset or "100%", dur))
    else:
        start = memory.get_resume_position() or PlaybackPosition(1, 0.0)

    session = NarratorSession(manifest, start)
    print(f'📖  {manifest["title"]}  —  Narrator (text mode, model={llm.config.LLM_MODEL})')
    print("   ", session.position_line())
    if session.memory_context:
        print("    (remembering earlier sessions)")
    print("    Type a question, or :pos / :goto N / :quit\n")

    try:
        while True:
            try:
                query = input("you  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            if query in (":quit", ":q", "exit"):
                break
            if query == ":pos":
                print("   ", session.position_line())
                continue
            if query.startswith(":goto"):
                parts = query.split()
                if len(parts) == 2 and parts[1].isdigit():
                    resp, note = session.handle(f"go to chapter {parts[1]}")
                    print("   ", note or f"↦ chapter {parts[1]}")
                    print("   ", session.position_line())
                continue

            try:
                resp, note = session.handle(query)
            except llm.LLMError as e:
                print("! LLM error:", e, file=sys.stderr)
                continue

            tag = " [spoiler revealed]" if resp.spoiler_used else ""
            print(f"narr > {resp.speech_text}{tag}")
            if note:
                print("   ", note)
                print("   ", session.position_line())
            print()
    finally:
        session.end()

    print("bye 👋  (progress + notes saved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
