"""WS-9 — Listener preference judge -> audiobook-generator handoff.

An LLM judge reads the listener's own utterances across sessions and infers TWO
preferences, then writes a self-describing, versioned JSON that a downstream
(black-box) audiobook *generator* can consume to regenerate LATER chapters:

  (a) vocabulary : "simplified" | "as_written"
        signal: how often the listener asks for word/passage MEANINGS.
  (b) pacing     : "condensed"  | "full"
        signal: how often they ask for SUMMARIES / RECAPS or SKIP chapters.

Design choices
--------------
- Deterministic signal tallies (keyword heuristics over listener turns) give
  robust evidence and a fallback; the local LLM judge produces the final call +
  natural-language evidence and the generator directive.
- Low confidence -> the SAFE default (as_written / full): never over-simplify or
  over-condense a book on weak evidence.
- Nothing here is specific to one story.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import config, corpus, llm

SCHEMA_VERSION = "1.0"

_MEANING_PAT = re.compile(
    r"\b(what does|what's|whats|meaning of|define|definition|explain|explanation|"
    r"what is the meaning|in simple|simpler|simplif|too hard|difficult word|"
    r"what do you mean|paraphrase)\b", re.IGNORECASE)
_SUMMARY_PAT = re.compile(
    r"\b(summar|recap|tl;?dr|in short|shorten|too long|get to the point|"
    r"brief|quick version|skip|fast forward|move on)\b", re.IGNORECASE)


def _book_id() -> str:
    title = corpus.resolve_title()
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "book"


def _listener_turns() -> list[str]:
    """All listener utterances across saved session transcripts."""
    turns: list[str] = []
    for f in sorted(config.SESSIONS_DIR.glob("session-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        turns += [t["text"] for t in data.get("turns", []) if t.get("role") == "listener"]
    return turns


def _tally(turns: list[str]) -> dict:
    meaning = sum(1 for t in turns if _MEANING_PAT.search(t))
    summary = sum(1 for t in turns if _SUMMARY_PAT.search(t))
    return {"n_turns": len(turns), "meaning_requests": meaning, "summary_or_skip_requests": summary}


_JUDGE_SYSTEM = """\
You infer a listener's preferences for how an audiobook should be (re)generated, \
from THEIR OWN questions/comments. Output ONLY JSON in this exact shape:

{
  "vocabulary": {"value": "simplified"|"as_written", "confidence": 0.0-1.0, "evidence": ["..."]},
  "pacing":     {"value": "condensed"|"full",        "confidence": 0.0-1.0, "evidence": ["..."]}
}

Guidance:
- vocabulary=simplified if they REPEATEDLY ask what words/sentences mean or ask for \
simpler language; otherwise as_written.
- pacing=condensed if they REPEATEDLY ask for summaries/recaps or skip ahead; else full.
- confidence reflects how strong/consistent the evidence is. With little or mixed \
evidence, use LOW confidence.
- evidence = short quotes/paraphrases of the listener's actual asks. Never invent.
No prose outside the JSON.
"""

_SAFE = {
    "vocabulary": {"value": "as_written", "confidence": 0.0, "evidence": []},
    "pacing": {"value": "full", "confidence": 0.0, "evidence": []},
}


def _judge(turns: list[str]) -> dict:
    if not turns:
        return dict(_SAFE)
    convo = "\n".join(f"- {t}" for t in turns[-60:])
    try:
        out = llm.chat_json(
            [{"role": "system", "content": _JUDGE_SYSTEM},
             {"role": "user", "content": f"Listener's utterances:\n{convo}"}],
            temperature=0.0,
        )
        prefs = {}
        for key, safe in _SAFE.items():
            p = out.get(key) or {}
            val = p.get("value")
            if val not in (("simplified", "as_written") if key == "vocabulary" else ("condensed", "full")):
                val = safe["value"]
            conf = float(p.get("confidence") or 0.0)
            ev = [str(e) for e in (p.get("evidence") or [])][:5]
            # low confidence -> safe default
            if conf < 0.34:
                val = safe["value"]
            prefs[key] = {"value": val, "confidence": round(conf, 2), "evidence": ev}
        return prefs
    except Exception:
        return dict(_SAFE)


def _directive(prefs: dict, applies_from: int) -> str:
    v = prefs["vocabulary"]["value"]
    p = prefs["pacing"]["value"]
    vocab = ("use simpler, everyday vocabulary and gently rephrase archaic or difficult wording"
             if v == "simplified" else "keep the author's original vocabulary and phrasing")
    pace = ("condense the narration (aim ~30% shorter), tightening description while keeping every plot beat"
            if p == "condensed" else "keep the full, unabridged narration")
    return (f"From chapter {applies_from} onward, {vocab}, and {pace}. "
            f"Preserve the plot, character names, and the order in which events/reveals occur "
            f"(introduce no new spoilers).")


def generate(applies_from_chapter: int = 1) -> dict:
    """Build the preference JSON, write it under data/preferences/, and return it."""
    turns = _listener_turns()
    prefs = _judge(turns)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "book_id": _book_id(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "applies_from_chapter": applies_from_chapter,
        "signals": _tally(turns),
        "preferences": prefs,
        "instructions_for_generator": _directive(prefs, applies_from_chapter),
    }
    config.PREFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PREFERENCES_DIR / f"{_book_id()}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Infer listener preferences for the audiobook generator")
    ap.add_argument("--from-chapter", type=int, default=1, help="regenerate from this chapter onward")
    args = ap.parse_args(argv)
    payload = generate(args.from_chapter)
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {config.PREFERENCES_DIR / (_book_id() + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
