"""B8 — Session Orchestrator.

The one place that ties the brain (B6), spoiler gate (B4), position/navigation
(B1), and memory (B7) together. Both the text CLI and the voice loop drive the
system through this class, so behaviour stays identical across modes.

    session = NarratorSession(manifest, start_pos)
    reply = session.handle("who is Herbert?")      # -> AgentResponse (+ position updated)
    session.end()                                  # persists summary + resume position
"""
from __future__ import annotations

from . import agent, memory, playback, position
from .position import PlaybackPosition


class NarratorSession:
    def __init__(self, manifest: dict, start_pos: PlaybackPosition):
        self.manifest = manifest
        self.pos = playback.clamp(manifest, start_pos)
        self.log = memory.SessionLog()
        self.memory_context = memory.recent_context()
        self.pending_spoiler: dict | None = None   # WS-2 warn-then-confirm state

    # ── introspection ────────────────────────────────────────────────────────
    def cutoff(self):
        return position.resolve(self.manifest, self.pos)

    def position_line(self) -> str:
        c = self.cutoff()
        return position.describe(self.manifest, self.pos, c)

    # ── one turn ──────────────────────────────────────────────────────────────
    def handle(self, query: str) -> tuple[agent.AgentResponse, str]:
        """Process one listener utterance.

        Returns (response, nav_note). Side effects: position updated on a
        navigation action; the turn is logged for memory.
        """
        cutoff = self.cutoff()
        self.log.add("listener", query)

        # Per-turn memory: recent-session continuity + semantic recall for THIS query.
        mem = memory.turn_context(query) or self.memory_context

        resp = agent.respond(
            self.manifest,
            query,
            self.pos,
            cutoff,
            memory_context=mem,
            pending_spoiler=self.pending_spoiler,
        )
        self.pending_spoiler = resp.pending_spoiler  # carry warn->confirm across turns
        self.log.add("narrator", resp.speech_text)

        new_pos, note = playback.apply_action(self.manifest, self.pos, resp.action)
        if note:
            self.pos = new_pos
        return resp, note

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def end(self) -> None:
        """Persist the session summary + resume point, and refresh preferences."""
        self.log.end(resume_pos=self.pos)
        try:
            from . import preferences
            preferences.generate(applies_from_chapter=self.pos.chapter_index + 1)
        except Exception:
            pass  # preferences are a nicety; never fail a session over them
