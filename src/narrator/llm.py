"""Thin local-LLM client for Ollama's chat API.

Uses only the standard library (urllib) so Milestone A has zero third-party
dependencies. Ollama serves an OpenAI-compatible endpoint too, but the native
`/api/chat` with `format="json"` is the most reliable way to force valid JSON
out of a local model, which is what the agent relies on.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config


class LLMError(RuntimeError):
    pass


def is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def model_available(model: str | None = None) -> bool:
    model = model or config.LLM_MODEL
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
        names = {m.get("name", "") for m in tags.get("models", [])}
        # match with or without an explicit :tag
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
    except Exception:
        return False


def chat(
    messages: list[dict],
    *,
    force_json: bool = False,
    temperature: float | None = None,
    model: str | None = None,
    num_predict: int | None = None,
    num_ctx: int = 8192,
) -> str:
    """Send a chat request to Ollama and return the assistant message content.

    num_ctx defaults to 8192 so the whole book (~5.5k tokens) plus the prompt
    fits without silent truncation. num_predict caps answer length to stop the
    model from rambling / retelling the story.
    """
    options: dict = {
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "num_ctx": num_ctx,
    }
    if num_predict is not None:
        options["num_predict"] = num_predict
    body = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if force_json:
        body["format"] = "json"

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_S) as r:
            payload = json.loads(r.read())
    except urllib.error.URLError as e:
        raise LLMError(
            f"Could not reach Ollama at {config.OLLAMA_URL}. Is `ollama serve` running? ({e})"
        ) from e
    return payload.get("message", {}).get("content", "")


def chat_json(messages: list[dict], **kw) -> dict:
    """chat() but parse the response as JSON, tolerating stray text/fences."""
    raw = chat(messages, force_json=True, **kw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise LLMError(f"Model did not return valid JSON:\n{raw}")


def chat_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    temperature: float | None = None,
    model: str | None = None,
    num_ctx: int = 8192,
) -> dict:
    """Send a tool-enabled chat request; return the assistant `message` dict.

    The returned dict may contain a `tool_calls` list (each with
    `function.name` and `function.arguments`) and/or `content`. Used by the
    agent's tool router (B6) to pick exactly one capability to run.
    """
    options: dict = {
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "num_ctx": num_ctx,
    }
    body = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": options,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_S) as r:
            payload = json.loads(r.read())
    except urllib.error.URLError as e:
        raise LLMError(
            f"Could not reach Ollama at {config.OLLAMA_URL}. Is `ollama serve` running? ({e})"
        ) from e
    return payload.get("message", {}) or {}


def embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Return embeddings for `texts` via Ollama's /api/embed (local, offline).

    Used by semantic cross-session memory (B7). Raises LLMError if the embed
    model isn't available so callers can fall back to keyword memory.
    """
    body = {"model": model or config.EMBED_MODEL, "input": texts}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/embed",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_S) as r:
            payload = json.loads(r.read())
    except urllib.error.URLError as e:
        raise LLMError(f"Could not reach Ollama embeddings: {e}") from e
    embs = payload.get("embeddings")
    if not embs:
        raise LLMError(f"No embeddings returned (is '{model or config.EMBED_MODEL}' pulled?)")
    return embs
