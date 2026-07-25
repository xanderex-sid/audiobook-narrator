"""LLM client — dispatches to OpenAI (cloud) or Ollama (local) by backend.

The public surface (`chat`, `chat_json`, `chat_tools`, `embed`, `is_up`,
`model_available`) is identical for both backends, so agent.py / memory.py /
preferences.py never know which brain is running. Selection is `config.BACKEND`.

- cloud: OpenAI Chat Completions (gpt-5.4-mini by default). Its tool-call shape
  (`function.name` + JSON-string `function.arguments`) already matches what the
  agent router expects, so no adapter is needed there.
- local: Ollama /api/chat (Qwen2.5) — the original fully-offline path.

Embeddings stay LOCAL (nomic via Ollama) in both backends; if Ollama is down the
caller falls back to non-semantic memory.

Uses only the standard library (urllib) so the brain has zero third-party deps.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config


class LLMError(RuntimeError):
    pass


# ── readiness ─────────────────────────────────────────────────────────────────
def is_up() -> bool:
    if config.BACKEND == "cloud":
        return bool(config.OPENAI_API_KEY)
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def model_available(model: str | None = None) -> bool:
    if config.BACKEND == "cloud":
        return bool(config.OPENAI_API_KEY)
    model = model or config.LLM_MODEL
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
        names = {m.get("name", "") for m in tags.get("models", [])}
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
    except Exception:
        return False


# ── OpenAI (cloud) ────────────────────────────────────────────────────────────
def _openai_post(path: str, body: dict) -> dict:
    if not config.OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY is not set (add it to .env).")
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OPENAI_BASE_URL}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_S) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "ignore")[:400]
        except Exception:
            pass
        raise LLMError(f"OpenAI {path} failed ({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"Could not reach OpenAI ({e}).") from e


def _openai_chat(messages, *, force_json, temperature, model, num_predict) -> str:
    body: dict = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
    }
    if num_predict is not None:
        body["max_completion_tokens"] = max(int(num_predict), 64)
    if force_json:
        body["response_format"] = {"type": "json_object"}
    payload = _openai_post("/chat/completions", body)
    try:
        return payload["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError):
        return ""


def _openai_chat_tools(messages, tools, *, temperature, model) -> dict:
    body = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
    }
    payload = _openai_post("/chat/completions", body)
    try:
        msg = payload["choices"][0]["message"]
    except (KeyError, IndexError):
        return {}
    # Already in the {content, tool_calls:[{function:{name, arguments(str)}}]} shape
    # the agent router consumes.
    return {"content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or []}


# ── Ollama (local) ────────────────────────────────────────────────────────────
def _ollama_post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_S) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        raise LLMError(
            f"Could not reach Ollama at {config.OLLAMA_URL}. Is `ollama serve` running? ({e})"
        ) from e


def _ollama_chat(messages, *, force_json, temperature, model, num_predict, num_ctx) -> str:
    options: dict = {
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "num_ctx": num_ctx,
    }
    if num_predict is not None:
        options["num_predict"] = num_predict
    body = {"model": model or config.LLM_MODEL, "messages": messages, "stream": False, "options": options}
    if force_json:
        body["format"] = "json"
    payload = _ollama_post("/api/chat", body)
    return payload.get("message", {}).get("content", "")


def _ollama_chat_tools(messages, tools, *, temperature, model, num_ctx) -> dict:
    options = {
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "num_ctx": num_ctx,
    }
    body = {"model": model or config.LLM_MODEL, "messages": messages, "tools": tools,
            "stream": False, "options": options}
    payload = _ollama_post("/api/chat", body)
    return payload.get("message", {}) or {}


# ── public API (backend-dispatched) ───────────────────────────────────────────
def chat(
    messages: list[dict],
    *,
    force_json: bool = False,
    temperature: float | None = None,
    model: str | None = None,
    num_predict: int | None = None,
    num_ctx: int = 8192,
) -> str:
    """Send a chat request; return the assistant message content.

    `num_predict` caps answer length (so the model can't ramble / retell the
    story); `num_ctx` is Ollama-only (cloud manages context automatically).
    """
    if config.BACKEND == "cloud":
        return _openai_chat(messages, force_json=force_json, temperature=temperature,
                            model=model, num_predict=num_predict)
    return _ollama_chat(messages, force_json=force_json, temperature=temperature,
                        model=model, num_predict=num_predict, num_ctx=num_ctx)


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
    """Tool-enabled chat; return the assistant `message` dict.

    The dict may contain `tool_calls` (each with `function.name` and
    `function.arguments`) and/or `content`. Used by the agent's tool router (B6).
    """
    if config.BACKEND == "cloud":
        return _openai_chat_tools(messages, tools, temperature=temperature, model=model)
    return _ollama_chat_tools(messages, tools, temperature=temperature, model=model, num_ctx=num_ctx)


def embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Return embeddings for `texts` via Ollama's /api/embed (LOCAL, both backends).

    Used by semantic cross-session memory (B7). Raises LLMError if the embed
    model isn't available so callers can fall back to keyword memory.
    """
    body = {"model": model or config.EMBED_MODEL, "input": texts}
    payload = _ollama_post("/api/embed", body)
    embs = payload.get("embeddings")
    if not embs:
        raise LLMError(f"No embeddings returned (is '{model or config.EMBED_MODEL}' pulled?)")
    return embs
