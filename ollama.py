import os
import re

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _ollama_error(r: httpx.Response) -> str:
    """Extract a human-readable error from an Ollama error response."""
    try:
        msg = r.json().get("error", r.text)
    except Exception:
        msg = r.text or f"HTTP {r.status_code}"
    if "not found" in msg and "try pulling" in msg:
        model = msg.split('"')[1] if '"' in msg else msg
        return f"Ollama model not found: '{model}'. Run: ollama pull {model}"
    return f"Ollama error ({r.status_code}): {msg}"


async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
            )
        except httpx.ConnectError:
            raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_HOST}. Is Ollama running?")
        if not r.is_success:
            raise RuntimeError(_ollama_error(r))
        return r.json()["embedding"]


async def ollama_generate(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": "llama3.2:latest", "prompt": prompt, "stream": False, "format": "json"},
            )
        except httpx.ConnectError:
            raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_HOST}. Is Ollama running?")
        if not r.is_success:
            raise RuntimeError(_ollama_error(r))
        raw = r.json()["response"]
        return _FENCE_RE.sub("", raw).strip()


async def ollama_chat(messages: list) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": "llama3.2:latest", "messages": messages, "stream": False},
            )
        except httpx.ConnectError:
            raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_HOST}. Is Ollama running?")
        if not r.is_success:
            raise RuntimeError(_ollama_error(r))
        return r.json()["message"]["content"]
