"""Utility functions: file hashing, retry logic, text chunking."""
from __future__ import annotations

import hashlib
import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def hash_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file's content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    """Return SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    import re
    # Strip markdown heading markers (# ## ###)
    text = re.sub(r"^#+\s*", "", text.strip())
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    # Remove leading/trailing hyphens that come from stripping non-word chars
    text = text.strip("-")
    return text[:80]


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at sentence boundary
        if end < len(text):
            for sep in (". ", "\n\n", "\n", " "):
                pos = text.rfind(sep, start, end)
                if pos > start + (chunk_size // 2):
                    end = pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator: retry a function with exponential backoff."""
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


def safe_json_parse(text: str) -> dict[str, Any] | None:
    """Parse JSON from LLM output, stripping markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None
