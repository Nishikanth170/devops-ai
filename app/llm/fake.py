"""Deterministic offline LLM gateway.

Why it exists: the platform must be runnable, testable and demo-able without an
OpenAI key or network access (CI, air-gapped environments, local development).
``FakeLLMGateway`` implements :class:`BaseLLMGateway` with:

* **feature hashed embeddings** - the same text always produces the same vector,
  and texts that share tokens are cosine-similar, so retrieval behaves like the
  real thing (approximate, but monotonic in lexical overlap);
* **structural chat responses** - it inspects the system prompt and returns
  answers shaped like the real agents expect (markdown + fenced JSON for reviews,
  risk sections for deployments, cited history for incidents);
* **token accounting** - a rough ``chars / 4`` estimate so cost plumbing can be
  exercised.

It is *not* a reasoning model: it never invents history it was not given.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from app.core.config import Settings, get_settings
from app.llm.gateway import BaseLLMGateway, Message

_WORD_RE = re.compile(r"[a-z0-9_\-]+")
_FILE_RE = re.compile(r"(?:\+\+\+ b/|diff --git a/)(\S+)")
_HISTORY_RE = re.compile(r"^\[(\d+)\]\s+(.*)$", re.MULTILINE)


class FakeLLMGateway(BaseLLMGateway):
    """Offline, deterministic stand-in for a hosted LLM."""

    provider = "fake"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.dimensions = self.settings.VECTOR_DIMENSIONS
        self._stats: dict[str, Any] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "failures": 0,
            "by_model": {},
        }

    # --------------------------------------------------------- embeddings
    def _tokens(self, text: str) -> list[str]:
        return _WORD_RE.findall((text or "").lower())

    async def generate_embedding(self, text: str, model: Optional[str] = None) -> list[float]:
        """Bigram-anchored bag-of-words embedding (L2 normalised).

        Tokens are anchored by their *preceding* token (word pairs), so "pool
        exhausted" and "database pool" hash differently, but the frequent
        domain phrases shared by near-duplicate incident reports ("connection
        pool exhausted", "payments traffic spike") land on overlapping buckets.
        Single tokens are also counted so similarity degrades gracefully.
        """
        vector = [0.0] * self.dimensions
        tokens = self._tokens(text)
        if not tokens:
            return vector

        shingles = list(tokens)
        for index, current in enumerate(tokens):
            previous = tokens[index - 1] if index else "^"
            shingles.append(f"{previous} {current}")

        for feature in shingles:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = sum(value * value for value in vector) ** 0.5
        if norm > 0:
            vector = [value / norm for value in vector]
        self._track(text, "", model or "fake-embedding")
        return vector

    async def generate_embeddings_batch(
        self, texts: list[str], model: Optional[str] = None, batch_size: int = 100
    ) -> list[list[float]]:
        return [await self.generate_embedding(text, model=model) for text in texts]

    def _track(self, prompt: str, completion: str, model: str) -> float:
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(0, len(completion) // 4)
        self._stats["calls"] += 1
        self._stats["prompt_tokens"] += prompt_tokens
        self._stats["completion_tokens"] += completion_tokens
        self._stats["total_tokens"] += prompt_tokens + completion_tokens
        per_model = self._stats["by_model"].setdefault(
            model, {"calls": 0, "total_tokens": 0, "cost_usd": 0.0}
        )
        per_model["calls"] += 1
        per_model["total_tokens"] += prompt_tokens + completion_tokens
        return 0.0

    # ---------------------------------------------------------- completions
    async def chat_completion(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        response_format: Optional[dict[str, str]] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str | dict] = None,
    ) -> str:
        """Return a deterministic, structurally realistic answer."""
        system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        blob = "\n".join(m.get("content", "") for m in messages)

        if "Code Review Agent" in system:
            content = self._review_answer(blob)
        elif "DevOps Pipeline Agent" in system:
            content = self._pipeline_answer(blob)
        elif "Incident Response Agent" in system:
            content = self._incident_answer(blob)
        else:
            content = self._generic_answer(blob)

        self._track(blob, content, model or self.settings.OPENAI_MODEL)
        return content

    async def structured_output(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
        tool_name: str = "structured_response",
    ) -> dict[str, Any]:
        """Emit a JSON object shaped by the requested schema."""
        blob = "\n".join(m.get("content", "") for m in messages)
        properties = (schema or {}).get("properties", {}) or {}
        if "inline_comments" in properties:
            payload: dict[str, Any] = {"inline_comments": self._inline_comments(blob)}
        else:
            payload = {"summary": self._generic_answer(blob)}
        self._track(blob, json.dumps(payload), model or self.settings.OPENAI_MODEL)
        return payload

    # ------------------------------------------------------------ builders
    @staticmethod
    def _diff_files(blob: str) -> list[str]:
        files: list[str] = []
        for match in _FILE_RE.finditer(blob):
            path = match.group(1).strip()
            if path in {"", "/dev/null"} or path in files:
                continue
            files.append(path)
        return files

    @staticmethod
    def _history_lines(blob: str) -> list[str]:
        return [match.group(0).strip() for match in _HISTORY_RE.finditer(blob)]

    def _review_answer(self, blob: str) -> str:
        files = self._diff_files(blob)
        pattern_types = [
            line.split(":", 1)[1].strip().split()[0]
            for line in self._history_lines(blob)
            if "pattern" in line.lower() and ":" in line
        ]
        payload = {
            "overall_assessment": (
                f"medium risk - {len(files)} file(s) changed; "
                "offline reviewer used deterministic heuristics"
            ),
            "inline_comments": self._inline_comments(blob),
            "patterns_matched": pattern_types,
            "positive_notes": ["Diff is small enough to review in one pass"],
        }
        summary = "\n".join(
            [
                "**Overall Assessment**: medium risk - review the flagged lines before merging.",
                "**Patterns Matched**: "
                + (", ".join(pattern_types) if pattern_types else "none from history"),
            ]
        )
        return f"{summary}\n\n```json\n{json.dumps(payload, indent=2)}\n```"

    def _inline_comments(self, blob: str) -> list[dict[str, Any]]:
        comments: list[dict[str, Any]] = []
        for index, path in enumerate(self._diff_files(blob)[:5], start=1):
            comments.append(
                {
                    "file_path": path,
                    "line_number": index,
                    "severity": "warning" if index == 1 else "info",
                    "category": "bug" if index == 1 else "style",
                    "message": f"Verify error handling and test coverage for the change in {path}.",
                    "confidence": round(0.75 - 0.05 * index, 2),
                }
            )
        return comments

    def _incident_answer(self, blob: str) -> str:
        history = self._history_lines(blob)[:3]
        citations = "\n".join(f"- {line}" for line in history) or "- No matching history"
        return "\n".join(
            [
                "**Likely Root Cause**: dependency/resource exhaustion matching the cited "
                "history (confidence: moderate).",
                "**Suggested Runbook**: "
                + ("see the runbook of the cited incident" if history else "No matching runbook"),
                "**Immediate Actions**:",
                "1. Confirm the alert against current metrics (error rate, saturation, latency).",
                "2. Apply the documented mitigation for this failure mode.",
                "3. Record symptoms and resolution in the incident for future retrieval.",
                "**Similar Past Incidents**:",
                citations,
                "**Escalation**: Page the service owner if mitigation fails within 15 minutes.",
            ]
        )

    def _pipeline_answer(self, blob: str) -> str:
        failures = [line for line in self._history_lines(blob) if "failed" in line.lower()]
        return "\n".join(
            [
                "**Risk Assessment**: Medium - mixed change types with limited comparable history.",
                "**Risk Factors**:",
                *(
                    [f"- {line}" for line in failures[:3]]
                    or ["- No comparable failed deployments in history"]
                ),
                "**Recommended Gates**: integration tests in staging, canary rollout with "
                "explicit rollback criteria.",
                "**Systemic Patterns**: database migrations carry the highest historical "
                "risk weight; dependency bumps are the second largest contributor.",
            ]
        )

    @staticmethod
    def _generic_answer(blob: str) -> str:
        last_line = blob.strip().splitlines()[-1][:200] if blob.strip() else ""
        return f"Offline gateway response (LLM_PROVIDER=fake). Request: {last_line}"

    # -------------------------------------------------------- observability
    def get_usage_stats(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "calls": self._stats["calls"],
            "failures": 0,
            "prompt_tokens": self._stats["prompt_tokens"],
            "completion_tokens": self._stats["completion_tokens"],
            "total_tokens": self._stats["total_tokens"],
            "total_cost_usd": 0.0,
            "cost_per_1k_tokens": 0.0,
            "by_model": self._stats["by_model"],
        }
