"""LLM gateway: prompts, retries, cost tracking and structured output.

``BaseLLMGateway`` is the interface every agent programs against; ``LLMGateway``
is the OpenAI implementation and ``FakeLLMGateway`` (app/llm/fake.py) is a
deterministic offline implementation used by tests and local development.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError, LLMGatewayError
from app.core.utils import truncate

logger = logging.getLogger(__name__)

Message = dict[str, str]

#: USD per 1M tokens (input, output). Keep in sync with the provider pricing page.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


@dataclass
class LLMResponse:
    """Detailed result of a completion call (use ``chat_completion`` for text)."""

    content: str
    model: str
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMGateway(ABC):
    """Interface used by agents and the memory service."""

    provider: str = "unknown"

    @abstractmethod
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
        """Return the assistant text for a chat conversation."""

    @abstractmethod
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> list[float]:
        """Return the embedding vector for ``text``."""

    @abstractmethod
    async def generate_embeddings_batch(
        self, texts: list[str], model: Optional[str] = None, batch_size: int = 100
    ) -> list[list[float]]:
        """Return embeddings for many texts."""

    @abstractmethod
    async def structured_output(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
        tool_name: str = "structured_response",
    ) -> dict[str, Any]:
        """Return a JSON object that validates against ``schema``."""

    @abstractmethod
    def get_usage_stats(self) -> dict[str, Any]:
        """Token/cost counters for observability."""

    async def aclose(self) -> None:  # pragma: no cover - overridden when needed
        """Release provider resources."""
        return None


class LLMGateway(BaseLLMGateway):
    """OpenAI chat/embeddings gateway with retries, timeouts and cost tracking."""

    provider = "openai"

    def __init__(self, settings: Optional[Settings] = None, client: Any = None) -> None:
        self.settings = settings or get_settings()
        self._stats: dict[str, Any] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "failures": 0,
            "by_model": {},
        }
        if client is not None:
            self.client = client
            return
        if not self.settings.OPENAI_API_KEY:
            raise ConfigurationError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "Set it in the environment or use LLM_PROVIDER=fake for offline runs."
            )
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            base_url=self.settings.OPENAI_BASE_URL,
            timeout=self.settings.LLM_TIMEOUT_SECONDS,
            max_retries=0,  # tenacity owns the retry policy
        )

    # ------------------------------------------------------------ helpers
    def _retrying(self) -> AsyncRetrying:
        settings = self.settings
        return AsyncRetrying(
            stop=stop_after_attempt(max(1, settings.LLM_MAX_RETRIES)),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

    async def _with_retry(
        self, operation: str, factory: Callable[[], Coroutine[Any, Any, Any]]
    ) -> Any:
        """Run ``factory`` with exponential backoff, mapping failures to a domain error."""
        try:
            async for attempt in self._retrying():
                with attempt:
                    return await factory()
        except Exception as exc:  # pragma: no cover - depends on provider
            logger.error("LLM %s failed after retries: %s", operation, exc)
            raise LLMGatewayError(f"LLM {operation} failed: {exc}") from exc
        return None  # pragma: no cover - unreachable

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
        return (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]

    def _track_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: int
    ) -> float:
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        self._stats["calls"] += 1
        self._stats["prompt_tokens"] += prompt_tokens
        self._stats["completion_tokens"] += completion_tokens
        self._stats["total_tokens"] += prompt_tokens + completion_tokens
        self._stats["cost_usd"] += cost
        per_model = self._stats["by_model"].setdefault(
            model, {"calls": 0, "total_tokens": 0, "cost_usd": 0.0}
        )
        per_model["calls"] += 1
        per_model["total_tokens"] += prompt_tokens + completion_tokens
        per_model["cost_usd"] += cost
        logger.info(
            "llm_call model=%s prompt=%s completion=%s cost=$%.6f latency=%sms",
            model,
            prompt_tokens,
            completion_tokens,
            cost,
            latency_ms,
        )
        return cost

    # -------------------------------------------------------- completions
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
        """Generate a chat completion (text) with retries and cost tracking."""
        model_name = model or self.settings.OPENAI_MODEL
        started = time.perf_counter()

        async def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice or "auto"
            return await self.client.chat.completions.create(**kwargs)

        response = await self._with_retry("chat_completion", _call)
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._track_usage(
                model_name,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                latency_ms,
            )
        return content

    async def structured_output(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
        tool_name: str = "structured_response",
    ) -> dict[str, Any]:
        """Force a JSON tool call so the model answer matches ``schema``."""
        import json

        model_name = model or self.settings.OPENAI_MODEL
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Return the structured response matching the schema.",
                    "parameters": schema,
                },
            }
        ]

        async def _call() -> Any:
            return await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": tool_name}},
                temperature=temperature,
            )

        response = await self._with_retry("structured_output", _call)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._track_usage(
                model_name,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                0,
            )

        tool_calls = getattr(response.choices[0].message, "tool_calls", None) or []
        if not tool_calls:
            raise LLMGatewayError("Model did not return a structured tool call")
        try:
            return json.loads(tool_calls[0].function.arguments)
        except (ValueError, AttributeError) as exc:
            raise LLMGatewayError(f"Could not parse structured output: {exc}") from exc

    # --------------------------------------------------------- embeddings
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> list[float]:
        """Embed one string (truncated to the model's safe input window)."""
        model_name = model or self.settings.OPENAI_EMBEDDING_MODEL
        payload = truncate(text or "", self.settings.LLM_MAX_EMBEDDING_CHARS, suffix="")
        if not payload:
            payload = " "

        async def _call() -> Any:
            return await self.client.embeddings.create(
                model=model_name, input=payload, encoding_format="float"
            )

        response = await self._with_retry("generate_embedding", _call)
        usage = getattr(response, "usage", None)
        if usage is not None:
            tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", 0) or 0
            self._track_usage(model_name, tokens, 0, 0)
        return list(response.data[0].embedding)

    async def generate_embeddings_batch(
        self, texts: list[str], model: Optional[str] = None, batch_size: int = 100
    ) -> list[list[float]]:
        """Embed many strings, sending ``batch_size`` inputs per request."""
        model_name = model or self.settings.OPENAI_EMBEDDING_MODEL
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [
                truncate(t or "", self.settings.LLM_MAX_EMBEDDING_CHARS, suffix="") or " "
                for t in texts[start : start + batch_size]
            ]

            async def _call(payload: list[str] = batch) -> Any:
                return await self.client.embeddings.create(
                    model=model_name, input=payload, encoding_format="float"
                )

            response = await self._with_retry("generate_embeddings_batch", _call)
            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens = (
                    getattr(usage, "prompt_tokens", None)
                    or getattr(usage, "total_tokens", 0)
                    or 0
                )
                self._track_usage(model_name, tokens, 0, 0)
            embeddings.extend(list(item.embedding) for item in response.data)
        return embeddings

    # -------------------------------------------------------- observability
    def get_usage_stats(self) -> dict[str, Any]:
        """Token/cost counters since process start."""
        total_tokens = int(self._stats["total_tokens"])
        cost = float(self._stats["cost_usd"])
        return {
            "provider": self.provider,
            "calls": self._stats["calls"],
            "failures": self._stats["failures"],
            "prompt_tokens": self._stats["prompt_tokens"],
            "completion_tokens": self._stats["completion_tokens"],
            "total_tokens": total_tokens,
            "total_cost_usd": round(cost, 6),
            "cost_per_1k_tokens": round(cost / (total_tokens / 1000), 6) if total_tokens else 0.0,
            "by_model": {
                name: {**values, "cost_usd": round(values["cost_usd"], 6)}
                for name, values in self._stats["by_model"].items()
            },
        }

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def build_llm_gateway(settings: Optional[Settings] = None) -> BaseLLMGateway:
    """Factory selecting the OpenAI or offline (fake) gateway."""
    from app.llm.fake import FakeLLMGateway

    settings = settings or get_settings()
    if settings.LLM_PROVIDER == "fake":
        logger.warning("LLM_PROVIDER=fake - using deterministic offline gateway")
        return FakeLLMGateway(settings)
    return LLMGateway(settings)
