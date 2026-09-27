"""Event queue with priority lanes.

Two interchangeable backends:

``redis``   (default, production) - durable LPUSH/BRPOP lists with a separate
            priority list, so incidents are never stuck behind routine events.
``memory``  (development / tests) - an in-process asyncio priority queue, so the
            platform runs without Redis.

Both implement :class:`BaseEventQueue`: register handlers, publish, run the
consumer loop.  Failed handler executions land in a dead-letter list instead of
being silently dropped.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import QueueError
from app.events.schemas import EventPriority, InternalEvent

logger = logging.getLogger(__name__)

EventHandler = Callable[[InternalEvent], Awaitable[Any]]

_PRIORITY_RANK: dict[str, int] = {"critical": 0, "high": 1, "normal": 2, "low": 3}
_WILDCARD = "*"


class BaseEventQueue(ABC):
    """Handler registry + consumer loop shared by both backends."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._handlers: dict[str, list[EventHandler]] = {}
        self._consumer_task: Optional[asyncio.Task] = None
        self.published = 0
        self.processed = 0
        self.failed = 0
        self.dead_letters: list[tuple[InternalEvent, str]] = []

    # -------------------------------------------------------------- registry
    def register_handler(self, event_type: str, handler: EventHandler) -> None:
        """Register ``handler`` for ``event_type`` (``"*"`` = every event)."""
        self._handlers.setdefault(event_type, []).append(handler)

    def register_handlers(self, mapping: dict[str, EventHandler]) -> None:
        for event_type, handler in mapping.items():
            self.register_handler(event_type, handler)

    def handler_count(self) -> int:
        return sum(len(handlers) for handlers in self._handlers.values())

    # ------------------------------------------------------------ lifecycle
    async def start_consumer(self) -> None:
        """Start the background consumer loop (idempotent)."""
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        """Stop the consumer loop and release backend resources."""
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover - shutdown must not raise
                logger.debug("consumer task ended with error: %s", exc)
            self._consumer_task = None
        await self._close()

    # ------------------------------------------------------------- contract
    @abstractmethod
    async def publish(self, event: InternalEvent) -> None:
        """Enqueue an event for asynchronous processing."""

    @abstractmethod
    async def _next_event(self) -> Optional[InternalEvent]:
        """Block until the next event is available (``None`` on timeout)."""

    async def _close(self) -> None:  # pragma: no cover - overridden by redis
        return None

    # ------------------------------------------------------------------ loop
    async def _consume_loop(self) -> None:
        while True:
            try:
                event = await self._next_event()
                if event is None:
                    continue
                await self.dispatch(event)
            except asyncio.CancelledError:
                logger.info("event consumer stopped")
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("event consumer error: %s", exc)
                await asyncio.sleep(1)

    async def dispatch(self, event: InternalEvent) -> None:
        """Invoke every handler registered for the event type (then wildcards)."""
        handlers = list(self._handlers.get(event.event_type, [])) + list(
            self._handlers.get(_WILDCARD, [])
        )
        if not handlers:
            logger.debug("no handler for event type %s", event.event_type)
            return

        for handler in handlers:
            try:
                await handler(event)
                self.processed += 1
            except Exception as exc:
                self.failed += 1
                self.dead_letters.append((event, str(exc)))
                logger.exception(
                    "handler %s failed for %s: %s",
                    getattr(handler, "__name__", handler),
                    event.event_type,
                    exc,
                )

    def stats(self) -> dict[str, Any]:
        return {
            "published": self.published,
            "processed": self.processed,
            "failed": self.failed,
            "dead_letters": len(self.dead_letters),
            "handlers": self.handler_count(),
        }


class RedisEventQueue(BaseEventQueue):
    """Durable Redis list queue with a dedicated priority lane."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        super().__init__(settings)
        self._client: Any = None

    async def connect(self) -> None:
        if self._client is not None:
            return
        try:
            import redis.asyncio as redis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise QueueError("the 'redis' package is required for QUEUE_BACKEND=redis") from exc
        self._client = redis.from_url(
            self.settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        try:
            await self._client.ping()
        except Exception as exc:  # pragma: no cover - requires a live Redis
            raise QueueError(
                f"Could not connect to Redis at {self.settings.REDIS_URL}: {exc}"
            ) from exc

    async def publish(self, event: InternalEvent) -> None:
        if self._client is None:
            await self.connect()
        queue_name = (
            self.settings.PRIORITY_QUEUE_NAME
            if event.priority in {"high", "critical"}
            else self.settings.EVENT_QUEUE_NAME
        )
        await self._client.lpush(queue_name, event.to_json())
        self.published += 1
        logger.debug("published %s to %s", event.event_type, queue_name)

    async def _next_event(self) -> Optional[InternalEvent]:
        if self._client is None:
            await self.connect()
        result = await self._client.brpop(
            [self.settings.PRIORITY_QUEUE_NAME, self.settings.EVENT_QUEUE_NAME],
            timeout=self.settings.EVENT_CONSUMER_BLOCK_SECONDS,
        )
        if not result:
            return None
        _, raw = result
        return InternalEvent.from_json(raw)

    async def _close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class InMemoryEventQueue(BaseEventQueue):
    """In-process priority queue (dev/test); no durability guarantees."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        super().__init__(settings)
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._counter = itertools.count()

    async def publish(self, event: InternalEvent) -> None:
        # Round-trip through JSON so serialisation bugs surface in dev too.
        clone = InternalEvent.from_json(event.to_json())
        rank = _PRIORITY_RANK.get(str(clone.priority), 2)
        await self._queue.put((rank, next(self._counter), clone))
        self.published += 1

    async def _next_event(self) -> Optional[InternalEvent]:
        try:
            _, _, event = await asyncio.wait_for(
                self._queue.get(), timeout=self.settings.EVENT_CONSUMER_BLOCK_SECONDS
            )
        except asyncio.TimeoutError:
            return None
        return event

    async def drain(self, timeout: float = 5.0) -> int:
        """Process everything currently queued; returns the number processed."""
        processed_before = self.processed
        while not self._queue.empty():
            try:
                _, _, event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:  # pragma: no cover
                break
            await self.dispatch(event)
        return self.processed - processed_before

    def queue_depth(self) -> int:
        return self._queue.qsize()


def build_event_queue(settings: Optional[Settings] = None) -> BaseEventQueue:
    """Factory selecting the Redis or in-memory queue backend."""
    settings = settings or get_settings()
    if settings.QUEUE_BACKEND == "memory":
        logger.warning("QUEUE_BACKEND=memory - events are not durable")
        return InMemoryEventQueue(settings)
    return RedisEventQueue(settings)


def priority_for(event_type: str) -> EventPriority:
    """Convenience re-export used by the ingestion layer."""
    from app.events.schemas import determine_priority

    return determine_priority(event_type)
