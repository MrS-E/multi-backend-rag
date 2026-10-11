"""Redis coordination primitives shared by API and worker paths."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, TypeVar

KB_LOCK_TIMEOUT_S = 3600
SOURCE_LOCK_TIMEOUT_S = 900
_LEASE_RENEWAL_FRACTION = 3

logger = logging.getLogger(__name__)
_ResultT = TypeVar("_ResultT")
_LOCAL_KB_LOCKS: dict[str, asyncio.Lock] = {}


class LeaseLostError(RuntimeError):
    """Raised when a Redis lock can no longer fence the current owner."""


def kb_lock(redis: Any, kb_id: str, *, blocking: bool = False) -> Any:
    """Return the per-KB lock used by every index and collection mutation."""
    return redis.lock(f"lock:kb:{kb_id}", timeout=KB_LOCK_TIMEOUT_S, blocking=blocking)


async def acquire_kb_lock(redis: Any, kb_id: str) -> Any | None:
    """Acquire the shared per-KB lock without waiting."""
    lock = kb_lock(redis, kb_id)
    return lock if await lock.acquire() else None


async def acquire_kb_coordination(redis: Any | None, kb_id: str) -> Any | None:
    """Acquire the shared KB lock, using a process-local fallback without Redis."""
    if redis is not None:
        return await acquire_kb_lock(redis, kb_id)
    lock = _LOCAL_KB_LOCKS.setdefault(kb_id, asyncio.Lock())
    await lock.acquire()
    return lock


async def release_kb_coordination(redis: Any | None, lock: Any) -> None:
    """Release a Redis or process-local KB coordination lock."""
    if redis is None:
        lock.release()
    else:
        await release_lock(lock)


class LockLease:
    """Renew a Redis lock while synchronous provider/index work is running."""

    def __init__(self, lock: Any, timeout_s: int = KB_LOCK_TIMEOUT_S) -> None:
        self._lock = lock
        self._timeout_s = timeout_s
        self._lost = False
        self._task: asyncio.Task[None] | None = None
        self._operations: set[asyncio.Task[Any]] = set()

    async def __aenter__(self) -> LockLease:
        if hasattr(self._lock, "extend"):
            self._task = asyncio.create_task(self._renew())
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._operations:
            await asyncio.gather(
                *(asyncio.shield(operation) for operation in self._operations),
                return_exceptions=True,
            )
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def run_blocking(self, function: Callable[..., _ResultT], *args: Any) -> _ResultT:
        """Run blocking work without releasing this lease before its thread ends.

        Cancelling the coroutine returned by :func:`asyncio.to_thread` does not
        stop the worker thread.  Keep that thread registered and wait for it in
        both the cancelled call and lease shutdown, so a lock can never be
        released while a mutation is still executing.
        """
        operation = asyncio.create_task(asyncio.to_thread(partial(function, *args)))
        self._operations.add(operation)
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            await asyncio.shield(operation)
            raise
        finally:
            self._operations.discard(operation)

    def ensure_valid(self) -> None:
        """Fail before the next mutation if Redis reported that the lease was lost."""
        if self._lost:
            raise LeaseLostError("Redis coordination lock lease expired")

    async def _renew(self) -> None:
        interval = max(self._timeout_s / _LEASE_RENEWAL_FRACTION, 1)
        while True:
            await asyncio.sleep(interval)
            try:
                renewed = await self._lock.extend(self._timeout_s)
                if renewed is False:
                    self._lost = True
                    return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the caller checks the lease before writes
                logger.exception("failed to renew Redis coordination lock")
                self._lost = True
                return


async def release_lock(lock: Any) -> None:
    """Release a lock without masking the job/API outcome if its lease expired."""
    try:
        await lock.release()
    except Exception:  # noqa: BLE001 - expired locks are already safe to abandon
        logger.warning("Redis coordination lock was already unavailable", exc_info=True)


@asynccontextmanager
async def held_lock(lock: Any) -> AsyncIterator[None]:
    """Keep a lock leased and release it safely when the operation completes."""
    async with LockLease(lock):
        try:
            yield
        finally:
            await release_lock(lock)


@asynccontextmanager
async def maybe_held_lock(lock: Any) -> AsyncIterator[None]:
    """Use :func:`held_lock` when Redis is configured, otherwise do nothing."""
    if lock is None:
        yield
        return
    async with held_lock(lock):
        yield