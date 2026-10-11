"""Focused tests for lease ownership around blocking work."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

import app.core.coordination as coordination
from app.core.coordination import (
    LeaseLostError,
    LockLease,
    acquire_kb_coordination,
    release_kb_coordination,
)


@pytest.mark.asyncio
async def test_given_cancelled_blocking_work_then_thread_finishes_before_lease_exit() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_mutation() -> None:
        started.set()
        release.wait()
        finished.set()

    lease = LockLease(object())
    task = asyncio.create_task(lease.run_blocking(blocking_mutation))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)

    assert not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await lease.__aexit__(None, None, None)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_given_cancelled_blocking_work_then_lock_release_waits_for_mutation() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    async def release_lock() -> None:
        assert finished.is_set()

    lock = type("Lock", (), {"release": AsyncMock(side_effect=release_lock)})()

    def blocking_mutation() -> None:
        started.set()
        release.wait()
        finished.set()

    lease = LockLease(lock)
    await lease.__aenter__()
    task = asyncio.create_task(lease.run_blocking(blocking_mutation))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)

    exit_task = asyncio.create_task(lease.__aexit__(None, None, None))
    await asyncio.sleep(0)
    assert not lock.release.await_count
    assert not finished.is_set()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await exit_task
    await lock.release()
    assert finished.is_set()
    assert lock.release.await_count == 1


@pytest.mark.asyncio
async def test_given_redis_renewal_loss_when_fencing_then_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = type("Lock", (), {"extend": AsyncMock(return_value=False)})()
    sleep = AsyncMock()
    monkeypatch.setattr(coordination.asyncio, "sleep", sleep)
    lease = LockLease(lock, timeout_s=30)

    await lease._renew()

    with pytest.raises(LeaseLostError):
        lease.ensure_valid()


@pytest.mark.asyncio
async def test_given_no_redis_when_acquiring_kb_coordination_then_local_lock_serializes() -> None:
    first = await acquire_kb_coordination(None, "local-kb")
    assert first is not None
    waiting = asyncio.create_task(acquire_kb_coordination(None, "local-kb"))
    await asyncio.sleep(0)
    assert not waiting.done()

    await release_kb_coordination(None, first)
    second = await waiting
    assert second is not None
    await release_kb_coordination(None, second)