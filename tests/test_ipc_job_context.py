"""Job context and cancellation when a native operation returns an asyncio Future."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any, cast

import aiohttp
import pytest

from livekit.agents.ipc.child import _JobProc
from livekit.agents.job import JobContext, _JobContextVar
from livekit.agents.utils import http_context

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_native_future_runs_in_job_context_without_changing_the_caller(cancel: bool) -> None:
    job = cast(JobContext, object())
    http = cast(aiohttp.ClientSession, object())
    proc = cast(_JobProc, SimpleNamespace(_job_ctx=job, _http_session=http))
    entered: list[tuple[JobContext, aiohttp.ClientSession]] = []

    class NativeFuture(asyncio.Future[int]):
        def __await__(self) -> Generator[Any, None, int]:
            entered.append((_JobContextVar.get(), http_context._ContextVar.get()))
            return super().__await__()

    caller_job = cast(JobContext, object())
    caller_http = cast(aiohttp.ClientSession, object())
    job_token = _JobContextVar.set(caller_job)
    http_token = http_context._ContextVar.set(caller_http)
    try:
        future = NativeFuture()
        task = _JobProc._in_job(proc, future, "native operation")
        await asyncio.sleep(0)
        assert entered == [(job, http)]
        assert _JobContextVar.get() is caller_job
        assert http_context._ContextVar.get() is caller_http
        assert task.get_name() == "native operation"
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert future.cancelled()
        else:
            future.set_result(42)
            assert await task == 42
    finally:
        _JobContextVar.reset(job_token)
        http_context._ContextVar.reset(http_token)
