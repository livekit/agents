from __future__ import annotations

import weakref
from builtins import ExceptionGroup
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from livekit.agents.ipc import JobExecutor
from tests.conformance.ipc_processes import close_owned_executor


@pytest.fixture
async def own_executor(tmp_path: Path) -> AsyncIterator[Callable[[JobExecutor], JobExecutor]]:
    executors: list[weakref.ReferenceType[JobExecutor]] = []

    def register(executor: JobExecutor) -> JobExecutor:
        executors.append(weakref.ref(executor))
        return executor

    yield register
    failures: list[Exception] = []
    for reference in executors:
        executor = reference()
        if executor is None:
            continue
        try:
            await close_owned_executor(executor, tmp_path / "ipc-cleanup")
        except Exception as error:
            failures.append(error)
    if failures:
        raise ExceptionGroup("IPC fixture cleanup failures", failures)
