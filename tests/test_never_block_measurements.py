"""Measurement contracts, including deliberately stalled workloads."""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from scripts.bench.metrics import (
    CallbackSlices,
    LoopLag,
    distribution,
)

pytestmark = pytest.mark.unit


def test_loop_monitor_retains_a_blocking_callback_stall() -> None:
    async def workload() -> None:
        monitor = LoopLag()
        heartbeat = asyncio.create_task(monitor.run())
        original_run = asyncio.Handle._run
        try:
            await monitor.ready.wait()
            # A deliberately bad integration must remain visible despite skipped deadlines.
            with CallbackSlices() as callbacks:
                asyncio.get_running_loop().call_soon(time.sleep, 0.025)
                await asyncio.sleep(0.03)
            assert distribution(monitor.samples).max_ms >= 20
            assert distribution(callbacks.wall).max_ms >= 20
            assert asyncio.Handle._run is original_run
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    asyncio.run(workload())
