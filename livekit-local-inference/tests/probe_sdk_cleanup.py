"""Exercise actual SDK cleanup methods with checked native close probes."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import TYPE_CHECKING, override

import numpy as np
import pytest


sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent))

from livekit import agents
from livekit.agents import utils, vad
from livekit.agents.inference.eot.base import (
    TurnDetectorOptions,
    _BaseStreamingTurnDetector,
    _BaseStreamingTurnDetectorStream,
)
from livekit.agents.inference.eot.languages import ThresholdOptions
from livekit.agents.inference.eot.transports import _LocalTransport
from livekit.agents.inference.vad import VAD as SDKVAD, _VADStream
from livekit.agents.utils.aio import join_task
from livekit.local_inference import EOT, VAD


if TYPE_CHECKING:
    from numpy.typing import NDArray


if len(sys.argv) != 3:
    message = "Lifecycle probe requires its exact SDK source and receipt path"
    raise ValueError(message)
SOURCE = Path(sys.argv[1]).resolve()
OUTPUT = Path(sys.argv[2]).resolve()
SELECTED_SDK = Path(agents.__file__).resolve()


class CloseError(RuntimeError):
    pass


class DetachError(RuntimeError):
    pass


class InputCloseError(RuntimeError):
    pass


class CancelRequestError(RuntimeError):
    pass


class CloseProbe:
    def __init__(self, *, fail: bool = False) -> None:
        self.started = threading.Event()
        self.predicted = threading.Event()
        self.release_predict = threading.Event()
        self.closing = threading.Event()
        self.release_close = threading.Event()
        self.closed = threading.Event()
        self.fail = fail
        self.close_count = 0
        self.logical_release_count = 0
        self.failures_remaining = 0
        self.close_thread = 0
        self.reset_started = threading.Event()
        self.release_reset = threading.Event()
        self.reset_done = threading.Event()
        self.reset_thread = 0

    def predict(self, pcm: NDArray[np.int16]) -> float:
        self.started.set()
        if not self.release_predict.wait(2):
            message = "Probe prediction was not released"
            raise TimeoutError(message)
        self.predicted.set()
        return float(pcm.sum()) / max(1, pcm.size * 65536) + 0.5

    def reset(self) -> None:
        self.reset_thread = threading.get_ident()
        self.reset_started.set()
        if not self.release_reset.wait(2):
            message = "Probe reset was not released"
            raise TimeoutError(message)
        self.reset_done.set()

    def close(self) -> None:
        self.close_count += 1
        self.close_thread = threading.get_ident()
        self.closing.set()
        if not self.release_close.wait(2):
            message = "Probe close was not released"
            raise TimeoutError(message)
        self.release_predict.set()
        if self.started.is_set() and not self.predicted.wait(2):
            message = "Physical prediction did not stop"
            raise TimeoutError(message)
        if self.failures_remaining:
            self.failures_remaining -= 1
            message = "Injected first retirement failure"
            raise CloseError(message)
        if not self.closed.is_set():
            self.logical_release_count += 1
            self.closed.set()
        if self.fail:
            message = "expected close failure"
            raise CloseError(message)


class EOTProbe(EOT):
    def __init__(self, control: CloseProbe) -> None:
        self.control = control

    @override
    def predict(self, pcm: NDArray[np.int16]) -> float:
        return self.control.predict(pcm)

    @override
    def close(self) -> None:
        self.control.close()


class VADProbe(VAD):
    def __init__(self, control: CloseProbe) -> None:
        self.control = control

    @override
    def predict(self, pcm: NDArray[np.int16]) -> float:
        return self.control.predict(pcm)

    @override
    def reset(self) -> None:
        self.control.reset()

    @override
    def close(self) -> None:
        self.control.close()


class TransportProbe(_LocalTransport):
    """Use real lifecycle methods; the separate SDK proof owns normal construction."""

    def __init__(self, control: CloseProbe) -> None:
        self._eot = EOTProbe(control)
        self._buf = utils.AudioArrayBuffer(buffer_size=19200, sample_rate=16000)
        self._stream_ref = None
        self._tasks = set()
        self._close_task = None
        self._closing = False


class VADStreamProbe(_VADStream):
    def __init__(self, control: CloseProbe, executor: ThreadPoolExecutor, *, owns: bool) -> None:
        self._native_vad = VADProbe(control)
        self._executor = executor
        self._owns_executor = owns
        self._close_task = None
        self._block = asyncio.Event()
        vad.VADStream.__init__(self, SDKVAD())

    @override
    async def _main_task(self) -> None:
        await self._block.wait()


class VADResetStreamProbe(VADStreamProbe):
    def __init__(self, control: CloseProbe, executor: ThreadPoolExecutor) -> None:
        super().__init__(control, executor, owns=True)
        self._opts = SDKVAD()._opts
        self._input_sample_rate = 0
        self._speech_buffer = None
        self._speech_buffer_index = 0
        self._speech_buffer_max_reached = False
        self._prefix_padding_samples = 0

    @override
    async def _main_task(self) -> None:
        await _VADStream._main_task(self)


class DetachFaultTransport(TransportProbe):
    @override
    def detach(self) -> None:
        super().detach()
        message = "expected detach failure"
        raise DetachError(message)


class InputFaultStream(_BaseStreamingTurnDetectorStream):
    @override
    def end_input(self) -> None:
        message = "expected input close failure"
        raise InputCloseError(message)


class CancelFaultStream(_BaseStreamingTurnDetectorStream):
    @override
    def cancel_inference(self, *, timed_out: bool = False) -> None:
        message = "expected inference cancellation failure"
        raise CancelRequestError(message)


async def _event(event: threading.Event) -> None:
    if not await asyncio.to_thread(event.wait, 1):
        message = "Probe did not reach the required phase"
        raise TimeoutError(message)


async def _helper_cases() -> list[str]:
    completed = asyncio.Event()
    release = asyncio.Event()

    async def owned() -> int:
        await release.wait()
        completed.set()
        return 7

    task = asyncio.create_task(owned())
    observer = asyncio.create_task(join_task(task))
    await asyncio.sleep(0)
    for _ in range(3):
        observer.cancel()
        await asyncio.sleep(0)
        assert not observer.done()
        assert not task.done()
    release.set()
    try:
        await observer
    except asyncio.CancelledError:
        pass
    else:
        message = "Caller cancellation was lost"
        raise AssertionError(message)
    assert completed.is_set()
    assert task.result() == 7
    assert await join_task(asyncio.create_task(asyncio.sleep(0, result=11))) == 11

    gate = asyncio.Event()

    async def failing() -> None:
        await gate.wait()
        message = "expected close failure"
        raise CloseError(message)

    failure = asyncio.create_task(failing())
    waiter = asyncio.create_task(join_task(failure))
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    gate.set()
    with pytest.raises(CloseError) as failure_info:
        await waiter
    assert isinstance(failure_info.value.__cause__, asyncio.CancelledError)

    cancelled = asyncio.create_task(asyncio.sleep(60))
    cancelled.cancel()
    try:
        await join_task(cancelled)
    except asyncio.CancelledError:
        pass
    else:
        message = "Owned-task cancellation was hidden"
        raise AssertionError(message)
    other_loop = asyncio.new_event_loop()
    other = other_loop.create_task(asyncio.sleep(0))
    try:
        with pytest.raises(ValueError, match="another event loop"):
            await join_task(other)
    finally:
        await asyncio.to_thread(other_loop.run_until_complete, other)
        other_loop.close()
    return [
        "join repeated cancellation",
        "join result",
        "join cleanup failure cause",
        "join cancelled owner",
        "join wrong-loop owner rejection",
    ]


async def _eot_cases() -> list[str]:
    control = CloseProbe()
    transport = TransportProbe(control)
    transport.run_inference("owned")
    await _event(control.started)
    closing = asyncio.create_task(transport.aclose())
    await _event(control.closing)
    for _ in range(3):
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        assert not control.closed.is_set()
    ticks = 0
    for _ in range(5):
        await asyncio.sleep(0.002)
        ticks += 1
    control.release_close.set()
    try:
        await closing
    except asyncio.CancelledError:
        pass
    else:
        message = "EOT close did not preserve cancellation"
        raise AssertionError(message)
    assert ticks == 5
    assert control.predicted.is_set()
    assert control.closed.is_set()
    assert control.close_thread != threading.get_ident()
    await transport.aclose()
    assert control.close_count == 1
    assert not transport._tasks

    early = CloseProbe()
    early.release_close.set()
    local = TransportProbe(early)
    options = TurnDetectorOptions(
        sample_rate=16000,
        thresholds=ThresholdOptions(model="turn-detector-v1-mini"),
    )
    stream = _BaseStreamingTurnDetectorStream(
        detector=_BaseStreamingTurnDetector(opts=options), opts=options, transport=local
    )
    stream._task.cancel()  # The actual stream never enters _run.
    await stream.aclose()
    await stream.aclose()
    assert early.close_count == 1
    assert early.closed.is_set()
    return [
        "EOT physical work joined",
        "EOT repeated cancellation",
        "EOT off-loop/idempotent close",
        "EOT early stream close",
    ]


async def _vad_cases() -> list[str]:
    result: list[str] = []
    for fail in (False, True):
        control = CloseProbe(fail=fail)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"lifecycle.vad.{fail}")
        stream = VADStreamProbe(control, executor, owns=True)
        prediction = executor.submit(control.predict, np.zeros(512, dtype=np.int16))
        await _event(control.started)
        closing = asyncio.create_task(stream.aclose())
        await _event(control.closing)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        control.release_close.set()
        if fail:
            with pytest.raises(CloseError) as close_info:
                await closing
            assert isinstance(close_info.value.__cause__, asyncio.CancelledError)
        else:
            with pytest.raises(asyncio.CancelledError):
                await closing
        assert prediction.done()
        assert control.closed.is_set()
        assert control.close_thread != threading.get_ident()
        assert not any(t.name.startswith(f"lifecycle.vad.{fail}") for t in threading.enumerate())
        result.append(
            "VAD failure still joins executor"
            if fail
            else "VAD repeated cancellation joins executor"
        )

    shared = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lifecycle.shared")
    control = CloseProbe()
    control.release_close.set()
    stream = VADStreamProbe(control, shared, owns=False)
    try:
        await stream.aclose()
        assert await asyncio.wrap_future(shared.submit(lambda: 17)) == 17
    finally:
        await asyncio.to_thread(shared.shutdown, wait=True, cancel_futures=True)
    result.append("VAD leaves caller executor owned")
    return result


async def _retry_cases() -> list[str]:
    results: list[str] = []
    for kind in ("local", "stream", "vad"):
        for failed in (True, False):
            control = CloseProbe()
            control.failures_remaining = int(failed)
            if failed:
                control.release_close.set()
            executor: ThreadPoolExecutor | None = None
            resource: _LocalTransport | _BaseStreamingTurnDetectorStream | _VADStream
            if kind == "local":
                resource = TransportProbe(control)
            elif kind == "stream":
                options = TurnDetectorOptions(
                    sample_rate=16000,
                    thresholds=ThresholdOptions(model="turn-detector-v1-mini"),
                )
                resource = _BaseStreamingTurnDetectorStream(
                    detector=_BaseStreamingTurnDetector(opts=options),
                    opts=options,
                    transport=TransportProbe(control),
                )
                resource._task.cancel()
            else:
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lifecycle.retry")
                resource = VADStreamProbe(control, executor, owns=True)
            closing = asyncio.create_task(resource.aclose())
            if failed:
                with pytest.raises(CloseError):
                    await closing
                assert not control.closed.is_set()
                assert control.logical_release_count == 0
            else:
                await _event(control.closing)
                owned = resource._close_task
                assert owned is not None
                owned.cancel()
                await asyncio.sleep(0)
                owned.cancel()
                for _ in range(5):
                    await asyncio.sleep(0.001)
                    assert not owned.done()
                    assert not control.closed.is_set()
                control.release_close.set()
                with pytest.raises(asyncio.CancelledError):
                    await closing
                assert control.closed.is_set()
                assert control.logical_release_count == 1
            await asyncio.gather(resource.aclose(), resource.aclose())
            await resource.aclose()
            assert control.close_count == (1 if kind == "stream" and not failed else 2)
            assert control.logical_release_count == 1
            assert not any(t.name.startswith("lifecycle.retry") for t in threading.enumerate())
            if executor is not None:
                with pytest.raises(
                    RuntimeError, match="cannot schedule new futures after shutdown"
                ):
                    executor.submit(lambda: 0)
            results.append(
                f"{kind} {'failed' if failed else 'cancelled'} close retries once without duplicate release"
            )
    # Failed cleanup must preserve a caller-owned executor through retry too.
    control = CloseProbe()
    control.failures_remaining = 1
    control.release_close.set()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lifecycle.retry.caller")
    stream = VADStreamProbe(control, executor, owns=False)
    try:
        with pytest.raises(CloseError):
            await stream.aclose()
        await stream.aclose()
        assert await asyncio.wrap_future(executor.submit(lambda: 23)) == 23
        assert control.close_count == 2
        assert control.logical_release_count == 1
    finally:
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
    results.append("VAD failed-close retry preserves caller executor")
    return results


async def _teardown_faults() -> list[str]:
    results: list[str] = []
    options = TurnDetectorOptions(
        sample_rate=16000, thresholds=ThresholdOptions(model="turn-detector-v1-mini")
    )
    for kind in ("detach", "input", "cancel", "native-and-detach"):
        control = CloseProbe(fail=kind == "native-and-detach")
        control.release_close.set()
        transport = (
            DetachFaultTransport(control)
            if kind in {"detach", "native-and-detach"}
            else TransportProbe(control)
        )
        stream_type = (
            InputFaultStream
            if kind == "input"
            else CancelFaultStream
            if kind == "cancel"
            else _BaseStreamingTurnDetectorStream
        )
        stream = stream_type(
            detector=_BaseStreamingTurnDetector(opts=options), opts=options, transport=transport
        )
        stream._task.cancel()
        expected = (
            InputCloseError
            if kind == "input"
            else CancelRequestError
            if kind == "cancel"
            else CloseError
            if kind == "native-and-detach"
            else DetachError
        )
        with pytest.raises(expected) as failure_info:
            await stream.aclose()
        if kind == "native-and-detach":
            assert isinstance(failure_info.value.__context__, DetachError)
        assert stream._task.done()
        assert stream._audio_ch.closed
        assert control.close_count == 1
        assert control.closed.is_set()
        assert not transport._tasks
        results.append(f"native retirement despite {kind} failure")
    return results


async def _reset_case() -> list[str]:
    control = CloseProbe()
    control.release_close.set()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lifecycle.reset")
    stream = VADResetStreamProbe(control, executor)
    try:
        stream.flush()
        await _event(control.reset_started)
        assert control.reset_thread != threading.get_ident()
        for _ in range(5):
            await asyncio.sleep(0.002)
            assert not control.reset_done.is_set()
        control.release_reset.set()
        await _event(control.reset_done)
    finally:
        control.release_reset.set()
        await stream.aclose()
    assert control.closed.is_set()
    assert not any(t.name.startswith("lifecycle.reset") for t in threading.enumerate())
    return ["VAD reset runs on owned executor without blocking event loop"]


async def main() -> None:
    assert sys.version_info[:3] == (3, 15, 2)
    assert SELECTED_SDK.is_relative_to(SOURCE)
    started = time.monotonic()
    results = [
        *await _helper_cases(),
        *await _eot_cases(),
        *await _vad_cases(),
        *await _teardown_faults(),
        *await _retry_cases(),
        *await _reset_case(),
    ]
    paths = [
        SOURCE / "livekit/agents" / name
        for name in (
            "utils/aio/utils.py",
            "utils/aio/__init__.py",
            "inference/eot/transports.py",
            "inference/eot/base.py",
            "inference/vad.py",
        )
    ]
    receipt = {
        "python": sys.version,
        "sdk": str(agents.__file__),
        "cases": results,
        "case_count": len(results),
        "elapsed_seconds": time.monotonic() - started,
        "opaque_vendor_code_executed": False,
        "scope": "Actual SDK cleanup methods with checked native close subclasses; normal construction/model quality belongs to the separate genuine SDK proof",
        "sources": {
            str(path.relative_to(SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": len(results), "elapsed_seconds": receipt["elapsed_seconds"]}))


if __name__ == "__main__":
    asyncio.run(main())
