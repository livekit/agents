"""Measure the actual Python media boundary, its GIL progress and asyncio scheduling."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import platform
import tempfile
import threading
import time
import wave
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TypeVar

from pydantic import AliasPath, BaseModel, ConfigDict, Field
from scripts.bench.metrics import CallbackSlices, Distribution, LoopLag, distribution

T = TypeVar("T")


class MediaLimits(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, allow_inf_nan=False)

    loop_lag_p999_ms_exclusive: float = Field(gt=0, le=2)
    loop_lag_max_ms_exclusive: float = Field(gt=0, le=5)
    poll_thread_cpu_p999_ms: float = Field(gt=0, le=1)
    poll_thread_cpu_max_ms: float = Field(gt=0, le=2)


class MediaBudget(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, allow_inf_nan=False)

    minimum_observations: int = Field(
        ge=1000,
        validation_alias=AliasPath("interfaces", "media", "workload", "minimum_observations"),
    )
    limits: MediaLimits = Field(validation_alias=AliasPath("interfaces", "media", "limits"))

    def qualified(
        self, lag: Distribution, cpu: Distribution, complete: bool, gil_ticks: int
    ) -> bool:
        return (
            complete
            and gil_ticks > 0
            and lag.count >= self.minimum_observations
            and lag.p999_ms < self.limits.loop_lag_p999_ms_exclusive
            and lag.max_ms < self.limits.loop_lag_max_ms_exclusive
            and cpu.p999_ms <= self.limits.poll_thread_cpu_p999_ms
            and cpu.max_ms <= self.limits.poll_thread_cpu_max_ms
        )


def load_budget(path: Path | None = None) -> MediaBudget:
    return MediaBudget.model_validate_json(
        (path or Path(__file__).with_name("budgets.json")).read_bytes()
    )


async def measure(
    name: str, budget: MediaBudget, workload: Callable[[], Awaitable[dict[str, object]]]
) -> dict[str, object]:
    monitor = LoopLag()
    heartbeat = asyncio.create_task(monitor.run())
    stopped = threading.Event()
    ticks: list[int] = []

    def probe() -> None:
        while not stopped.wait(0.001):
            ticks.append(time.monotonic_ns())

    thread = threading.Thread(target=probe, name="media-gil-probe")
    thread.start()
    try:
        await monitor.ready.wait()
        started = time.monotonic_ns()
        with CallbackSlices() as callbacks:
            await asyncio.sleep(0)
            details = await workload()
            await asyncio.sleep(0)
        finished = time.monotonic_ns()
        # Retain the final overdue heartbeat, including synchronous calls that return before
        # the loop has another chance to run. Collect idle padding separately from workload lag.
        await asyncio.sleep(0.002)
        active = list(monitor.samples)
        while len(monitor.samples) < 1000:
            await asyncio.sleep(0.01)
        cpu = distribution(callbacks.cpu)
        lag = distribution(active)
        gil_ticks = sum(started <= tick <= finished for tick in ticks)
        result: dict[str, object] = {
            "name": name,
            "elapsed_ms": (finished - started) / 1e6,
            "loop_lag": asdict(lag),
            "heartbeat_count": len(active),
            "heartbeat_count_with_idle_padding": len(monitor.samples),
            "callback_wall": asdict(distribution(callbacks.wall)),
            "callback_thread_cpu": asdict(cpu),
            "largest_callbacks": callbacks.largest("wall"),
            "gil_probe_ticks": gil_ticks,
            **details,
        }
        result["qualified"] = budget.qualified(
            lag, cpu, details["workload_complete"] is True, gil_ticks
        )
        return result
    finally:
        stopped.set()
        # Joining is outside the measured workload; the probe only waits on the stop event.
        thread.join()
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat


async def run(budget: MediaBudget) -> list[dict[str, object]]:
    from livekit import rtc
    from livekit.agents import _core

    results: list[dict[str, object]] = []
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(b"\x10\x00" * 480_000)
    data = buffer.getvalue()

    async def cold(recording_path: str) -> dict[str, object]:
        calls = []

        def timed(name: str, operation: Callable[[], T]) -> T:
            wall, cpu = time.monotonic_ns(), time.thread_time_ns()
            result = operation()
            calls.append(
                {
                    "name": name,
                    "wall_ms": (time.monotonic_ns() - wall) / 1e6,
                    "thread_cpu_ms": (time.thread_time_ns() - cpu) / 1e6,
                }
            )
            return result

        decoder = timed(
            "cold_constructor_after_import",
            lambda: _core.AudioDecoder("audio/wav", sample_rate=24000, num_channels=1),
        )
        pcm = timed("push_10s_48khz_to_24khz", lambda: decoder.push(data))
        tail = timed("flush", decoder.flush)
        samples = (len(pcm) + len(tail)) // 2
        if samples != 240_000:
            raise RuntimeError(f"ten-second WAV yielded {samples} samples, expected 240000")
        recorder = timed(
            "cold_recorder_constructor_after_import",
            lambda: _core.SessionRecorder(recording_path, sample_rate=48000, t0=0.0, bitrate=64000),
        )
        await recorder.ready()
        await recorder.finish(0.0)
        return {
            "workload_complete": True,
            "delivered_samples": samples,
            "source_samples": 480_000,
            "source_bytes": len(data),
            "delivered_pcm_bytes": len(pcm) + len(tail),
            "calls": calls,
        }

    with tempfile.TemporaryDirectory(prefix="lk-media-bench-") as directory:
        results.append(
            await measure(
                "cold_constructors_and_wav_resample",
                budget,
                partial(cold, str(Path(directory) / "cold.ogg")),
            )
        )

    async def idle() -> dict[str, object]:
        await asyncio.sleep(1.2)
        return {"workload_complete": True}

    results.append(await measure("idle", budget, idle))

    async def encode(frame: rtc.VideoFrame, format: str) -> dict[str, object]:
        outputs = await asyncio.gather(
            *(
                _core.encode_image(
                    frame,
                    format=format,
                    quality=75,
                    resize_width=None,
                    resize_height=None,
                    resize_strategy=None,
                )
                for _ in range(4)
            )
        )
        if not all(outputs):
            raise RuntimeError("image encoder returned an empty payload")
        return {
            "workload_complete": True,
            "width": frame.width,
            "height": frame.height,
            "format": format,
            "streams": 4,
            "rgba_bytes_per_frame": frame.width * frame.height * 4,
            "encoded_bytes": [len(output) for output in outputs],
        }

    for width, height in [(1920, 1080), (3840, 2160)]:
        # Fixture construction is excluded; encoding retains the SDK frame's real conversion.
        frame = rtc.VideoFrame(
            width, height, rtc.VideoBufferType.RGBA, bytes(range(256)) * (width * height * 4 // 256)
        )
        for format in ["JPEG", "PNG"]:
            results.append(await measure("image_encode", budget, partial(encode, frame, format)))
    return results


async def main(args: argparse.Namespace) -> int:
    from livekit.agents import _core

    extension = Path(_core.__file__)
    budget = load_budget()
    results = await run(budget)
    capture = {
        "platform": platform.platform(),
        "extension": str(extension),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "measurements": results,
        "evidence": {
            "gil_probe": "Python thread ticks during the workload; counts do not measure GIL hold time",
            "cold_constructor": "first decoder and recorder construction after package import in this process",
            "native_load": "real image encoding and WAV decode/resampling, without provider fakes",
            "native_adapter_suite": "media_measurements Rust test covers 32 Ogg streams and ten-second recording bursts",
        },
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(capture, indent=2) + "\n")
    temporary.replace(args.output)
    return int(args.check and not all(result["qualified"] for result in results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
