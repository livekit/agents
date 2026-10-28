"""Capture the FakeRoom never-block workload. See README.md for evidence limits."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from livekit.agents import Agent, VoiceSession, _core
from scripts.bench.fakes import PacedLLM, PacedSTT, PacedTTS, PacedVAD
from scripts.bench.metrics import (
    LoopLag,
    NativeCalls,
    cadence_budget_pass,
    distribution,
    sampled_stacks,
)


@dataclass
class Call:
    room: _core.FakeRoom
    session: VoiceSession
    stt: PacedSTT
    tts: PacedTTS
    vad: PacedVAD


def positive(value: str) -> int:
    result = int(value)
    if not 0 < result <= 256:
        raise argparse.ArgumentTypeError("session count must be between 1 and 256")
    return result


def duration(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 4 <= result <= 120:
        raise argparse.ArgumentTypeError("duration must be finite and between 4 and 120 seconds")
    return result


async def run(args: argparse.Namespace, count: int, cadence: int) -> dict[str, object]:
    calls: list[Call] = []
    errors: list[str] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def failed(loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        errors.append(str(context))
        loop.default_exception_handler(context)

    loop.set_exception_handler(failed)
    monitor = LoopLag()
    heartbeat = asyncio.create_task(monitor.run())
    await monitor.ready.wait()
    sampler: asyncio.subprocess.Process | None = None
    raw_sample: Path | None = None
    profiler = NativeCalls()
    startup = time.perf_counter()
    try:
        async with asyncio.timeout(30):
            for index in range(count):
                room = _core.FakeRoom(name=f"bench-{index}", sample_rate=48000)
                room.join("user")
                stt, tts, vad = PacedSTT(), PacedTTS(), PacedVAD()
                session = VoiceSession(stt=stt, tts=tts, vad=vad, llm=PacedLLM())
                session.on("error", lambda ev: errors.append(str(ev)))
                calls.append(Call(room, session, stt, tts, vad))
                await session.start(Agent(instructions="Help the caller.", name="bench"), room=room)
        startup_ms = (time.perf_counter() - startup) * 1000
        startup_lag = distribution(monitor.samples) if monitor.samples else None
        monitor.samples.clear()
        if args.sample:
            descriptor, sample_name = tempfile.mkstemp(
                prefix=f"{args.output.stem}-{count}-{cadence}-",
                suffix=".sample.txt",
                dir=args.output.parent,
            )
            os.close(descriptor)
            raw_sample = Path(sample_name)
            sampler = await asyncio.create_subprocess_exec(
                "/usr/bin/sample",
                str(os.getpid()),
                str(max(1, int(args.seconds) - 1)),
                "1",
                "-file",
                str(raw_sample),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        injected = 0
        frame_lateness: list[int] = []
        started = loop.time()
        before_cpu = time.process_time()
        ticks = math.ceil(args.seconds * 1000 / cadence)
        with profiler if args.profile else contextlib.nullcontext():
            for tick in range(ticks):
                deadline = started + tick * cadence / 1000
                await asyncio.sleep(max(0, deadline - loop.time()))
                frame_lateness.append(max(0, int((loop.time() - deadline) * 1e9)))
                for call in calls:
                    # FakeRoom's public speak() emits fixed 20 ms PCM; never relabel it as 10 ms.
                    if (tick * cadence) % 20 == 0:
                        call.room.speak(frames=1)
                        injected += 1
                    call.room.advance(cadence / 1000)
            await asyncio.sleep(cadence / 1000)
        elapsed = loop.time() - started
        cpu_seconds = time.process_time() - before_cpu
        steady_lag = distribution(monitor.samples)
        # Outcome checks prevent empty or broken provider paths becoming a good baseline.
        for call in calls:
            if not (
                call.stt.finals
                and call.vad.predictions
                and call.tts.frames
                and call.tts.completed == call.tts.started
                and call.stt.frames == injected // count
                and call.vad.frames == injected // count
                and call.room.played_bytes()
            ):
                raise RuntimeError(
                    f"incomplete provider/playback: STT={call.stt.frames}, VAD={call.vad.frames}, "
                    f"input={injected // count}, TTS={call.tts.completed}/{call.tts.started}"
                )
        frames = sum(call.stt.frames + call.vad.frames + call.tts.frames for call in calls)
        evidence: dict[str, object] = {
            "sessions": count,
            "control_cadence_ms": cadence,
            "input_frame_ms": 20,
            "seconds": elapsed,
            "process_cpu_seconds": cpu_seconds,
            "loop_timing_is_perturbed": bool(args.profile or args.sample),
            "startup_ms": startup_ms,
            "startup_loop_lag": asdict(startup_lag) if startup_lag else None,
            "loop_lag": asdict(steady_lag),
            "frame_dispatch_lag": asdict(distribution(frame_lateness)),
            "injected_input_frames": injected,
            "provider_frames": frames,
            "stt_frames": sum(c.stt.frames for c in calls),
            "vad_frames": sum(c.vad.frames for c in calls),
            "vad_predictions": sum(c.vad.predictions for c in calls),
            "tts_frames": sum(c.tts.frames for c in calls),
            "tts_started": sum(c.tts.started for c in calls),
            "tts_completed": sum(c.tts.completed for c in calls),
            "played_bytes": sum(c.room.played_bytes() for c in calls),
            "native_call_wall": asdict(distribution(profiler.wall)) if profiler.wall else None,
            "native_call_thread_cpu": asdict(distribution(profiler.cpu)) if profiler.cpu else None,
            "native_calls_by_operation": dict(profiler.calls),
            "visible_copy_calls": profiler.copy_calls if args.profile else None,
            "visible_copy_bytes": profiler.copy_bytes if args.profile else None,
            "visible_copies_per_provider_frame": profiler.copy_calls / frames
            if args.profile
            else None,
            "binding_pcm_copies_lower_bound": sum(c.stt.frames + c.vad.frames for c in calls),
            "binding_pcm_copy_bytes_lower_bound": sum(
                c.stt.pcm_bytes + c.vad.pcm_bytes for c in calls
            ),
            "binding_copies_per_input_frame_lower_bound": sum(
                c.stt.frames + c.vad.frames for c in calls
            )
            / injected,
            "exact_driver_step_cpu": None,
            "per_worker_heartbeat": None,
        }
    finally:
        close_started = time.perf_counter()
        try:
            async with asyncio.timeout(30):
                await asyncio.gather(*(call.session.aclose() for call in calls))
        finally:
            for call in calls:
                call.room.close()
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            loop.set_exception_handler(previous)
            if sampler is not None:
                if sampler.returncode is None:
                    try:
                        async with asyncio.timeout(10):
                            _, stderr = await sampler.communicate()
                    except TimeoutError:
                        sampler.kill()
                        await sampler.wait()
                        raise RuntimeError("native sampler failed to finish") from None
                else:
                    _, stderr = await sampler.communicate()
                if sampler.returncode:
                    raise RuntimeError(f"native sample failed: {stderr.decode()}")
    if errors:
        raise RuntimeError(f"provider/asyncio errors: {errors}")
    evidence["close_ms"] = (time.perf_counter() - close_started) * 1000
    if raw_sample is not None:
        raw = raw_sample.read_text()
        if "Call graph:" not in raw:
            raise RuntimeError("native sampler produced no call graph")
        evidence["native_sample_file"] = str(raw_sample)
        observations = sampled_stacks(raw)
        if not observations:
            raise RuntimeError("native sampler produced no recognized native stacks")
        evidence["native_stack_observations"] = [asdict(s) for s in observations]
    evidence["measured_budget_pass"] = cadence_budget_pass(
        steady_lag, cadence, distribution(profiler.cpu) if args.profile else None
    )
    return evidence


async def main(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    library = Path(_core.__file__)
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--"])
    results = []
    for count in args.sessions:
        for cadence in args.cadence_ms:
            result = await run(args, count, cadence)
            results.append(result)
            print(f"captured N={count} cadence={cadence}: {result['loop_lag']}", flush=True)
    if hashlib.sha256(library.read_bytes()).hexdigest() != library_hash:
        raise RuntimeError("native library changed during capture; repeat on a stable build")
    report = {
        "schema_version": 1,
        "revision": revision,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "native_library": str(library),
        "native_library_sha256": library_hash,
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "profile_enabled": args.profile,
        "sample_enabled": args.sample,
        "results": results,
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    return int(args.check and any(not result["measured_budget_pass"] for result in results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=positive, nargs="+", default=[1, 8, 32])
    parser.add_argument("--cadence-ms", type=int, choices=(10, 20), nargs="+", default=[10, 20])
    parser.add_argument("--seconds", type=duration, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="instrument native calls and copies")
    parser.add_argument(
        "--sample", action="store_true", help="external macOS native stack sampling"
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 for an exceeded measured budget"
    )
    arguments = parser.parse_args()
    if arguments.sample and sys.platform != "darwin":
        parser.error("--sample currently requires macOS /usr/bin/sample")
    logging.basicConfig(level=logging.ERROR)
    raise SystemExit(asyncio.run(main(arguments)))
