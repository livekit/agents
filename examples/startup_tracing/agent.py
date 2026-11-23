"""Instrument existing agent startup paths without introducing a new publication path."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import json
import logging
import math
import os
import signal
import time
from array import array
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, room_io, utils
from livekit.agents.telemetry import set_tracer_provider
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput

TRACE_DIR = Path(os.environ["STARTUP_TRACE_DIR"])
AGENT_NAME = os.environ["STARTUP_AGENT_NAME"]
CURRENT: dict[str, Any] = {}
SEEN: set[str] = set()
FIRST_FRAME_ACCEPTED = asyncio.Event()
INSTRUMENTED = False


def record(name: str, *, at_ms: float | None = None, **fields: Any) -> None:
    run = CURRENT.get("run_id", "worker")
    data = {
        "kind": "mark",
        "name": name,
        "at_ms": at_ms or time.time() * 1000,
        "run_id": run,
        "pid": os.getpid(),
        **fields,
    }
    path = TRACE_DIR / f"{run}.jsonl"
    with path.open("a") as output:
        output.write(json.dumps(data, default=str) + "\n")


class FileExporter(SpanExporter):
    def export(self, spans: Any) -> SpanExportResult:
        for span in spans:
            attrs = {
                k: v
                for k, v in (span.attributes or {}).items()
                if not any(
                    s in k.lower()
                    for s in ("content", "token", "key", "prompt", "message", "transcript")
                )
            }
            record(
                "span",
                span_name=span.name,
                start_ms=span.start_time / 1e6,
                end_ms=span.end_time / 1e6,
                trace_id=f"{span.context.trace_id:032x}",
                span_id=f"{span.context.span_id:016x}",
                parent_id=f"{span.parent.span_id:016x}" if span.parent else None,
                attributes=attrs,
                status=span.status.status_code.name,
            )
        return SpanExportResult.SUCCESS


def wrap_async(cls: Any, method: str, label: str) -> None:
    original = getattr(cls, method)

    @functools.wraps(original)
    async def measured(self: Any, *args: Any, **kwargs: Any) -> Any:
        record(label + "_begin")
        if cls is JobContext:
            kwargs.setdefault("single_peer_connection", True)
            # A/B the ordinary and connect-time publication paths with the same native build.
            if (
                CURRENT.get("profile") == "automatic"
                and "publish_tracks" in inspect.signature(original).parameters
            ):
                kwargs["publish_tracks"] = ()
            record("connect_initial_tracks", count=len(kwargs.get("publish_tracks", ())))
        if cls is rtc.Room:
            options = kwargs.get("options")
            record(
                "native_connect_config",
                initial_track_count=len(getattr(options, "publish_tracks", ())),
            )
        try:
            if label == "toolsets_setup" and CURRENT.get("activity_delay_ms", 0):
                record("injected_activity_delay_begin")
                await asyncio.sleep(CURRENT["activity_delay_ms"] / 1000)
                record("injected_activity_delay_end")
            result = await original(self, *args, **kwargs)
            record(label + "_end")
            return result
        except BaseException as exc:
            record(label + "_error", error_type=type(exc).__name__)
            raise

    setattr(cls, method, measured)


def instrument() -> None:
    global INSTRUMENTED
    if INSTRUMENTED:
        return
    INSTRUMENTED = True
    from livekit.agents.llm._realtime.openai import RealtimeSession

    wrap_async(RealtimeSession, "_create_ws_conn", "provider_websocket")
    for cls, method, name in [
        (JobContext, "connect", "room_connect"),
        (AgentSession, "start", "session_start"),
        (AgentActivity, "start", "activity_start"),
        (AgentActivity, "_setup_toolsets", "toolsets_setup"),
        (room_io.RoomIO, "start", "roomio_setup"),
        (_ParticipantAudioOutput, "start", "audio_output_start"),
        (rtc.Room, "connect", "native_room_connect"),
        (rtc.LocalParticipant, "publish_track", "track_publish"),
        (rtc.LocalTrackPublication, "wait_for_subscription", "output_subscription"),
    ]:
        wrap_async(cls, method, name)
    original = _ParticipantAudioOutput.capture_frame

    async def output_frame(self: Any, frame: rtc.AudioFrame) -> None:
        if "pcm" not in SEEN:
            SEEN.add("pcm")
            record("first_pcm_at_output")
        samples = memoryview(frame.data).cast("B").cast("h")
        if "nonzero_pcm" not in SEEN and any(abs(x) > 160 for x in samples):
            SEEN.add("nonzero_pcm")
            record("first_nonsilent_pcm_at_output")
        await original(self, frame)
        if "accepted" not in SEEN:
            SEEN.add("accepted")
            record("first_output_frame_accepted")
            FIRST_FRAME_ACCEPTED.set()

    _ParticipantAudioOutput.capture_frame = output_frame


def setup_process(proc: Any) -> None:
    import livekit.agents

    if os.environ.get("STARTUP_NATIVE_TRACE"):

        class NativeTraceHandler(logging.Handler):
            def emit(self, entry: logging.LogRecord) -> None:
                # Native diagnostics contain only event names, counts and timestamps.
                # Drop every other native log; never persist SDP, tokens or ICE credentials.
                if (
                    isinstance(entry.args, tuple)
                    and len(entry.args) == 4
                    and entry.args[0] == "publication_trace"
                ):
                    event = json.loads(entry.args[3])
                    record(
                        "native_" + event["name"],
                        at_ms=event["at_ms"],
                        native_fields=event["fields"],
                    )

        rtc_logger = logging.getLogger("livekit")
        rtc_logger.handlers.clear()
        rtc_logger.addHandler(NativeTraceHandler())
        rtc_logger.propagate = False
        rtc_logger.setLevel(logging.INFO)

    record(
        "process_prepared",
        agents_source=livekit.agents.__file__,
        rtc_source=rtc.__file__,
        native_library=os.environ.get("LIVEKIT_LIB_PATH"),
        connect_accepts_initial_tracks="publish_tracks"
        in inspect.signature(JobContext.connect).parameters,
        rtc_accepts_initial_tracks="publish_tracks" in rtc.RoomOptions.__dataclass_fields__,
    )


async def tone_frames():
    if CURRENT.get("pcm_delay_ms", 0):
        record("injected_pcm_delay_begin")
        await asyncio.sleep(CURRENT["pcm_delay_ms"] / 1000)
        record("injected_pcm_delay_end")
    data = array("h", (int(8000 * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(240)))
    for index in range(100):
        if index == 0:
            record("first_pcm_generated")
        yield rtc.AudioFrame(data.tobytes(), 24000, 1, 240)
        await asyncio.sleep(0)


class Greeter(Agent):
    def __init__(self, real_provider: bool) -> None:
        super().__init__(
            instructions="Greet the user in one short English sentence. Do not call tools."
        )
        self.real_provider = real_provider

    async def on_enter(self) -> None:
        record("on_enter")
        if self.real_provider:
            self.session.generate_reply(instructions="Say: Hello! I'm ready to help you.")
        else:
            self.session.say("Hello, ready to help.", audio=tone_frames())


server = AgentServer(
    num_idle_processes=int(os.environ.get("STARTUP_IDLE_PROCESSES", "1")),
    port=int(os.environ.get("STARTUP_WORKER_PORT", "18793")),
    host="127.0.0.1",
    load_fnc=lambda _: 0.0,
    setup_fnc=setup_process,
    log_level="INFO",
    shutdown_process_timeout=5,
)
server.on("worker_registered", lambda *_: record("worker_registered"))


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    global CURRENT, TRACE_DIR
    CURRENT = json.loads(ctx.job.metadata)
    TRACE_DIR = Path(CURRENT.get("trace_dir", str(TRACE_DIR)))
    SEEN.clear()
    FIRST_FRAME_ACCEPTED.clear()
    record("entrypoint")
    info = ctx._info
    for field in ("received_at", "accepted_at", "assigned_at", "launched_at"):
        value = getattr(info, field, None)
        if value:
            record(field, at_ms=value * 1000)
    record("process_assignment", launched_warm=getattr(info, "launched_warm", None))
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(FileExporter()))
    set_tracer_provider(provider, allow_pii=False)
    instrument()
    probe_tasks: set[asyncio.Task[None]] = set()

    def on_probe(packet: rtc.DataPacket) -> None:
        if packet.topic != "startup.probe" or packet.participant is None:
            return
        record("client_probe_received")

        async def echo() -> None:
            await ctx.room.local_participant.publish_data(
                packet.data,
                topic="startup.ack",
                destination_identities=[packet.participant.identity],
            )
            record("client_probe_ack_sent")

        task = asyncio.create_task(echo(), name="startup_probe_echo")
        probe_tasks.add(task)
        task.add_done_callback(probe_tasks.discard)

    ctx.room.on("data_received", on_probe)
    if CURRENT["profile"] not in ("automatic", "prepared", "prepared-custom"):
        await ctx.connect(single_peer_connection=True)
        record("participant_wait_begin")
        await ctx.wait_for_participant()
        record("participant_wait_end")
    real = CURRENT["mode"] == "realtime"
    record("model_construction_begin")
    model = None
    if real:
        from livekit.plugins import openai

        model = openai.realtime.RealtimeModel(
            model=os.environ.get("STARTUP_REALTIME_MODEL", "gpt-realtime"),
            voice="marin",
            turn_detection=None,
        )
    record("model_construction_end")
    session = AgentSession(llm=model)
    custom = None
    if CURRENT["profile"] not in ("automatic", "prepared"):
        custom = _ParticipantAudioOutput(
            ctx.room,
            sample_rate=24000,
            num_channels=1,
            track_publish_options=rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        session.output.audio = custom
    session.on(
        "error",
        lambda event: record(
            "session_error",
            error_type=type(event.error).__name__,
            recoverable=getattr(event.error, "recoverable", None),
        ),
    )
    session.on("agent_state_changed", lambda event: record("agent_state", state=event.new_state))
    session.on(
        "metrics_collected",
        lambda event: record(
            "provider_metrics",
            metrics=dataclasses.asdict(event.metrics)
            if dataclasses.is_dataclass(event.metrics)
            else event.metrics.model_dump(),
        ),
    )
    ctx.room.on("connected", lambda *_: record("agent_room_connected"))
    ctx.room.on(
        "participant_connected",
        lambda participant: record("participant_seen", identity=participant.identity),
    )
    ctx.room.on("local_track_subscribed", lambda *_: record("track_subscription_event"))

    async def collect_transport_stats() -> None:
        await FIRST_FRAME_ACCEPTED.wait()
        await asyncio.sleep(0.4)
        try:
            stats = await ctx.room.get_rtc_stats()
            record(
                "agent_transport_stats",
                publisher_stat_count=len(stats.publisher_stats),
                subscriber_stat_count=len(stats.subscriber_stats),
            )
        except Exception as error:
            record("transport_stats_unavailable", error_type=type(error).__name__)

    diagnostics = asyncio.create_task(
        collect_transport_stats(), name="startup_transport_diagnostics"
    )

    async def observe_outbound() -> None:
        while not ctx.room.isconnected():
            await asyncio.sleep(0.01)
        while not FIRST_FRAME_ACCEPTED.is_set():
            stats = await ctx.room.get_rtc_stats()
            packets = sum(
                s.outbound_rtp.sent.packets_sent
                for s in stats.publisher_stats
                if s.WhichOneof("stats") == "outbound_rtp"
            )
            if packets:
                record("outbound_rtp_before_pcm", packets=packets)
                return
            await asyncio.sleep(0.02)
        record("no_outbound_rtp_observed_before_pcm")

    outbound_observer = (
        asyncio.create_task(observe_outbound()) if os.environ.get("STARTUP_NATIVE_TRACE") else None
    )

    async def cleanup() -> None:
        record("session_cleanup_begin")
        ctx.room.off("data_received", on_probe)
        if outbound_observer is not None:
            await utils.aio.cancel_and_wait(outbound_observer)
        await utils.aio.cancel_and_wait(*probe_tasks)
        diagnostics.cancel()
        await asyncio.gather(diagnostics, return_exceptions=True)
        await session.aclose()
        if custom:
            await custom.aclose()
        provider.force_flush()
        provider.shutdown()
        record("session_cleanup_end", probe_tasks_remaining=len(probe_tasks))

    ctx.add_shutdown_callback(cleanup)
    starting = session.start(
        agent=Greeter(real),
        room=ctx.room,
        room_options=room_io.RoomOptions(audio_input=False, video_input=False, text_output=False),
        record=False,
        session_host=False,
    )
    if CURRENT["profile"] == "prepared-custom":

        async def start_custom() -> None:
            await ctx.connect(single_peer_connection=True, publish_tracks=[custom._prepare_track()])
            await custom.start()

        tasks = [asyncio.create_task(starting), asyncio.create_task(start_custom())]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
    elif CURRENT["profile"] == "parallel-output":
        await asyncio.gather(starting, custom.start())
    else:
        await starting
        if custom:
            await custom.start()
    record("entrypoint_returned")


async def main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    worker = asyncio.create_task(server.run())
    stopping = asyncio.create_task(stop.wait())
    try:
        done, _ = await asyncio.wait([worker, stopping], return_when=asyncio.FIRST_COMPLETED)
        if worker in done:
            await worker
    finally:
        await server.aclose()
        stopping.cancel()
        await asyncio.gather(worker, stopping, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
