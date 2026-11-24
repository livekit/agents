from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import signal
import struct
import sys
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from livekit import rtc
from livekit.agents.cli import _legacy, proto
from livekit.agents.cli._legacy import (
    SAMPLE_RATE,
    ConsoleAudioOutput,
    _build_cli,
    _configure_logger,
)
from livekit.agents.cli.cli import _ExitCli, _run_worker
from livekit.agents.cli.log import setup_logging
from livekit.agents.cli.tcp_console import AGENT_SAMPLE_RATE, WIRE_SAMPLE_RATE, TcpAudioOutput
from livekit.agents.voice.remote_session import SessionHost, TcpSessionTransport
from livekit.agents.voice.transcription import TranscriptSynchronizer
from livekit.agents.worker import AgentServer, ServerEnvOption, ServerOptions
from livekit.protocol.agent_pb import agent_session as agent_pb

# signal handlers, the root logger and sys.stdout are process-global
pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


async def _noop_entrypoint(ctx: Any) -> None:
    pass


def _make_server(**kwargs: Any) -> AgentServer:
    return AgentServer.from_server_options(ServerOptions(entrypoint_fnc=_noop_entrypoint, **kwargs))


# -- SIGTERM must reliably shut the worker down. A signal handler runs at an arbitrary bytecode
# boundary, so its raise can land inside a task or loop callback: _ExitCli is a SystemExit so
# asyncio re-raises it, the first signal only schedules the exit, and a watchdog re-signals the
# main thread when synchronous code blocks the loop.


def test_exit_cli_is_systemexit() -> None:
    # Task.__step / Handle._run re-raise only SystemExit and KeyboardInterrupt;
    # anything else is swallowed by the loop when the raise lands inside a task
    # or callback
    assert issubclass(_ExitCli, SystemExit)
    assert issubclass(_legacy._ExitCli, SystemExit)


def test_exit_cli_escapes_a_loop_callback() -> None:
    # the raise lands inside Handle._run (a loop callback)
    loop = asyncio.new_event_loop()
    try:

        def _raise_exit() -> None:
            raise _ExitCli()

        swallowed: list[dict[str, Any]] = []
        loop.set_exception_handler(lambda _loop, ctx: swallowed.append(ctx))
        loop.call_soon(_raise_exit)
        loop.call_later(0.5, loop.stop)  # safety stop for the broken case

        with pytest.raises(_ExitCli):
            loop.run_forever()

        assert not swallowed, "exit was reported to the exception handler instead of raised"
    finally:
        loop.close()


def test_exit_cli_escapes_a_task() -> None:
    # the raise lands inside a coroutine, e.g. _job_request_task while a request_fnc blocks
    loop = asyncio.new_event_loop()
    try:

        async def _job_request_task() -> None:
            raise _ExitCli()

        async def _main() -> None:
            await asyncio.sleep(30)  # the worker's main task, still running

        main_task = loop.create_task(_main())
        request_tasks: list[asyncio.Task[None]] = []
        loop.call_soon(lambda: request_tasks.append(loop.create_task(_job_request_task())))
        loop.call_later(0.5, loop.stop)  # safety stop for the broken case

        with pytest.raises(_ExitCli):
            loop.run_until_complete(main_task)
    finally:
        # drain the cancellation and retrieve the request task's exception so
        # closing the loop doesn't emit destroyed-pending / never-retrieved noise
        main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, _ExitCli):
            loop.run_until_complete(main_task)
        for task in request_tasks:
            if task.done() and not task.cancelled():
                task.exception()
        loop.close()


class _StubServer:
    """Stands in for AgentServer: run() until aclose(), with an optional task
    that blocks the event loop the way a synchronous request_fnc does."""

    def __init__(self, *, block_loop_for: float = 0.0) -> None:
        self._block_loop_for = block_loop_for
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self.calls: list[str] = []
        self.run_active_during_drain = False

    async def run(self, *, devmode: bool, unregistered: bool) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()
        if self._block_loop_for > 0:

            async def _blocking_job_request() -> None:
                time.sleep(self._block_loop_for)  # noqa: ASYNC251

            self._loop.create_task(_blocking_job_request())
        await self._stopped.wait()

    async def drain(self) -> None:
        self.calls.append("drain")
        # drain must run while run() is still active: jobs are joined by drain,
        # and aclose() is what ends run()
        self.run_active_during_drain = self._stopped is not None and not self._stopped.is_set()

    async def aclose(self) -> None:
        self.calls.append("aclose")
        if self._stopped is not None:
            self._stopped.set()

    def force_finish_threadsafe(self) -> None:
        """Test safety net: end run() so a regression fails by timing instead of
        hanging pytest forever."""
        if self._loop is not None and self._stopped is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._stopped.set)


@contextlib.contextmanager
def _run_worker_test_env() -> Iterator[None]:
    """_run_worker installs signal handlers (and leaves force-exit ones behind on
    purpose); restore the pytest process state afterwards."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with patch("livekit.agents.cli.cli.setup_logging"):
            yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        asyncio.set_event_loop(None)


def _sigterm_main_thread_after(delay: float) -> threading.Timer:
    main_id = threading.main_thread().ident
    assert main_id is not None
    timer = threading.Timer(delay, lambda: signal.pthread_kill(main_id, signal.SIGTERM))
    timer.daemon = True
    timer.start()
    return timer


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs signal.pthread_kill (POSIX)")
def test_sigterm_triggers_graceful_shutdown() -> None:
    """One SIGTERM on a healthy loop: drain + aclose run and _run_worker returns."""
    server = _StubServer()
    args = proto.CliArgs(log_level="DEBUG", dev=False)

    with _run_worker_test_env():
        _sigterm_main_thread_after(0.05)
        safety = threading.Timer(8.0, server.force_finish_threadsafe)
        safety.daemon = True
        safety.start()

        start = time.monotonic()
        _run_worker(server, args)  # type: ignore[arg-type]
        elapsed = time.monotonic() - start
        safety.cancel()

    assert server.calls == ["drain", "aclose"]
    assert server.run_active_during_drain
    assert elapsed < 5.0, f"shutdown took {elapsed:.1f}s — the exit signal was lost"


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs signal.pthread_kill (POSIX)")
def test_sigterm_with_blocked_event_loop_escalates() -> None:
    """A SIGTERM while synchronous work blocks the loop preempts the blocking frame."""
    block_for = 30.0
    server = _StubServer(block_loop_for=block_for)
    args = proto.CliArgs(log_level="DEBUG", dev=False)

    with (
        _run_worker_test_env(),
        patch("livekit.agents.cli.cli._EXIT_ESCALATION_TIMEOUT", 0.1),
    ):
        # deliver while time.sleep() holds the loop; the handler must not lose it
        _sigterm_main_thread_after(0.1)
        safety = threading.Timer(8.0, server.force_finish_threadsafe)
        safety.daemon = True
        safety.start()

        start = time.monotonic()
        _run_worker(server, args)  # type: ignore[arg-type]
        elapsed = time.monotonic() - start
        safety.cancel()

    assert server.calls == ["drain", "aclose"]
    assert server.run_active_during_drain
    assert elapsed < block_for / 2, (
        f"shutdown took {elapsed:.1f}s — the blocked loop was never preempted"
    )


# -- log level: CLI flag > LIVEKIT_LOG_LEVEL > server option > INFO (start) / DEBUG (dev)


@pytest.mark.parametrize(
    ("server_level", "argv", "env", "expected"),
    [
        (None, ["start"], {}, "INFO"),
        (None, ["start", "--log-level", "error"], {}, "ERROR"),
        (None, ["start"], {"LIVEKIT_LOG_LEVEL": "CRITICAL"}, "CRITICAL"),
        (None, ["start", "--log-level", "warn"], {"LIVEKIT_LOG_LEVEL": "CRITICAL"}, "WARN"),
        ("error", ["start"], {}, "ERROR"),
        ("ERROR", ["start", "--log-level", "debug"], {}, "DEBUG"),
        (None, ["dev", "--no-reload"], {}, "DEBUG"),
        (None, ["dev", "--no-reload", "--log-level", "info"], {}, "INFO"),
        (None, ["dev", "--no-reload"], {"LIVEKIT_LOG_LEVEL": "ERROR"}, "ERROR"),
        ("CRITICAL", ["dev", "--no-reload"], {}, "CRITICAL"),
    ],
)
def test_cli_log_level(
    server_level: str | None, argv: list[str], env: dict[str, str], expected: str
) -> None:
    server = _make_server(**({} if server_level is None else {"log_level": server_level}))
    with patch("livekit.agents.cli.cli._run_worker") as run_worker:
        result = CliRunner().invoke(_build_cli(server), argv, env=env)

    assert result.exit_code == 0, result.output
    assert run_worker.call_args.kwargs["args"].log_level == expected


@pytest.mark.parametrize(
    ("log_level", "prod", "dev"),
    [
        (None, "INFO", "DEBUG"),
        ("WARN", "WARN", "WARN"),
        ("info", "INFO", "INFO"),
        (ServerEnvOption(dev_default="ERROR", prod_default="CRITICAL"), "CRITICAL", "ERROR"),
        (ServerEnvOption(dev_default="debug", prod_default="error"), "ERROR", "DEBUG"),
    ],
)
@pytest.mark.parametrize("via", ["server_options", "agent_server"])
def test_log_level_option(log_level: Any, prod: str, dev: str, via: str) -> None:
    kwargs = {} if log_level is None else {"log_level": log_level}
    if via == "server_options":
        level = ServerOptions(entrypoint_fnc=_noop_entrypoint, **kwargs).log_level
    else:
        level = AgentServer(**kwargs).log_level

    assert ServerEnvOption.getvalue(level, False) == prod
    assert ServerEnvOption.getvalue(level, True) == dev


@pytest.mark.parametrize(
    "build",
    [
        lambda: ServerOptions(entrypoint_fnc=_noop_entrypoint, log_level="WARNING"),
        lambda: AgentServer(log_level="NOTSET"),
        lambda: ServerOptions(
            entrypoint_fnc=_noop_entrypoint,
            log_level=ServerEnvOption(dev_default="WARNING", prod_default="INFO"),
        ),
    ],
)
def test_invalid_log_level_is_rejected(build: Any) -> None:
    with pytest.raises(ValueError, match="Invalid log level"):
        build()


@pytest.mark.parametrize("level", ["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"])
def test_valid_log_levels_are_accepted(level: str) -> None:
    assert ServerOptions(entrypoint_fnc=_noop_entrypoint, log_level=level).log_level == level
    assert AgentServer(log_level=level).log_level == level


# -- logging setup on a non-UTF-8 stdout


@pytest.fixture
def stdout_as(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(encoding: str) -> io.BytesIO:
        output = io.BytesIO()
        root = logging.getLogger()
        monkeypatch.setattr(root, "handlers", [])
        monkeypatch.setattr(root, "level", logging.NOTSET)
        monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(output, encoding=encoding))
        monkeypatch.setattr(logging, "raiseExceptions", False)
        return output

    return install


@pytest.mark.parametrize(
    "configure",
    [
        lambda: setup_logging("INFO", devmode=False, console=False),
        lambda: _configure_logger(None, "INFO"),
    ],
    ids=["production", "legacy"],
)
@pytest.mark.parametrize("encoding", ["cp1252", "utf-8"])
def test_json_logging_preserves_unicode(stdout_as: Any, configure: Any, encoding: str) -> None:
    output = stdout_as(encoding)
    configure()
    logging.getLogger().info("café عاوز 🗣️")
    sys.stdout.flush()

    text = output.getvalue().decode(encoding, errors="replace")
    assert json.loads(text)["message"] == "café عاوز 🗣️"
    if encoding == "utf-8":
        assert '"message": "café' in text  # not escaped when stdout can encode it


def test_development_logging_escapes_unencodable_text(stdout_as: Any) -> None:
    output = stdout_as("cp1252")
    setup_logging("INFO", devmode=True, console=False)
    logging.getLogger().info("café عاوز 🗣️")
    sys.stdout.flush()

    message = output.getvalue().decode("cp1252", errors="replace")
    assert "café" in message
    assert r"\u0639" in message
    assert r"\U0001f5e3" in message


def test_logging_setup_keeps_the_stdout_encoding_for_print(stdout_as: Any) -> None:
    output = stdout_as("iso8859-1")
    setup_logging("INFO", devmode=False, console=False)
    print("café")
    sys.stdout.flush()

    assert output.getvalue() == b"caf\xe9\n"


# -- connect


class _FakeRoomService:
    async def list_rooms(self, request: Any) -> SimpleNamespace:
        return SimpleNamespace(rooms=[SimpleNamespace(name=request.names[0])])


class _FakeLiveKitAPI:
    def __init__(self, url: str, api_key: str, api_secret: str) -> None:
        self.room = _FakeRoomService()

    async def __aenter__(self) -> _FakeLiveKitAPI:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def test_connect_applies_cli_connection_options_before_simulate_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect mints the job token and URL from --url/--api-key/--api-secret, like start/dev."""
    for var in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)

    server = _make_server()
    simulated: list[tuple[str, str, str]] = []
    observed = asyncio.Event()

    async def fake_simulate_job(**kwargs: Any) -> None:
        simulated.append((server._ws_url, server._api_key, server._api_secret))
        observed.set()

    async def fake_run(*, devmode: bool, unregistered: bool) -> None:
        server.emit("worker_started")
        await asyncio.wait_for(observed.wait(), timeout=1.0)

    monkeypatch.setattr(server, "simulate_job", fake_simulate_job)
    monkeypatch.setattr(server, "run", fake_run)

    argv = ["connect", "--room", "myroom", "--url", "ws://flag-host:7880"]
    argv += ["--api-key", "cli-key", "--api-secret", "cli-secret", "--participant-identity", "x"]
    with patch("livekit.agents.cli._legacy.api.LiveKitAPI", _FakeLiveKitAPI):
        result = CliRunner().invoke(_build_cli(server), argv)

    assert result.exit_code == 0, result.output
    assert simulated == [("ws://flag-host:7880", "cli-key", "cli-secret")]


# -- the console speaker holds back each segment's head until PREBUFFER_DURATION is queued or
# the segment is flushed, so network jitter on a real-time stream doesn't click mid-word

_BLOCK = SAMPLE_RATE // 10  # one sounddevice callback


def _frame(value: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=np.full(_BLOCK, value, dtype=np.int16).tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=_BLOCK,
    )


def _read(out: ConsoleAudioOutput) -> np.ndarray:
    buf = np.empty((_BLOCK, 1), dtype=np.int16)
    out.read_into(buf, _BLOCK)
    return buf[:, 0]


async def test_segment_head_waits_for_prebuffer() -> None:
    out = ConsoleAudioOutput(asyncio.get_running_loop())
    await out.capture_frame(_frame(1))
    assert not _read(out).any()
    await out.capture_frame(_frame(2))
    await out.capture_frame(_frame(3))
    assert _read(out)[0] == 1


async def test_flush_releases_a_short_segment() -> None:
    out = ConsoleAudioOutput(asyncio.get_running_loop())
    await out.capture_frame(_frame(1))
    out.flush()
    assert _read(out)[0] == 1
    await asyncio.sleep(0.01)


async def test_underrun_mid_segment_does_not_reprime() -> None:
    out = ConsoleAudioOutput(asyncio.get_running_loop())
    for v in (1, 2, 3):
        await out.capture_frame(_frame(v))
    for _ in range(3):
        _read(out)
    assert not _read(out).any()
    await out.capture_frame(_frame(4))
    assert _read(out)[0] == 4


async def test_next_segment_primes_again() -> None:
    out = ConsoleAudioOutput(asyncio.get_running_loop())
    await out.capture_frame(_frame(1))
    out.flush()
    _read(out)
    await asyncio.sleep(0.01)
    await out.capture_frame(_frame(2))
    assert not _read(out).any()


# -- TcpAudioOutput: resampler tail draining, empty-segment flush gating, and pause capability


class _RecordingTcpTransport:
    def __init__(self) -> None:
        self.sent: list[agent_pb.AgentSessionMessage] = []

    def send_message_threadsafe(self, msg: agent_pb.AgentSessionMessage) -> None:
        self.sent.append(msg)


def _tcp_frame(duration_s: float, value: int = 1000) -> rtc.AudioFrame:
    samples = round(AGENT_SAMPLE_RATE * duration_s)
    return rtc.AudioFrame(
        data=np.full(samples, value, dtype=np.int16).tobytes(),
        sample_rate=AGENT_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )


async def test_tcp_audio_output_reports_cannot_pause() -> None:
    transport = _RecordingTcpTransport()
    out = TcpAudioOutput(transport)  # type: ignore[arg-type]
    sync = TranscriptSynchronizer(next_in_chain_audio=out, next_in_chain_text=None)
    try:
        assert not out.can_pause
        assert not sync.audio_output.can_pause
    finally:
        await sync.aclose()


async def test_tcp_audio_output_flush_drains_resampler_tail() -> None:
    transport = _RecordingTcpTransport()
    out = TcpAudioOutput(transport)  # type: ignore[arg-type]
    try:
        # A 20 ms segment stays inside the 24->48 kHz resampler until flush().
        await out.capture_frame(_tcp_frame(0.02))
        out.flush()

        kinds = [msg.WhichOneof("message") for msg in transport.sent]
        assert kinds == ["audio_output", "audio_playback_flush"]
        audio_samples = sum(
            msg.audio_output.samples_per_channel
            for msg in transport.sent
            if msg.HasField("audio_output")
        )
        assert audio_samples == round(WIRE_SAMPLE_RATE * 0.02)

        out.notify_playout_finished()
        ev = await asyncio.wait_for(out.wait_for_playout(), timeout=1.0)
        assert not ev.interrupted
        assert ev.playback_position == pytest.approx(0.02)
    finally:
        if out._flush_task is not None and not out._flush_task.done():
            out._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await out._flush_task


async def test_tcp_audio_output_empty_flush_sends_no_wire_flush() -> None:
    transport = _RecordingTcpTransport()
    out = TcpAudioOutput(transport)  # type: ignore[arg-type]

    out.flush()
    assert transport.sent == []

    # A following real segment still waits for its own AudioPlaybackFinished ack.
    await out.capture_frame(_tcp_frame(0.1))
    out.flush()
    out.notify_playout_finished()
    ev = await asyncio.wait_for(out.wait_for_playout(), timeout=1.0)
    assert not ev.interrupted
    assert ev.playback_position == pytest.approx(0.1)


async def test_tcp_audio_output_clear_before_flush_finishes_and_discards_resampler_tail() -> None:
    transport = _RecordingTcpTransport()
    out = TcpAudioOutput(transport)  # type: ignore[arg-type]

    # Interrupted mid-capture (e.g. proxy sink swap calls flush() + clear_buffer(), or
    # clear_buffer() arrives before flush()): 20 ms of value=-10000 is held in the resampler.
    try:
        await out.capture_frame(_tcp_frame(0.02, value=-10000))
        out.clear_buffer()
        out.flush()
        ev = await asyncio.wait_for(out.wait_for_playout(), timeout=0.2)
        assert ev.interrupted

        transport.sent.clear()

        await out.capture_frame(_tcp_frame(0.05, value=10000))
        out.flush()
        out.notify_playout_finished()
        await asyncio.wait_for(out.wait_for_playout(), timeout=1.0)

        audio_msgs = [msg.audio_output for msg in transport.sent if msg.HasField("audio_output")]
        samples = np.concatenate(
            [np.frombuffer(frame.data, dtype=np.int16) for frame in audio_msgs]
        )
        assert len(samples) == round(WIRE_SAMPLE_RATE * 0.05)
        assert np.all(samples[:400] > 0)
    finally:
        if out._flush_task is not None and not out._flush_task.done():
            out._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await out._flush_task


async def test_tcp_audio_output_flush_then_clear_discards_resampler_tail() -> None:
    transport = _RecordingTcpTransport()
    out = TcpAudioOutput(transport)  # type: ignore[arg-type]

    # Cancelled _audio_forwarding_task runs flush() then clear_buffer() in finally;
    # if only clear_buffer() ran without flushing, the stale -10000 samples must still
    # not leak into the next segment.
    await out.capture_frame(_tcp_frame(0.02, value=-10000))
    out.flush()
    out.clear_buffer()
    ev = await asyncio.wait_for(out.wait_for_playout(), timeout=0.2)
    assert ev.interrupted

    transport.sent.clear()

    await out.capture_frame(_tcp_frame(0.05, value=10000))
    out.flush()
    out.notify_playout_finished()
    await asyncio.wait_for(out.wait_for_playout(), timeout=1.0)

    audio_msgs = [msg.audio_output for msg in transport.sent if msg.HasField("audio_output")]
    samples = np.concatenate([np.frombuffer(frame.data, dtype=np.int16) for frame in audio_msgs])
    assert len(samples) == round(WIRE_SAMPLE_RATE * 0.05)
    assert np.all(samples[:400] > 0)


async def test_tcp_audio_output_end_to_end_over_tcp_transport() -> None:
    """Drive TcpAudioOutput -> TcpSessionTransport -> loopback broker -> SessionHost."""
    broker_received: list[str] = []
    broker_samples = 0
    client_tasks: set[asyncio.Task[None]] = set()

    async def _broker(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal broker_samples
        try:
            while True:
                header = await reader.readexactly(4)
                (size,) = struct.unpack(">I", header)
                payload = await reader.readexactly(size)
                msg = agent_pb.AgentSessionMessage()
                msg.ParseFromString(payload)
                kind = msg.WhichOneof("message")
                assert kind is not None
                broker_received.append(kind)
                if kind == "audio_output":
                    broker_samples += msg.audio_output.samples_per_channel
                elif kind == "audio_playback_flush":
                    ack = agent_pb.AgentSessionMessage(
                        audio_playback_finished=agent_pb.AgentSessionMessage.ConsoleIO.AudioPlaybackFinished()
                    ).SerializeToString()
                    writer.write(struct.pack(">I", len(ack)) + ack)
                    await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()

    def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(_broker(reader, writer))
        client_tasks.add(task)
        task.add_done_callback(client_tasks.discard)

    server = await asyncio.start_server(_on_connect, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    transport = TcpSessionTransport("127.0.0.1", port)
    out = TcpAudioOutput(transport)
    host = SessionHost(transport, audio_output=out)
    await host.start()
    try:
        # Empty flush sends nothing on the wire, so no stray ack poisons the next turn.
        out.flush()

        # Short 20 ms utterance flushes its resampled tail and completes via TCP ack.
        await out.capture_frame(_tcp_frame(0.02, value=4000))
        out.flush()
        ev = await asyncio.wait_for(out.wait_for_playout(), timeout=1.0)
        assert not ev.interrupted
        assert ev.playback_position == pytest.approx(0.02)
        assert broker_received == ["audio_output", "audio_playback_flush"]
        assert broker_samples == round(WIRE_SAMPLE_RATE * 0.02)
    finally:
        await host.aclose()
        server.close()
        await server.wait_closed()
        if client_tasks:
            await asyncio.gather(*client_tasks, return_exceptions=True)
