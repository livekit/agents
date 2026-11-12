"""Native integration checks with scripted providers and local transports."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from livekit.agents import (
    Agent,
    APIConnectionError,
    VoiceSession,
    _core,
    function_tool,
    llm,
)
from livekit.agents.llm._realtime import RealtimeModel as OpenAIRealtimeModel
from livekit.agents.voice import SpeechHandle

from .native_smoke_fixtures import (
    QUICK_RETRIES,
    DyingRealtime,
    FlakyLLM,
    FlakySTT,
    FlakyTTS,
    ScriptedDuplex,
    ScriptedLLM,
    ScriptedRealtime,
    ScriptedSTT,
    SilentTTS,
    _openai_server,
    _square,
    error_events,
    front_desk,
    metrics_events,
    print_item,
    until,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_logged_tracebacks(caplog: pytest.LogCaptureFixture):
    yield
    _core.flush_logs()
    assert not [record for record in caplog.get_records("call") if record.exc_info]


async def test_realtime() -> None:
    session = VoiceSession(llm=ScriptedRealtime(), vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are realtime.", name="realtime"), room=None)
    handle = session.generate_reply(instructions="greet the user")
    await until(handle.done, "the realtime reply did not finish")
    await session.aclose()


async def test_realtime_fallback() -> None:
    primary, backup = DyingRealtime(), ScriptedRealtime()
    adapter = llm.RealtimeModelFallbackAdapter([primary, backup])
    changes: list[tuple[bool, bool]] = []
    adapter.on(
        "realtime_availability_changed",
        lambda ev: changes.append((ev.realtime_model is primary, ev.available)),
    )
    session = VoiceSession(llm=adapter, vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are realtime.", name="fallback"), room=None)
    handle = session.generate_reply(instructions="greet the user")
    await until(handle.done, "the backup model did not answer")
    assert changes == [(True, False)], changes
    await session.aclose()
    print(f"availability {changes}")


async def test_duplex() -> None:
    model = ScriptedDuplex()
    session = VoiceSession(llm=llm.DuplexRealtimeAdapter(model), vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are duplex.", name="duplex"), room=None)
    handle = session.generate_reply()
    await until(handle.done, "the duplex reply did not finish")
    script = model.sessions[0]
    assert script._configured.is_set(), "the configuration never reached the model as one unit"
    assert script.configured_with == ["You are duplex."], script.configured_with
    await session.aclose()


async def test_openai_protocol_realtime() -> None:
    sessions: list[dict[str, Any]] = []

    async def socket(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await _openai_server(ws, sessions)
        return ws

    app = web.Application()
    app.router.add_get("/v1/realtime", socket)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    model = OpenAIRealtimeModel(
        api_key="sk-local", base_url=f"http://127.0.0.1:{port}/v1", voice="cedar"
    )
    session = VoiceSession(llm=model, vad=None)
    session.on("conversation_item_added", print_item)
    await session.start(Agent(instructions="You are realtime.", name="openai_realtime"), room=None)
    handle = session.generate_reply(instructions="greet the user")
    await until(handle.done, "the OpenAI-protocol realtime reply did not finish")
    assert sessions[0]["audio"]["output"]["voice"] == "cedar", sessions[0]
    await session.aclose()
    await runner.cleanup()
    usage = session.usage
    assert usage.settled and [u.model for u in usage.models] == ["gpt-realtime"], usage
    print(f"usage_requests={usage.requests}")


async def test_console() -> None:
    """``lk agent console``'s socket: session state, a typed turn, events, playback acks, close."""
    import struct

    from livekit.agents.voice._session_bindings import ConsoleCall
    from livekit.protocol.agent_pb import agent_session as pb

    print("== console")
    accepted: asyncio.Future[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
        asyncio.get_running_loop().create_future()
    )
    server = await asyncio.start_server(lambda r, w: accepted.set_result((r, w)), "127.0.0.1", 0)
    ConsoleCall.addr = f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    await session.start(front_desk())
    reader, writer = await asyncio.wait_for(accepted, 10)

    def send(msg: pb.AgentSessionMessage) -> None:
        data = msg.SerializeToString()
        writer.write(struct.pack(">I", len(data)) + data)

    async def recv() -> pb.AgentSessionMessage | None:
        try:
            (size,) = struct.unpack(">I", await reader.readexactly(4))
            return pb.AgentSessionMessage.FromString(await reader.readexactly(size))
        except asyncio.IncompleteReadError:
            return None

    events: list[str] = []
    responses: dict[str, pb.SessionResponse] = {}

    async def until(request_id: str) -> pb.SessionResponse:
        while request_id not in responses:
            msg = await asyncio.wait_for(recv(), 10)
            assert msg is not None, "the socket ended early"
            kind = msg.WhichOneof("message")
            if kind == "event":
                events.append(msg.event.WhichOneof("event"))
            elif kind == "response":
                responses[msg.response.request_id] = msg.response
            elif kind == "audio_playback_flush":
                send(
                    pb.AgentSessionMessage(
                        audio_playback_finished=pb.AgentSessionMessage.ConsoleIO.AudioPlaybackFinished()
                    )
                )
        return responses[request_id]

    send(
        pb.AgentSessionMessage(
            request=pb.SessionRequest(
                request_id="req_state", get_session_state=pb.SessionRequest.GetSessionState()
            )
        )
    )
    state = await until("req_state")
    assert not state.error and state.get_session_state.agent_id, state
    send(
        pb.AgentSessionMessage(
            request=pb.SessionRequest(
                request_id="req_run", run_input=pb.SessionRequest.RunInput(text="hello there")
            )
        )
    )
    run = await until("req_run")
    assert not run.error, run.error
    items = [i.WhichOneof("item") for i in run.run_input.items]
    print(f"console run_input items {items}")
    assert "message" in items, items
    await session.aclose()
    while (msg := await asyncio.wait_for(recv(), 10)) is not None:
        if msg.WhichOneof("message") == "event":
            events.append(msg.event.WhichOneof("event"))
    print(f"console events {sorted(set(events))}")
    assert {"agent_state_changed", "conversation_item_added"} <= set(events), events
    writer.close()
    server.close()


async def test_job() -> None:
    """A console job over IPC: connect (twice, deduplicated), wait for the participant, a skipped
    API call, and a requested shutdown that runs the callbacks and reaches the parent."""
    from livekit.agents.ipc import JobRunnerOptions, JobStatus
    from livekit.agents.ipc.child import JobArgs, _JobProc
    from livekit.agents.job import JobAcceptArguments, JobContext, JobExecutorType, RunningJobInfo
    from livekit.protocol import agent as agent_proto, models

    print("== job")
    seen: list[str] = []

    async def on_shutdown(reason: str) -> None:
        seen.append(f"shutdown callback {reason!r}")

    async def on_shutdown_no_reason() -> None:
        seen.append("shutdown callback, no reason")

    async def entrypoint(ctx: JobContext) -> None:
        ctx.add_shutdown_callback(on_shutdown)
        ctx.add_shutdown_callback(on_shutdown_no_reason)
        await asyncio.gather(ctx.connect(), ctx.connect())
        seen.append(f"waited for {(await ctx.wait_for_participant()).identity}")
        seen.append(f"delete_room {type(await ctx.delete_room()).__name__}")
        ctx.shutdown("done")

    runner = _JobProc(
        JobArgs(
            job_entrypoint_fnc=entrypoint,
            session_end_fnc=None,
            simulation_end_fnc=None,
            session_end_timeout=5.0,
        ),
        JobExecutorType.THREAD,
    )

    class Logs(logging.Handler):
        def __init__(self) -> None:
            super().__init__(logging.DEBUG)
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    logs = Logs()
    agents_logger = logging.getLogger("livekit.agents")
    agents_logger.addHandler(logs)
    agents_logger.setLevel(logging.DEBUG)

    class Parent:
        def __init__(self) -> None:
            self.endpoint: asyncio.Future[int] = asyncio.get_running_loop().create_future()
            self.launch: Any = None

        async def _core_launch(self, endpoint: int, launch: Any) -> None:
            self.launch = launch
            self.endpoint.set_result(endpoint)

        async def _core_completed(self, completion: Any) -> None:
            pass

    # a thread job served on this loop: no pings, nothing to kill
    parent = Parent()
    sup = _core.JobSupervisor(
        parent,
        JobRunnerOptions(
            thread=True,
            initialize_timeout=30,
            close_timeout=30,
            ping_interval=0,
            ping_timeout=0,
            high_ping_threshold=10,
        ),
    )
    await sup.start()
    cch = await parent.endpoint
    initialized = asyncio.ensure_future(sup.initialize())
    client = await asyncio.to_thread(_core.IpcClient, cch, runner.initialize)
    await initialized
    runner._client = client
    serving = asyncio.ensure_future(client.serve(runner))
    job = agent_proto.Job(id="AJ_smoke", room=models.Room(name="console", sid="RM_smoke"))
    await sup.launch_job(
        RunningJobInfo(
            accept_arguments=JobAcceptArguments(
                name="", identity="agent", metadata="", attributes={}
            ),
            job=job,
            url="",
            token="",
            worker_id="smoke",
            fake_job=True,
            received_at=0,
            accepted_at=0,
            assigned_at=0,
        )
    )
    await asyncio.wait_for(serving, 30)
    assert runner._job_task is not None
    await asyncio.wait_for(runner._job_task, 30)
    await client.aclose()
    parent.launch.thread_exited()
    assert await sup.join()
    assert sup.status == JobStatus.SUCCESS, sup.status
    _core.flush_logs()
    agents_logger.removeHandler(logs)
    print(f"job {seen} told its parent {[r.getMessage() for r in logs.records]}")
    assert sorted(seen) == [
        "delete_room DeleteRoomResponse",
        "shutdown callback 'done'",
        "shutdown callback, no reason",
        "waited for mock_user",
    ], seen
    assert any(
        r.getMessage() == "job thread exiting"
        and (r.reason, r.job_id, r.room, r.room_id) == ("done", "AJ_smoke", "console", "RM_smoke")  # type: ignore[attr-defined]
        for r in logs.records
    ), logs.records
    assert sup.running_job is not None
    assert sup.running_job.job.id == "AJ_smoke"
    assert sup.running_job.launched_at > 0


async def test_workflows() -> None:
    from livekit.agents.beta import workflows as wf
    from livekit.agents.llm import ToolError

    print("== workflows")
    assert wf.validate_card_number("4242424242424242")
    assert not wf.validate_card_number("4242424242424241")
    assert not wf.is_expired(12, 99)
    assert wf.normalize_email_address(" theo@livekit.io ") == "theo@livekit.io"
    assert wf.normalize_phone_number("+1 (510) 555-0123") == "+15105550123"
    assert str(wf.normalize_date_of_birth(90, 5, 2)) == "1990-05-02"
    for check, value in ((wf.normalize_email_address, "theo"), (wf.normalize_phone_number, "12")):
        with contextlib.suppress(ToolError):
            check(value)
            raise AssertionError(f"{value!r} was accepted")
    assert wf.format_dtmf([wf.DtmfEvent.ONE, wf.DtmfEvent.POUND]) == "1 #"
    assert wf.dtmf_event_to_code(wf.DtmfEvent.D) == 15


async def test_toolsets() -> None:
    """A search loads the matching tools; the proxy returns their schemas and calls one by name."""
    from livekit.agents.beta import KeywordSearchStrategy, ToolProxyToolset, ToolSearchToolset
    from livekit.agents.llm import ToolError

    print("== toolsets")

    @function_tool
    async def get_weather(city: str) -> str:
        """The weather in a city."""
        return f"sunny in {city}"

    @function_tool
    async def get_stock(symbol: str) -> str:
        """A stock's price."""
        return f"{symbol}: 100"

    search = ToolSearchToolset(id="search", tools=[get_weather, get_stock])
    await search.setup()
    print(await search._handle_search({"query": "weather"}))
    assert [t.id for t in search.tools] == ["tool_search", "get_weather"], search.tools
    with contextlib.suppress(ToolError):
        await search._handle_search({"query": "nothing matches"})
        raise AssertionError("an empty search loaded tools")
    keyword = ToolSearchToolset(
        id="kw", tools=[get_weather, get_stock], search_strategy=KeywordSearchStrategy()
    )
    await keyword.setup()
    await keyword._handle_search({"query": "stock"})
    assert [t.id for t in keyword.tools] == ["tool_search", "get_stock"], keyword.tools

    proxy = ToolProxyToolset(id="proxy", tools=[get_weather, get_stock])
    await proxy.setup()
    schemas = await proxy._handle_search({"query": "stock"})
    assert json.loads(schemas)["name"] == "get_stock", schemas
    called = await proxy._handle_call(None, {"name": "get_weather", "parameters": {"city": "Oslo"}})
    print(f"proxy called: {called}")
    assert called == "sunny in Oslo", called
    for bad in ({"name": "", "parameters": {}}, {"name": "nope", "parameters": {}}):
        with contextlib.suppress(ToolError):
            await proxy._handle_call(None, bad)
            raise AssertionError(f"{bad} was called")


async def test_cli() -> None:
    """What ``lk agent`` drives: argument parsing, JSON log lines, the dev channel, shutdown."""
    import struct

    from livekit.agents import cli as agents_cli
    from livekit.agents._generated.cli import CliArgsStart, CliParseResultDisplay
    from livekit.protocol.agent_pb import agent_dev

    print("== cli")
    args = _core.parse_cli(["agent", "start", "--dev", "--reload-addr", "127.0.0.1:1"]).args
    assert isinstance(args, CliArgsStart)
    assert args.cli_addr == "127.0.0.1:1", args
    for bad in (["start", "--cli-addr", "x"], ["console"], ["serve"]):
        assert isinstance(_core.parse_cli(["agent", *bad]), CliParseResultDisplay)

    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("livekit.agents", logging.ERROR, __file__, 1, "x=%d", (3,), exc_info)
    record.job_id = "AJ_1"
    line = json.loads(agents_cli._LogFormatter("json", ensure_ascii=True).format(record))
    assert (line["message"], line["level"], line["job_id"]) == ("x=3", "ERROR", "AJ_1"), line
    assert "ValueError: boom" in line["exception"], line

    async def read(reader: asyncio.StreamReader) -> agent_dev.AgentDevMessage:
        (length,) = struct.unpack("!I", await reader.readexactly(4))
        return agent_dev.AgentDevMessage.FromString(await reader.readexactly(length))

    def write(writer: asyncio.StreamWriter, msg: agent_dev.AgentDevMessage) -> None:
        data = msg.SerializeToString()
        writer.write(struct.pack("!I", len(data)) + data)

    seen: list[str] = []
    running: list[agent_dev.RunningAgentJobInfo] = []
    done = asyncio.Event()

    async def lk_cli(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        seen.append((await read(reader)).server_info.agent_name)
        assert (await read(reader)).HasField("get_running_jobs_request")
        write(
            writer,
            agent_dev.AgentDevMessage(
                get_running_jobs_response=agent_dev.GetRunningAgentJobsResponse(jobs=[])
            ),
        )
        write(
            writer,
            agent_dev.AgentDevMessage(
                get_running_jobs_request=agent_dev.GetRunningAgentJobsRequest()
            ),
        )
        await writer.drain()
        running.extend((await read(reader)).get_running_jobs_response.jobs)
        writer.close()
        done.set()

    stand_in = await asyncio.start_server(lk_cli, "127.0.0.1", 0)
    port = stand_in.sockets[0].getsockname()[1]
    # no job process runs yet, so the dev channel reports none
    from livekit.agents import AgentServer

    server = AgentServer(
        ws_url="wss://x", api_key="key", api_secret="secret_of_32_bytes_or_more_ok!!"
    )
    await asyncio.wait_for(
        _core.dev_channel(
            f"127.0.0.1:{port}",
            "dev-agent",
            "wss://x",
            "key",
            "secret_of_32_bytes_or_more_ok!!",
            server._ensure_runtime(),
        ),
        10,
    )
    await asyncio.wait_for(done.wait(), 10)
    stand_in.close()
    print(f"dev channel {seen}, running {len(running)}")
    assert seen == ["dev-agent"], seen
    assert running == [], running

    shutdown = _core.Shutdown(1.0, 0.0)
    assert shutdown.begin()
    assert not shutdown.begin()
    shutdown.closed()


async def test_decoder() -> None:
    """``sample_rate=None`` keeps a 16 kHz WAV's own rate; otherwise it is resampled."""
    import io
    import wave

    from livekit.agents.utils.media import AudioStreamDecoder

    print("== decoder")
    wav = io.BytesIO()
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    for asked, expected in ((None, 16000), (24000, 24000)):
        decoder = AudioStreamDecoder(sample_rate=asked, format="audio/wav")
        decoder.push(wav.getvalue())
        decoder.end_input()
        rates = {frame.sample_rate async for frame in decoder}
        print(f"decoded at {rates} with sample_rate={asked}")
        assert rates == {expected}, (asked, rates)


async def test_providers() -> None:
    """A standalone provider retries per ``conn_options``: each retried failure emits a
    recoverable ``error`` event, the last one a non-recoverable one."""
    print("== standalone provider retries")
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")
    model = FlakyLLM(fails=2)
    seen = error_events(model)
    measured = metrics_events(model)
    reply = await model.chat(chat_ctx=chat_ctx).collect()
    assert (reply.text, model.calls, seen) == ("You said: hello", 3, [True, True]), (reply, seen)
    # one metrics record, for the request that answered
    assert [type(m).__name__ for m in measured] == ["LLMMetrics"], measured

    model = FlakyLLM(fails=9)
    seen = error_events(model)
    try:
        await model.chat(chat_ctx=chat_ctx).collect()
        raise AssertionError("an exhausted budget raises")
    except APIConnectionError as e:
        assert "after 3 attempts" in str(e), e
    assert (model.calls, seen) == (3, [True, True, False]), (model.calls, seen)

    voice = FlakyTTS(fails=1)
    seen = error_events(voice)
    measured = metrics_events(voice)
    frame = await voice.synthesize("hi").collect()
    assert frame.duration > 0 and voice.attempts == {"hi": 2} and seen == [True], seen
    assert [(type(m).__name__, m.characters_count) for m in measured] == [("TTSMetrics", 2)]

    # a retried stream is fed again what was pushed before the failure
    voice = FlakyTTS(fails=1, streaming=True)
    seen = error_events(voice)
    stream = voice.stream()
    stream.push_text("hello ")
    stream.push_text("world")
    stream.end_input()
    assert [ev async for ev in stream] and voice.attempts == {"hello world": 2}, voice.attempts
    assert seen == [True], seen

    ears = FlakySTT(fails=1)
    seen = error_events(ears)
    measured = metrics_events(ears)
    said = await ears.recognize([_square(0.0)], conn_options=QUICK_RETRIES)
    assert said.alternatives[0].text == "batch words" and seen == [True], seen
    assert [type(m).__name__ for m in measured] == ["STTMetrics"], measured

    # a gateway LLM gets a native leg per session, unless a subclass overrides chat()
    from livekit.agents import _bindings, inference

    class Logged(inference.LLM):
        def chat(self, **kwargs: Any) -> llm.LLMStream:
            return super().chat(**kwargs)

    gateway = inference.LLM("openai/gpt-4.1-mini", api_key="k", api_secret="s")
    first, second = _bindings.core_leg(gateway), _bindings.core_leg(gateway)
    assert isinstance(first, _core.InferenceLlm) and first is not second, (first, second)
    logged = Logged("openai/gpt-4.1-mini", api_key="k", api_secret="s")
    assert _bindings.core_leg(logged) is logged


async def test_llm_prewarm() -> None:
    """The gateway LLM is prewarmed (``GET /models``) as the session starts."""
    from livekit.agents import inference

    print("== llm prewarm")
    seen: list[str] = []

    async def models(request: web.Request) -> web.Response:
        seen.append(f"{request.method} {request.path}")
        return web.json_response({"object": "list", "data": []})

    app = web.Application()
    app.router.add_get("/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    llm_ = inference.LLM(
        "openai/gpt-4.1-mini", base_url=f"http://127.0.0.1:{port}", api_key="k", api_secret="s"
    )
    session = VoiceSession(stt=ScriptedSTT(), llm=llm_, tts=SilentTTS(), vad=None)
    await session.start(Agent(instructions="You answer."), room=None)
    await until(lambda: bool(seen), "the gateway saw no request", 2)
    await session.aclose()
    await runner.cleanup()
    print(f"gateway saw {seen}")
    assert seen == ["GET /models"], seen


def test_job_logs() -> None:
    """A job process logs JSON to its inherited stdout, stamped with its pid and job; non-JSON
    extras use their string representation."""
    import os

    from livekit.agents import Agent

    print("== a job process's own log lines")
    pii = str(Agent(instructions="x", name="pii")).split(" object at ")[0]
    script = Path(__file__).with_name("native_job_logs.py")
    for style in ("json", "none"):
        run = subprocess.run(
            [sys.executable, str(script), style], capture_output=True, text=True, timeout=120
        )
        lines = [line for line in run.stdout.splitlines() if "from the job" in line]
        assert run.returncode == 0 and len(lines) == 1, (run.stdout, run.stderr)
        record = json.loads(lines[0])
        assert record["job_id"] == "AJ_logs" and record["room"] == "logs", record
        assert record["pid"] not in (os.getpid(), None), record
        assert record["lk.pii.output"].startswith(pii), record


async def test_voice_and_text() -> None:
    session = VoiceSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)

    def on_tools(ev: object) -> None:
        calls = [(c.name, o.output) for c, o in ev.zipped()]  # type: ignore[attr-defined]
        print(f"function_tools_executed {calls}")

    def on_state(ev: object) -> None:
        assert session.agent_state == ev.new_state  # type: ignore[attr-defined]
        print(f"agent_state {ev.new_state}")  # type: ignore[attr-defined]

    session.on("agent_state_changed", on_state)
    session.on("conversation_item_added", print_item)
    session.on("function_tools_executed", on_tools)
    session.on("speech_created", lambda ev: print(f"speech_created {ev.source}"))
    session.on("close", lambda ev: print(f"close {ev.reason.value}"))

    print("== start")
    await session.start(front_desk(), room=None)

    print("== say")
    # without audio output the say can end before the next line; callbacks wait for speech_done
    handle = session.say("Let me say something.")
    # set before the speech is created
    handle.allow_interruptions = False
    calls: list[str] = []
    handle.add_done_callback(lambda h: calls.append(f"first {h.id}"))
    handle.add_done_callback(dropped := lambda h: calls.append("dropped"))
    handle.add_done_callback(lambda h: calls.append("second"))
    handle.remove_done_callback(dropped)
    await until(lambda: len(calls) >= 2, "say()'s done callbacks did not run")
    assert handle.done() and not handle.allow_interruptions, "the setter before creation was lost"
    assert calls == [f"first {handle.id}", "second"], calls
    assert handle.exception() is None
    late: list[str] = []
    handle.add_done_callback(lambda h: late.append("late"))
    await asyncio.sleep(0)
    assert late == ["late"], "a callback added after done runs soon"

    print("== wait_if_not_interrupted")

    async def unfinished() -> AsyncIterator[str]:
        # never ends, so the speech is still running at interrupt() even without audio output
        yield "A long sentence the agent will not finish saying."
        await asyncio.Event().wait()

    handle = session.say(unfinished())
    work: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    waiting = asyncio.ensure_future(handle.wait_if_not_interrupted([work]))
    handle.interrupt()
    await until(waiting.done, "an interrupt ends the wait")
    assert not work.done(), "an interrupt ends the wait, never the work"
    work.cancel()
    assert SpeechHandle.SPEECH_PRIORITY_LOW < SpeechHandle.SPEECH_PRIORITY_HIGH

    print("== say, streamed words")

    async def words() -> AsyncIterator[str]:
        for word in ("Streamed ", "words."):
            yield word

    handle = session.say(words())
    await until(handle.done, "say(AsyncIterable) did not finish")

    print("== generate_reply")
    handle = session.generate_reply(user_input="hello there")
    await until(handle.done, "generate_reply did not finish")

    print("== a reply, asserted")
    result = await asyncio.wait_for(session._run(user_input="what's the weather like?"), 10)
    assert result.status == "completed", result.status
    result.expect.next_event().is_function_call(name="get_weather")
    result.expect.next_event().is_function_call_output(output="sunny in Paris")
    result.expect.next_event().is_message(role="assistant")
    result.expect.no_more_events()
    print(f"run items: {[item.type for item in result.new_messages()]}")

    print("== handoff")
    session.generate_reply(user_input="put me through to billing")
    await until(lambda: session.current_agent.name == "Billing", "no handoff to billing")

    print("== close")
    await session.aclose()

    print("history:")
    for item in session.history.items:
        detail = getattr(item, "text_content", None) or getattr(item, "name", "")
        print(f"  {item.type} {getattr(item, 'role', '')} {detail or ''}".rstrip())

    print("== agent.run(), the text path")
    text = await Agent(instructions="You answer by text.", name="text", llm=ScriptedLLM()).run(
        "hello by text"
    )
    print(f"output {text.output!r}")
    assert text.output == "You said: hello by text", text.output
