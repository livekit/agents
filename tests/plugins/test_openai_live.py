"""OpenAI GPT Live: a VoiceSession against a local Live API peer speaking its literal JSON."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal, TypedDict

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    Agent,
    APIConnectOptions,
    ErrorEvent,
    NotGivenOr,
    VoiceSession,
    _core,
    function_tool,
    http,
    llm,
)
from livekit.agents.metrics import LLMMetrics, RealtimeModelMetrics
from livekit.plugins.openai.realtime import ResponsesDelegationOptions, gpt_live_types
from livekit.plugins.openai.realtime.gpt_live_model import (
    GPTLiveDelegation,
    GPTLiveModel,
    GPTVoiceSession,
)
from tests.fakes import FakeAudioOutput, audio_frame, until


class _Model(GPTLiveModel):
    """Keeps the session the VoiceSession opens, to reach its mute and append methods, and the
    metrics it reports from its creation on."""

    live: GPTVoiceSession
    metrics: list[RealtimeModelMetrics | LLMMetrics]

    def session(self) -> GPTVoiceSession:
        self.live = super().session()
        self.metrics = []
        self.live.on("metrics_collected", self.metrics.append)
        return self.live

    def acquired(self) -> list[RealtimeModelMetrics]:
        """The connection metrics: each one a session the service started."""
        return [
            m
            for m in self.metrics
            if isinstance(m, RealtimeModelMetrics) and m.acquire_time is not None
        ]

    def used(self) -> list[float]:
        """The session seconds each usage report added."""
        return [
            m.connection_duration
            for m in self.metrics
            if isinstance(m, RealtimeModelMetrics) and m.connection_duration is not None
        ]


class _Tool(TypedDict):
    name: str


class _Responses(TypedDict):
    tools: list[_Tool]
    parallel_tool_calls: bool


class _Delegation(TypedDict):
    responses: _Responses


class _Part(TypedDict):
    text: str


class _Input(TypedDict):
    role: str
    content: list[_Part]


class _Session(TypedDict):
    instructions: str
    audio: dict[str, object]
    delegation: _Delegation
    input: list[_Input]


class _Start(TypedDict):
    """The parts of a session.start event the tests read."""

    type: str
    session: _Session


class _Peer:
    """A Live API endpoint: answers session.start and session.close, and keeps every event each
    connection sent, audio aside."""

    def __init__(self, *, start: bool = True, closed_seconds: float = 0) -> None:
        self.start = start
        self.closed_seconds = closed_seconds
        self.connections: list[list[dict[str, object]]] = []
        self.starts: list[_Start] = []
        self.audio: list[bytes] = []
        self.sockets: list[web.WebSocketResponse] = []

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.sockets.append(socket)
        sent: list[dict[str, object]] = []
        self.connections.append(sent)
        async for message in socket:
            event = json.loads(message.data)
            if event["type"] == "session.input_audio.append":
                self.audio.append(base64.b64decode(event["audio"]))
                continue
            sent.append(event)
            if event["type"] == "session.start":
                self.starts.append(event)
            if event["type"] == "session.start" and self.start:
                await socket.send_json({"type": "session.started", "session": {"id": "live_1"}})
            elif event["type"] == "session.close":
                usage = {"seconds": self.closed_seconds}
                await socket.send_json(
                    {"type": "session.closed", "reason": "close_requested", "usage": usage}
                )
        return socket

    @property
    def socket(self) -> web.WebSocketResponse:
        return self.sockets[-1]

    def sent(self, connection: int = -1) -> list[str]:
        """The event types a connection was sent after its session.start."""
        return [str(e["type"]) for e in self.connections[connection][1:]]

    async def backend(self, *events: dict[str, object], delegation: str = "d1") -> None:
        for event in events:
            await self.socket.send_json(
                {"type": "response.event", "delegation_id": delegation, "event": event}
            )

    async def barrier(self, model: _Model) -> None:
        """Wait until everything the session queued before now has reached the peer."""
        model.live.append_thinking("barrier")
        await until(
            lambda: any(e.get("content") == "barrier" for e in self.connections[-1]),
            "the session stopped sending",
        )
        self.connections[-1][:] = [e for e in self.connections[-1] if e.get("content") != "barrier"]


@asynccontextmanager
async def _voice(
    peer: _Peer,
    agent: Agent,
    *,
    output: FakeAudioOutput | None = None,
    voice: str | gpt_live_types.CustomVoice = "marin",
    responses_options: NotGivenOr[ResponsesDelegationOptions] = NOT_GIVEN,
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    delegation: Literal["responses", "client"] = "responses",
    max_session_duration: float | None = None,
) -> AsyncIterator[_Model]:
    app = web.Application()
    app.router.add_get("/v1/live/sessions", peer.handle)
    async with TestServer(app) as server, http.open():
        model = _Model(
            api_key="sk-test",
            base_url=str(server.make_url("/v1")),
            voice=voice,
            responses_options=responses_options,
            conn_options=conn_options,
            delegation=delegation,
            max_session_duration=max_session_duration,
        )
        session = VoiceSession(llm=llm.DuplexRealtimeAdapter(model), vad=None)
        session.output.audio = output or FakeAudioOutput()
        await session.start(agent, room=None)
        await until(lambda: peer.connections and peer.connections[0], "no session.start")
        try:
            yield model
        finally:
            await session.aclose()
            await model.aclose()


def _created(response_id: str) -> dict[str, object]:
    return {"type": "response.created", "response": {"id": response_id}}


def _call(call: str, key: str = "a", /, **item: str | None) -> dict[str, object]:
    """A completed `lookup` call; a field given as None is left out."""
    done: dict[str, object] = {
        "id": f"fc_{call}",
        "type": "function_call",
        "status": "completed",
        "call_id": call,
        "name": "lookup",
        "arguments": json.dumps({"key": key}),
    }
    fields = {k: v for k, v in {**done, **item}.items() if v is not None}
    return {"type": "response.output_item.done", "item": fields}


def _completed(response_id: str) -> dict[str, object]:
    return {"type": "response.completed", "response": {"id": response_id, "status": "completed"}}


class _Lookups:
    """A `lookup` tool whose answer for each key waits until the test releases it."""

    def __init__(self, *held: str) -> None:
        self.calls: list[str] = []
        self.gates = {key: asyncio.Event() for key in held}

        @function_tool
        async def lookup(key: str) -> str:
            """Look a key up."""
            self.calls.append(key)
            if key in self.gates:
                await self.gates[key].wait()
            return key

        self.agent = Agent(instructions="Answer.", tools=[lookup])


async def test_the_session_starts_with_its_whole_configuration() -> None:
    """The first event carries instructions, voice and the backend model with the agent's tools.

    https://github.com/livekit/agents/pull/7212
    """
    peer = _Peer()
    lookups = _Lookups()

    async with _voice(
        peer,
        lookups.agent,
        voice={"id": "voice_test"},
        responses_options={"parallel_tool_calls": False},
    ):
        [start] = peer.starts

    assert start["type"] == "session.start"
    config = start["session"]
    assert config["instructions"] == "Answer."
    assert config["audio"] == {
        "format": {"type": "audio/pcm", "rate": 24000},
        "output": {"voice": {"id": "voice_test"}},
    }
    backend = config["delegation"]["responses"]
    assert [tool["name"] for tool in backend["tools"]] == ["lookup"]
    assert backend["parallel_tool_calls"] is False


async def test_output_audio_is_played() -> None:
    """The model's audio deltas reach the session's audio output.

    https://github.com/livekit/agents/pull/7212
    """
    peer = _Peer()
    output = FakeAudioOutput()
    delta = base64.b64encode(audio_frame(0.1).data.tobytes()).decode()
    async with _voice(peer, Agent(instructions="Answer."), output=output):
        for _ in range(3):
            await peer.socket.send_json({"type": "session.output_audio.delta", "delta": delta})
        await peer.socket.send_json(
            {"type": "session.output_transcript.delta", "delta": "Hi", "start_ms": 0, "end_ms": 300}
        )

        await asyncio.wait_for(output.captured.wait(), 5)


async def test_the_mute_state_survives_a_reconnect() -> None:
    """A reconnect re-sends the mute before anything else.

    https://github.com/livekit/agents/issues/7662
    """
    peer = _Peer()
    options = APIConnectOptions(max_retry=1, retry_interval=0)
    async with _voice(peer, Agent(instructions="Answer."), conn_options=options) as model:
        model.live.mute_input()
        await until(lambda: "session.input_audio.mute" in peer.sent(0), "the mute was not sent")

        await peer.sockets[0].close()
        await until(lambda: len(peer.connections) == 2 and peer.sent(1), "no reconnect")

        assert peer.sent(1)[0] == "session.input_audio.mute"


async def test_a_continuation_waits_for_every_open_call() -> None:
    """A fast tool result cannot continue the backend while a slower response's call is open.

    https://github.com/livekit/agents/pull/7266
    """
    peer = _Peer()
    lookups = _Lookups("slow", "fast")
    async with _voice(peer, lookups.agent) as model:
        await peer.backend(_created("resp_1"), _call("call_slow", "slow"), _completed("resp_1"))
        await peer.backend(_created("resp_2"), _call("call_fast", "fast"), _completed("resp_2"))
        await until(lambda: len(lookups.calls) == 2, "the tools did not run")

        lookups.gates["fast"].set()
        await until(lambda: "response.item.create" in peer.sent(), "no fast result")
        await peer.barrier(model)
        lookups.gates["slow"].set()
        await until(lambda: "response.create" in peer.sent(), "no continuation")

        assert peer.sent() == ["response.item.create", "response.item.create", "response.create"]


async def test_a_failed_response_releases_the_continuation_it_held_back() -> None:
    """A failed response stops blocking an answered one, and its own call's result is context.

    https://github.com/livekit/agents/pull/7266
    """
    peer = _Peer()
    lookups = _Lookups("two")
    async with _voice(peer, lookups.agent) as model:
        await peer.backend(_created("resp_1"), _call("call_1", "one"), _completed("resp_1"))
        await peer.backend(_created("resp_2"), _call("call_2", "two"), delegation="d2")
        await until(lambda: "response.item.create" in peer.sent(), "no first result")
        await peer.barrier(model)
        assert peer.sent() == ["response.item.create"]

        await peer.backend(
            {"type": "response.failed", "response": {"id": "resp_2"}}, delegation="d2"
        )
        await until(lambda: "response.create" in peer.sent(), "the failure held the chain")
        lookups.gates["two"].set()
        await until(lambda: "session.thinking.append" in peer.sent(), "the result was lost")

        assert peer.sent() == ["response.item.create", "response.create", "session.thinking.append"]


@pytest.mark.parametrize(
    "item",
    [
        {"status": None},
        {"status": "incomplete"},
        {"status": "in_progress"},
        {"status": "failed"},
        {"call_id": None},
        {"name": None},
        {"arguments": None},
    ],
)
async def test_an_unfinished_backend_call_does_not_run_its_tool(
    item: dict[str, str | None],
) -> None:
    """Only a completed, whole function call runs its tool; a later complete one still does.

    https://github.com/livekit/agents/pull/7230
    """
    peer = _Peer()
    lookups = _Lookups()
    unfinished = _call("call_1", "early", **item)
    async with _voice(peer, lookups.agent) as model:
        await peer.backend(_created("resp_1"), unfinished, _call("call_1", "late"))
        await until(lambda: lookups.calls, "the completed call did not run")
        await peer.barrier(model)

    assert lookups.calls == ["late"]


async def test_a_duplicated_backend_call_runs_once_and_continues_once() -> None:
    """A repeated call event neither runs the tool twice nor changes when the backend continues.

    https://github.com/livekit/agents/pull/7230
    """
    peer = _Peer()
    lookups = _Lookups("b")
    async with _voice(peer, lookups.agent) as model:
        calls = [_call("call_a", "a"), _call("call_a", "a"), _call("call_b", "b")]
        await peer.backend(_created("resp_1"), *calls, _call("call_b", "b"), _completed("resp_1"))
        await until(lambda: "response.item.create" in peer.sent(), "no first result")
        await peer.barrier(model)
        assert peer.sent() == ["response.item.create"]

        lookups.gates["b"].set()
        await until(lambda: "response.create" in peer.sent(), "no continuation")

    assert sorted(lookups.calls) == ["a", "b"]
    assert peer.sent().count("response.create") == 1


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/issues/8441 reproduces: response.create is sent "
    "inside a continuation that had no response.created",
)
async def test_a_fast_answer_continues_only_after_its_continuation_completes() -> None:
    """A tool answered while its continuation still runs continues the chain once it completes.

    https://github.com/livekit/agents/issues/8441
    """
    peer = _Peer()
    lookups = _Lookups()
    async with _voice(peer, lookups.agent) as model:
        await peer.backend(_created("resp_1"), _call("call_1", "one"), _completed("resp_1"))
        await until(lambda: "response.create" in peer.sent(), "no continuation")

        # the continuation streams its call with no response.created before it
        await peer.backend(_call("call_2", "two"))
        await until(lambda: peer.sent().count("response.item.create") == 2, "no second result")
        await peer.barrier(model)

        assert peer.sent() == ["response.item.create", "response.create", "response.item.create"]


async def test_a_delayed_start_keeps_only_recent_audio_and_every_command() -> None:
    """Before session.started, only the newest audio is kept and queued commands all go out.

    https://github.com/livekit/agents/pull/8387
    """
    peer = _Peer(start=False)
    async with _voice(peer, Agent(instructions="Answer.")) as model:
        for index in range(80):
            model.live.push_audio(
                rtc.AudioFrame(index.to_bytes(2, "little") * 2400, 24000, 1, 2400)
            )
        model.live.append_instructions("first rule")
        model.live.append_thinking("then context")

        await peer.socket.send_json({"type": "session.started", "session": {"id": "late"}})
        await until(lambda: "session.thinking.append" in peer.sent(), "the commands were lost")
        last = (79).to_bytes(2, "little") * 2400
        await until(lambda: b"".join(peer.audio).endswith(last), "the newest audio was lost")

    audio = b"".join(peer.audio)
    assert len(audio) <= 4 * 24000 * 2
    assert [e["content"] for e in peer.connections[0] if "content" in e] == [
        "first rule",
        "then context",
    ]


async def test_provider_content_is_only_logged_under_pii_fields(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With debug logging on, the customer's words appear only under `lk.pii.*` record fields.

    https://github.com/livekit/agents/pull/7212
    """
    monkeypatch.setenv("LK_OPENAI_DEBUG", "1")
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.openai")
    private = "private-customer-payload"
    peer = _Peer()
    async with _voice(peer, Agent(instructions=private)) as model:
        model.live.append_thinking(private)
        await peer.socket.send_json(
            {"type": "session.input_transcript.delta", "delta": private, "start_ms": 0}
        )
        failed = {"error": {"message": private}, "incomplete_details": {"reason": private}}
        await peer.backend(
            {"type": "response.failed", "response": failed},
            _call("call_1", status="incomplete", arguments=private),
        )
        await peer.socket.send_json({"type": "error", "error": {"message": private}})
        await peer.barrier(model)

    leaks = [
        key
        for record in caplog.records
        for key, value in vars(record).items()
        if private in repr(value) and ".pii." not in key
    ]
    assert leaks == []
    assert any("lk.pii.event" in vars(record) for record in caplog.records)


async def test_a_backend_tool_call_gets_its_output_back() -> None:
    """The tool's result goes back to the backend as the output of the call it answers.

    https://github.com/livekit/agents/pull/7230
    """
    # Catches: the result returned on the voice channel or under another call id, which left
    # the backend waiting on its call.
    peer = _Peer()
    lookups = _Lookups()
    async with _voice(peer, lookups.agent):
        await peer.backend(_created("resp_1"), _call("call_1", "weather"), _completed("resp_1"))
        await until(lambda: "response.item.create" in peer.sent(), "no tool result")

    [result] = [e for e in peer.connections[0] if e["type"] == "response.item.create"]
    assert result["item"] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "weather",
    }


async def test_client_delegation_reaches_the_application_and_is_answered() -> None:
    """A delegation to the client is handed to the application, whose answer goes back under its id.

    https://github.com/livekit/agents/pull/7212
    """
    # Catches: a client delegation never surfacing, or its answer sent without the delegation
    # id, so the model never hears back.
    peer = _Peer()
    async with _voice(peer, Agent(instructions="Answer."), delegation="client") as model:
        delegations: list[GPTLiveDelegation] = []
        model.live.on("delegation_created", delegations.append)
        delegation = {"id": "item_1", "type": "delegation", "target": "client"}
        await peer.socket.send_json(
            {"type": "session.delegation.created", "offset_ms": 0, "delegation": delegation}
        )
        await until(lambda: delegations, "the delegation did not reach the application")
        model.live.append_commentary("62 and raining.", delegation_id=delegations[0].id)
        await until(lambda: "session.commentary.append" in peer.sent(), "no answer sent")

    [answer] = [e for e in peer.connections[0] if e["type"] == "session.commentary.append"]
    assert (answer["delegation_id"], answer["content"]) == ("item_1", "62 and raining.")


async def test_client_delegation_refuses_tools_it_could_never_call() -> None:
    """An agent with tools under client delegation fails the session instead of losing its tools.

    https://github.com/livekit/agents/pull/7212
    """
    # Catches: dropping the tools silently, an agent whose tools never run that reads as a
    # model fault.
    peer = _Peer()
    app = web.Application()
    app.router.add_get("/v1/live/sessions", peer.handle)
    async with TestServer(app) as server, http.open():
        model = _Model(api_key="sk-test", base_url=str(server.make_url("/v1")), delegation="client")
        session = VoiceSession(llm=llm.DuplexRealtimeAdapter(model), vad=None)
        errors: list[ErrorEvent] = []
        session.on("error", errors.append)
        await session.start(_Lookups().agent, room=None)
        await until(lambda: errors, "the tools were accepted")
        await session.aclose()
        await model.aclose()

    [error] = errors
    assert isinstance(error.error, llm.RealtimeModelError)
    assert not error.error.recoverable
    assert "no tool channel" in str(error.error.error) and "lookup" in str(error.error.error)
    assert not peer.starts


async def _lose_the_connection(peer: _Peer, model: _Model) -> None:
    """Close the peer's socket and wait for the session to retry, with nothing reconnecting
    until the test skips the retry interval."""
    retried: list[llm.RealtimeModelError] = []
    model.live.on("error", retried.append)
    await peer.socket.close()
    await until(lambda: retried, "the lost connection was not retried")
    assert retried[0].recoverable


async def test_commands_for_a_lost_session_never_reach_the_next() -> None:
    """A tool result answered while the connection is down starts the next session as history.

    https://github.com/livekit/agents/pull/7230
    """
    # Catches: commands queued for a session that ended sent to the next one, whose backend
    # never made the call: the service refuses the response.item.create.
    peer = _Peer()
    lookups = _Lookups("a")
    options = APIConnectOptions(max_retry=1, retry_interval=5)
    async with _voice(peer, lookups.agent, conn_options=options) as model:
        queued: list[gpt_live_types.ClientEvent] = []
        model.live.on("openai_client_event_queued", queued.append)
        await peer.backend(_created("resp_1"), _call("call_1", "a"), _completed("resp_1"))
        await until(lambda: lookups.calls, "the tool did not run")
        await _lose_the_connection(peer, model)

        lookups.gates["a"].set()
        await until(lambda: any(e["type"] == "response.create" for e in queued), "no result")
        _core.skip_host_time(10)  # past the retry interval
        await until(lambda: len(peer.starts) == 2, "no reconnect")
        await peer.barrier(model)

        assert peer.sent(1) == []
    history = [
        part["text"] for item in peer.starts[1]["session"]["input"] for part in item["content"]
    ]
    assert any(text.endswith(" returned a") for text in history)


async def test_appends_while_disconnected_start_the_next_session() -> None:
    """Instructions and commentary added while the connection is down are in the next start."""
    # Catches: an append made between sessions replayed after the next one started, out of
    # order with the history it was added after, or lost with the session it was queued for.
    peer = _Peer()
    options = APIConnectOptions(max_retry=1, retry_interval=5)
    async with _voice(peer, Agent(instructions="Answer."), conn_options=options) as model:
        await _lose_the_connection(peer, model)
        model.live.append_instructions("Speak French.")
        model.live.append_commentary("The order shipped.")
        _core.skip_host_time(10)  # past the retry interval
        await until(lambda: len(peer.starts) == 2, "no reconnect")
        await peer.barrier(model)

        assert peer.sent(1) == []
    history = [
        part["text"] for item in peer.starts[1]["session"]["input"] for part in item["content"]
    ]
    assert history[-2:] == ["Speak French.", "The order shipped."]


async def test_an_append_after_the_service_ended_the_session_starts_the_next_one() -> None:
    """An append made after the service closed its session is in the next session's start, which
    the next audio opens."""
    # Catches: a command queued once the session was over holding back the audio that opens the
    # next one, so the conversation never resumed, or sent to the next session after its start.
    peer = _Peer()
    async with _voice(peer, Agent(instructions="Answer.")) as model:
        await until(model.acquired, "the session did not start")
        closed = {"reason": "idle_timeout", "usage": {"seconds": 1}}
        await peer.socket.send_json({"type": "session.closed", **closed})
        await until(lambda: 1 in model.used(), "the session's end was not read")
        model.live.append_instructions("Speak French.")
        model.live.push_audio(audio_frame(0.1))
        await until(lambda: len(peer.starts) == 2, "no new session")
        await peer.barrier(model)

        assert peer.sent(1) == []
    history = [
        part["text"] for item in peer.starts[1]["session"]["input"] for part in item["content"]
    ]
    assert history[-1:] == ["Speak French."]


async def test_the_connection_is_acquired_once_the_session_started() -> None:
    """The acquire time covers the handshake and the session's start, the wait before input."""
    # Catches: the acquire time measured at the handshake, before session.start even went, so
    # a slow session start never showed in the metric.
    peer = _Peer(start=False)
    async with _voice(peer, Agent(instructions="Answer.")) as model:
        # the peer holds session.start, sent after the handshake
        assert model.acquired() == []

        await peer.socket.send_json({"type": "session.started", "session": {"id": "live_1"}})
        await until(model.acquired, "the connection was never reported acquired")


async def test_a_session_past_its_duration_is_closed_then_replaced() -> None:
    """At max_session_duration the session is asked to close, its final usage kept, then a new
    one starts from the conversation so far."""
    # Catches: a replacement that drops the socket without session.close, losing the session's
    # final usage, or that leaves the session running past its duration.
    peer = _Peer(closed_seconds=30)
    async with _voice(peer, Agent(instructions="Answer."), max_session_duration=20) as model:
        await until(model.acquired, "the session did not start")
        _core.skip_host_time(25)  # past the duration
        await until(lambda: "session.close" in peer.sent(0), "the session was not closed")
        # the next session opens for the next input
        model.live.push_audio(audio_frame(0.1))
        await until(lambda: len(peer.starts) == 2, "no new session")

        assert 30 in model.used()


async def test_closing_ends_the_session_and_keeps_its_final_usage() -> None:
    """Closing asks the service to close the session and reports the usage it closes with."""
    # Catches: a close that drops the socket at once, losing the usage session.closed carries.
    peer = _Peer(closed_seconds=12)
    async with _voice(peer, Agent(instructions="Answer.")) as model:
        await until(model.acquired, "the session did not start")

    assert peer.sent(0)[-1] == "session.close"
    assert 12 in model.used()


async def test_every_server_frame_reaches_the_application_unknown_ones_included() -> None:
    """Each frame is handed to the application as received; one the session does not read is
    skipped, as is a field it does not know."""
    # Catches: a new event type or field from the service failing a live session, or never
    # reaching the application's raw event handler.
    peer = _Peer()
    async with _voice(peer, Agent(instructions="Answer.")) as model:
        frames: list[str | bytes] = []
        errors: list[llm.RealtimeModelError] = []
        model.live.on("openai_server_event_received", frames.append)
        model.live.on("error", errors.append)
        await peer.socket.send_json({"type": "session.future_event", "payload": 1})
        delta = {"type": "session.output_transcript.delta", "delta": "Hi", "speaker": "agent"}
        await peer.socket.send_json(delta)
        await until(lambda: len(frames) >= 2, "the frames did not reach the application")
        await peer.barrier(model)

    read = [json.loads(frame)["type"] for frame in frames]
    # the close reads the answer to its session.close last
    unknown, known = "session.future_event", "session.output_transcript.delta"
    assert read[-3:] == [unknown, known, "session.closed"]
    assert errors == []
