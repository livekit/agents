from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import llm
from livekit.agents.metrics import LLMMetrics, RealtimeModelMetrics
from livekit.plugins.openai.realtime import gpt_live_model
from livekit.plugins.openai.realtime.gpt_live_model import (
    GPTLiveDelegation,
    GPTLiveModel,
    GPTLiveSession,
)
from livekit.plugins.openai.tools import WebSearch

pytestmark = pytest.mark.unit


class _FakeWS:
    """A websocket that accepts everything and delivers nothing but, by default, the startup ack.

    Every command waits for ``session.started``, so a session that never hears it sends only
    ``session.start``; tests about that hold turn the ack off.
    """

    def __init__(self, *, auto_start: bool) -> None:
        self.sent: list[dict[str, Any]] = []
        self.auto_start = auto_start
        self.session: GPTLiveSession | None = None

    async def send_str(self, data: str) -> None:
        event = json.loads(data)
        self.sent.append(event)
        if self.session is None or not self.auto_start:
            return
        if event["type"] == "session.start":
            self.session._handle_event({"type": "session.started", "session": {"id": "live_test"}})
        elif event["type"] == "session.close":
            self.session._handle_event({"type": "session.closed", "usage": {"seconds": 0}})

    async def receive(self) -> None:
        await asyncio.Event().wait()

    async def close(self) -> None:
        pass


def _connect_hook(monkeypatch: pytest.MonkeyPatch, *, auto_start: bool = True) -> _FakeWS:
    """Replace the handshake before any session exists, so nothing can reach the network."""
    ws = _FakeWS(auto_start=auto_start)

    async def _create_ws_conn(self: GPTLiveSession) -> _FakeWS:
        ws.session = self
        return ws

    monkeypatch.setattr(GPTLiveSession, "_create_ws_conn", _create_ws_conn)
    return ws


def _chat_ctx() -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="a prior turn", id="m1")
    return ctx


@llm.function_tool
async def _get_weather(location: str) -> str:
    """Get the weather."""
    return "rainy"


def _response_event(delegation_id: str, inner: dict[str, Any]) -> dict[str, Any]:
    return {"type": "response.event", "delegation_id": delegation_id, "event": inner}


def _function_call_done(call_id: str, name: str = "_get_weather") -> dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "item": {
            "id": f"fc_{call_id}",
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": name,
            "arguments": '{"location": "Paris"}',
        },
    }


def _transcript(role: str, text: str, start_ms: int) -> dict[str, Any]:
    return {
        "type": f"session.{'input' if role == 'user' else 'output'}_transcript.delta",
        "delta": text,
        "start_ms": start_ms,
        "end_ms": start_ms + 200,
    }


def _completed(response_id: str) -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 376,
                "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 20},
                "output_tokens": 18,
                "output_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 394,
            },
        },
    }


async def test_a_session_awaiting_config_sends_nothing_until_it_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The connection may win the race with the configuration, and must still wait for it: the
    adapter configures every session before use, and a session built directly is configured by
    the same call."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    try:
        await asyncio.sleep(0.05)  # let the connection open well ahead of any configuration
        assert not ws.sent, "session.start went out before the configuration arrived"

        await session._update_session(
            instructions="Be concise.", chat_ctx=_chat_ctx(), tools=[_get_weather]
        )
        await asyncio.sleep(0.05)
        assert [e["type"] for e in ws.sent] == ["session.start"]
        assert ws.sent[0]["session"]["instructions"] == "Be concise."
    finally:
        await session.aclose()
        await model.aclose()


async def test_closing_releases_a_session_still_waiting_for_its_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session closed before it was ever configured has to finish closing.

    Judged on elapsed time: ``aclose`` suppresses ``CancelledError``, so a deadline never fails.
    """
    _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    await asyncio.sleep(0.05)

    started = time.perf_counter()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(session.aclose(), timeout=2)  # bounds a real hang
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"aclose blocked {elapsed:.2f}s on a configuration that never came"
    await model.aclose()


async def test_first_event_is_a_session_start_carrying_the_whole_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(
        api_key="sk-test", voice="marin", responses_options={"parallel_tool_calls": False}
    )
    session = model.session()
    try:
        await session._update_session(
            instructions="Be concise.", chat_ctx=_chat_ctx(), tools=[_get_weather]
        )
        await asyncio.sleep(0.1)  # let the connection open and the send loop drain

        assert ws.sent, "nothing reached the wire"
        first = ws.sent[0]
        assert first["type"] == "session.start"
        config = first["session"]
        assert config["model"] == gpt_live_model.DEFAULT_MODEL
        assert config["instructions"] == "Be concise."
        assert config["audio"] == {
            "format": {"type": "audio/pcm", "rate": 24000},
            "output": {"voice": "marin"},
        }
        responses = config["delegation"]["responses"]
        assert responses["model"] == gpt_live_model.DEFAULT_BACKEND_MODEL
        assert [t["name"] for t in responses["tools"]] == ["_get_weather"]
        assert responses["parallel_tool_calls"] is False
        assert config["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "a prior turn"}],
            }
        ]
    finally:
        await session.aclose()
        await model.aclose()


async def test_instructions_cannot_change_once_the_session_has_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent.update_instructions promises a RealtimeError when the session cannot apply it."""
    _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    try:
        await session._update_session(instructions="Be concise.")
        await asyncio.sleep(0.05)
        await session._update_instructions("Be concise.")  # the same text is not a change
        with pytest.raises(llm.RealtimeError, match="immutable"):
            await session._update_instructions("Be verbose.")
    finally:
        await session.aclose()
        await model.aclose()


async def test_a_custom_voice_is_sent_as_an_object(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test", voice={"id": "voice_123"})
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.05)
        assert ws.sent[0]["session"]["audio"]["output"]["voice"] == {"id": "voice_123"}
    finally:
        await session.aclose()
        await model.aclose()


async def test_a_hosted_tool_is_delegated_to_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OpenAI provider tool goes to the backend as its own entry, next to the function tools."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    try:
        await session._update_session(
            instructions="Be concise.",
            chat_ctx=llm.ChatContext.empty(),
            tools=[_get_weather, WebSearch(search_context_size="low")],
        )
        await asyncio.sleep(0.1)

        tools = ws.sent[0]["session"]["delegation"]["responses"]["tools"]
        assert tools[0]["name"] == "_get_weather"
        assert tools[1] == {"type": "web_search", "search_context_size": "low"}
    finally:
        await session.aclose()
        await model.aclose()


async def test_a_running_session_only_updates_its_backend_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later updates are sparse: the mode, model and startup fields are immutable."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    try:
        await session._update_session(instructions="Be concise.", tools=[])
        await asyncio.sleep(0.05)
        assert "tools" not in ws.sent[0]["session"]["delegation"]["responses"]

        await session._update_tools([_get_weather])
        await asyncio.sleep(0.05)
        update = ws.sent[-1]
        assert update["type"] == "session.update"
        responses = update["session"]["delegation"]["responses"]
        assert [t["name"] for t in responses["tools"]] == ["_get_weather"]
        assert "model" not in responses
        assert update["session"] == {"delegation": {"type": "responses", "responses": responses}}

        session._update_options(tool_choice="required")
        await asyncio.sleep(0.05)
        assert ws.sent[-1]["session"]["delegation"]["responses"] == {"tool_choice": "required"}
    finally:
        await session.aclose()
        await model.aclose()


async def test_client_delegation_reaches_the_application_and_is_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test", delegation="client")
    assert model.capabilities.mutable_tools is False
    session = model.session()
    delegations: list[GPTLiveDelegation] = []
    session.on("delegation_created", delegations.append)
    try:
        await session._update_session(instructions="Be concise.")
        await asyncio.sleep(0.05)
        assert ws.sent[0]["session"]["delegation"] == {"type": "client"}

        # a responses-targeted delegation is the backend's, so it must not reach the app
        session._handle_event(
            {
                "type": "session.delegation.created",
                "offset_ms": 1000,
                "delegation": {
                    "id": "item_1",
                    "type": "delegation",
                    "target": "responses",
                    "response_id": "resp_1",
                },
            }
        )
        assert not delegations

        # the model delegates before the caller's turn closes, so the words that triggered it
        # ride on the event rather than waiting a second for the chat context
        session._handle_event(_transcript("user", "What is the weather", 1000))
        session._handle_event(
            {
                "type": "session.delegation.created",
                "offset_ms": 2000,
                "delegation": {
                    "id": "item_delegation_123",
                    "type": "delegation",
                    "target": "client",
                },
            }
        )
        # a plugin type, not the wire event: the wire shape must not reach the application
        assert delegations == [
            GPTLiveDelegation(id="item_delegation_123", pending_transcript="What is the weather")
        ]

        session.append_commentary("62 and raining.", delegation_id=delegations[0].id)
        session.append_thinking("Still checking the forecast.", delegation_id=delegations[0].id)
        await asyncio.sleep(0.05)
        speak, think = ws.sent[-2], ws.sent[-1]
        assert speak["type"] == "session.commentary.append"
        assert speak["delegation_id"] == "item_delegation_123"
        assert speak["content"] == "62 and raining."
        assert think["type"] == "session.thinking.append"
        assert think["content"] == "Still checking the forecast."
    finally:
        await session.aclose()
        await model.aclose()


async def test_client_delegation_refuses_tools_it_could_never_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping them silently leaves an agent whose tools never run, which reads as a model fault."""
    _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test", delegation="client")
    session = model.session()
    try:
        with pytest.raises(llm.RealtimeError, match="no tool channel"):
            await session._update_session(instructions="Be concise.", tools=[_get_weather])

        # an agent with no tools is the supported shape, and the activity passes an empty list
        await session._update_session(instructions="Be concise.", tools=[])

        # responses delegation is the way to have tools at all, so it still takes them
        responses_model = GPTLiveModel(api_key="sk-test")
        responses_session = responses_model.session()
        try:
            await responses_session._update_session(tools=[_get_weather])
        finally:
            await responses_session.aclose()
            await responses_model.aclose()
    finally:
        await session.aclose()
        await model.aclose()


async def test_general_context_carries_an_explicit_null_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service requires ``delegation_id`` on every append, null included."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.05)
        session.append_thinking("The caller is a premium customer.")
        session.append_instructions("Speak slowly.")
        await asyncio.sleep(0.05)
        for event in ws.sent[-2:]:
            assert "delegation_id" in event and event["delegation_id"] is None
        assert ws.sent[-1]["type"] == "session.instructions.append"
    finally:
        await session.aclose()
        await model.aclose()


async def test_a_reply_is_asked_for_as_commentary_to_speak_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.05)
        session._generate_reply(instructions="Greet the caller.")
        await asyncio.sleep(0.05)
        ask = ws.sent[-1]
        assert ask["type"] == "session.commentary.append"
        assert ask["delegation_id"] is None
        assert ask["content"] == f"{gpt_live_model._ASK_INSTRUCTED}\n\nGreet the caller."
    finally:
        await session.aclose()
        await model.aclose()


async def test_only_session_started_releases_the_queued_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A greeting queued at startup must not race session.start; only the acknowledgment lets it out."""
    ws = _connect_hook(monkeypatch, auto_start=False)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.05)
        session._generate_reply(instructions="Greet the caller.")
        session._handle_event(
            {"type": "session.updated", "event_id": "e1", "session": {"id": "s1"}}
        )
        await asyncio.sleep(0.05)
        assert [e["type"] for e in ws.sent] == ["session.start"]  # an update receipt is no startup
        assert session.session_id is None

        session._handle_event({"type": "session.started", "session": {"id": "live_1"}})
        await asyncio.sleep(0.05)
        assert [e["type"] for e in ws.sent] == ["session.start", "session.commentary.append"]
        assert session.session_id == "live_1"
    finally:
        await session.aclose()
        await model.aclose()


async def test_output_audio_is_forwarded_as_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every delta reaches the stream, silence included; the framework decides what plays."""
    _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.05)
        import base64

        hundred_ms = base64.b64encode(b"\x00" * 4800).decode()
        for _ in range(3):
            session._handle_event({"type": "session.output_audio.delta", "delta": hundred_ms})
        session._audio_ch.close()
        frames = [f async for f in session.audio_stream]
        assert [f.frame.duration for f in frames] == pytest.approx([0.1, 0.1, 0.1])
        assert all(f.frame.sample_rate == 24000 and f.start_ms is None for f in frames)
    finally:
        await session.aclose()
        await model.aclose()


def _pcm(level: float, *, duration_ms: int = 100) -> rtc.AudioFrame:
    """A frame whose RMS is exactly ``level`` (0..1), from an alternating square wave."""
    num_samples = gpt_live_model.SAMPLE_RATE * duration_ms // 1000
    samples = np.empty(num_samples, dtype=np.int16)
    samples[0::2] = int(level * 32767)
    samples[1::2] = -int(level * 32767)
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=gpt_live_model.SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=num_samples,
    )


def test_the_gate_holds_through_the_tick_each_connection_opens_with() -> None:
    """Measured against the alpha: silence is all zeros bar a 0.4 s tick reaching 0.0006 RMS."""
    gate = GPTLiveModel(api_key="sk-test").audio_gate()

    assert not gate.update(_pcm(0.000578))
    assert not any(gate.update(_pcm(level)) for level in (0.000093, 0.000027, 0.000013))
    assert not gate.update(_pcm(0.0))

    # the quietest speech frame the alpha produced was 0.005
    assert gate.update(_pcm(0.005))


def test_the_gate_rides_out_a_pause_between_sentences() -> None:
    """A sentence pause is ~0.5 s of true silence; closing on one would split the utterance."""
    gate = GPTLiveModel(api_key="sk-test").audio_gate()
    assert gate.update(_pcm(0.05))

    assert all(gate.update(_pcm(0.0)) for _ in range(5))
    assert gate.update(_pcm(0.05))

    # the turn itself still ends
    for _ in range(9):
        gate.update(_pcm(0.0))
    assert not gate.update(_pcm(0.0))


async def test_a_backend_function_call_is_answered_and_the_response_continued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A result is queued at once, but response.create waits for the response to finish asking."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    calls: list[llm.FunctionCall] = []
    session.on("function_call", calls.append)
    try:
        await session._update_session(instructions="Be concise.", tools=[_get_weather])
        await asyncio.sleep(0.05)

        session._handle_event(
            _response_event("item_d1", {"type": "response.created", "response": {"id": "resp_1"}})
        )
        session._handle_event(_response_event("item_d1", _function_call_done("call_1")))
        assert [c.call_id for c in calls] == ["call_1"]

        # the framework runs the tool and the adapter hands over the new output
        await session._append_items(
            [
                llm.FunctionCallOutput(
                    call_id="call_1", name="_get_weather", output="rainy", is_error=False
                )
            ]
        )
        await asyncio.sleep(0.05)
        assert ws.sent[-1]["type"] == "response.item.create"
        assert ws.sent[-1]["item"] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "rainy",
        }

        session._handle_event(_response_event("item_d1", _completed("resp_1")))
        await asyncio.sleep(0.05)
        assert ws.sent[-1]["type"] == "response.create"
        assert not session._delegated_responses and not session._fnc_call_to_delegation
    finally:
        await session.aclose()
        await model.aclose()


async def test_a_response_continues_only_once_every_call_has_its_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial batch is rejected by the service, so the continuation waits for the last one."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.05)
        session._handle_event(
            _response_event("item_d1", {"type": "response.created", "response": {"id": "resp_1"}})
        )
        session._handle_event(_response_event("item_d1", _function_call_done("call_a")))
        session._handle_event(_response_event("item_d1", _function_call_done("call_b")))
        session._handle_event(_response_event("item_d1", _completed("resp_1")))

        await session._append_items(
            [llm.FunctionCallOutput(call_id="call_a", output="one", is_error=False)]
        )
        await asyncio.sleep(0.05)
        assert [e["type"] for e in ws.sent[1:]] == ["response.item.create"]

        await session._append_items(
            [llm.FunctionCallOutput(call_id="call_b", output="two", is_error=False)]
        )
        await asyncio.sleep(0.05)
        assert [e["type"] for e in ws.sent[1:]] == [
            "response.item.create",
            "response.item.create",
            "response.create",
        ]
    finally:
        await session.aclose()
        await model.aclose()


async def test_a_delegated_model_is_billed_under_its_own_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frontend is billed by duration; each backend response names the model that spent it."""
    _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    collected: list[Any] = []
    session.on("metrics_collected", collected.append)
    try:
        await session._update_session()
        await asyncio.sleep(0.05)
        session._handle_event(
            {
                "type": "session.usage.updated",
                "usage": {"seconds": 14.0},
                "context_window": {"usage_ratio": 0.2},
            }
        )
        session._handle_event(_response_event("item_d1", _completed("resp_1")))
        session._handle_event(
            {"type": "session.closed", "reason": "client_request", "usage": {"seconds": 27.0}}
        )

        backend = [m for m in collected if isinstance(m, LLMMetrics)]
        assert [m.metadata.model_name for m in backend] == ["gpt-5.6-sol"]
        assert backend[0].request_id == "resp_1"
        assert backend[0].prompt_tokens == 376
        assert backend[0].prompt_cached_tokens == 100
        assert backend[0].cache_creation_tokens == 20
        assert backend[0].completion_tokens == 18
        assert backend[0].reasoning_tokens == 5

        # the session's own rows carry the duration deltas and none of the delegated tokens
        frontend = [m for m in collected if isinstance(m, RealtimeModelMetrics)]
        assert [round(m.session_duration, 1) for m in frontend] == [14.0, 13.0]
        assert all(m.input_tokens == 0 and m.output_tokens == 0 for m in frontend)
        assert session._session_closed_fut.done()
    finally:
        await session.aclose()
        await model.aclose()


def _user_events(session: GPTLiveSession) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []
    for name in (
        "input_speech_started",
        "input_audio_transcription_completed",
        "input_speech_stopped",
    ):
        session.on(name, lambda ev, name=name: events.append((name, ev)))
    return events


def _silence(duration_ms: int) -> rtc.AudioFrame:
    samples = 24000 * duration_ms // 1000
    return rtc.AudioFrame(
        data=b"\x00" * samples * 2, sample_rate=24000, num_channels=1, samples_per_channel=samples
    )


async def test_fragments_are_forwarded_and_mirrored_as_growing_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model's fragments go to the adapter as they are; the history a reconnect reseeds from
    still needs messages, so a speaker's fragments extend one until a pause on the model's clock."""
    _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    deltas: list[llm.DuplexOutputTranscriptDelta] = []
    session.on("transcript_delta", deltas.append)
    try:
        await session._update_session()
        await asyncio.sleep(0.05)
        for role, text, start in (
            ("user", " What is", 7000),
            ("assistant", "Let me", 7400),
            ("user", " the weather", 7200),
            ("assistant", " check.", 7600),
            ("assistant", "Sixty-two.", 9000),
        ):
            session._handle_event(_transcript(role, text, start))

        assert [(d.text, d.start_ms, d.end_ms) for d in deltas] == [
            ("Let me", 7400, 7600),
            (" check.", 7600, 7800),
            ("Sixty-two.", 9000, 9200),
        ]
        history = session._session_start_event().session.input or []
        assert [(m.role, m.content[0].text) for m in history] == [
            ("user", " What is the weather"),
            ("assistant", "Let me check."),
            ("assistant", "Sixty-two."),
        ]
    finally:
        await session.aclose()
        await model.aclose()


async def test_the_callers_turn_ends_on_their_own_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """There are no turn events: the caller's fragments accumulate, and a second of their audio
    pushed with no new fragment ends the turn."""
    _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    events = _user_events(session)
    try:
        await session._update_session()
        await asyncio.sleep(0.05)
        for _ in range(20):
            session.push_audio(_silence(100))  # audio already pushed must not count as quiet
        session._handle_event(_transcript("user", " What", 1800))
        session._handle_event(_transcript("user", " is the", 2000))

        assert [name for name, _ in events] == [
            "input_speech_started",
            "input_audio_transcription_completed",
            "input_audio_transcription_completed",
        ]
        interim = events[-1][1]
        assert isinstance(interim, llm.InputTranscriptionCompleted)
        assert (interim.transcript, interim.is_final) == (" What is the", False)

        for _ in range(9):
            session.push_audio(_silence(100))
        assert len(events) == 3  # quiet, but not yet for the gap

        session.push_audio(_silence(100))
        assert [name for name, _ in events[-2:]] == [
            "input_audio_transcription_completed",
            "input_speech_stopped",
        ]
        final = events[-2][1]
        assert isinstance(final, llm.InputTranscriptionCompleted)
        assert (final.transcript, final.is_final) == (" What is the", True)
        assert final.item_id == interim.item_id
        assert final.turn_started_at == interim.turn_started_at
    finally:
        await session.aclose()
        await model.aclose()


async def test_a_pause_on_the_models_clock_splits_fragments_that_arrive_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bursty delivery must not merge two utterances into one turn."""
    _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test")
    session = model.session()
    events = _user_events(session)
    try:
        await session._update_session()
        await asyncio.sleep(0.05)
        session._handle_event(_transcript("user", "Hello", 1000))
        session._handle_event(_transcript("user", "Again", 4000))

        finals = [
            ev.transcript
            for name, ev in events
            if name == "input_audio_transcription_completed" and ev.is_final
        ]
        assert finals == ["Hello"]
        assert [name for name, _ in events].count("input_speech_started") == 2
        history = session._session_start_event().session.input or []
        assert [m.content[0].text for m in history] == ["Hello", "Again"]
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.parametrize("delegation", ["responses", "client"])
async def test_a_typed_message_rides_in_the_ask_while_it_is_the_newest_thing_said(
    monkeypatch: pytest.MonkeyPatch, delegation: str
) -> None:
    """A typed message is context like any other item, and the ask that follows carries it: read
    from the history rather than tracked, and only once."""
    ws = _connect_hook(monkeypatch)

    model = GPTLiveModel(api_key="sk-test", delegation=delegation)
    session = model.session()
    try:
        await session._update_session(instructions="Be concise.", tools=[])
        await asyncio.sleep(0.05)
        sent_before = len(ws.sent)

        await session._append_items(
            [
                llm.ChatMessage(role="system", content=["Answer in French."], id="rule_1"),
                llm.ChatMessage(
                    role="user", content=["What is the weather in Paris?"], id="typed_1"
                ),
            ]
        )
        session._generate_reply()
        await asyncio.sleep(0.05)

        # a system message is a standing rule, everything else is context
        new = ws.sent[sent_before:]
        assert [e["type"] for e in new] == [
            "session.instructions.append",
            "session.thinking.append",
            "session.commentary.append",
        ]
        assert new[0]["content"] == "Answer in French."
        assert new[1]["content"] == "user: What is the weather in Paris?"
        assert new[2]["content"] == f"{gpt_live_model._ASK_TYPED}\n\nWhat is the weather in Paris?"

        session._generate_reply()  # the same message is not asked about twice
        await asyncio.sleep(0.05)
        assert ws.sent[-1]["content"] == gpt_live_model._ASK_BARE

        await session._append_items(
            [llm.ChatMessage(role="user", content=["Never mind."], id="typed_2")]
        )
        session._handle_event(_transcript("assistant", "Okay.", 1))
        session._generate_reply()  # speech has moved the conversation on since the typed message
        await asyncio.sleep(0.05)
        assert ws.sent[-1]["content"] == gpt_live_model._ASK_BARE
    finally:
        await session.aclose()
        await model.aclose()
