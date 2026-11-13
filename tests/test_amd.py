from __future__ import annotations

import asyncio
import json
import runpy
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import aclosing, asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock

import httpx2 as httpx
import openai
import pytest
from opentelemetry import trace
from pydantic import ValidationError

from livekit import agents, rtc
from livekit.agents import (
    AMD,
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    AMDCompletedEvent,
    AMDPredictionEvent,
    APIConnectionError,
    LanguageCode,
    StopResponse,
    UserStateChangedEvent,
    inference,
    llm,
    stt,
    utils,
    vad,
)
from livekit.agents.inference._utils import HEADER_CLIENT_FEATURE
from livekit.agents.llm.tool_context import get_raw_function_info
from livekit.agents.types import APIConnectOptions, NotGivenOr
from livekit.agents.voice._turn_hooks import TurnHooks
from livekit.agents.voice.amd import (
    AMDCategory,
    AMDLifecycle,
    AMDReason,
    _fsm,
    _inference,
    detector as detector_module,
)
from livekit.agents.voice.amd._chat_context import AGENT_REPLY, AMDRequest, AMDTranscript
from livekit.agents.voice.amd._stt import AMDTranscriber
from livekit.agents.voice.amd.detector import (
    _DEFAULT_HUMAN_INSTRUCTIONS,
    _DEFAULT_IVR_INSTRUCTIONS,
    _DEFAULT_SCREENING_INSTRUCTIONS,
    _DEFAULT_VOICEMAIL_INSTRUCTIONS,
    _LATE_AMD_FINAL_WAIT,
)
from livekit.agents.voice.audio_recognition import (
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
    _PreemptiveGenerationInfo,
)
from livekit.agents.voice.events import (
    FunctionToolsExecutedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
)
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, _audio_frame, fake_capabilities
from .fake_stt import DrainingStream, DrainingSTT, FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.fixture(autouse=True)
def detector_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detector_module,
        "time",
        SimpleNamespace(monotonic=lambda: asyncio.get_running_loop().time(), time=time.time),
    )


def next_deadline(detector: AMD) -> float | None:
    """Earliest armed detector deadline on the (patched) monotonic clock."""
    return detector._deadline_at(asyncio.get_running_loop().time())


# region: Stage transitions (_fsm.py)


def test_transition_is_repeatable_and_does_not_mutate_input() -> None:
    state = AMDCategory.UNCERTAIN
    event = AMDCategory.MACHINE_VM
    first = _fsm.transition(state, event)
    assert _fsm.transition(state, event) == first
    assert state is AMDCategory.UNCERTAIN
    assert first.next_state is AMDCategory.MACHINE_VM
    assert first.effects == ()


STAGES = [c for c in AMDCategory if c is not AMDCategory.WAIT]


@pytest.mark.parametrize("current", STAGES)
@pytest.mark.parametrize("category", list(AMDCategory))
def test_stage_transitions(current: AMDCategory, category: AMDCategory) -> None:
    recommended = {
        AMDCategory.UNCERTAIN: set(AMDCategory),
        AMDCategory.MACHINE_SCREENING: {
            AMDCategory.MACHINE_SCREENING,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_UNAVAILABLE,
            AMDCategory.UNCERTAIN,
            AMDCategory.WAIT,
        },
        AMDCategory.MACHINE_VM: {
            AMDCategory.MACHINE_VM,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_IVR,
            AMDCategory.MACHINE_UNAVAILABLE,
            AMDCategory.UNCERTAIN,
            AMDCategory.WAIT,
        },
        AMDCategory.MACHINE_IVR: {
            AMDCategory.MACHINE_IVR,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_UNAVAILABLE,
            AMDCategory.UNCERTAIN,
            AMDCategory.WAIT,
        },
        AMDCategory.HUMAN: set(),
        AMDCategory.MACHINE_UNAVAILABLE: set(),
    }
    state = current
    event = category
    if not recommended[current]:
        with pytest.raises(ValueError, match="invalid AMD transition"):
            _fsm.transition(state, event)
        return
    result = _fsm.transition(state, event)
    if category in {AMDCategory.UNCERTAIN, AMDCategory.WAIT}:
        assert result.next_state is current
    else:
        assert result.next_state is category
    if category in {AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE}:
        assert result.effects == (_fsm.Effect.COMPLETE,)
    elif category is AMDCategory.MACHINE_IVR:
        assert result.effects == (_fsm.Effect.EXTRACT_MENU,)
    else:
        assert result.effects == ()


@pytest.mark.parametrize(
    ("initial", "corrected"),
    [
        (AMDCategory.MACHINE_SCREENING, AMDCategory.MACHINE_IVR),
        (AMDCategory.MACHINE_VM, AMDCategory.MACHINE_SCREENING),
        (AMDCategory.MACHINE_IVR, AMDCategory.MACHINE_SCREENING),
    ],
)
@pytest.mark.parametrize("bridge", [AMDCategory.UNCERTAIN, AMDCategory.WAIT])
def test_wait_and_uncertain_keep_the_stage(
    initial: AMDCategory, corrected: AMDCategory, bridge: AMDCategory
) -> None:
    intermediate = _fsm.transition(initial, bridge)
    assert intermediate.next_state is initial
    assert intermediate.effects == ()
    assert _fsm.transition(intermediate.next_state, corrected).next_state is corrected


def test_same_ivr_state_extracts_each_menu() -> None:
    state = AMDCategory.MACHINE_IVR
    result = _fsm.transition(state, AMDCategory.MACHINE_IVR)
    assert result.next_state == state
    assert result.effects == (_fsm.Effect.EXTRACT_MENU,)


@pytest.mark.parametrize(
    ("initial", "corrected"),
    [
        (AMDCategory.MACHINE_SCREENING, AMDCategory.MACHINE_IVR),
        (AMDCategory.MACHINE_VM, AMDCategory.MACHINE_SCREENING),
        (AMDCategory.MACHINE_IVR, AMDCategory.MACHINE_SCREENING),
    ],
)
def test_leaving_the_recommendations_corrects_the_stage(
    initial: AMDCategory, corrected: AMDCategory
) -> None:
    assert corrected not in _fsm.RECOMMENDED[initial]
    result = _fsm.transition(initial, corrected)
    assert result.next_state == corrected
    assert result.effects == (
        (_fsm.Effect.EXTRACT_MENU,) if corrected == AMDCategory.MACHINE_IVR else ()
    )


@pytest.mark.parametrize("stage", [AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE])
def test_terminal_stages_cannot_reopen_detection(stage: AMDCategory) -> None:
    with pytest.raises(ValueError, match="invalid AMD transition"):
        _fsm.transition(stage, AMDCategory.MACHINE_SCREENING)


# endregion


# region: Structured model calls and output validation (_inference.py)

CHAT_CTX = llm.ChatContext()
CHAT_CTX.add_message(role="user", content="input")
REQUEST = AMDRequest(
    stage=AMDCategory.UNCERTAIN,
    recommended_next_categories=sorted(AMDCategory),
    previous_prediction=None,
    chat_ctx=CHAT_CTX,
    speech_duration=0.5,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        "not JSON",
        "[]",
        '{"category":"unknown"}',
        '```json\n{"category":"human"}\n```',
    ],
)
async def test_classifier_rejects_invalid_tool_arguments(arguments: str) -> None:
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="input",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name="record_result", arguments=arguments, call_id="result"
                    )
                ],
            )
        ]
    )
    with pytest.raises(ValidationError):
        await _inference.classify(model, REQUEST)


@pytest.mark.asyncio
@pytest.mark.parametrize("menu", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
async def test_amd_uses_a_required_structured_tool(
    monkeypatch: pytest.MonkeyPatch, menu: bool, fallback: bool
) -> None:
    payload = (
        {"menu": "Main menu", "options": [{"label": "Talk to a person", "dtmf": "1"}]}
        if menu
        else {"category": "human"}
    )
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input=json.dumps({"transcript": "input"}) if menu else "input",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name="record_result", arguments=json.dumps(payload), call_id="result"
                    )
                ],
            )
        ]
    )
    chat = Mock(wraps=model.chat)
    monkeypatch.setattr(model, "chat", chat)
    adapter = llm.FallbackAdapter([model]) if fallback else model
    try:
        if menu:
            result = await _inference.extract_ivr_menu(adapter, "input")
            schema = _inference.AMDIVRMenuResponse
        else:
            result = await _inference.classify(adapter, REQUEST)
            schema = _inference.AMDResponse
        assert result == schema.model_validate(payload)
        chat.assert_called_once()
        assert chat.call_args.kwargs["tool_choice"] == "required"
        assert chat.call_args.kwargs["parallel_tool_calls"] is False
        tools = chat.call_args.kwargs["tools"]
        assert len(tools) == 1
        expected_schema = schema.model_json_schema()
        assert get_raw_function_info(tools[0]).raw_schema["parameters"] == expected_schema
        tool_ctx = llm.ToolContext(tools)
        assert tool_ctx.parse_function_tools("openai")
        assert tool_ctx.parse_function_tools("google")
        assert tool_ctx.parse_function_tools("anthropic")
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [[], ["unexpected"], ["record_result", "record_result"]])
async def test_classifier_requires_exactly_one_result_tool(names: list[str]) -> None:
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="input",
                content='{"category":"human"}',
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name=name, arguments='{"category":"human"}', call_id=str(i)
                    )
                    for i, name in enumerate(names)
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="exactly one record_result"):
        await _inference.classify(model, REQUEST)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [
        AMDCategory.UNCERTAIN,
        AMDCategory.MACHINE_SCREENING,
        AMDCategory.MACHINE_VM,
        AMDCategory.MACHINE_IVR,
    ],
)
@pytest.mark.parametrize("category", list(AMDCategory))
async def test_classifier_recommends_next_states_without_restricting_predictions(
    stage: AMDCategory,
    category: AMDCategory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice.amd import _fsm

    request = replace(
        REQUEST, stage=stage, recommended_next_categories=sorted(_fsm.RECOMMENDED[stage])
    )
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="input",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name="record_result",
                        arguments=json.dumps({"category": category}),
                        call_id="result",
                    )
                ],
            )
        ]
    )
    chat = Mock(wraps=model.chat)
    monkeypatch.setattr(model, "chat", chat)
    try:
        assert (await _inference.classify(model, request)).category == category
        tools = chat.call_args.kwargs["tools"]
        schema = get_raw_function_info(tools[0]).raw_schema["parameters"]
        assert schema["$defs"]["AMDCategory"]["enum"] == [c.value for c in AMDCategory]
        context = chat.call_args.kwargs["chat_ctx"]
        constraints = json.loads(context.items[1].text_content)
        assert constraints["recommended_next_categories"] == request.recommended_next_categories
        tool_ctx = llm.ToolContext(tools)
        for provider in ("openai", "google", "anthropic"):
            assert tool_ctx.parse_function_tools(provider)
    finally:
        await model.aclose()


# Long enough to keep PyJWT's short-key warning out of the suite output.
_SECRET = "f" * 32


def _sse(choices: list[dict[str, Any]]) -> bytes:
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "google/gemma-4-31b-it",
        "choices": choices,
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


_RECORD_RESULT = b"".join(
    [
        _sse(
            [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "record_result",
                                    "arguments": json.dumps({"category": "human"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        ),
        _sse([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
        b"data: [DONE]\n\n",
    ]
)


def _inference_llm() -> tuple[inference.LLM, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_RECORD_RESULT
        )

    model = inference.LLM(model="google/gemma-4-31b-it", api_key=_SECRET, api_secret=_SECRET)
    model._client = openai.AsyncClient(
        api_key=_SECRET,
        base_url="http://inference.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return model, requests


@pytest.mark.asyncio
async def test_classifier_tags_only_its_own_request() -> None:
    model, requests = _inference_llm()
    model._opts.extra_kwargs["extra_headers"] = {"X-Custom": "kept"}

    assert (await _inference.classify(model, REQUEST)).category == AMDCategory.HUMAN
    async with model.chat(chat_ctx=CHAT_CTX) as stream:
        async for _ in stream:
            pass

    amd_request, agent_request = requests
    assert amd_request.headers[HEADER_CLIENT_FEATURE] == "AMD"
    assert amd_request.headers["X-Custom"] == "kept"
    # the agent's next turn on the same LLM is ordinary traffic
    assert HEADER_CLIENT_FEATURE not in agent_request.headers
    assert agent_request.headers["X-Custom"] == "kept"
    # the stream stamps SDK headers on a copy, never on the model's own dict
    assert model._opts.extra_kwargs["extra_headers"] == {"X-Custom": "kept"}


@pytest.mark.asyncio
async def test_plugin_llm_gets_no_header() -> None:
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="input",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name="record_result", arguments='{"category":"human"}', call_id="r"
                    )
                ],
            )
        ]
    )
    chat = Mock(wraps=model.chat)
    model.chat = chat  # type: ignore[method-assign]

    await _inference.classify(model, REQUEST)

    assert chat.call_args.kwargs["extra_kwargs"] is NOT_GIVEN


def test_fallback_adapter_is_tagged_only_when_every_llm_is_inference() -> None:
    gateway = inference.LLM(model="google/gemma-4-31b-it", api_key=_SECRET, api_secret=_SECRET)
    plugin = FakeLLM()

    assert _inference._reaches_only_inference(gateway)
    assert _inference._reaches_only_inference(llm.FallbackAdapter([gateway, gateway]))
    assert not _inference._reaches_only_inference(llm.FallbackAdapter([gateway, plugin]))
    assert not _inference._reaches_only_inference(plugin)


# endregion


# region: Transcript attribution and failure handling in AMDTranscriber (_stt.py)


@pytest.fixture
async def transcriber() -> AsyncIterator[tuple[AMDTranscriber, DrainingStream]]:
    model = DrainingSTT()
    async with aclosing(AMDTranscriber(model, APIConnectOptions(max_retry=0))) as transcriber:
        transcriber.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        yield transcriber, model.streams[0]
    await model.aclose()


async def wait_for_transcript(predicate: Callable[[], bool]) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 1)


async def amd_heard(transcriber: AMDTranscriber, stream: DrainingStream, text: str) -> None:
    before = transcriber._current.get("amd")
    stream.send_fake_transcript(text)
    await wait_for_transcript(lambda: transcriber._current.get("amd") != before)


@pytest.mark.asyncio
@pytest.mark.parametrize("session_first", [True, False])
async def test_each_turn_keeps_both_transcripts_whichever_arrives_first(
    transcriber: tuple[AMDTranscriber, DrainingStream], session_first: bool
) -> None:
    amd, stream = transcriber
    if session_first:
        amd.push_transcript("session transcript")
    await amd_heard(amd, stream, "amd transcript")
    first = amd.end_turn("session transcript")
    assert first == AMDTranscript(session="session transcript", amd="amd transcript")

    await amd_heard(amd, stream, "amd next turn")
    second = amd.end_turn("session next turn")
    assert second == AMDTranscript(session="session next turn", amd="amd next turn")
    assert amd.amd_stt_active


@pytest.mark.asyncio
async def test_a_late_amd_transcript_leaves_the_session_transcript(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    await amd_heard(amd, stream, "Hi, you've reached Sam.")
    first = amd.end_turn("Hi, you've reached Sam.")
    second = amd.end_turn("Leave your name and number.")
    assert first.amd == "Hi, you've reached Sam."
    assert second == AMDTranscript(session="Leave your name and number.", amd="")
    assert second.transcript == "Leave your name and number."

    # the late words belong to the next turn
    await amd_heard(amd, stream, "Leave your name and number.")
    assert amd.end_turn("").amd == "Leave your name and number."


@pytest.mark.asyncio
async def test_a_final_within_the_wait_belongs_to_the_turn_just_ended(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    ended = amd.end_turn("Sam Lee speaking.", late_wait=0.3)
    assert ended == AMDTranscript(session="Sam Lee speaking.", amd="")
    late = amd.late_final
    assert late is not None
    waiting = asyncio.create_task(late.wait())
    await asyncio.sleep(0.1)
    stream.send_fake_transcript("Sam Lee speaking.")
    assert await waiting == "Sam Lee speaking."
    # the next turn starts clean
    await amd_heard(amd, stream, "Yes, that works.")
    assert amd.end_turn("Yes, that works.").amd == "Yes, that works."


@pytest.mark.asyncio
async def test_a_final_after_the_wait_belongs_to_the_next_turn(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    amd.end_turn("Sam Lee speaking.", late_wait=0.3)
    late = amd.late_final
    assert late is not None
    assert await late.wait() is None
    await amd_heard(amd, stream, "Sam Lee speaking.")
    assert amd.end_turn("Yes.").amd == "Sam Lee speaking."


@pytest.mark.asyncio
async def test_a_final_before_the_wait_starts_still_belongs_to_the_turn(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    amd.end_turn("Sam Lee speaking.", late_wait=0.3)
    late = amd.late_final
    assert late is not None
    stream.send_fake_transcript("Sam Lee speaking.")
    await wait_for_transcript(lambda: amd.late_final is None)
    assert await late.wait() == "Sam Lee speaking."
    assert amd.end_turn("").amd == ""


@pytest.mark.asyncio
async def test_without_session_stt_a_final_after_the_wait_belongs_to_the_next_turn() -> None:
    model = DrainingSTT()
    async with aclosing(
        AMDTranscriber(model, APIConnectOptions(max_retry=0), session_stt=False)
    ) as amd:
        amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        amd.end_turn("Sam Lee speaking.", late_wait=0.3)
        late = amd.late_final
        assert late is not None
        assert await late.wait() is None
        await amd_heard(amd, model.streams[0], "Sam Lee speaking.")
        assert amd.end_turn("").amd == "Sam Lee speaking."
    await model.aclose()


@pytest.mark.asyncio
async def test_a_turn_with_its_amd_final_needs_no_wait(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    await amd_heard(amd, stream, "Sam Lee speaking.")
    amd.end_turn("Sam Lee speaking.", late_wait=0.3)
    assert amd.late_final is None


def test_the_classifier_sees_both_transcripts_only_with_an_amd_stt() -> None:
    assert AMDTranscript(session="Hi.").for_classifier() == "Hi."
    assert AMDTranscript(session=None, amd="Hi.").for_classifier() == "Hi."
    assert (
        AMDTranscript(session="Hi.", amd="Hi.").for_classifier()
        == '{"transcript_1": "Hi.", "transcript_2": "Hi."}'
    )
    assert (
        AMDTranscript(session="Hi.", amd="").for_classifier()
        == '{"transcript_1": "Hi.", "transcript_2": ""}'
    )
    assert AMDTranscript(session="", amd="Hi.").transcript == "Hi."


async def test_without_session_stt_amd_uses_its_own_transcript_until_it_fails() -> None:
    model = DrainingSTT()
    async with aclosing(
        AMDTranscriber(model, APIConnectOptions(max_retry=0), session_stt=False)
    ) as amd:
        amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        amd.push_transcript("Session transcript.")
        assert amd.end_turn("Session transcript.") == AMDTranscript(session=None, amd="")

        await amd_heard(amd, model.streams[0], "AMD transcript.")
        assert amd.end_turn("Session transcript.") == AMDTranscript(
            session=None, amd="AMD transcript."
        )

        amd.push_transcript("Fallback transcript.")
        amd.fail(RuntimeError("connection lost"))
        assert amd.end_turn("") == AMDTranscript(session="Fallback transcript.", amd=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("started", ["no_audio", "reader_pending", "reader_running", "amd_heard"])
async def test_close_stops_stream_and_prevents_more_audio(started: str) -> None:
    model = DrainingSTT()
    model.aclose = AsyncMock(wraps=model.aclose)
    frame = rtc.AudioFrame.create(16000, 1, 320)
    async with aclosing(AMDTranscriber(model, APIConnectOptions(max_retry=0))) as amd:
        if started != "no_audio":
            amd.push_audio(frame)
            stream = model.streams[0]
            stream.aclose = AsyncMock(wraps=stream.aclose)
            stream.push_frame = Mock(wraps=stream.push_frame)
            if started in {"reader_running", "amd_heard"}:
                await wait_for_transcript(lambda: bool(stream.frames))
            if started == "amd_heard":
                amd.push_transcript("Hello from AMD.", source="amd")

        await amd.aclose()
        amd.push_audio(frame)
        await amd.aclose()
        amd.push_audio(frame)

        if started != "no_audio":
            stream.aclose.assert_awaited_once()
            stream.push_frame.assert_not_called()
            assert stream._task.done()
            assert stream._metrics_task.done()
        assert len(model.streams) == (0 if started == "no_audio" else 1)
    model.aclose.assert_not_awaited()
    await model.aclose()


@pytest.mark.asyncio
async def test_turns_commit_immediately_without_flushing_stt(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    stream.flush = Mock(wraps=stream.flush)
    assert amd.end_turn("") == AMDTranscript(session="", amd="")

    await amd_heard(amd, stream, "Hello.")
    assert amd.end_turn("Hello from session.") == AMDTranscript(
        session="Hello from session.", amd="Hello."
    )
    stream.flush.assert_not_called()
    assert not stream.input_ended.is_set()


@pytest.mark.asyncio
async def test_late_final_after_empty_turns_belongs_to_the_next_turn(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    earlier = [amd.end_turn("") for _ in range(3)]
    await amd_heard(amd, stream, "Hello.")
    assert amd.end_turn("").amd == "Hello."
    assert all(turn.transcript == "" for turn in earlier)


@pytest.mark.asyncio
async def test_final_segments_accumulate_and_ignore_interim_transcript(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    stream.send_fake_transcript("partial", is_final=False)
    stream.send_fake_transcript("")
    stream.send_fake_transcript("Please leave")
    stream.send_fake_transcript("a message.")
    await wait_for_transcript(lambda: amd._current.get("amd") == "Please leave a message.")
    assert amd.end_turn("Leave a message.") == AMDTranscript(
        session="Leave a message.", amd="Please leave a message."
    )


@pytest.mark.asyncio
async def test_amd_failure_leaves_the_session_transcript_for_the_open_turn_on(
    transcriber: tuple[AMDTranscriber, DrainingStream],
) -> None:
    amd, stream = transcriber
    await amd_heard(amd, stream, "Hello.")
    first = amd.end_turn("Hello from session.")
    amd.push_transcript("Session turn two.")

    stream.error = RuntimeError("connection lost")
    amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    await wait_for_transcript(lambda: not amd.amd_stt_active)
    second = amd.end_turn("")
    assert first == AMDTranscript(session="Hello from session.", amd="Hello.")
    assert second == AMDTranscript(session="Session turn two.", amd=None)
    assert amd.end_turn("Session turn three.").amd is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["push", "read"])
async def test_failure_drops_the_amd_transcript_even_when_the_session_is_empty(
    transcriber: tuple[AMDTranscriber, DrainingStream], failure: str
) -> None:
    amd, stream = transcriber
    await amd_heard(amd, stream, "Please leave a message.")
    if failure == "push":
        stream.push_frame = Mock(side_effect=RuntimeError("failed to push"))
    else:
        stream.error = RuntimeError("connection lost")
    amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    await wait_for_transcript(lambda: not amd.amd_stt_active)
    assert amd.end_turn("") == AMDTranscript(session="", amd=None)
    assert amd.end_turn("Session transcript.") == AMDTranscript(
        session="Session transcript.", amd=None
    )


@pytest.mark.asyncio
async def test_stream_open_failure_keeps_session_transcript() -> None:
    model = DrainingSTT()
    model.stream = Mock(side_effect=RuntimeError("failed to open"))
    async with aclosing(AMDTranscriber(model, APIConnectOptions(max_retry=0))) as amd:
        amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        amd.push_audio(rtc.AudioFrame.create(16000, 1, 320))
        model.stream.assert_called_once()
        assert amd.end_turn("Hello.") == AMDTranscript(session="Hello.", amd=None)
    await model.aclose()


@pytest.mark.asyncio
async def test_amd_stt_stream_is_tagged_only_for_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draining = DrainingSTT()
    gateway = inference.STT(model="cartesia/ink-whisper", api_key=_SECRET, api_secret=_SECRET)
    calls: list[dict[str, Any]] = []

    def stream(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return draining.stream(conn_options=kwargs["conn_options"])

    monkeypatch.setattr(gateway, "stream", stream)
    frame = rtc.AudioFrame.create(16000, 1, 320)

    async with aclosing(AMDTranscriber(gateway, APIConnectOptions(max_retry=0))) as amd:
        amd.push_audio(frame)
    assert calls[0]["extra_headers"] == {HEADER_CLIENT_FEATURE: "AMD"}

    plugin = DrainingSTT()
    plugin_stream = Mock(wraps=plugin.stream)
    monkeypatch.setattr(plugin, "stream", plugin_stream)
    async with aclosing(AMDTranscriber(plugin, APIConnectOptions(max_retry=0))) as amd:
        amd.push_audio(frame)
    assert "extra_headers" not in plugin_stream.call_args.kwargs

    await draining.aclose()
    await plugin.aclose()
    await gateway.aclose()


# endregion


# region: Detector lifecycle, turns and replies (detector.py)


def test_prediction_event_carries_its_stage() -> None:
    event = AMDPredictionEvent(
        category=AMDCategory.WAIT,
        reason="prediction",
        transcript="hello",
        speech_duration=0,
        delay=0,
    )
    assert event.stage is AMDCategory.UNCERTAIN
    dumped = json.loads(event.model_dump_json())
    assert dumped["stage"] == "uncertain"

    event = AMDPredictionEvent(
        category=AMDCategory.MACHINE_VM,
        stage=AMDCategory.MACHINE_VM,
        reason="prediction",
        transcript="hello",
        speech_duration=0,
        delay=0,
    )
    assert event.model_dump()["stage"] == "machine-vm"


class ClassifierLLM(FakeLLM):
    """Controllable provider model. Tests use the real SDK inference path."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: asyncio.Queue[Any] = asyncio.Queue()
        self.menu_requests: asyncio.Queue[asyncio.Future[str]] = asyncio.Queue()
        self.responses: dict[int, asyncio.Future[str]] = {}

    def chat(self, **kwargs: Any) -> llm.LLMStream:
        chat_ctx = kwargs["chat_ctx"]
        response = asyncio.get_running_loop().create_future()
        if chat_ctx.items[0].text_content == _inference.CLASSIFY_PROMPT:
            data = json.loads(chat_ctx.items[1].text_content)
            messages = [item for item in chat_ctx.messages() if item.role == "user"]
            self.responses[messages[-1].extra["turn_id"]] = response
            self.requests.put_nowait(
                SimpleNamespace(
                    **data,
                    current_turn=messages[-1],
                    earlier_turns=messages[:-1],
                    chat_ctx=chat_ctx.copy(),
                )
            )
        else:
            self.menu_requests.put_nowait(response)
        return ControlledStream(self, response=response, **kwargs)

    async def request(self) -> Any:
        return await asyncio.wait_for(self.requests.get(), 2)

    def prediction(self, turn_id: int, category: AMDCategory, **kwargs: Any) -> None:
        self.respond(turn_id, json.dumps({"category": category.value, **kwargs}))

    def respond(self, turn_id: int, arguments: str) -> None:
        response = self.responses[turn_id]
        if not response.done():
            response.set_result(arguments)


class ControlledStream(llm.LLMStream):
    def __init__(
        self, model: ClassifierLLM, *, response: asyncio.Future[str], **kwargs: Any
    ) -> None:
        super().__init__(
            model,
            chat_ctx=kwargs["chat_ctx"],
            tools=kwargs["tools"],
            conn_options=kwargs.get("conn_options", APIConnectOptions(max_retry=0)),
        )
        self.response = response

    async def _run(self) -> None:
        arguments = await self.response
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="amd-test",
                delta=llm.ChoiceDelta(
                    role="assistant",
                    tool_calls=[
                        llm.FunctionToolCall(
                            name="record_result", arguments=arguments, call_id="amd-test"
                        )
                    ],
                ),
            )
        )


class RecordingLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__(
            fake_responses=[
                FakeLLMResponse(input=prompt, content="Please call me back.", ttft=0, duration=0)
                for prompt in (
                    _DEFAULT_SCREENING_INSTRUCTIONS,
                    _DEFAULT_VOICEMAIL_INSTRUCTIONS,
                    _DEFAULT_IVR_INSTRUCTIONS,
                    _DEFAULT_HUMAN_INSTRUCTIONS,
                    "hello",
                )
            ]
        )
        self.calls: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def chat(self, **kwargs: Any) -> llm.LLMStream:
        self.calls.put_nowait({**kwargs, "chat_ctx": kwargs["chat_ctx"].copy()})
        return super().chat(**kwargs)


class CustomerAgent(Agent):
    def __init__(self, *, stop: bool = False) -> None:
        super().__init__(instructions="You are calling about an appointment.")
        self.hooks: list[llm.ChatContext] = []
        self.hook_started = asyncio.Event()
        self.hook_release = asyncio.Event()
        self.hook_release.set()
        self.stop = stop

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        self.hooks.append(turn_ctx.copy())
        self.hook_started.set()
        await self.hook_release.wait()
        if self.stop:
            raise StopResponse()
        turn_ctx.add_message(role="system", content="Customer hook edit")


def end_of_turn(transcript: str = "hello", *, skip_reply: bool = False) -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=skip_reply,
        new_transcript=transcript,
        transcript_confidence=1,
        metrics=_EndOfTurnMetrics(None, None, None, None),
    )


def push_audio(detector: AMD, stt: DrainingSTT) -> DrainingStream:
    detector.push_audio(rtc.AudioFrame.create(16000, 1, 320))
    return stt.streams[-1]


def commit_turn(detector: AMD, info: _EndOfTurnInfo) -> TurnHooks:
    hooks = detector._session._turn_hooks
    assert hooks is not None
    return hooks.on_user_turn_committed(info.new_transcript, info.metrics.end_of_turn_delay)


def speech_started(detector: AMD) -> None:
    detector._session._update_user_state("speaking")


def speech_ended(detector: AMD, silence_duration: float) -> None:
    detector._session._update_user_state(
        "listening", last_speaking_time=time.time() - silence_duration
    )


def transcribe(detector: AMD, transcript: str) -> None:
    detector._session._user_input_transcribed(
        UserInputTranscribedEvent(transcript=transcript, is_final=True)
    )


def dtmf_executed(
    session: AgentSession,
    digits: str,
    *,
    is_error: bool = False,
    name: str = "send_dtmf_events",
    created_at: float | None = None,
) -> FunctionToolsExecutedEvent:
    call = llm.FunctionCall(
        call_id=utils.shortuuid("dtmf_"),
        name=name,
        arguments=json.dumps({"events": list(digits)}),
        created_at=created_at if created_at is not None else time.time(),
    )
    event = FunctionToolsExecutedEvent(
        function_calls=[call],
        function_call_outputs=[
            llm.FunctionCallOutput(
                call_id=call.call_id,
                name=name,
                output="Send failed" if is_error else f"Successfully sent DTMF events: {digits}",
                is_error=is_error,
            )
        ],
    )
    session.emit("function_tools_executed", event)
    return event


def dtmf_calls(chat_ctx: llm.ChatContext) -> list[llm.FunctionCall]:
    return [item for item in chat_ctx.items if isinstance(item, llm.FunctionCall)]


async def eventually(predicate: Any) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 2)


@asynccontextmanager
async def running(
    *,
    classifier: ClassifierLLM | None = None,
    agent: CustomerAgent | None = None,
    session_options: dict[str, Any] | None = None,
    **options: Any,
) -> AsyncIterator[tuple[AMD, AgentSession, ClassifierLLM, RecordingLLM]]:
    classifier = classifier or ClassifierLLM()
    agent = agent or CustomerAgent()
    model = RecordingLLM()
    session = AgentSession(
        **{
            "llm": model,
            "tts": FakeTTS(fake_audio_duration=0.05),
            "turn_handling": {"turn_detection": "manual"},
            "aec_warmup_duration": None,
            **(session_options or {}),
        }
    )
    session.output.audio = FakeAudioOutput()
    await session.start(agent)
    try:
        detector = AMD(
            session,
            llm=classifier,
            **{"machine_silence_threshold": 0, "stt": None, **options},
        )
        async with detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            yield detector, session, classifier, model
    finally:
        await session.aclose()


async def commit(
    detector: AMD, session: AgentSession, classifier: ClassifierLLM, *, reply: bool = False
) -> TurnHooks | None:
    info = end_of_turn()
    hooks = None
    if reply:
        assert session._activity is not None
        session._activity.on_end_of_turn(info)
    else:
        hooks = commit_turn(detector, info)
    request = await classifier.request()
    assert request.current_turn.extra["turn_id"] == detector._turn_id
    return hooks


async def confirm_human(
    detector: AMD, classifier: ClassifierLLM, text: str = "Hello, who is this?"
) -> TurnHooks:
    """Commit and classify the turn that confirms a human after a menu or screener."""
    held = detector._turn_id
    await eventually(lambda: detector._turns[held].prediction is not None)
    hooks = commit_turn(detector, end_of_turn(text))
    request = await classifier.request()
    classifier.prediction(request.current_turn.extra["turn_id"], AMDCategory.HUMAN)
    return hooks


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_reply", [False, True])
async def test_turn_hooks_control_replies_without_amd(allow_reply: bool) -> None:
    @llm.function_tool
    def reply_tool() -> str:
        """A tool available only to the generated reply."""
        return "done"

    replies: list[SpeechHandle] = []
    calls: list[str] = []

    class Hooks:
        reply_instructions = None

        def on_user_turn_committed(
            self, transcript: str, end_of_turn_delay: float | None
        ) -> TurnHooks:
            assert transcript == "hello"
            assert end_of_turn_delay is None
            assert agent.hooks == []
            calls.append("user_turn_committed")
            return self

        def on_user_turn_completed(self) -> None:
            assert activity._user_turn_completed_atask.done()
            calls.append("user_turn_completed")

        def on_reply_generation(
            self, tools: list[llm.Tool | llm.Toolset]
        ) -> list[llm.Tool | llm.Toolset]:
            calls.append("reply_generation")
            return [*tools, reply_tool]

        async def should_reply(self, chat_ctx: llm.ChatContext) -> bool:
            assert len(agent.hooks) == 1
            calls.append("should_reply")
            chat_ctx.add_message(role="system", content="Turn instructions")
            return allow_reply

        def on_agent_turn_committed(self, handle: SpeechHandle) -> None:
            calls.append("agent_turn_committed")
            replies.append(handle)

    model = RecordingLLM()
    agent = CustomerAgent()
    session = AgentSession(llm=model, turn_handling={"turn_detection": "manual"})
    await session.start(agent)
    try:
        assert session.amd is None
        session._turn_hooks = Hooks()
        activity = session._activity
        assert activity is not None
        info = end_of_turn()
        activity.on_end_of_turn(info)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)

        if allow_reply:
            assert calls == [
                "user_turn_committed",
                "should_reply",
                "reply_generation",
                "agent_turn_committed",
                "user_turn_completed",
            ]
            call = await asyncio.wait_for(model.calls.get(), 2)
            assert any(
                m.text_content == "Turn instructions"
                for m in call["chat_ctx"].items
                if m.type == "message"
            )
            assert reply_tool in call["tools"]
            assert len(replies) == 1
            await asyncio.wait_for(replies[0], 2)
        else:
            assert calls == ["user_turn_committed", "should_reply", "user_turn_completed"]
            assert model.calls.empty()
            assert replies == []
        assert len(agent.hooks) == 1
        assert not any(
            m.text_content == "Turn instructions"
            for m in agent.chat_ctx.items
            if m.type == "message"
        )
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["adopted", "replaced", "suppressed"])
async def test_preemptive_generation_commits_only_the_accepted_reply(outcome: str) -> None:
    hooks = Mock(spec=TurnHooks)
    hooks.on_user_turn_committed.return_value = hooks
    hooks.on_reply_generation.side_effect = lambda tools: tools

    async def should_reply(chat_ctx: llm.ChatContext) -> bool:
        if outcome == "replaced":
            chat_ctx.add_message(role="system", content=_DEFAULT_HUMAN_INSTRUCTIONS)
        return outcome != "suppressed"

    hooks.should_reply.side_effect = should_reply
    model = RecordingLLM()
    session = AgentSession(llm=model, turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        session._turn_hooks = hooks
        session.options.preemptive_generation["enabled"] = True
        activity = session._activity
        assert activity is not None
        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="hello", transcript_confidence=1, started_speaking_at=None
            )
        )
        assert activity._preemptive_generation is not None
        preemptive = activity._preemptive_generation.speech_handle
        await asyncio.wait_for(model.calls.get(), 2)

        hooks.on_reply_generation.assert_called_once()
        hooks.on_user_turn_committed.assert_not_called()
        hooks.on_user_turn_completed.assert_not_called()
        hooks.should_reply.assert_not_awaited()
        hooks.on_agent_turn_committed.assert_not_called()

        activity.on_end_of_turn(end_of_turn())
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)

        hooks.on_user_turn_committed.assert_called_once_with("hello", None)
        hooks.on_user_turn_completed.assert_called_once_with()
        hooks.should_reply.assert_awaited_once()
        if outcome == "suppressed":
            hooks.on_agent_turn_committed.assert_not_called()
        else:
            hooks.on_agent_turn_committed.assert_called_once()
            reply = hooks.on_agent_turn_committed.call_args.args[0]
            assert (reply is preemptive) == (outcome == "adopted")
            await asyncio.wait_for(reply, 2)
        assert preemptive.interrupted == (outcome != "adopted")
        if outcome == "replaced":
            await asyncio.wait_for(model.calls.get(), 2)
        assert hooks.on_reply_generation.call_count == (2 if outcome == "replaced" else 1)
        assert model.calls.empty()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_cancel_pending_speech_includes_preemptive_and_queued_speech() -> None:
    async with running() as (_, session, _, _):
        activity = session._activity
        assert activity is not None
        session.options.preemptive_generation["enabled"] = True
        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="hello", transcript_confidence=1, started_speaking_at=None
            )
        )
        assert activity._preemptive_generation is not None
        preemptive = activity._preemptive_generation.speech_handle
        queued = session.say("Please call me back.")

        activity._cancel_pending_speeches()

        await asyncio.wait_for(asyncio.gather(preemptive, queued), 2)
        assert preemptive.interrupted and queued.interrupted
        assert activity._preemptive_generation is None
        assert session.output.audio.captured_playout_segments == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["reused", "inference_error", "inference_timeout"])
async def test_a_fallback_is_uncertain_and_a_reuse_repeats_the_prediction(reason: str) -> None:
    async with running(inference_timeout=0.02) as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(detector._should_reply(1, llm.ChatContext()), 2)
        first = events[-1]
        assert first.stage is AMDCategory.MACHINE_VM

        if reason == "reused":
            commit_turn(detector, end_of_turn(""))
        else:
            await commit(detector, session, classifier)
            if reason == "inference_error":
                classifier.respond(2, "invalid JSON")
        await asyncio.wait_for(detector._should_reply(2, llm.ChatContext()), 2)
        fallback = detector._turns[2].prediction
        assert fallback is not None
        assert len(events) == 2
        assert fallback.reason == reason
        # a turn with no prediction is unknown; an empty turn repeats the last one
        expected = AMDCategory.MACHINE_VM if reason == "reused" else AMDCategory.UNCERTAIN
        assert fallback.category == expected
        assert fallback.stage is first.stage

        await detector.aclose()
        completed = await detector.execute()
        latest = first if reason == "reused" else fallback
        assert completed.last_prediction == latest


@pytest.mark.asyncio
async def test_customer_hook_and_prediction_overlap_controls_are_temporary() -> None:
    agent = CustomerAgent()
    agent.hook_release.clear()
    async with running(agent=agent) as (detector, session, classifier, model):
        completed = asyncio.create_task(detector.execute())
        info = end_of_turn()
        info.metrics.end_of_turn_delay = 0.5
        session._activity.on_end_of_turn(info)
        assert detector._turn_id == 1
        assert not agent.hook_started.is_set()
        request = await classifier.request()
        await agent.hook_started.wait()
        assert request.current_turn.extra["turn_id"] == 1
        assert request.current_turn.text_content == "hello"
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._turns[1].prediction is not None)
        assert model.calls.empty()
        agent.hook_release.set()
        call = await asyncio.wait_for(model.calls.get(), 2)
        messages = [m for m in call["chat_ctx"].items if m.type == "message"]
        assert messages[-1].text_content == _DEFAULT_SCREENING_INSTRUCTIONS
        assert any(m.text_content == "Customer hook edit" for m in messages)
        assert any(m.text_content == "hello" for m in messages)
        assert len(agent.hooks) == 1
        assert not completed.done()
        await eventually(lambda: session._activity._no_pending_speech)
        assert not any(m.id.startswith(detector._control_prefix) for m in agent.chat_ctx.items)
        await detector.aclose()
        assert (await completed).reason == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "instructions"),
    [
        (AMDCategory.MACHINE_SCREENING, _DEFAULT_SCREENING_INSTRUCTIONS),
        (AMDCategory.MACHINE_VM, _DEFAULT_VOICEMAIL_INSTRUCTIONS),
        # a first human after screening is likely human, counted as uncertain, with no stage instructions
        (AMDCategory.HUMAN, None),
    ],
)
async def test_turn_hooks_keep_their_turn_while_an_older_hook_is_blocked(
    category: AMDCategory, instructions: str | None
) -> None:
    agent = CustomerAgent()
    agent.hook_release.clear()
    async with running(agent=agent) as (detector, session, classifier, model):
        activity = session._activity
        assert activity is not None
        try:
            await commit(detector, session, classifier, reply=True)
            await asyncio.wait_for(agent.hook_started.wait(), 2)
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
            await eventually(lambda: detector._turns[1].prediction is not None)

            activity.on_end_of_turn(end_of_turn("Can you hear me?"))
            assert (await classifier.request()).current_turn.extra["turn_id"] == 2
            classifier.prediction(2, category)
            await eventually(lambda: detector._turns[2].prediction is not None)
            assert len(agent.hooks) == 1
            assert model.calls.empty()
        finally:
            agent.hook_release.set()

        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        call = await asyncio.wait_for(model.calls.get(), 2)
        messages = [m for m in call["chat_ctx"].items if m.type == "message"]
        if instructions is None:
            assert not any(m.id.startswith(detector._control_prefix) for m in messages)
        else:
            assert messages[-1].text_content == instructions
        assert any(m.text_content == "Can you hear me?" for m in messages)
        assert len(agent.hooks) == 2
        if category == AMDCategory.MACHINE_VM:
            await eventually(lambda: detector._voicemail_message_played)
        await eventually(lambda: activity._no_pending_speech)
        assert model.calls.empty()


@pytest.mark.asyncio
async def test_turn_ids_are_local_to_each_amd_run() -> None:
    session = AgentSession(llm=RecordingLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(CustomerAgent())
    try:
        activity = session._activity
        assert activity is not None
        for _ in range(2):
            activity.on_end_of_turn(end_of_turn(skip_reply=True))
            await asyncio.wait_for(activity._user_turn_completed_atask, 2)
            classifier = ClassifierLLM()
            async with AMD(
                session, llm=classifier, stt=None, machine_silence_threshold=0
            ) as detector:
                await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
                hooks = await commit(detector, session, classifier)
                assert detector._turn_id == 1
                classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
                assert await hooks.should_reply(llm.ChatContext())
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category", [AMDCategory.MACHINE_SCREENING, AMDCategory.MACHINE_UNAVAILABLE]
)
async def test_amd_turn_hooks_retain_the_adopted_user_turn_span(category: AMDCategory) -> None:
    async with running() as (detector, session, classifier, model):
        span = Mock(spec=trace.Span)
        span.get_span_context.return_value = trace.INVALID_SPAN_CONTEXT
        info = end_of_turn()
        info.user_turn_span = span
        session._activity.on_end_of_turn(info)
        await classifier.request()
        await session.current_agent.hook_started.wait()
        await asyncio.sleep(0)

        assert info.user_turn_span_adopted
        span.end.assert_not_called()
        assert model.calls.empty()

        classifier.prediction(1, category)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        span.end.assert_called_once_with()
        assert not info.user_turn_span_adopted
        if category == AMDCategory.MACHINE_UNAVAILABLE:
            assert (await detector.execute()).category == category
            assert model.calls.empty()
        else:
            call = await asyncio.wait_for(model.calls.get(), 2)
            assert call["chat_ctx"].items[-1].text_content == _DEFAULT_SCREENING_INSTRUCTIONS


@pytest.mark.asyncio
async def test_voicemail_sends_one_delivered_message_and_records_playback() -> None:
    async with running() as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: detector._voicemail_message_played)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(2, AMDCategory.MACHINE_VM)
        await eventually(lambda: detector._turns[2].prediction is not None)
        await session._activity._user_turn_completed_atask
        assert model.calls.empty()
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert [
            m.text_content
            for m in session.current_agent.chat_ctx.items
            if m.type == "message" and m.role == "user"
        ] == ["hello", "hello"]


@pytest.mark.asyncio
async def test_superseded_voicemail_reply_does_not_consume_the_stage_message() -> None:
    async with running() as (detector, session, classifier, _):
        activity = session._activity
        assert activity is not None
        speeches = []

        def advance_turn(event: Any) -> None:
            speeches.append(event.speech_handle)
            if len(speeches) == 1:
                assert activity.on_end_of_turn(end_of_turn("Please leave your message."))

        session.on("speech_created", advance_turn)
        await commit(detector, session, classifier, reply=True)
        first_turn_task = activity._user_turn_completed_atask
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert (await classifier.request()).current_turn.extra["turn_id"] == 2
        await asyncio.wait_for(first_turn_task, 2)
        assert len(speeches) == 1
        assert speeches[0].interrupted
        assert detector._voicemail_handle is None

        classifier.prediction(2, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        assert len(speeches) == 2
        assert detector._voicemail_handle is speeches[1]
        await asyncio.wait_for(speeches[1], 2)
        assert not speeches[1].interrupted
        assert detector._voicemail_message_played
        assert session.output.audio.captured_playout_segments == 1

        await commit(detector, session, classifier, reply=True)
        classifier.prediction(3, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        assert len(speeches) == 2
        assert session.output.audio.captured_playout_segments == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_commit", ["before_new_handle", "after_new_handle"])
async def test_stale_voicemail_handle_does_not_clear_a_newer_reservation(
    stale_commit: str,
) -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        second = await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.MACHINE_VM)
        assert await second.should_reply(llm.ChatContext())

        old_handle = SpeechHandle.create()
        old_handle.interrupt()
        new_handle = SpeechHandle.create()
        if stale_commit == "before_new_handle":
            first.on_agent_turn_committed(old_handle)
        second.on_agent_turn_committed(new_handle)
        if stale_commit == "after_new_handle":
            first.on_agent_turn_committed(old_handle)
        assert detector._voicemail_handle is new_handle

        output = session.output.audio
        assert output is not None
        await output.capture_frame(rtc.AudioFrame.create(24000, 1, 24))
        output.flush()
        await output.wait_for_playout()
        completed = asyncio.Event()
        new_handle.add_done_callback(lambda _: completed.set())
        new_handle._mark_done()
        old_handle._mark_done()
        await asyncio.wait_for(completed.wait(), 2)
        assert detector._voicemail_message_played

        third = await commit(detector, session, classifier)
        classifier.prediction(3, AMDCategory.MACHINE_VM)
        assert not await third.should_reply(llm.ChatContext())


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["played", "interrupted", "failed", "no_audio", "closed"])
async def test_voicemail_playback_is_recorded_by_its_handle(outcome: str) -> None:
    async with running() as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        assert hooks is not None
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await hooks.should_reply(llm.ChatContext())

        handle = SpeechHandle.create()
        hooks.on_agent_turn_committed(handle)
        if outcome != "no_audio":
            output = session.output.audio
            assert output is not None
            await output.capture_frame(rtc.AudioFrame.create(24000, 1, 24))
            output.flush()
            await output.wait_for_playout()

        assert not detector._voicemail_message_played
        if outcome == "interrupted":
            handle.interrupt()
        elif outcome == "closed":
            await detector.aclose()

        completed = asyncio.Event()
        handle.add_done_callback(lambda _: completed.set())
        handle._mark_done(RuntimeError("speech failed") if outcome == "failed" else None)
        await asyncio.wait_for(completed.wait(), 2)
        assert detector._voicemail_message_played == (outcome == "played")


@pytest.mark.asyncio
async def test_voicemail_handle_commit_does_not_signal_a_prediction_or_update_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with running() as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await hooks.should_reply(llm.ChatContext())
        notify_prediction = Mock(wraps=detector._prediction_changed.set)
        update_idle = Mock(wraps=detector._update_idle)
        monkeypatch.setattr(detector._prediction_changed, "set", notify_prediction)
        monkeypatch.setattr(detector, "_update_idle", update_idle)

        handle = SpeechHandle.create()
        hooks.on_agent_turn_committed(handle)

        notify_prediction.assert_not_called()
        update_idle.assert_not_called()
        handle._mark_done()
        await asyncio.wait_for(handle, 2)


@pytest.mark.asyncio
async def test_voicemail_idle_defaults_to_one_minute_after_playback() -> None:
    async with running() as (detector, session, classifier, model):
        loop = asyncio.get_running_loop()
        assert next_deadline(detector) is not None
        assert next_deadline(detector) - loop.time() == pytest.approx(10, abs=0.1)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(
            lambda: detector._voicemail_message_played and session._activity._no_pending_speech
        )
        assert next_deadline(detector) is not None
        assert next_deadline(detector) - loop.time() == pytest.approx(60, abs=0.1)
        await detector.aclose()
        assert next_deadline(detector) is None


@pytest.mark.asyncio
async def test_late_post_voicemail_menu_uses_the_normal_ivr_idle_timeout() -> None:
    async with running(idle_timeout=0.03, voicemail_idle_timeout=1.0) as (
        detector,
        session,
        classifier,
        model,
    ):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(
            lambda: detector._voicemail_message_played and session._activity._no_pending_speech
        )
        voicemail_deadline = next_deadline(detector)
        assert voicemail_deadline is not None
        await asyncio.sleep(0.06)
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        speech_started(detector)
        assert next_deadline(detector) is not None
        assert next_deadline(detector) > voicemail_deadline
        speech_ended(detector, 0)
        menu = (
            "To replay your message, press 1. To continue recording, press 2. "
            "To delete and re-record your message, press 3. For delivery options, press 4. "
            "To send a fax, press 6. To cancel this message, press star. "
            "To send this message now, press pound or hang up."
        )
        session._activity.on_end_of_turn(end_of_turn(menu))
        await classifier.request()
        classifier.prediction(2, AMDCategory.MACHINE_IVR)
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert call["chat_ctx"].items[-1].text_content.startswith(_DEFAULT_IVR_INSTRUCTIONS)
        assert any(tool.id == "send_dtmf_events" for tool in call["tools"])
        result = await asyncio.wait_for(detector.execute(), 0.5)
        assert result.reason == "idle_timeout"
        assert result.category == AMDCategory.MACHINE_IVR
        assert result.voicemail_message_played


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("menu", "last", "expected"),
    [
        (False, AMDCategory.MACHINE_UNAVAILABLE, AMDCategory.MACHINE_VM),
        (True, AMDCategory.MACHINE_UNAVAILABLE, AMDCategory.MACHINE_IVR),
        (True, AMDCategory.HUMAN, AMDCategory.HUMAN),
    ],
)
async def test_a_goodbye_after_the_voicemail_message_keeps_the_stage(
    menu: bool, last: AMDCategory, expected: AMDCategory
) -> None:
    async with running() as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(
            lambda: detector._voicemail_message_played and session._activity._no_pending_speech
        )
        if menu:
            hooks = await commit(detector, session, classifier)
            classifier.prediction(2, AMDCategory.MACHINE_IVR)
            assert hooks is not None
            await hooks.should_reply(llm.ChatContext())
        await commit(detector, session, classifier)
        classifier.prediction(detector._turn_id, last)
        if menu and last is AMDCategory.HUMAN:
            await confirm_human(detector, classifier)

        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "finished"
        # the box's goodbye is not an unreachable call; a person who picks up is human
        assert result.category == expected
        assert result.voicemail_message_played


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [AMDCategory.MACHINE_VM, AMDCategory.MACHINE_IVR])
async def test_late_prediction_preserves_stage_and_idle_timer(category: AMDCategory) -> None:
    async with running(idle_timeout=1.0, voicemail_idle_timeout=2.0, inference_timeout=0.01) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        first = await commit(detector, session, classifier)
        previous = (
            AMDCategory.MACHINE_IVR
            if category == AMDCategory.MACHINE_VM
            else AMDCategory.MACHINE_VM
        )
        classifier.prediction(1, previous)
        assert await first.should_reply(llm.ChatContext())
        second = await commit(detector, session, classifier)
        await second.should_reply(llm.ChatContext())
        deadline = next_deadline(detector)
        classifier.prediction(2, category)
        await asyncio.sleep(0)
        assert classifier.responses[2].cancelled()
        assert events[-1].reason == "inference_timeout"
        assert len(events) == 2
        assert detector._state == previous
        assert next_deadline(detector) == deadline
        assert next_deadline(detector) is not None
        expected = 2 if previous == AMDCategory.MACHINE_VM else 1
        assert next_deadline(detector) - asyncio.get_running_loop().time() == pytest.approx(
            expected, abs=0.1
        )


@pytest.mark.asyncio
async def test_voicemail_idle_does_not_extend_the_hard_timeout() -> None:
    async with running(timeout=0.08) as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.category == AMDCategory.MACHINE_VM
        assert result.reason == "timeout"
        assert next_deadline(detector) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("voicemail_idle_timeout", [0, -1])
async def test_nonpositive_voicemail_idle_timeout_is_rejected(
    voicemail_idle_timeout: float,
) -> None:
    with pytest.raises(ValueError, match="timeouts"):
        AMD(AgentSession(), voicemail_idle_timeout=voicemail_idle_timeout)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "prompt"),
    [
        (
            AMDCategory.MACHINE_VM,
            "I'm not available right now. Please leave a message after the beep.",
        ),
        (AMDCategory.MACHINE_SCREENING, "Please state your name and the reason you are calling."),
        (AMDCategory.MACHINE_IVR, "Say who you want to speak to."),
    ],
)
@pytest.mark.parametrize("hook_finishes_after_cleanup", [False, True])
@pytest.mark.parametrize(
    "human_instructions",
    [None, "Greet the person and ask whether they can discuss the appointment."],
)
async def test_human_takeover_updates_the_reply_context_during_machine_speech(
    category: AMDCategory,
    prompt: str,
    hook_finishes_after_cleanup: bool,
    human_instructions: str | None,
) -> None:
    agent = CustomerAgent()
    options = {"human_instructions": human_instructions} if human_instructions is not None else {}
    async with running(agent=agent, **options) as (detector, session, classifier, model):
        if human_instructions is not None:
            model.fake_response_map[human_instructions] = FakeLLMResponse(
                input=human_instructions,
                content="Hello, can we discuss your appointment?",
                ttft=0,
                duration=0,
            )
        activity = session._activity
        assert activity is not None
        assert isinstance(activity.tts, FakeTTS)
        activity.tts.update_options(fake_audio_duration=10)
        activity.on_end_of_turn(end_of_turn(prompt))
        await classifier.request()
        classifier.prediction(1, category)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: session.agent_state == "speaking")
        held = category in _fsm.LIKELY_HUMAN_STAGES
        if held:
            # the first human after a menu or screener is likely human, counted as uncertain; its reply
            # is the machine-stage speech that the confirming turn interrupts
            # the held turn's reply carries no AMD instructions: the hook's edit is last
            model.fake_response_map["Customer hook edit"] = FakeLLMResponse(
                input="Customer hook edit", content="Hi, this is Alex.", ttft=0, duration=0
            )
            first_speech = activity._current_speech
            activity.on_end_of_turn(end_of_turn("Hello?"))
            await classifier.request()
            classifier.prediction(2, AMDCategory.HUMAN)
            await asyncio.wait_for(model.calls.get(), 2)
            await eventually(
                lambda: (
                    activity._current_speech not in (None, first_speech)
                    and session.agent_state == "speaking"
                )
            )
            assert detector.lifecycle is AMDLifecycle.ACTIVE
        machine_speech = activity._current_speech
        assert machine_speech is not None and not machine_speech.done()
        activity.tts.update_options(fake_audio_duration=0.05)

        agent.hook_started.clear()
        if hook_finishes_after_cleanup:
            agent.hook_release.clear()
        activity.on_end_of_turn(end_of_turn("Can you hear me?"))
        await classifier.request()
        await asyncio.wait_for(agent.hook_started.wait(), 2)
        classifier.prediction(3 if held else 2, AMDCategory.HUMAN)
        if hook_finishes_after_cleanup:
            await asyncio.wait_for(detector.execute(), 2)
            assert session.amd is None
            assert session._turn_hooks is None
            agent.hook_release.set()

        call = await asyncio.wait_for(model.calls.get(), 2)
        messages = [m for m in call["chat_ctx"].items if m.type == "message"]
        controls = [m for m in messages if m.id.startswith(detector._control_prefix)]
        assert len(controls) == 1
        assert controls[0].extra["amd_stage"] == "human"
        assert controls[0] is messages[-1]
        assert controls[0].text_content == (human_instructions or _DEFAULT_HUMAN_INSTRUCTIONS)
        assert any(m.text_content == "Customer hook edit" for m in messages)
        assert any(m.text_content == prompt for m in messages)
        assert any(m.text_content == "Can you hear me?" for m in messages)
        assert machine_speech.interrupted
        assert len(agent.hooks) == (3 if held else 2)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.category == AMDCategory.HUMAN
        assert not result.voicemail_message_played
        await eventually(lambda: activity._no_pending_speech)
        assert not any(m.id.startswith(detector._control_prefix) for m in agent.chat_ctx.items)

        activity.on_end_of_turn(end_of_turn("Thank you."))
        normal = await asyncio.wait_for(model.calls.get(), 2)
        assert not any(m.id.startswith(detector._control_prefix) for m in normal["chat_ctx"].items)
        assert detector._turn_id == (3 if held else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("uncertain_first", [False, True])
async def test_human_instructions_are_skipped_without_a_machine_stage(
    uncertain_first: bool,
) -> None:
    async with running(human_instructions="The automated screening has finished.") as (
        detector,
        session,
        classifier,
        model,
    ):
        if uncertain_first:
            hooks = await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.UNCERTAIN)
            assert await hooks.should_reply(llm.ChatContext())

        await commit(detector, session, classifier, reply=True)
        classifier.prediction(detector._turn_id, AMDCategory.HUMAN)
        call = await asyncio.wait_for(model.calls.get(), 2)
        messages = [m for m in call["chat_ctx"].items if m.type == "message"]
        assert not any(m.id.startswith(detector._control_prefix) for m in messages)
        assert any(m.text_content == "Customer hook edit" for m in messages)
        assert any(m.text_content == "hello" for m in messages)
        assert (await detector.execute()).category == AMDCategory.HUMAN
        await eventually(lambda: session._activity._no_pending_speech)


@pytest.mark.asyncio
async def test_menu_is_observability_only_and_dtmf_tool_is_temporary() -> None:
    async with running() as (detector, session, classifier, model):
        menus = []
        detector.on("amd_menu_observed", menus.append)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        context = llm.ChatContext()
        assert await hooks.should_reply(context)
        before = session.current_agent.tools.copy()
        tools = hooks.on_reply_generation(before)
        assert any(t.id == "send_dtmf_events" for t in tools)
        assert session.current_agent.tools == before
        response = await asyncio.wait_for(classifier.menu_requests.get(), 2)
        response.set_result(
            json.dumps(
                {
                    "menu": "Choose a department",
                    "options": [{"label": "Billing", "dtmf": "1"}],
                }
            )
        )
        await eventually(lambda: len(menus) == 1)
        assert menus[0].options[0].dtmf == "1"
        assert model.calls.empty()
        assert session._activity._no_pending_speech
        assert detector._state == AMDCategory.MACHINE_IVR


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE])
async def test_terminal_result_completes_once_and_releases_turn_hooks(
    category: AMDCategory,
) -> None:
    async with running() as (detector, session, classifier, _):
        assert session._turn_hooks is not None
        events = []
        detector.on("amd_completed", events.append)
        first = asyncio.create_task(detector.execute())
        second = asyncio.create_task(detector.execute())
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, category)
        a, b = await asyncio.wait_for(asyncio.gather(first, second), 2)
        assert a is b
        assert a.reason == "finished"
        assert len(events) == 1
        assert session.amd is None
        assert session._turn_hooks is None
        assert session._activity._authorization_allowed.is_set()
        assert not detector._tasks
        for event, callback in (
            ("user_state_changed", detector._on_user_state_changed),
            ("user_input_transcribed", detector._on_user_input_transcribed),
        ):
            assert callback not in session._events[event]
        assert await hooks.should_reply(llm.ChatContext()) == (category == AMDCategory.HUMAN)
        await detector.aclose()
        assert len(events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "option"),
    [
        (AMDCategory.MACHINE_SCREENING, "screening_instructions"),
        (AMDCategory.MACHINE_VM, "voicemail_instructions"),
        (AMDCategory.MACHINE_IVR, "ivr_instructions"),
    ],
)
async def test_stage_without_instructions_completes_without_a_reply(
    category: AMDCategory, option: str
) -> None:
    async with running(**{option: None}) as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, category)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.category == category
        assert result.reason == "finished"
        assert result.last_prediction is not None and result.last_prediction.turn_id == 1
        assert not result.voicemail_message_played
        await eventually(lambda: session._activity._no_pending_speech)
        assert model.calls.empty()
        assert classifier.menu_requests.empty()
        assert session._turn_hooks is None


@pytest.mark.asyncio
async def test_human_instructions_none_keeps_agent_instructions() -> None:
    async with running(human_instructions=None) as (detector, session, classifier, model):
        hooks = await commit(detector, session, classifier)
        # voicemail has no likely-human turn, so the first human prediction completes AMD
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await hooks.should_reply(llm.ChatContext())

        await commit(detector, session, classifier, reply=True)
        classifier.prediction(2, AMDCategory.HUMAN)
        call = await asyncio.wait_for(model.calls.get(), 2)
        messages = [m for m in call["chat_ctx"].items if m.type == "message"]
        assert not any(m.id.startswith(detector._control_prefix) for m in messages)
        assert (await detector.execute()).category == AMDCategory.HUMAN
        await eventually(lambda: session._activity._no_pending_speech)


@pytest.mark.asyncio
@pytest.mark.parametrize("playback_started", [False, True])
async def test_late_unavailable_does_not_cancel_reply(
    playback_started: bool,
) -> None:
    async with running(inference_timeout=0.02) as (detector, session, classifier, model):
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._should_reply(1, llm.ChatContext())

        activity = session._activity
        assert activity is not None
        assert isinstance(activity.tts, FakeTTS)
        activity.tts.update_options(
            fake_audio_duration=10,
            fake_timeout=None if playback_started else 2,
        )
        await commit(detector, session, classifier, reply=True)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: activity._current_speech is not None)
        speech = activity._current_speech
        assert speech is not None
        if playback_started:
            await eventually(lambda: session.agent_state == "speaking")
        else:
            assert session.agent_state != "speaking"

        classifier.prediction(2, AMDCategory.MACHINE_UNAVAILABLE)
        await asyncio.sleep(0.1)
        assert classifier.responses[2].cancelled()
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._state == AMDCategory.MACHINE_SCREENING
        assert not speech.interrupted
        if playback_started:
            assert session.agent_state == "speaking"
        else:
            assert not speech.done()


@pytest.mark.asyncio
async def test_cancelled_execute_waiter_does_not_cancel_detection() -> None:
    async with running() as (detector, session, classifier, _):
        waiter = asyncio.create_task(detector.execute())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.HUMAN)
        assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_cancelled_reply_waiter_does_not_block_other_turns() -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        cancelled = asyncio.create_task(detector._should_reply(1, llm.ChatContext()))
        surviving = asyncio.create_task(detector._should_reply(1, llm.ChatContext()))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        await commit(detector, session, classifier)
        current = asyncio.create_task(detector._should_reply(2, llm.ChatContext()))
        await asyncio.wait_for(surviving, 2)
        assert detector._turns[1].prediction is None
        assert not current.done()
        classifier.prediction(2, AMDCategory.HUMAN)
        await asyncio.wait_for(current, 2)
        assert detector._turns[2].prediction.category == AMDCategory.HUMAN


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["cancelled", "participant_disconnected", "timeout"])
async def test_nonterminal_finish_releases_a_waiter_without_a_prediction(reason: str) -> None:
    async with running(timeout=0.03, inference_timeout=1) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        await commit(detector, session, classifier)
        waiter = asyncio.create_task(detector._should_reply(1, llm.ChatContext()))
        await asyncio.sleep(0)
        assert not waiter.done()
        if reason == "cancelled":
            await detector.aclose()
        elif reason == "participant_disconnected":
            detector._participant_identity = "callee"
            detector._on_disconnected(SimpleNamespace(identity="callee"))
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == reason
        assert result.category == AMDCategory.UNCERTAIN
        await asyncio.wait_for(waiter, 2)
        assert detector._turns[1].prediction is None
        assert events == []
        assert next_deadline(detector) is None


@pytest.mark.asyncio
async def test_empty_turn_does_not_cancel_a_pending_terminal_prediction() -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        commit_turn(detector, end_of_turn(""))
        assert detector._turns[2].prediction is None
        classifier.prediction(1, AMDCategory.HUMAN)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.last_prediction is not None
        assert result.last_prediction.turn_id == 1
        assert result.category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_completion_is_forwarded_to_session_observability() -> None:
    async with running() as (detector, session, classifier, _):
        host = Mock()
        session._session_host = host
        try:
            await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.HUMAN)
            result = await asyncio.wait_for(detector.execute(), 2)
            host._on_amd_completed.assert_called_once_with(result)
        finally:
            session._session_host = None


@pytest.mark.asyncio
async def test_timeout_rearms_and_late_result_cannot_change_a_newer_turn() -> None:
    async with running(inference_timeout=0.02) as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        hooks = await commit(detector, session, classifier)
        assert await hooks.should_reply(llm.ChatContext())
        assert events[-1].reason == "inference_timeout"
        assert next_deadline(detector) is not None
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert classifier.responses[1].cancelled()
        assert [event.reason for event in events] == ["inference_timeout"]
        assert detector._state == AMDCategory.UNCERTAIN
        classifier.prediction(2, AMDCategory.MACHINE_IVR)
        await eventually(lambda: events[-1].turn_id == 2 and events[-1].reason == "prediction")
        assert detector._state == AMDCategory.MACHINE_IVR


@pytest.mark.asyncio
async def test_stop_response_remains_owned_by_session() -> None:
    agent = CustomerAgent(stop=True)
    async with running(agent=agent, idle_timeout=0.03) as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert (await asyncio.wait_for(detector.execute(), 2)).reason == "idle_timeout"
        assert len(agent.hooks) == 1
        assert model.calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("handoff", ["direct", "inline"])
async def test_agent_handoff_is_rejected_until_amd_finishes(handoff: str) -> None:
    entered = asyncio.Event()

    class ReplacementAgent(Agent):
        async def on_enter(self) -> None:
            entered.set()

    replacement = ReplacementAgent(instructions="Continue the call.")
    async with running() as (detector, session, _, _):
        original = session.current_agent
        activity = session._activity
        with pytest.raises(RuntimeError, match="handoffs are not supported while AMD is running"):
            if handoff == "direct":
                session.update_agent(replacement)
            else:
                await session._update_activity(replacement, previous_activity="pause")

        assert session.current_agent is original
        assert session._activity is activity
        assert not activity._new_turns_blocked
        assert not entered.is_set()
        assert session.amd is detector
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        await detector.aclose()
        session.update_agent(replacement)
        await asyncio.wait_for(entered.wait(), 2)
        assert session.current_agent is replacement


@pytest.mark.asyncio
@pytest.mark.parametrize("draining", [False, True])
async def test_amd_cannot_enter_during_an_agent_handoff(draining: bool) -> None:
    exiting = asyncio.Event()
    release_exit = asyncio.Event()

    class OriginalAgent(Agent):
        async def on_exit(self) -> None:
            exiting.set()
            await release_exit.wait()

    async with AgentSession(llm=FakeLLM()) as session:
        await session.start(OriginalAgent(instructions="Test"))
        replacement = Agent(instructions="Continue the call.")
        session.update_agent(replacement)
        try:
            if draining:
                await asyncio.wait_for(exiting.wait(), 2)
            detector = AMD(session, llm=None, stt=None)
            with pytest.raises(RuntimeError, match="agent handoff to finish"):
                await detector.__aenter__()
            assert session.amd is None
            assert detector.lifecycle is AMDLifecycle.INITIALIZED
        finally:
            release_exit.set()
            await session._update_activity_atask

        async with detector:
            assert session.current_agent is replacement
            assert session.amd is detector


@pytest.mark.asyncio
async def test_tool_handoff_is_cancelled_while_amd_is_running(
    caplog: pytest.LogCaptureFixture,
) -> None:
    replacement = Agent(instructions="Continue the call.")

    class HandoffAgent(CustomerAgent):
        @llm.function_tool
        async def transfer(self) -> Agent:
            """Transfer the call to the next agent."""
            return replacement

    original = HandoffAgent()
    async with running(agent=original) as (detector, session, classifier, reply_model):
        reply_model.fake_response_map[_DEFAULT_SCREENING_INSTRUCTIONS] = FakeLLMResponse(
            input=_DEFAULT_SCREENING_INSTRUCTIONS,
            content="",
            ttft=0,
            duration=0,
            tool_calls=[llm.FunctionToolCall(name="transfer", arguments="{}", call_id="move")],
        )
        handles: list[SpeechHandle] = []
        session.once("speech_created", lambda ev: handles.append(ev.speech_handle))
        executed: list[FunctionToolsExecutedEvent] = []
        session.on("function_tools_executed", executed.append)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: bool(executed))
        await asyncio.wait_for(handles[0], 2)

        assert session.current_agent is original
        assert session.amd is detector
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert not executed[0].has_agent_handoff
        assert executed[0].function_call_outputs[0] in session.history.items
        assert "agent handoff cancelled while AMD is running" in caplog.text

        await detector.aclose()
        session.generate_reply(instructions=_DEFAULT_SCREENING_INSTRUCTIONS)
        await eventually(lambda: session.current_agent is replacement)


@pytest.mark.asyncio
async def test_invalid_model_output_falls_back_and_releases_the_reply() -> None:
    async with running() as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        classifier.respond(1, "not JSON")
        assert await hooks.should_reply(llm.ChatContext())
        assert detector._turns[1].prediction.reason == "inference_error"
        assert detector._state == AMDCategory.UNCERTAIN
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert next_deadline(detector) is not None


@pytest.mark.asyncio
async def test_amd_commits_short_turns_during_agent_speech() -> None:
    session_options = {
        "stt": FakeSTT(),
        "turn_handling": {"turn_detection": "stt", "interruption": {"min_words": 3}},
    }
    async with running(session_options=session_options) as (detector, session, _, _):
        activity = session._activity
        assert activity is not None
        speech = SpeechHandle.create(allow_interruptions=True)
        activity._current_speech = speech
        # AMD classifies every turn, so the min_words filter does not drop a short one
        assert activity.on_end_of_turn(end_of_turn("hi"))
        assert detector._turn_id == 1
        assert detector._turns[1].transcript.transcript == "hi"
        speech._mark_done()
        activity._current_speech = None
        await detector.aclose()
        # without AMD the same short turn is dropped
        assert not activity.on_end_of_turn(end_of_turn("hi"))


@pytest.mark.asyncio
async def test_false_interruption_settlement_rearms_idle_without_a_speech_handle() -> None:
    async with running(idle_timeout=0.03) as (detector, session, _, _):
        activity = session._activity
        activity._false_interruption_pending = True
        session._update_agent_state("thinking")
        session._update_agent_state("listening")
        await asyncio.sleep(0.04)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        session.emit("agent_false_interruption", AgentFalseInterruptionEvent(resumed=False))
        activity._false_interruption_pending = False
        assert (await asyncio.wait_for(detector.execute(), 2)).reason == "idle_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["inference_error", "reused", "prediction"])
async def test_unchanged_voicemail_rearms_idle_without_another_reply(reason: str) -> None:
    async with running(voicemail_idle_timeout=0.03) as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        handle = SpeechHandle.create()
        first.on_agent_turn_committed(handle)
        output = session.output.audio
        await output.capture_frame(rtc.AudioFrame.create(24000, 1, 24))
        output.flush()
        await output.wait_for_playout()
        handle._mark_done()
        await asyncio.wait_for(handle, 2)
        speech_started(detector)
        speech_ended(detector, 0)
        if reason == "reused":
            second = commit_turn(detector, end_of_turn(""))
        else:
            second = await commit(detector, session, classifier)
            if reason == "inference_error":
                classifier.respond(2, "not JSON")
            else:
                classifier.prediction(2, AMDCategory.MACHINE_VM)
        assert not await second.should_reply(llm.ChatContext())
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "idle_timeout"
        assert result.category == AMDCategory.MACHINE_VM


@pytest.mark.asyncio
async def test_three_uncertain_predictions_complete_detection() -> None:
    async with running() as (detector, session, classifier, _):
        for turn_id in range(1, 4):
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, AMDCategory.UNCERTAIN)
            await eventually(
                lambda expected=turn_id: detector._turns[expected].prediction is not None
            )
        assert (await detector.execute()).reason == "max_uncertain_turns"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 1, 4])
async def test_inference_timeout_limit_completes_detection(limit: int | None) -> None:
    options = {} if limit is None else {"max_inference_timeouts": limit}
    expected_limit = 3 if limit is None else limit
    async with running(inference_timeout=0.01, **options) as (detector, session, classifier, _):
        for turn_id in range(1, expected_limit + 1):
            await commit(detector, session, classifier)
            await eventually(
                lambda expected=turn_id: detector._turns[expected].prediction is not None
            )
            assert (detector.lifecycle is AMDLifecycle.ACTIVE) == (turn_id < expected_limit)
        assert (await detector.execute()).reason == "inference_timeout"


@pytest.mark.parametrize("limit", [0, -1])
def test_max_inference_timeouts_must_be_positive(limit: int) -> None:
    with pytest.raises(ValueError, match="max_inference_timeouts must be positive"):
        AMD(AgentSession(), llm=None, stt=None, max_inference_timeouts=limit)


@pytest.mark.asyncio
async def test_supplied_model_is_not_closed_by_amd() -> None:
    async with running() as (detector, session, classifier, _):
        classifier.aclose = AsyncMock()
        await detector.aclose()
        classifier.aclose.assert_not_called()
        assert session.amd is None


@pytest.mark.asyncio
@pytest.mark.parametrize("track_published", [False, True])
async def test_missing_audio_subscription_times_out_and_releases_session(
    monkeypatch: pytest.MonkeyPatch, track_published: bool
) -> None:
    room = utils.EventEmitter()
    room.isconnected = lambda: True
    publication = SimpleNamespace(
        sid="track", kind=rtc.TrackKind.KIND_AUDIO, subscribed=False, track=None
    )
    participant = SimpleNamespace(identity="callee", track_publications={"track": publication})
    room.remote_participants = {"callee": participant} if track_published else {}
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Hello"))
    activity = session._activity
    assert activity is not None
    session_audio = Mock()
    monkeypatch.setattr(activity._audio_recognition, "_push_audio", session_audio)
    session._room_io = SimpleNamespace(room=room, set_participant=Mock())
    frame = rtc.AudioFrame.create(16000, 1, 320)
    started = asyncio.get_running_loop().time()
    try:
        async with AMD(
            session, llm=None, stt=None, participant_identity="callee", timeout=0.1
        ) as detector:
            await eventually(lambda: bool(room._events.get("track_subscribed")))
            assert detector.lifecycle is AMDLifecycle.PENDING
            assert detector._hard_deadline is None
            assert not activity._authorization_allowed.is_set()
            activity.push_audio(frame)
            session_audio.assert_not_called()

            result = await asyncio.wait_for(detector.execute(), 6)
            assert asyncio.get_running_loop().time() - started == pytest.approx(5, abs=0.01)
            assert result.reason is AMDReason.PARTICIPANT_MISSING
            assert result.category is AMDCategory.UNCERTAIN
            assert detector.lifecycle is AMDLifecycle.FINISHED
            assert session.amd is None
            assert session._turn_hooks is None
            assert activity._authorization_allowed.is_set()
            assert not detector._tasks
            assert not any(room._events.values())
            for emitter, event, handler in detector._subscriptions:
                assert handler not in emitter._events.get(event, ())
            activity.push_audio(frame)
            session_audio.assert_called_once_with(frame, stt_frame=None)
    finally:
        session._room_io = None
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting_for", ["track", "answer"])
@pytest.mark.parametrize("room_disconnected", [False, True])
async def test_disconnect_during_setup_reports_disconnected(
    waiting_for: str, room_disconnected: bool
) -> None:
    room = utils.EventEmitter()
    room.isconnected = Mock(return_value=True)
    publication = SimpleNamespace(
        sid="track",
        kind=rtc.TrackKind.KIND_AUDIO,
        subscribed=waiting_for == "answer",
        track=object() if waiting_for == "answer" else None,
    )
    participant = SimpleNamespace(
        identity="callee",
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        attributes={"sip.callStatus": "ringing"},
        track_publications={"track": publication},
    )
    room.remote_participants = {"callee": participant}
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Hello"))
    session._room_io = SimpleNamespace(room=room, set_participant=Mock())
    try:
        async with AMD(session, llm=None, stt=None, participant_identity="callee") as detector:
            event = (
                "track_subscribed" if waiting_for == "track" else "participant_attributes_changed"
            )
            await eventually(lambda: bool(room._events.get(event)))
            if room_disconnected:
                room.isconnected.return_value = False
                room.emit("connection_state_changed", rtc.ConnectionState.CONN_DISCONNECTED)
            else:
                room.remote_participants.clear()
                room.emit("participant_disconnected", participant)
            assert (await detector.execute()).reason is AMDReason.PARTICIPANT_DISCONNECTED
            assert session.amd is None
            assert session._activity._authorization_allowed.is_set()
            assert not any(room._events.values())
    finally:
        session._room_io = None
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_until_answered", [True, False])
async def test_sip_answer_gating_and_early_media(
    monkeypatch: pytest.MonkeyPatch, wait_until_answered: bool
) -> None:
    from livekit.agents.voice.amd import detector as module

    room = utils.EventEmitter()
    participant = SimpleNamespace(
        identity="callee",
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        track_publications={"track": SimpleNamespace(sid="track")},
    )
    room.remote_participants = {"callee": participant}
    room_io = SimpleNamespace(room=room, set_participant=Mock())
    subscribed = asyncio.Event()
    answered = asyncio.Event()

    async def wait_for_track(**kwargs: Any) -> Any:
        assert kwargs["identity"] == "callee"
        assert kwargs["wait_for_subscription"]
        subscribed.set()
        return participant.track_publications["track"]

    async def wait_for_answer(*args: Any, **kwargs: Any) -> None:
        assert kwargs["attribute"] == "sip.callStatus"
        assert kwargs["value"] == "active"
        await answered.wait()

    monkeypatch.setattr(module, "wait_for_track_publication", wait_for_track)
    answer_mock = AsyncMock(side_effect=wait_for_answer)
    monkeypatch.setattr(module, "wait_for_participant_attribute", answer_mock)
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Hello"))
    session._room_io = room_io
    try:
        async with AMD(
            session,
            llm=None,
            stt=None,
            participant_identity="callee",
            wait_until_answered=wait_until_answered,
        ) as detector:
            await subscribed.wait()
            room_io.set_participant.assert_called_once_with("callee")
            if wait_until_answered:
                await asyncio.sleep(5.1)
                assert detector.lifecycle is AMDLifecycle.PENDING
                assert next_deadline(detector) is None
                answered.set()
            else:
                answer_mock.assert_not_called()
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            room.emit("participant_disconnected", participant)
            assert (await detector.execute()).reason == "participant_disconnected"
            assert detector.lifecycle is AMDLifecycle.FINISHED
    finally:
        session._room_io = None
        await session.aclose()


@pytest.mark.asyncio
async def test_session_close_completes_amd_and_releases_resources() -> None:
    async with running() as (detector, session, classifier, _):
        await session.aclose()
        assert (await detector.execute()).reason == "cancelled"
        assert session.amd is None
        assert not detector._tasks
        assert not detector._tasks


@pytest.mark.asyncio
async def test_dtmf_results_are_collected_once_without_changing_inflight_requests() -> None:
    async with running() as (detector, session, classifier, _):
        first_send = dtmf_executed(session, "12#")
        session.emit("function_tools_executed", first_send)
        assert classifier.requests.empty()
        assert detector._state == AMDCategory.UNCERTAIN
        assert detector._turn_id == 0

        commit_turn(detector, end_of_turn())
        second_send = dtmf_executed(session, "3")
        first = await classifier.request()
        assert dtmf_calls(first.chat_ctx) == first_send.function_calls
        commit_turn(detector, end_of_turn())
        second = await classifier.request()
        assert dtmf_calls(second.chat_ctx) == (
            first_send.function_calls + second_send.function_calls
        )
        assert dtmf_calls(first.chat_ctx) == first_send.function_calls
        assert [item.type for item in second.chat_ctx.items[2:]] == [
            "function_call",
            "function_call_output",
            "message",
            "function_call",
            "function_call_output",
            "message",
        ]


@pytest.mark.asyncio
async def test_amd_collects_only_successful_dtmf_calls_from_its_run() -> None:
    async with running() as (detector, session, classifier, _):
        session.history.add_message(role="user", content="unrelated session transcript")
        session.history.add_message(role="assistant", content="agent response")
        dtmf_executed(session, "1", is_error=True)
        dtmf_executed(session, "2", name="another_tool")
        dtmf_executed(session, "3", created_at=detector._started_at - 1)
        success = dtmf_executed(session, "4")
        assert success.has_tool_reply
        commit_turn(detector, end_of_turn())
        request = await classifier.request()
        assert dtmf_calls(request.chat_ctx) == success.function_calls
        assert request.current_turn.text_content == "hello"
        assert request.earlier_turns == []
        classifier.prediction(1, AMDCategory.HUMAN)
        await detector.execute()
        dtmf_executed(session, "5")
        assert dtmf_calls(detector._chat_ctx) == success.function_calls


@pytest.mark.asyncio
async def test_classifier_sees_that_the_agent_spoke_but_not_its_words() -> None:
    async with running() as (detector, session, classifier, _):
        session._conversation_item_added(
            llm.ChatMessage(
                role="assistant", content=["before AMD"], created_at=detector._started_at - 1
            )
        )
        commit_turn(detector, end_of_turn("Leave a message after the tone."))
        await classifier.request()
        session._conversation_item_added(llm.ChatMessage(role="assistant", content=[""]))
        session._conversation_item_added(
            llm.ChatMessage(role="assistant", content=["Hi Sam, this is Alex."])
        )
        commit_turn(detector, end_of_turn("To erase it and start over, press two."))
        request = await classifier.request()

        history = [(m.role, m.text_content) for m in request.chat_ctx.messages()[2:]]
        assert history == [
            ("user", "Leave a message after the tone."),
            ("assistant", AGENT_REPLY),
            ("user", "To erase it and start over, press two."),
        ]


@pytest.mark.asyncio
async def test_agent_reply_keeps_speaking_order_when_a_turn_commits_during_it() -> None:
    async with running() as (detector, session, classifier, _):
        commit_turn(detector, end_of_turn("Please state your name."))
        first = await classifier.request()
        commit_turn(detector, end_of_turn("Okay."))
        second = await classifier.request()
        assert first.current_turn.created_at < second.current_turn.created_at
        # the reply started between the two turns but is added when its speech ends
        session._conversation_item_added(
            llm.ChatMessage(
                role="assistant",
                content=["This is Alex from Acme."],
                created_at=(first.current_turn.created_at + second.current_turn.created_at) / 2,
            )
        )
        commit_turn(detector, end_of_turn("Please hold."))
        request = await classifier.request()

        history = [(m.role, m.text_content) for m in request.chat_ctx.messages()[2:]]
        assert history == [
            ("user", "Please state your name."),
            ("assistant", AGENT_REPLY),
            ("user", "Okay."),
            ("user", "Please hold."),
        ]


@pytest.mark.asyncio
async def test_classifier_sees_the_agent_words_while_a_human_is_held() -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        await eventually(lambda: detector._turns[1].prediction is not None)
        session._conversation_item_added(
            llm.ChatMessage(role="assistant", content=["Pressing one."])
        )
        commit_turn(detector, end_of_turn("Hello?"))
        await classifier.request()
        classifier.prediction(2, AMDCategory.HUMAN)
        await eventually(lambda: detector._turns[2].prediction is not None)
        session._conversation_item_added(
            llm.ChatMessage(role="assistant", content=["Hi Sam, can you make it tomorrow?"])
        )
        commit_turn(detector, end_of_turn("Yes. Goodbye."))
        request = await classifier.request()

        history = [(m.role, m.text_content) for m in request.chat_ctx.messages()[2:]]
        assert history == [
            ("user", "hello"),
            ("assistant", AGENT_REPLY),
            ("user", "Hello?"),
            ("assistant", "Hi Sam, can you make it tomorrow?"),
            ("user", "Yes. Goodbye."),
        ]


@pytest.mark.asyncio
async def test_dtmf_tool_works_without_amd() -> None:
    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    publisher = AsyncMock()
    tool_session = SimpleNamespace(
        room_io=SimpleNamespace(
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publisher))
        ),
    )
    result = await send_dtmf_events(SimpleNamespace(session=tool_session), [DtmfEvent.ONE])
    publisher.assert_awaited_once_with(code=1, digit="1")
    assert result == llm.ToolResult("Successfully sent DTMF events: 1", reply_required=False)


@pytest.mark.asyncio
async def test_dtmf_publish_failure_does_not_expose_provider_error() -> None:
    import traceback

    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    error = RuntimeError("private provider payload")
    publisher = AsyncMock(side_effect=error)
    tool_session = SimpleNamespace(
        room_io=SimpleNamespace(
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publisher))
        ),
    )
    with pytest.raises(llm.ToolError, match="^Failed to send DTMF events\\.$") as caught:
        await send_dtmf_events(SimpleNamespace(session=tool_session), [DtmfEvent.ONE])
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert "private provider payload" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "error", "cancelled"])
async def test_amd_observes_dtmf_tool_completion_from_session(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from livekit.agents.beta.tools import send_dtmf

    publisher = AsyncMock(
        side_effect=[None, RuntimeError("publish failed")]
        if failure == "error"
        else [None, asyncio.CancelledError()]
        if failure == "cancelled"
        else None
    )
    room = SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publisher))
    monkeypatch.setattr(send_dtmf, "get_job_context", lambda: SimpleNamespace(room=room))
    async with running() as (detector, session, classifier, reply_model):
        executed = []
        session.on("function_tools_executed", executed.append)
        handles: list[SpeechHandle] = []
        session.once("speech_created", lambda ev: handles.append(ev.speech_handle))
        reply_model.fake_response_map[_DEFAULT_IVR_INSTRUCTIONS] = FakeLLMResponse(
            input=_DEFAULT_IVR_INSTRUCTIONS,
            content="",
            ttft=0,
            duration=0,
            tool_calls=[
                llm.FunctionToolCall(
                    name="send_dtmf_events", arguments='{"events":["1","2"]}', call_id="dtmf"
                )
            ],
        )
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        await eventually(lambda: bool(executed))
        await asyncio.wait_for(handles[0], 2)
        assert publisher.await_count == 2
        output = executed[0].function_call_outputs[0]
        assert output.is_error == (failure == "error")
        assert output.reply_required == (failure == "error")
        assert reply_model.calls.qsize() == (2 if failure == "error" else 1)
        assert session.agent_state == "listening"
        if failure == "cancelled":
            assert not output.output
        if failure == "error":
            assert output.output == "Failed to send DTMF events."
        commit_turn(detector, end_of_turn("Connecting you."))
        request = await classifier.request()
        calls = dtmf_calls(request.chat_ctx)
        assert calls == ([] if failure else executed[0].function_calls)
        if not failure:
            assert output.output == "Successfully sent DTMF events: 1, 2"
            assert output in request.chat_ctx.items
            assert output in session.history.items


@pytest.mark.asyncio
async def test_dtmf_tool_propagates_cancelled_publish() -> None:
    from livekit.agents.beta.tools.send_dtmf import send_dtmf_events
    from livekit.agents.beta.workflows.utils import DtmfEvent

    started = asyncio.Event()

    async def publish(**kwargs: Any) -> None:
        started.set()
        await asyncio.Future()

    tool_session = SimpleNamespace(
        room_io=SimpleNamespace(
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publish))
        ),
    )
    task = asyncio.create_task(
        send_dtmf_events(SimpleNamespace(session=tool_session), [DtmfEvent.ONE])
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("session_stt", [False, True])
async def test_classifier_sees_both_transcripts_without_changing_agent_transcript(
    session_stt: bool,
) -> None:
    stt = DrainingSTT()
    async with running(stt=stt, session_options={"stt": FakeSTT() if session_stt else None}) as (
        detector,
        session,
        classifier,
        reply_model,
    ):
        reply_model.fake_response_map["session transcript"] = FakeLLMResponse(
            input="session transcript", content="Hello.", ttft=0, duration=0
        )
        stream = push_audio(detector, stt)
        transcribe(detector, "session transcript")
        stream.send_fake_transcript("AMD transcript")
        await eventually(lambda: detector._resources.stt._current.get("amd") == "AMD transcript")

        info = end_of_turn("session transcript")
        session._activity.on_end_of_turn(info)
        request = await classifier.request()
        assert request.current_turn.text_content == (
            '{"transcript_1": "session transcript", "transcript_2": "AMD transcript"}'
            if session_stt
            else "AMD transcript"
        )
        assert not stream.flushed.is_set()
        classifier.prediction(1, AMDCategory.UNCERTAIN)
        reply = await asyncio.wait_for(reply_model.calls.get(), 2)
        message_contents = [m.text_content for m in reply["chat_ctx"].items if m.type == "message"]
        assert "session transcript" in message_contents
        assert not any("AMD transcript" in (text or "") for text in message_contents)
        assert info.new_transcript == "session transcript"
        await session._activity._user_turn_completed_atask
        await eventually(lambda: session._activity._no_pending_speech)
        assert any(
            m.type == "message" and m.text_content == "session transcript"
            for m in session.current_agent.chat_ctx.items
        )


@pytest.mark.asyncio
async def test_a_late_amd_transcript_does_not_empty_the_turn() -> None:
    # the session committed the turn before AMD's STT delivered its words
    stt = DrainingSTT()
    async with running(stt=stt, session_options={"stt": FakeSTT()}) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        stream.send_fake_transcript("Hi, you've reached Sam.")
        await eventually(
            lambda: detector._resources.stt._current.get("amd") == "Hi, you've reached Sam."
        )
        commit_turn(detector, end_of_turn("Hi, you've reached Sam."))
        first = await classifier.request()
        classifier.prediction(1, AMDCategory.UNCERTAIN)

        commit_turn(detector, end_of_turn("Leave your name and number."))
        second = await classifier.request()
        assert second.current_turn.extra["turn_id"] == 2
        assert (
            second.current_turn.text_content
            == '{"transcript_1": "Leave your name and number.", "transcript_2": ""}'
        )
        assert second.earlier_turns == [first.current_turn]


@pytest.mark.asyncio
async def test_transcript_received_after_eot_is_kept_in_next_request() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        commit_turn(detector, end_of_turn(""))
        await asyncio.sleep(
            _LATE_AMD_FINAL_WAIT + 0.01
        )  # after the wait, a final is the next turn's
        assert detector._turns[1].prediction.transcript == ""
        stream.send_fake_transcript("Please state your name.")
        await eventually(
            lambda: detector._resources.stt._current.get("amd") == "Please state your name."
        )
        assert classifier.requests.empty()
        assert detector._turns[1].prediction.transcript == ""

        commit_turn(detector, end_of_turn(""))
        first = await classifier.request()
        assert first.current_turn.extra["turn_id"] == 2
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        await asyncio.wait_for(detector._should_reply(2, llm.ChatContext()), 2)
        prediction = detector._turns[2].prediction
        assert prediction.transcript == first.current_turn.text_content == "Please state your name."

        stream.send_fake_transcript("Okay, connecting you.")
        await eventually(
            lambda: detector._resources.stt._current.get("amd") == "Okay, connecting you."
        )
        commit_turn(detector, end_of_turn("Okay."))
        second = await classifier.request()
        assert second.current_turn.extra["turn_id"] == 3
        assert second.current_turn.text_content == "Okay, connecting you."
        assert second.earlier_turns == [first.current_turn]


@pytest.mark.asyncio
async def test_without_session_stt_a_turn_is_classified_when_its_final_arrives() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        commit_turn(detector, end_of_turn(""))
        await asyncio.sleep(0.1)
        assert detector._turns[1].prediction is None
        stream.send_fake_transcript("Please state your name.")
        request = await classifier.request()
        assert request.current_turn.extra["turn_id"] == 1
        assert request.current_turn.text_content == "Please state your name."
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._turns[1].prediction is not None)
        assert detector._turns[1].prediction.transcript == "Please state your name."


@pytest.mark.asyncio
@pytest.mark.parametrize("session_stt", [False, True])
async def test_a_final_before_the_classifier_waits_still_belongs_to_the_turn(
    session_stt: bool,
) -> None:
    stt = DrainingSTT()
    session_options = {"stt": FakeSTT()} if session_stt else None
    async with running(stt=stt, session_options=session_options) as (
        detector,
        _,
        classifier,
        _,
    ):
        stream = push_audio(detector, stt)
        commit_turn(detector, end_of_turn("Hello." if session_stt else ""))
        stream.send_fake_transcript("Please state your name.")  # before the task starts
        request = await classifier.request()
        assert request.current_turn.extra["turn_id"] == 1
        assert detector._turns[1].transcript.amd == "Please state your name."


@pytest.mark.asyncio
async def test_a_reused_prediction_is_kept_when_the_turn_final_arrives() -> None:
    # without session STT, turn 2 waits for its final while turn 1 is still classified
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, _, classifier, _):
        events: list[AMDPredictionEvent] = []
        detector.on("amd_prediction", events.append)
        stream = push_audio(detector, stt)
        commit_turn(detector, end_of_turn(""))
        stream.send_fake_transcript("Please state your name.")
        await classifier.request()
        commit_turn(detector, end_of_turn(""))
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._turns[2].prediction is not None)
        assert detector._turns[2].prediction.reason is AMDReason.REUSED

        stream.send_fake_transcript("Why are you calling?")
        await eventually(lambda: detector._turns[2].transcript.amd == "Why are you calling?")
        await asyncio.sleep(0.01)
        assert classifier.requests.empty()
        assert [event.turn_id for event in events] == [1, 2]

        # the next request covers the late text
        commit_turn(detector, end_of_turn(""))
        stream.send_fake_transcript("Hello?")
        request = await classifier.request()
        assert request.current_turn.extra["turn_id"] == 3
        assert [turn.text_content for turn in request.earlier_turns] == [
            "Please state your name.",
            "Why are you calling?",
        ]


@pytest.mark.asyncio
async def test_late_final_after_many_empty_turns_is_committed_at_the_next_eot() -> None:
    stt = DrainingSTT()
    async with running(stt=stt) as (detector, _, classifier, _):
        stream = push_audio(detector, stt)
        dtmf_executed(detector._session, "1")
        commit_turn(detector, end_of_turn(""))
        for _ in range(20):
            commit_turn(detector, end_of_turn(""))
        await asyncio.sleep(
            _LATE_AMD_FINAL_WAIT + 0.01
        )  # after the wait, a final is the next turn's
        assert detector._turn_id == 21
        assert all(detector._turns[turn_id].prediction is not None for turn_id in range(1, 22))
        stream.send_fake_transcript("Hello, can you hear me?")
        await eventually(
            lambda: detector._resources.stt._current.get("amd") == "Hello, can you hear me?"
        )
        assert classifier.requests.empty()
        commit_turn(detector, end_of_turn(""))
        request = await classifier.request()
        assert request.current_turn.extra["turn_id"] == 22
        assert request.current_turn.text_content == "Hello, can you hear me?"
        assert [json.loads(call.arguments) for call in dtmf_calls(request.chat_ctx)] == [
            {"events": ["1"]}
        ]
        assert request.earlier_turns == []
        classifier.prediction(22, AMDCategory.HUMAN)
        assert (await detector.execute()).category == AMDCategory.HUMAN


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", sorted(_fsm.LIKELY_HUMAN_STAGES))
async def test_a_first_human_after_a_menu_or_screener_waits_for_the_next_turn(
    stage: AMDCategory,
) -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        classifier.prediction(1, stage)
        await eventually(lambda: detector._turns[1].prediction is not None)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.HUMAN)
        context = llm.ChatContext()
        assert await hooks.should_reply(context)
        assert not context.items  # no stage instructions while it may be a person
        held = detector._turns[2].prediction
        assert held.category is AMDCategory.UNCERTAIN
        assert held.stage is stage
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        hooks = commit_turn(detector, end_of_turn("Hello, who is this?"))
        request = await classifier.request()
        # the model is told the held turn is likely a person
        assert request.previous_prediction["category"] == _inference.LIKELY_HUMAN
        assert request.stage == stage
        classifier.prediction(3, AMDCategory.HUMAN)
        context = llm.ChatContext()
        assert await hooks.should_reply(context)
        assert context.items[-1].text_content == _DEFAULT_HUMAN_INSTRUCTIONS
        result = await detector.execute()
        assert result.category is AMDCategory.HUMAN
        assert result.last_prediction.turn_id == 3


@pytest.mark.asyncio
async def test_a_held_human_followed_by_a_voicemail_greeting_is_voicemail() -> None:
    async with running() as (detector, session, classifier, _):
        for turn_id, category in enumerate(
            (AMDCategory.MACHINE_IVR, AMDCategory.WAIT, AMDCategory.HUMAN), start=1
        ):
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, category)
            await eventually(lambda t=turn_id: detector._turns[t].prediction is not None)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(4, AMDCategory.MACHINE_VM)
        context = llm.ChatContext()
        assert await hooks.should_reply(context)
        assert context.items[-1].text_content == _DEFAULT_VOICEMAIL_INSTRUCTIONS
        assert detector._state is AMDCategory.MACHINE_VM
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        # a person after the voicemail stage is not held
        await commit(detector, session, classifier)
        classifier.prediction(5, AMDCategory.HUMAN)
        assert (await detector.execute()).category is AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_a_human_at_call_start_completes_at_once() -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.HUMAN)
        assert (await detector.execute()).category is AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_the_classifier_waits_briefly_for_a_late_amd_final() -> None:
    stt = DrainingSTT()
    async with running(stt=stt, session_options={"stt": FakeSTT()}) as (
        detector,
        _,
        classifier,
        _,
    ):
        stream = push_audio(detector, stt)
        commit_turn(detector, end_of_turn("Sam Lee speaking."))
        await asyncio.sleep(0.1)
        assert classifier.requests.empty()
        stream.send_fake_transcript("Sam Lee speaking.")
        request = await classifier.request()
        assert json.loads(request.current_turn.text_content) == {
            "transcript_1": "Sam Lee speaking.",
            "transcript_2": "Sam Lee speaking.",
        }
        assert detector._turns[1].transcript.amd == "Sam Lee speaking."


@pytest.mark.asyncio
async def test_superseded_inference_keeps_the_classification_context() -> None:
    async with running() as (detector, _, classifier, _):
        dtmf_executed(detector._session, "1")
        commit_turn(detector, end_of_turn("Hello, can you hear me?"))
        previous = await classifier.request()
        hooks = commit_turn(detector, end_of_turn("Yes, let's schedule that."))
        request = await classifier.request()
        assert request.earlier_turns == [previous.current_turn]
        assert request.earlier_turns[0].text_content == "Hello, can you hear me?"
        assert request.earlier_turns[0].extra == {"turn_id": 1}
        assert [json.loads(call.arguments) for call in dtmf_calls(request.chat_ctx)] == [
            {"events": ["1"]}
        ]
        assert request.current_turn.text_content == "Yes, let's schedule that."
        assert request.stage == previous.stage == "uncertain"
        assert detector._turns[1].prediction is None
        assert classifier.responses[1].cancelled()
        classifier.prediction(2, AMDCategory.HUMAN)
        assert await hooks.should_reply(llm.ChatContext())
        result = await detector.execute()
        assert result.last_prediction is not None
        assert result.last_prediction.transcript == request.current_turn.text_content


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["prediction", "inference_error", "inference_timeout"])
async def test_empty_turns_do_not_cancel_or_count_against_pending_inference(reason: str) -> None:
    async with running(
        inference_timeout=0.1 if reason == "inference_timeout" else 10, max_uncertain_turns=2
    ) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        first = await commit(detector, session, classifier)
        first_wait = asyncio.create_task(first.should_reply(llm.ChatContext()))
        await asyncio.sleep(0)
        second = commit_turn(detector, end_of_turn(""))
        assert not await asyncio.wait_for(first_wait, 2)
        second_wait = asyncio.create_task(second.should_reply(llm.ChatContext()))
        await asyncio.sleep(0)
        current = commit_turn(detector, end_of_turn(""))
        assert not await asyncio.wait_for(second_wait, 2)
        current_wait = asyncio.create_task(current.should_reply(llm.ChatContext()))
        await asyncio.sleep(0)
        assert not current_wait.done()
        assert detector._turns[1].prediction is None
        assert not classifier.responses[1].done()
        assert detector._turns[2].prediction is None
        if reason != "inference_timeout":
            classifier.respond(
                1, "invalid" if reason == "inference_error" else '{"category":"uncertain"}'
            )
        assert await asyncio.wait_for(current_wait, 2)
        source = detector._turns[1].prediction
        for turn_id in (2, 3):
            reused = detector._turns[turn_id].prediction
            assert reused.turn_id == turn_id
            assert reused.reason == "reused"
            assert reused.category == source.category
            assert reused.stage == source.stage
            assert reused.transcript == ""
        assert [(e.turn_id, e.reason) for e in events] == [
            (1, reason),
            (2, "reused"),
            (3, "reused"),
        ]
        assert classifier.requests.empty()
        final_turn_id = 4 if reason == "prediction" else 5
        for turn_id in range(4, final_turn_id + 1):
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, AMDCategory.UNCERTAIN)
            await asyncio.wait_for(detector._should_reply(turn_id, llm.ChatContext()), 2)
            assert (detector.lifecycle is AMDLifecycle.ACTIVE) == (turn_id < final_turn_id)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "max_uncertain_turns"
        assert result.last_prediction is not None
        assert result.last_prediction.turn_id == final_turn_id


@pytest.mark.asyncio
async def test_dtmf_on_an_empty_turn_is_context_not_an_inference_trigger() -> None:
    async with running() as (detector, _, classifier, _):
        dtmf_executed(detector._session, "1")
        commit_turn(detector, end_of_turn(""))
        assert classifier.requests.empty()
        commit_turn(detector, end_of_turn("Hello."))
        request = await classifier.request()
        assert [json.loads(call.arguments) for call in dtmf_calls(request.chat_ctx)] == [
            {"events": ["1"]}
        ]
        assert request.earlier_turns == []


@pytest.mark.asyncio
async def test_inference_error_falls_back_and_uncertain_keeps_the_stage() -> None:
    async with running() as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await first.should_reply(llm.ChatContext())
        state = detector._state
        for turn_id, arguments, reason in (
            (2, '{"category":"unknown"}', "inference_error"),
            (3, '{"category":"uncertain"}', "prediction"),
            (4, '{"category":"unknown"}', "inference_error"),
        ):
            hooks = await commit(detector, session, classifier)
            classifier.respond(turn_id, arguments)
            ctx = llm.ChatContext()
            assert await hooks.should_reply(ctx)
            assert events[-1].reason == reason
            assert detector._state is state
            assert events[-1].stage is state
            # the stage instructions still apply while the stage is kept
            assert ctx.items[-1].text_content == _DEFAULT_VOICEMAIL_INSTRUCTIONS
            assert events[-1].category is AMDCategory.UNCERTAIN
            if turn_id == 3:
                assert classifier.requests.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("menu", [False, True])
@pytest.mark.parametrize(
    ("failures", "max_retry", "retryable", "succeeds", "expected_attempts"),
    [
        (1, 1, True, True, 2),
        (2, 1, True, False, 2),
        (1, 0, True, False, 1),
        (1, 2, False, False, 1),
    ],
)
async def test_llm_calls_use_session_connection_options(
    monkeypatch: pytest.MonkeyPatch,
    menu: bool,
    failures: int,
    max_retry: int,
    retryable: bool,
    succeeds: bool,
    expected_attempts: int,
) -> None:
    original_run = ControlledStream._run
    attempts = 0

    async def run(stream: ControlledStream) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= failures:
            raise APIConnectionError("provider failure", retryable=retryable)
        await original_run(stream)

    monkeypatch.setattr(ControlledStream, "_run", run)
    async with running() as (detector, session, classifier, _):
        options = APIConnectOptions(max_retry=max_retry, retry_interval=0, timeout=0.25)
        session.conn_options.llm_conn_options = options
        chat = Mock(wraps=classifier.chat)
        monkeypatch.setattr(classifier, "chat", chat)
        if menu:
            menus = []
            detector.on("amd_menu_observed", menus.append)
            task = detector._menu_atask = asyncio.create_task(
                detector._extract_menu(1, "Press 1 for sales.")
            )
            response = await asyncio.wait_for(classifier.menu_requests.get(), 2)
            response.set_result('{"menu":"Sales","options":[{"label":"Sales","dtmf":"1"}]}')
            await task
            assert len(menus) == int(succeeds)
        else:
            hooks = await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
            assert await hooks.should_reply(llm.ChatContext())
            prediction = detector._turns[1].prediction
            assert prediction.reason == ("prediction" if succeeds else "inference_error")
            assert prediction.category == (
                AMDCategory.MACHINE_SCREENING if succeeds else AMDCategory.UNCERTAIN
            )
        assert attempts == expected_attempts
        chat.assert_called_once()
        assert chat.call_args.kwargs["conn_options"] is options


@pytest.mark.asyncio
async def test_prediction_deadline_does_not_wait_for_llm_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = ControlledStream._run
    attempts = 0

    async def run(stream: ControlledStream) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise APIConnectionError("provider failure")
        await original_run(stream)

    monkeypatch.setattr(ControlledStream, "_run", run)
    async with running(inference_timeout=0.02) as (detector, session, classifier, _):
        session.conn_options.llm_conn_options = APIConnectOptions(max_retry=1, retry_interval=0)
        predictions = []
        detector.on("amd_prediction", predictions.append)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.HUMAN)
        assert await hooks.should_reply(llm.ChatContext())
        fallback = detector._turns[1].prediction
        assert fallback.reason == "inference_timeout"
        assert attempts == 1
        await asyncio.sleep(0.2)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._state == AMDCategory.UNCERTAIN
        assert attempts == 1
        assert detector._classifier_atask is None
        assert [event.reason for event in predictions] == ["inference_timeout"]
        assert detector._turns[1].prediction == fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["classification", "prediction", "listener"])
async def test_unexpected_inference_errors_finish_with_internal_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, source: str
) -> None:
    error_type = TypeError if source == "listener" else RuntimeError
    error = error_type("private customer content")
    error.__cause__ = ValueError("private provider payload")
    async with running() as (detector, session, classifier, _):
        if source == "classification":
            monkeypatch.setattr(_inference, "classify", AsyncMock(side_effect=error))
            commit_turn(detector, end_of_turn())
        else:
            if source == "prediction":
                monkeypatch.setattr(_fsm, "transition", Mock(side_effect=error))
            else:
                detector.on("amd_prediction", Mock(side_effect=error))
            await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert (await asyncio.wait_for(detector.execute(), 2)).reason == "internal_error"
        failures = [record for record in caplog.records if record.levelname == "ERROR"]
        assert failures
        assert all(record.exc_info is None for record in failures)
        assert all(record.error_type == error_type.__name__ for record in failures)
        assert "private customer content" not in caplog.text
        assert "private provider payload" not in caplog.text


@pytest.mark.asyncio
async def test_unexpected_menu_error_does_not_end_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract_menu = AsyncMock(side_effect=RuntimeError("menu bug"))
    monkeypatch.setattr(_inference, "extract_ivr_menu", extract_menu)
    async with running() as (detector, session, classifier, _):
        menus = []
        detector.on("amd_menu_observed", menus.append)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert await first.should_reply(llm.ChatContext())
        await eventually(lambda: extract_menu.await_count == 1)
        await asyncio.sleep(0.01)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert menus == []
        await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.HUMAN)
        await confirm_human(detector, classifier)
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.category == AMDCategory.HUMAN
        assert result.reason == "finished"


@pytest.mark.asyncio
async def test_menu_listener_error_ends_detection_with_internal_error() -> None:
    async with running() as (detector, session, classifier, _):

        def fail_listener(_event: object) -> None:
            raise TypeError("listener bug")

        detector.on("amd_menu_observed", fail_listener)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert await first.should_reply(llm.ChatContext())
        menu_response = await asyncio.wait_for(classifier.menu_requests.get(), 2)
        menu_response.set_result(
            '{"menu":"Choose a department","options":[{"label":"Sales","dtmf":"1"}]}'
        )
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "internal_error"
        assert result.category == AMDCategory.MACHINE_IVR


@pytest.mark.asyncio
async def test_slow_menu_does_not_block_classification_or_next_turn() -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert await first.should_reply(llm.ChatContext())
        menu_response = await asyncio.wait_for(classifier.menu_requests.get(), 2)
        assert not menu_response.done()
        await commit(detector, session, classifier)
        assert menu_response.cancelled()
        classifier.prediction(2, AMDCategory.HUMAN)
        await confirm_human(detector, classifier)
        assert (await detector.execute()).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_prediction_listener_next_turn_cancels_the_previous_menu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract_menu = AsyncMock()
    monkeypatch.setattr(_inference, "extract_ivr_menu", extract_menu)
    async with running() as (detector, session, classifier, _):

        def advance_turn(event: AMDPredictionEvent) -> None:
            if event.turn_id == 1:
                commit_turn(detector, end_of_turn("Hello, can you hear me?"))

        detector.on("amd_prediction", advance_turn)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        assert (await classifier.request()).current_turn.extra["turn_id"] == 2
        await asyncio.sleep(0.01)
        extract_menu.assert_not_called()
        classifier.prediction(2, AMDCategory.HUMAN)
        await confirm_human(detector, classifier)
        assert (await asyncio.wait_for(detector.execute(), 2)).category == AMDCategory.HUMAN


@pytest.mark.asyncio
async def test_delayed_deadline_cancels_reply_hold_without_repeating_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice.amd import detector as detector_module

    extract_menu = AsyncMock(return_value=_inference.AMDIVRMenuResponse(menu="", options=[]))
    monkeypatch.setattr(_inference, "extract_ivr_menu", extract_menu)
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2, timeout=2) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        await asyncio.sleep(0.01)
        assert detector._turns[1].prediction.category == AMDCategory.MACHINE_IVR
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        extract_menu.assert_awaited_once()
        state = detector._state
        loop = asyncio.get_running_loop()
        timer = detector._timer
        assert timer is not None
        with monkeypatch.context() as delayed_clock:
            # fire the armed timer as if the loop ran it late
            delayed_clock.setattr(detector_module.time, "monotonic", lambda: loop.time() + 3)
            timer._run()
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.reason == "timeout"
        assert result.category == AMDCategory.MACHINE_IVR
        assert detector._state is state
        assert len(events) == 1
        extract_menu.assert_awaited_once()


@pytest.mark.asyncio
async def test_screening_prediction_is_forwarded_to_session_observability() -> None:
    async with running() as (detector, session, classifier, _):
        host = Mock()
        session._session_host = host
        try:
            hooks = await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
            await hooks.should_reply(llm.ChatContext())
            assert (
                host._on_amd_prediction.call_args.args[0].category == AMDCategory.MACHINE_SCREENING
            )
        finally:
            session._session_host = None


@pytest.mark.asyncio
async def test_split_screening_rollover_retains_all_turns() -> None:
    async with running() as (detector, _, classifier, _):
        history = [
            ("Please state your name and why you are calling.", AMDCategory.MACHINE_SCREENING),
            ("Okay.", AMDCategory.UNCERTAIN),
            ("They can't take the call.", AMDCategory.UNCERTAIN),
            ("Feel free to leave a message.", AMDCategory.MACHINE_VM),
        ]
        for index, (transcript, category) in enumerate(history, start=1):
            commit_turn(detector, end_of_turn(transcript))
            request = await classifier.request()
            assert [e.text_content for e in request.earlier_turns] == [
                prior[0] for prior in history[: index - 1]
            ]
            classifier.prediction(index, category)
            await eventually(lambda turn_id=index: detector._turns[turn_id].prediction is not None)
        assert detector._state == AMDCategory.MACHINE_VM
        assert detector._previous_stage == AMDCategory.MACHINE_SCREENING
        assert detector._turns[3].prediction.category == AMDCategory.UNCERTAIN
        assert detector._turns[3].prediction.stage == AMDCategory.MACHINE_SCREENING


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled_by", ["timeout", "new_turn", "finish"])
@pytest.mark.parametrize(
    "outcome", ["human", "wait", "correction", "provider_error", "unexpected_error"]
)
async def test_classifier_that_ignores_cancellation_cannot_change_state(
    monkeypatch: pytest.MonkeyPatch, cancelled_by: str, outcome: str
) -> None:
    requests = asyncio.Queue()
    cancelled = asyncio.Event()

    async def classify(model, request, **kwargs):
        response = asyncio.get_running_loop().create_future()
        requests.put_nowait(response)
        while True:
            try:
                result = await asyncio.shield(response)
                break
            except asyncio.CancelledError:
                if request.chat_ctx.messages()[-1].extra["turn_id"] != 1:
                    raise
                cancelled.set()
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(_inference, "classify", classify)
    async with running(inference_timeout=0.02) as (detector, _, _, _):
        events = []
        detector.on("amd_prediction", events.append)
        commit_turn(detector, end_of_turn())
        response = await asyncio.wait_for(requests.get(), 2)
        old_task = detector._classifier_atask
        if cancelled_by == "new_turn":
            hooks = commit_turn(detector, end_of_turn("a new prompt"))
            newer_response = await asyncio.wait_for(requests.get(), 2)
        elif cancelled_by == "finish":
            detector._finish(AMDReason.CANCELLED)
        await asyncio.wait_for(cancelled.wait(), 2)
        state = detector._state
        predictions = list(events)
        if outcome == "human":
            response.set_result(_inference.AMDResponse(category=AMDCategory.HUMAN))
        elif outcome == "wait":
            response.set_result(_inference.AMDResponse(category=AMDCategory.WAIT))
        elif outcome == "correction":
            response.set_result(_inference.AMDResponse(category=AMDCategory.MACHINE_SCREENING))
        elif outcome == "provider_error":
            response.set_result(APIConnectionError("stale provider failure"))
        else:
            response.set_result(RuntimeError("stale internal failure"))
        await asyncio.gather(old_task, return_exceptions=True)
        await asyncio.sleep(0)
        assert detector._state == state
        assert events == predictions
        if cancelled_by == "new_turn":
            assert detector._turns[1].prediction is None
            newer_response.set_result(_inference.AMDResponse(category=AMDCategory.MACHINE_VM))
            assert await hooks.should_reply(llm.ChatContext())
            assert detector._state == AMDCategory.MACHINE_VM
        elif cancelled_by == "timeout":
            assert detector._turns[1].prediction.reason == "inference_timeout"
        else:
            assert (await detector.execute()).reason == "cancelled"


@pytest.mark.asyncio
async def test_result_after_deadline_is_ignored_before_timer_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice.amd import detector as detector_module

    async with running(inference_timeout=1) as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        hooks = await commit(detector, session, classifier)
        task = detector._classifier_atask
        loop = asyncio.get_running_loop()
        with monkeypatch.context() as delayed_clock:
            delayed_clock.setattr(detector_module.time, "monotonic", lambda: loop.time() + 2)
            classifier.prediction(1, AMDCategory.HUMAN)
            await task
        assert await hooks.should_reply(llm.ChatContext())
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._state == AMDCategory.UNCERTAIN
        assert [event.reason for event in events] == ["inference_timeout"]


@pytest.mark.asyncio
async def test_prediction_listener_cannot_mutate_saved_decision() -> None:
    async with running() as (detector, session, classifier, _):

        def mutate(event: AMDPredictionEvent) -> None:
            event.category = AMDCategory.HUMAN
            event.transcript = "changed by listener"

        detector.on("amd_prediction", mutate)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await hooks.should_reply(llm.ChatContext())
        prediction = detector._turns[1].prediction
        assert prediction.category == AMDCategory.MACHINE_VM
        assert prediction.transcript == "hello"
        assert detector._state == AMDCategory.MACHINE_VM
        await detector.aclose()
        result = await detector.execute()
        assert result.category == AMDCategory.MACHINE_VM
        assert result.last_prediction is not None
        assert result.last_prediction.transcript == "hello"


@pytest.mark.asyncio
async def test_completed_amd_cannot_be_entered_again() -> None:
    async with running() as (detector, session, _, _):
        await detector.aclose()
        result = await detector.execute()
        with pytest.raises(RuntimeError, match="new AMD instance"):
            await detector.__aenter__()
        assert await detector.execute() is result
        assert session.amd is None
        assert session._turn_hooks is None


@pytest.mark.asyncio
async def test_classifier_history_is_bounded_and_saved_decisions_survive() -> None:
    async with running() as (detector, _, classifier, _):
        commit_turn(detector, end_of_turn("first prompt"))
        await classifier.request()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._should_reply(1, llm.ChatContext())
        first_prediction = detector._turns[1].prediction
        for turn_id in range(2, 24):
            hooks = commit_turn(detector, end_of_turn(f"prompt {turn_id}"))
            await classifier.request()
            classifier.prediction(turn_id, AMDCategory.MACHINE_SCREENING)
            await hooks.should_reply(llm.ChatContext())
        dtmf_executed(detector._session, "2#")
        commit_turn(detector, end_of_turn("latest prompt"))
        request = await classifier.request()
        assert request.current_turn.extra["turn_id"] == 24
        assert [json.loads(call.arguments) for call in dtmf_calls(request.chat_ctx)] == [
            {"events": ["2", "#"]}
        ]
        assert [turn.extra["turn_id"] for turn in request.earlier_turns] == list(range(7, 24))
        assert len(request.chat_ctx.items[2:]) == 20
        assert all(turn.text_content for turn in request.earlier_turns)
        assert detector._turns[1].prediction is first_prediction
        assert first_prediction.category == AMDCategory.MACHINE_SCREENING


@pytest.mark.asyncio
async def test_empty_turns_preserve_transcript_and_dtmf_history() -> None:
    async with running() as (detector, session, classifier, _):
        hooks = commit_turn(detector, end_of_turn("For sales press 1."))
        first = await classifier.request()
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        await hooks.should_reply(llm.ChatContext())
        sent = dtmf_executed(session, "1")
        for _ in range(22):
            commit_turn(detector, end_of_turn(""))
        assert classifier.requests.empty()
        commit_turn(detector, end_of_turn("One moment please."))
        request = await classifier.request()
        assert request.earlier_turns == [first.current_turn]
        assert dtmf_calls(request.chat_ctx) == sent.function_calls
        assert request.current_turn.extra["turn_id"] == 24


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", ["idle", "classification", "silence", "speech"])
async def test_amd_enforces_overall_deadline_independently_of_fsm(pending: str) -> None:
    async with running(timeout=0.3, inference_timeout=1, machine_silence_threshold=1.5) as (
        detector,
        session,
        classifier,
        _,
    ):
        deadline = detector._hard_deadline
        assert deadline is not None
        if pending != "idle":
            await commit(detector, session, classifier)
            if pending != "classification":
                classifier.prediction(1, AMDCategory.MACHINE_VM)
                await eventually(lambda: detector._reply_held_at(asyncio.get_running_loop().time()))
                if pending == "speech":
                    speech_started(detector)
        state = detector._state
        assert not hasattr(state, "next_deadline")
        assert next_deadline(detector) == deadline
        assert detector._hard_deadline == deadline
        result = await asyncio.wait_for(detector.execute(), 1)
        assert result.reason == "timeout"
        assert result.category == (
            AMDCategory.MACHINE_VM if pending in {"silence", "speech"} else AMDCategory.UNCERTAIN
        )
        assert detector._hard_deadline is None
        assert detector._state is state
        assert not detector._reply_held_at(asyncio.get_running_loop().time())
        assert next_deadline(detector) is None
        assert detector._timer is None
        assert session.amd is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bridge", [None, AMDCategory.UNCERTAIN, AMDCategory.WAIT, AMDCategory.MACHINE_IVR]
)
@pytest.mark.parametrize("outcome", ["played", "interrupted", "failed", "no_audio"])
async def test_voicemail_delivery_guard_survives_transitions_and_allows_retry(
    bridge: AMDCategory | None,
    outcome: str,
) -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        handle = SpeechHandle.create()
        first.on_agent_turn_committed(handle)
        turn_id = 1
        if bridge is not None:
            turn_id += 1
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, bridge)
            await detector._should_reply(turn_id, llm.ChatContext())
        turn_id += 1
        overlapping = await commit(detector, session, classifier)
        classifier.prediction(turn_id, AMDCategory.MACHINE_VM)
        assert not await overlapping.should_reply(llm.ChatContext())
        state = detector._state
        if outcome != "no_audio":
            output = session.output.audio
            await output.capture_frame(rtc.AudioFrame.create(24000, 1, 24))
            output.flush()
            await output.wait_for_playout()
        if outcome == "interrupted":
            handle.interrupt()
        handle._mark_done(RuntimeError("playback failed") if outcome == "failed" else None)
        await handle
        await eventually(lambda: detector._voicemail_handle is None)
        assert detector._state is state
        assert detector._voicemail_message_played == (outcome == "played")

        if bridge is not None:
            turn_id += 1
            await commit(detector, session, classifier)
            classifier.prediction(turn_id, bridge)
            await detector._should_reply(turn_id, llm.ChatContext())
        turn_id += 1
        retry = await commit(detector, session, classifier)
        classifier.prediction(turn_id, AMDCategory.MACHINE_VM)
        assert await retry.should_reply(llm.ChatContext()) == (outcome != "played")


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["inference_error", "inference_timeout", "reused"])
@pytest.mark.parametrize("category", [AMDCategory.MACHINE_IVR, AMDCategory.WAIT])
async def test_fallback_and_reuse_leave_fsm_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    category: AMDCategory,
) -> None:
    async with running(inference_timeout=0.1) as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, category)
        assert await first.should_reply(llm.ChatContext()) == (category is not AMDCategory.WAIT)
        state = detector._state
        transition = Mock(wraps=_fsm.transition)
        monkeypatch.setattr(_fsm, "transition", transition)
        if reason == "reused":
            hooks = commit_turn(detector, end_of_turn(""))
        else:
            hooks = await commit(detector, session, classifier)
            if reason == "inference_error":
                classifier.respond(2, "invalid JSON")
        # only an empty turn repeats a wait; a turn without a prediction may be a person
        allowed = category is not AMDCategory.WAIT or reason != "reused"
        assert await hooks.should_reply(llm.ChatContext()) == allowed
        assert detector._turns[2].prediction.reason == reason
        assert detector._state is state
        transition.assert_not_called()


@pytest.mark.asyncio
async def test_valid_prediction_resets_timeout_budget_before_reply_is_allowed() -> None:
    async with running(
        inference_timeout=0.1,
        max_inference_timeouts=2,
        machine_silence_threshold=1.5,
    ) as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        assert await first.should_reply(llm.ChatContext())
        assert detector._inference_timeouts == 1
        await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._turns[2].prediction is not None)
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        assert detector._inference_timeouts == 0
        third = await commit(detector, session, classifier)
        assert await third.should_reply(llm.ChatContext())
        assert detector._inference_timeouts == 1
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        fourth = await commit(detector, session, classifier)
        await fourth.should_reply(llm.ChatContext())
        assert (await detector.execute()).reason == "inference_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge", [AMDCategory.UNCERTAIN, AMDCategory.WAIT])
async def test_human_instructions_remember_machine_history(bridge: AMDCategory) -> None:
    async with running() as (detector, session, classifier, _):
        for turn_id, category in enumerate(
            (AMDCategory.MACHINE_SCREENING, bridge, AMDCategory.HUMAN),
            start=1,
        ):
            hooks = await commit(detector, session, classifier)
            classifier.prediction(turn_id, category)
            ctx = llm.ChatContext()
            assert await hooks.should_reply(ctx) == (category is not AMDCategory.WAIT)
            if category is bridge:
                # wait blocks the reply; uncertain keeps the screening stage and its instructions
                if bridge is AMDCategory.WAIT:
                    assert not ctx.items
                else:
                    assert ctx.items[-1].extra["amd_stage"] == "machine-screening"
        # the first human after screening is likely human, counted as uncertain, with no stage instructions
        assert not ctx.items
        hooks = await confirm_human(detector, classifier)
        ctx = llm.ChatContext()
        assert await hooks.should_reply(ctx)
        assert ctx.items[-1].text_content == _DEFAULT_HUMAN_INSTRUCTIONS
        await detector.execute()
        assert detector._previous_stage == AMDCategory.MACHINE_SCREENING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [None, AMDCategory.MACHINE_SCREENING, AMDCategory.MACHINE_VM, AMDCategory.MACHINE_IVR],
)
async def test_wait_skips_reply_and_allows_a_fresh_prediction(
    monkeypatch: pytest.MonkeyPatch, stage: AMDCategory | None
) -> None:
    async with running(machine_silence_threshold=1.5) as (
        detector,
        session,
        classifier,
        model,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        if stage is not None:
            await commit(detector, session, classifier)
            classifier.prediction(1, stage)
            await detector._should_reply(1, llm.ChatContext())
        previous = detector._state
        previous_stage = detector._previous_stage
        predictions = list(events)
        transition = Mock(wraps=_fsm.transition)
        monkeypatch.setattr(_fsm, "transition", transition)
        activity = session._activity
        activity.on_end_of_turn(end_of_turn("While you wait, hear about our special offers."))
        request = await classifier.request()
        classifier.prediction(request.current_turn.extra["turn_id"], AMDCategory.WAIT)
        await asyncio.wait_for(activity._user_turn_completed_atask, 0.1)

        # wait blocks the reply but keeps the stage
        assert detector._should_wait
        assert detector._state is previous
        assert detector._previous_stage is previous_stage
        assert events[:-1] == predictions
        assert events[-1].category is AMDCategory.WAIT
        assert events[-1].stage is previous
        assert not detector._reply_held_at(asyncio.get_running_loop().time())
        assert detector._voicemail_turn_id is None
        assert model.calls.empty()
        transition.assert_called_once_with(previous, AMDCategory.WAIT)
        await asyncio.sleep(1.6)
        assert model.calls.empty()
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        activity.on_end_of_turn(end_of_turn())
        following = await classifier.request()
        assert following.stage == previous
        assert set(following.recommended_next_categories) == set(_fsm.RECOMMENDED[previous])
        classifier.prediction(following.current_turn.extra["turn_id"], AMDCategory.UNCERTAIN)
        # a kept machine stage still waits for participant silence before replying
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        assert not detector._should_wait
        assert detector._state is previous
        await asyncio.wait_for(model.calls.get(), 2)
        assert transition.call_count == 2
        transition.assert_called_with(previous, AMDCategory.UNCERTAIN)


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
async def test_empty_turns_preserve_wait_until_new_classification(pending: bool) -> None:
    async with running() as (detector, session, classifier, _):
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_IVR)
        await detector._should_reply(1, llm.ChatContext())
        hooks = await commit(detector, session, classifier)
        if pending:
            hooks = commit_turn(detector, end_of_turn(""))
        classifier.prediction(2, AMDCategory.WAIT)
        assert not await asyncio.wait_for(hooks.should_reply(llm.ChatContext()), 0.1)

        for _ in range(2):
            hooks = commit_turn(detector, end_of_turn(""))
            assert not await asyncio.wait_for(hooks.should_reply(llm.ChatContext()), 0.1)
            assert detector._turns[detector._turn_id].prediction.category is AMDCategory.WAIT
        assert detector._should_wait
        assert detector._state is AMDCategory.MACHINE_IVR
        assert classifier.requests.empty()
        hooks = await commit(detector, session, classifier)
        classifier.prediction(detector._turn_id, AMDCategory.HUMAN)
        assert await hooks.should_reply(llm.ChatContext())
        await confirm_human(detector, classifier)
        assert (await detector.execute()).category is AMDCategory.HUMAN


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [AMDReason.TIMEOUT, AMDReason.CANCELLED])
async def test_wait_does_not_block_completion_or_allow_a_reply_afterward(reason: AMDReason) -> None:
    async with running(
        idle_timeout=0.1,
        timeout=0.1 if reason is AMDReason.TIMEOUT else 10,
    ) as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.WAIT)
        assert not await hooks.should_reply(llm.ChatContext())
        if reason is AMDReason.CANCELLED:
            await detector.aclose()
        assert (await asyncio.wait_for(detector.execute(), 0.2)).reason is reason
        assert not await hooks.should_reply(llm.ChatContext())


@pytest.mark.asyncio
async def test_wait_pauses_the_idle_timer_until_the_next_prediction() -> None:
    async with running(idle_timeout=0.1) as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.WAIT)
        assert not await hooks.should_reply(llm.ChatContext())
        await asyncio.sleep(0.3)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._idle_deadline is None
        hooks = commit_turn(detector, end_of_turn(""))
        assert not await hooks.should_reply(llm.ChatContext())
        await asyncio.sleep(0.3)
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        hooks = await commit(detector, session, classifier)
        classifier.prediction(3, AMDCategory.UNCERTAIN)
        assert await hooks.should_reply(llm.ChatContext())
        assert detector._idle_deadline is not None
        assert (await asyncio.wait_for(detector.execute(), 1)).reason is AMDReason.IDLE_TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", ["uncertain", "timeout"])
async def test_wait_resets_inference_timeout_and_uncertain_turn_budgets(prior: str) -> None:
    async with running(inference_timeout=0.05, max_inference_timeouts=2, max_uncertain_turns=2) as (
        detector,
        session,
        classifier,
        _,
    ):
        first = await commit(detector, session, classifier)
        if prior == "uncertain":
            classifier.prediction(1, AMDCategory.UNCERTAIN)
        assert await first.should_reply(llm.ChatContext())
        assert detector._uncertain_turns + detector._inference_timeouts == 1

        hooks = await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.WAIT)
        assert not await hooks.should_reply(llm.ChatContext())
        assert detector._uncertain_turns == detector._inference_timeouts == 0
        following = await commit(detector, session, classifier)
        if prior == "uncertain":
            classifier.prediction(3, AMDCategory.UNCERTAIN)
        # a timeout after a wait is uncertain, so it does not block the reply
        assert await following.should_reply(llm.ChatContext())
        assert detector.lifecycle is AMDLifecycle.ACTIVE


# endregion


# region: Stage corrections


@pytest.mark.parametrize(
    ("initial", "corrected", "transcript", "instructions"),
    [
        (
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_SCREENING,
            "Please state your name and why you are calling.",
            _DEFAULT_SCREENING_INSTRUCTIONS,
        ),
        (
            AMDCategory.MACHINE_IVR,
            AMDCategory.MACHINE_SCREENING,
            "Please state your name and why you are calling.",
            _DEFAULT_SCREENING_INSTRUCTIONS,
        ),
        (
            AMDCategory.MACHINE_SCREENING,
            AMDCategory.MACHINE_IVR,
            "Press 1 for appointments.",
            _DEFAULT_IVR_INSTRUCTIONS,
        ),
    ],
)
@pytest.mark.parametrize("bridge", [None, AMDCategory.WAIT, AMDCategory.UNCERTAIN])
async def test_correction_uses_current_turn_and_persists(
    initial: AMDCategory,
    corrected: AMDCategory,
    transcript: str,
    instructions: str,
    bridge: AMDCategory | None,
) -> None:
    async with running() as (detector, _, classifier, _):
        first = commit_turn(detector, end_of_turn("Please record your name."))
        request = await classifier.request()
        assert request.previous_prediction is None
        assert set(request.recommended_next_categories) == set(AMDCategory)
        classifier.prediction(1, initial)
        assert await first.should_reply(llm.ChatContext())
        saved_prediction = detector._turns[1].prediction.model_copy(deep=True)
        previous_turn = 1
        if bridge is not None:
            hooks = commit_turn(detector, end_of_turn("One moment."))
            await classifier.request()
            classifier.prediction(2, bridge)
            assert await hooks.should_reply(llm.ChatContext()) == (bridge != AMDCategory.WAIT)
            previous_turn = 2

        current_turn = previous_turn + 1
        hooks = commit_turn(detector, end_of_turn(transcript))
        request = await classifier.request()
        assert request.stage == initial
        assert request.previous_prediction == {
            "turn_id": previous_turn,
            "category": bridge or initial,
            "reason": "prediction",
        }
        assert corrected not in request.recommended_next_categories
        classifier.prediction(current_turn, corrected)
        context = llm.ChatContext()
        assert await hooks.should_reply(context)
        assert context.items[-1].text_content == instructions
        prediction = detector._turns[current_turn].prediction
        assert prediction.reason == AMDReason.PREDICTION
        assert prediction.stage is corrected
        assert detector._previous_stage is initial
        assert prediction.inference_duration is not None
        assert detector._turns[1].prediction == saved_prediction

        human = commit_turn(detector, end_of_turn("Hello, Sam speaking."))
        request = await classifier.request()
        assert request.stage == corrected
        assert request.previous_prediction["category"] == corrected
        assert request.previous_prediction["turn_id"] == current_turn
        assert AMDCategory.HUMAN in request.recommended_next_categories
        classifier.prediction(current_turn + 1, AMDCategory.HUMAN)
        assert await human.should_reply(llm.ChatContext())
        if corrected in _fsm.LIKELY_HUMAN_STAGES:
            assert detector._turns[current_turn + 1].prediction.category is AMDCategory.UNCERTAIN
            assert await (await confirm_human(detector, classifier)).should_reply(llm.ChatContext())
        assert (await detector.execute()).category == AMDCategory.HUMAN


async def test_reused_turns_keep_the_corrected_stage() -> None:
    async with running() as (detector, session, classifier, _):
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        correction = await commit(detector, session, classifier)
        empty = commit_turn(detector, end_of_turn(""))
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        assert not await correction.should_reply(llm.ChatContext())
        assert await empty.should_reply(llm.ChatContext())
        assert detector._turns[2].prediction.stage == AMDCategory.MACHINE_SCREENING
        empty = commit_turn(detector, end_of_turn(""))
        assert await empty.should_reply(llm.ChatContext())
        for turn_id in (3, 4):
            prediction = detector._turns[turn_id].prediction
            assert prediction.reason == AMDReason.REUSED
            assert prediction.stage == AMDCategory.MACHINE_SCREENING
        assert classifier.requests.empty()


async def test_correction_preserves_delivered_voicemail_and_dtmf_history() -> None:
    async with running() as (detector, session, classifier, model):
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        await eventually(lambda: detector._voicemail_message_played)
        dtmf_executed(session, "1")

        activity = session._activity
        assert activity is not None
        transcript = "Please state your name and why you are calling."
        assert activity.on_end_of_turn(end_of_turn(transcript))
        request = await classifier.request()
        assert [json.loads(call.arguments) for call in dtmf_calls(request.chat_ctx)] == [
            {"events": ["1"]}
        ]
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert call["chat_ctx"].items[-1].text_content == _DEFAULT_SCREENING_INSTRUCTIONS
        await eventually(lambda: activity._no_pending_speech)

        await commit(detector, session, classifier, reply=True)
        classifier.prediction(3, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        assert detector._voicemail_message_played
        assert model.calls.empty()
        assert detector._turns[3].prediction.stage == AMDCategory.MACHINE_VM


# endregion


# region: Model selection and resource ownership


@pytest.fixture
def cloud_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "wss://test.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")


@pytest.fixture
def model_factories(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    model = ClassifierLLM()
    model.aclose = AsyncMock()
    stt = DrainingSTT()
    stt.aclose = AsyncMock()
    llm_factory = Mock(return_value=model)
    stt_factory = Mock(return_value=stt)
    monkeypatch.setattr(inference.LLM, "from_model_string", llm_factory)
    monkeypatch.setattr(inference.STT, "from_model_string", stt_factory)
    return llm_factory, stt_factory


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", ["project", "inference", "mixed"])
@pytest.mark.parametrize("finish", [False, True])
async def test_default_models_are_auto_selected_and_closed(
    monkeypatch: pytest.MonkeyPatch,
    cloud_credentials: None,
    model_factories: tuple[Mock, Mock],
    credentials: str,
    finish: bool,
) -> None:
    if credentials != "project":
        monkeypatch.delenv("LIVEKIT_API_KEY")
        monkeypatch.setenv("LIVEKIT_INFERENCE_API_KEY", "test-key")
    if credentials == "inference":
        monkeypatch.delenv("LIVEKIT_API_SECRET")
        monkeypatch.setenv("LIVEKIT_INFERENCE_API_SECRET", "test-secret")
    llm_factory, stt_factory = model_factories
    model, stt = llm_factory.return_value, stt_factory.return_value
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        async with AMD(session) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            llm_factory.assert_called_once_with("google/gemma-4-31b-it")
            stt_factory.assert_called_once_with("cartesia/ink-whisper")
            assert detector._llm is model
            assert detector._stt is stt
            stream = push_audio(detector, stt)
            if finish:
                stream.send_fake_transcript("hello")
                await eventually(lambda: detector._resources.stt._current.get("amd") == "hello")
                await commit(detector, session, model)
                model.prediction(1, AMDCategory.HUMAN)
                assert (await detector.execute()).category == AMDCategory.HUMAN
        assert stream._task.done()
        model.aclose.assert_awaited_once()
        stt.aclose.assert_awaited_once()
        assert session.amd is None
        assert session._activity._authorization_allowed.is_set()
    finally:
        await session.aclose()


@pytest.mark.parametrize("missing", ["LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"])
def test_auto_selection_requires_cloud_credentials(
    monkeypatch: pytest.MonkeyPatch,
    cloud_credentials: None,
    model_factories: tuple[Mock, Mock],
    missing: str,
) -> None:
    monkeypatch.delenv(missing)
    detector = AMD(AgentSession())
    assert detector._llm is None
    assert detector._stt is None
    for factory in model_factories:
        factory.assert_not_called()


def test_auto_selection_does_not_use_cloud_models_for_a_local_server(
    monkeypatch: pytest.MonkeyPatch,
    cloud_credentials: None,
    model_factories: tuple[Mock, Mock],
) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    detector = AMD(AgentSession())
    assert detector._llm is None
    assert detector._stt is None
    for factory in model_factories:
        factory.assert_not_called()


@pytest.mark.asyncio
async def test_none_always_inherits_the_active_agent_models(
    cloud_credentials: None, model_factories: tuple[Mock, Mock]
) -> None:
    model = ClassifierLLM()
    model.aclose = AsyncMock()
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment.", llm=model))
    try:
        async with AMD(session, llm=None, stt=None) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            assert detector._resources.llm is model
            assert detector._stt is None
            assert detector._resources.stt._model is None
            await commit(detector, session, model)
            model.prediction(1, AMDCategory.HUMAN)
            assert (await detector.execute()).category == AMDCategory.HUMAN
        model.aclose.assert_not_awaited()
        for factory in model_factories:
            factory.assert_not_called()
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("inherited", ["llm", "stt"])
async def test_model_selection_is_independent_for_each_model(
    cloud_credentials: None, model_factories: tuple[Mock, Mock], inherited: str
) -> None:
    llm_factory, stt_factory = model_factories
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        async with AMD(session, **{inherited: None}) as detector:
            if inherited == "llm":
                llm_factory.assert_not_called()
                stt_factory.assert_called_once_with("cartesia/ink-whisper")
                assert detector._llm is None
            else:
                llm_factory.assert_called_once_with("google/gemma-4-31b-it")
                stt_factory.assert_not_called()
                assert detector._stt is None
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("as_strings", [False, True])
async def test_explicit_models_override_auto_selection(
    cloud_credentials: None, model_factories: tuple[Mock, Mock], as_strings: bool
) -> None:
    llm_factory, stt_factory = model_factories
    model, stt = llm_factory.return_value, stt_factory.return_value
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    detector = AMD(
        session,
        llm="google/gemma-4-31b-it" if as_strings else model,
        stt="cartesia/ink-2" if as_strings else stt,
    )
    for factory in model_factories:
        factory.assert_not_called()
    try:
        async with detector:
            assert detector._llm is model
            assert detector._stt is stt
            assert detector._owns_llm == as_strings
            assert detector._owns_stt == as_strings
            if as_strings:
                llm_factory.assert_called_once_with("google/gemma-4-31b-it")
                stt_factory.assert_called_once_with("cartesia/ink-2")
            else:
                llm_factory.assert_not_called()
                stt_factory.assert_not_called()
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_stt", [None, NOT_GIVEN])
async def test_default_falls_back_to_session_model_without_amd_credentials(
    monkeypatch: pytest.MonkeyPatch, extra_stt: Any
) -> None:
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    model = ClassifierLLM()
    session = AgentSession(llm=model, turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        async with AMD(session, stt=extra_stt) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            assert detector._resources.llm is model
            await commit(detector, session, model)
            model.prediction(1, AMDCategory.HUMAN)
            assert (await detector.execute()).category == AMDCategory.HUMAN
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("session_enabled", [False, True])
@pytest.mark.parametrize("agent_enabled", [False, True, NOT_GIVEN])
async def test_amd_restores_interruption_settings(
    session_enabled: bool, agent_enabled: NotGivenOr[bool]
) -> None:
    agent = Agent(
        instructions="Call about an appointment.",
        turn_handling={"interruption": {"enabled": agent_enabled}}
        if utils.is_given(agent_enabled)
        else {},
    )
    async with running(
        agent=agent,
        session_options={
            "turn_handling": {
                "turn_detection": "manual",
                "interruption": {"enabled": session_enabled},
            }
        },
    ) as (detector, session, classifier, _):
        assert session.options.interruption["enabled"] is True
        assert agent.allow_interruptions is True
        handles = []
        session.on("speech_created", lambda event: handles.append(event.speech_handle))
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await session._activity._user_turn_completed_atask
        assert handles[0].allow_interruptions
        await handles[0]

        await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.HUMAN)
        await detector.execute()
        assert session.options.interruption["enabled"] is session_enabled
        assert agent.allow_interruptions is agent_enabled
        following = session.generate_reply(user_input="hello")
        expected = agent_enabled if utils.is_given(agent_enabled) else session_enabled
        assert following.allow_interruptions is expected
        await following


class ChoiceModel(llm.DecisionModel):
    """A decision model that answers each request with the next queued category."""

    def __init__(self, *kinds: llm.DecisionKind) -> None:
        super().__init__(
            capabilities=llm.DecisionCapabilities(decision_kinds=list(kinds or ("choice",)))
        )
        self.requests: asyncio.Queue[tuple[str, Mapping[str, llm.Decision]]] = asyncio.Queue()
        self.answers: asyncio.Queue[str | None] = asyncio.Queue()

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, llm.Decision],
        conn_options: APIConnectOptions,
    ) -> llm.DecisionResponse:
        self.requests.put_nowait((input, decisions))
        category = await self.answers.get()
        if category is None:
            return llm.DecisionResponse(results={}, errors={"category": "refused"})
        return llm.DecisionResponse(results={"category": llm.ChoiceResult(value=category)})


@pytest.mark.asyncio
async def test_decision_model_classifies_with_one_choice_and_skips_menus() -> None:
    model = ChoiceModel()
    async with running(classifier=model) as (detector, _, _, _):
        menus: list[Any] = []
        detector.on("amd_menu_observed", menus.append)
        commit_turn(detector, end_of_turn("Thanks for calling. For billing, press 1."))
        text, decisions = await asyncio.wait_for(model.requests.get(), 2)
        assert list(decisions) == ["category"]
        choice = decisions["category"]
        assert isinstance(choice, llm.Choice)
        assert choice.prompt == _inference.CLASSIFY_PROMPT
        assert list(choice.options) == [category.value for category in AMDCategory]
        assert not any(choice.options.values())
        assert text.startswith('Request: {"stage": "uncertain"')
        assert text.endswith("\nTranscript:\nuser: Thanks for calling. For billing, press 1.")

        model.answers.put_nowait("machine-ivr")
        await eventually(lambda: detector._state is AMDCategory.MACHINE_IVR)
        assert detector._menu_atask is None

        commit_turn(detector, end_of_turn("For billing, press 1."))
        text, _ = await asyncio.wait_for(model.requests.get(), 2)
        assert text.startswith('Request: {"stage": "machine-ivr"')
        assert "user: Thanks for calling. For billing, press 1.\nuser: For billing" in text
        model.answers.put_nowait("machine-unavailable")
        result = await asyncio.wait_for(detector.execute(), 2)
        assert result.category is AMDCategory.MACHINE_UNAVAILABLE
        assert not menus


@pytest.mark.asyncio
async def test_decision_model_error_is_an_inference_error() -> None:
    model = ChoiceModel()
    async with running(classifier=model) as (detector, _, _, _):
        predictions: list[Any] = []
        detector.on("amd_prediction", predictions.append)
        commit_turn(detector, end_of_turn("Hello?"))
        await asyncio.wait_for(model.requests.get(), 2)
        model.answers.put_nowait(None)
        await eventually(lambda: len(predictions) == 1)
        assert predictions[0].reason is AMDReason.INFERENCE_ERROR
        assert predictions[0].category is AMDCategory.UNCERTAIN


@pytest.mark.asyncio
async def test_decision_model_without_choice_is_rejected() -> None:
    session = AgentSession(turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        with pytest.raises(ValueError, match="choice decisions"):
            await AMD(session, llm=ChoiceModel("probability"), stt=None).__aenter__()
        assert session.amd is None
        assert session._turn_hooks is None
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_missing_llm_does_not_install_turn_hooks() -> None:
    session = AgentSession(turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        with pytest.raises(ValueError, match="requires an LLM"):
            await AMD(session, llm=None, stt=None).__aenter__()
        assert session.amd is None
        assert session._turn_hooks is None
        assert session._activity._authorization_allowed.is_set()
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_rejected_entry_does_not_create_owned_models(
    model_factories: tuple[Mock, Mock], started: bool
) -> None:
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    if started:
        await session.start(Agent(instructions="Call about an appointment."))
        session.options.ivr_detection = True
    detector = AMD(session, llm="google/gemini-3.1-flash-lite", stt="cartesia/ink-whisper")
    try:
        with pytest.raises((ValueError, RuntimeError)):
            await detector.__aenter__()
        for factory in model_factories:
            factory.assert_not_called()
        assert session.amd is None
        assert session._turn_hooks is None
    finally:
        await detector.aclose()
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["factory", "nonstreaming"])
async def test_partial_model_setup_closes_owned_models(
    model_factories: tuple[Mock, Mock], failure: str
) -> None:
    llm_factory, stt_factory = model_factories
    if failure == "factory":
        stt_factory.side_effect = ValueError("STT setup failed")
    else:
        stt_factory.return_value._capabilities.streaming = False
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    detector = AMD(session, llm="google/gemini-3.1-flash-lite", stt="cartesia/ink-whisper")
    try:
        with pytest.raises(ValueError):
            await detector.__aenter__()
        await detector.aclose()
        llm_factory.return_value.aclose.assert_awaited_once()
        if failure == "nonstreaming":
            stt_factory.return_value.aclose.assert_awaited_once()
        else:
            stt_factory.return_value.aclose.assert_not_awaited()
        assert session.amd is None
        assert session._turn_hooks is None
        assert session._activity._authorization_allowed.is_set()
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["pause", "participant", "session_on", "room_on", "listening"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_setup_failure_cleans_up_amd(
    monkeypatch: pytest.MonkeyPatch,
    model_factories: tuple[Mock, Mock],
    failure: str,
    error_type: type[BaseException],
) -> None:
    session = AgentSession(
        llm=FakeLLM(),
        turn_handling={"turn_detection": "manual", "interruption": {"enabled": False}},
    )
    agent = Agent(
        instructions="Call about an appointment.",
        turn_handling={"interruption": {"enabled": False}},
    )
    await session.start(agent)
    activity = session._activity
    assert activity is not None
    room = utils.EventEmitter()
    room_io = SimpleNamespace(room=room, set_participant=Mock())
    session._room_io = room_io
    detector = AMD(
        session,
        llm="google/gemma-4-31b-it",
        stt="cartesia/ink-2",
        participant_identity="callee",
    )
    target, method = {
        "pause": (activity, "_pause_authorization"),
        "participant": (room_io, "set_participant"),
        "session_on": (session, "on"),
        "room_on": (room, "on"),
        "listening": (asyncio.get_running_loop(), "create_task"),
    }[failure]
    original = getattr(target, method)
    error = error_type("setup failed")

    def fail_setup(*args: Any, **kwargs: Any) -> Any:
        if failure == "listening":
            if args[0].cr_code is AMD._setup_listening.__code__:
                raise error
            return original(*args, **kwargs)
        result = original(*args, **kwargs)
        if failure == "session_on" and args[0] != "speech_created":
            return result
        raise error

    monkeypatch.setattr(target, method, fail_setup)
    try:
        with pytest.raises(error_type, match="setup failed") as exc_info:
            await detector.__aenter__()
        assert exc_info.value is error
        assert session.amd is None
        assert session._turn_hooks is None
        assert activity._authorization_allowed.is_set()
        assert session.options.interruption["enabled"] is False
        assert agent.allow_interruptions is False
        assert detector.lifecycle is AMDLifecycle.FINISHED
        assert not detector._tasks
        for emitter, event, handler in detector._subscriptions:
            assert handler not in emitter._events.get(event, ())
        for factory in model_factories:
            factory.return_value.aclose.assert_awaited_once()
    finally:
        await detector.aclose()
        session._room_io = None
        await session.aclose()


@pytest.mark.asyncio
async def test_stt_model_string_uses_the_standard_factory_and_closes_owned_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents import inference

    model = DrainingSTT()
    model.aclose = AsyncMock()
    factory = Mock(return_value=model)
    monkeypatch.setattr(inference.STT, "from_model_string", factory)
    async with running(stt="cartesia/ink-2") as (detector, _, _, _):
        stream = push_audio(detector, model)
        factory.assert_called_once_with("cartesia/ink-2")
        await detector.aclose()
        assert stream._task.done()
    model.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_closes_the_run_stream_and_does_not_close_supplied_stt() -> None:
    stt = DrainingSTT()
    stt.aclose = AsyncMock()
    async with running(stt=stt) as (detector, session, classifier, _):
        stream = push_audio(detector, stt)
        stream.send_fake_transcript("hello")
        await eventually(lambda: detector._resources.stt._current.get("amd") == "hello")
        await commit(detector, session, classifier)
        assert push_audio(detector, stt) is stream
        await asyncio.sleep(0)
        await detector.aclose()
        assert stream._task.done()
        assert len(stt.streams) == 1
        assert not detector._tasks
        assert session.amd is None
        stt.aclose.assert_not_called()


@pytest.mark.asyncio
async def test_owned_model_cleanup_failure_still_detaches_and_releases_turn_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents import inference

    stt = DrainingSTT()
    stt.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
    monkeypatch.setattr(inference.STT, "from_model_string", Mock(return_value=stt))
    async with running(stt="cartesia/ink-2") as (detector, session, classifier, _):
        stream = push_audio(detector, stt)
        stream.send_fake_transcript("hello")
        await eventually(lambda: detector._resources.stt._current.get("amd") == "hello")
        await commit(detector, session, classifier)
        await asyncio.wait_for(detector.aclose(), 2)
        assert (await detector.execute()).reason == "cancelled"
        assert session.amd is None
        assert session._activity._authorization_allowed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE])
async def test_finish_detaches_before_task_cleanup_and_preserves_a_new_run(
    monkeypatch: pytest.MonkeyPatch, category: AMDCategory
) -> None:
    async with running() as (detector, session, classifier, _):
        activity = session._activity
        assert activity is not None
        started = asyncio.Event()
        cancelling = asyncio.Event()
        release = asyncio.Event()

        async def slow_cancellation() -> None:
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelling.set()
                await release.wait()

        detector._spawn(slow_cancellation())
        await started.wait()
        terminal_session_state = []
        detector.on(
            "amd_prediction",
            lambda _: terminal_session_state.append(
                (session.amd, session._turn_hooks, activity._authorization_allowed.is_set())
            ),
        )
        try:
            await commit(detector, session, classifier)
            classifier.prediction(1, category)
            await asyncio.wait_for(cancelling.wait(), 2)
            assert terminal_session_state == [(None, None, True)]
            assert session.current_agent.allow_interruptions is NOT_GIVEN
            assert not detector._resources.completion.done()

            session_audio = Mock()
            amd_audio = Mock(wraps=detector.push_audio)
            monkeypatch.setattr(activity._audio_recognition, "_push_audio", session_audio)
            monkeypatch.setattr(detector, "push_audio", amd_audio)
            frame = rtc.AudioFrame.create(16000, 1, 320)
            activity.push_audio(frame)
            session_audio.assert_called_once_with(frame, stt_frame=None)
            amd_audio.assert_not_called()

            following = AMD(session, llm=ClassifierLLM(), stt=None)

            async def wait_to_listen() -> None:
                await asyncio.Future()

            monkeypatch.setattr(following, "_setup_listening", wait_to_listen)
            async with following:
                assert following.lifecycle is AMDLifecycle.PENDING
                cancel_pending = Mock(wraps=activity._cancel_pending_speeches)
                cancel_preemptive = Mock(wraps=activity._cancel_preemptive_generation)
                monkeypatch.setattr(activity, "_cancel_pending_speeches", cancel_pending)
                monkeypatch.setattr(activity, "_cancel_preemptive_generation", cancel_preemptive)
                release.set()
                assert (await asyncio.wait_for(detector.execute(), 2)).category == category
                assert session.amd is following
                assert session._turn_hooks is following._turn_hooks
                assert not activity._authorization_allowed.is_set()
                assert session.current_agent.allow_interruptions is True
                cancel_pending.assert_not_called()
                cancel_preemptive.assert_not_called()
            assert session.current_agent.allow_interruptions is NOT_GIVEN
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_target", ["run_stt", "owned_stt", "owned_llm"])
@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE])
async def test_slow_resource_close_does_not_keep_amd_attached(
    monkeypatch: pytest.MonkeyPatch,
    model_factories: tuple[Mock, Mock],
    close_target: str,
    category: AMDCategory,
) -> None:
    llm_factory, stt_factory = model_factories
    classifier, stt_model = llm_factory.return_value, stt_factory.return_value
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        async with AMD(
            session,
            llm="google/gemma-4-31b-it",
            stt="cartesia/ink-2",
            machine_silence_threshold=0,
        ) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            stream = push_audio(detector, stt_model)
            target = {
                "run_stt": detector._resources.stt,
                "owned_stt": stt_model,
                "owned_llm": classifier,
            }[close_target]
            close_started = asyncio.Event()
            release_close = asyncio.Event()
            original_close = target.aclose

            async def slow_close() -> None:
                close_started.set()
                await release_close.wait()
                await original_close()

            stream.send_fake_transcript("hello")
            await eventually(lambda: detector._resources.stt._current.get("amd") == "hello")
            hooks = await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.MACHINE_VM)
            assert await hooks.should_reply(llm.ChatContext())
            speech = SpeechHandle.create()
            session.emit(
                "speech_created",
                SpeechCreatedEvent(
                    speech_handle=speech, user_initiated=False, source="generate_reply"
                ),
            )
            hooks.on_agent_turn_committed(speech)
            remove_callback = Mock(wraps=speech.remove_done_callback)
            monkeypatch.setattr(speech, "remove_done_callback", remove_callback)
            activity = session._activity
            assert activity is not None
            cancel_pending = Mock(wraps=activity._cancel_pending_speeches)
            cancel_preemptive = Mock(wraps=activity._cancel_preemptive_generation)
            monkeypatch.setattr(activity, "_cancel_pending_speeches", cancel_pending)
            monkeypatch.setattr(activity, "_cancel_preemptive_generation", cancel_preemptive)
            completed_events = []
            detector.on("amd_completed", completed_events.append)
            completion = asyncio.create_task(detector.execute())
            try:
                monkeypatch.setattr(target, "aclose", AsyncMock(side_effect=slow_close))
                stream.send_fake_transcript("hello")
                await eventually(lambda: detector._resources.stt._current.get("amd") == "hello")
                await commit(detector, session, classifier)
                classifier.prediction(2, category)
                await asyncio.wait_for(close_started.wait(), 2)

                assert session.amd is None
                assert session._turn_hooks is None
                assert activity._authorization_allowed.is_set()
                for emitter, event, handler in detector._subscriptions:
                    assert handler not in emitter._events.get(event, ())
                remove_callback.assert_any_call(detector._on_speech_done)
                remove_callback.assert_any_call(detector._on_voicemail_done)
                assert cancel_pending.called == (category == AMDCategory.MACHINE_UNAVAILABLE)
                cancel_preemptive.assert_called()
                assert not completion.done()
                assert completed_events == []
            finally:
                release_close.set()
                result = await asyncio.wait_for(completion, 2)
                speech._mark_done()
            assert result.category == category
            assert result.reason == "finished"
            assert completed_events == [result]
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_nonstreaming_stt_is_rejected() -> None:
    stt = FakeSTT()
    stt._capabilities.streaming = False
    with pytest.raises(ValueError, match="streaming STT"):
        AMD(AgentSession(), llm=None, stt=stt)


# endregion


# region: Machine silence wait and idle timeouts


def playback_allowed(session: AgentSession) -> bool:
    assert session._activity is not None
    return session._activity._authorization_allowed.is_set()


async def playback_released(session: AgentSession) -> None:
    assert session._activity is not None
    await session._activity._authorization_allowed.wait()


def test_machine_silence_default_and_validation() -> None:
    assert AMD(AgentSession(), llm=None, stt=None)._options.machine_silence_threshold == 1.5
    with pytest.raises(ValueError, match="machine_silence_threshold"):
        AMD(AgentSession(), machine_silence_threshold=-1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category",
    [
        AMDCategory.MACHINE_SCREENING,
        AMDCategory.MACHINE_VM,
        AMDCategory.MACHINE_IVR,
        AMDCategory.MACHINE_UNAVAILABLE,
    ],
)
async def test_machine_prediction_updates_state_before_reply_silence(category: AMDCategory) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.3) as (
        detector,
        session,
        classifier,
        model,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier, reply=True)
        await asyncio.sleep(0.2)
        classifier.prediction(1, category)
        await asyncio.sleep(0.7)
        assert len(events) == 1
        assert events[0].category == category
        assert events[0].reason == "prediction"
        assert events[0].delay == pytest.approx(0.2, abs=0.01)
        assert events[0].inference_duration == pytest.approx(0.2, abs=0.01)
        assert detector._state == category
        assert detector._turns[1].prediction is not None
        if category == AMDCategory.MACHINE_UNAVAILABLE:
            assert model.calls.empty()
            assert (await detector.execute()).category == category
            return
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        assert not model.calls.empty()
        assert not playback_allowed(session)
        assert next_deadline(detector) - asyncio.get_running_loop().time() == pytest.approx(
            0.1, abs=0.01
        )
        await asyncio.sleep(0.11)
        assert len(events) == 1
        assert playback_allowed(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.UNCERTAIN])
async def test_human_and_initial_uncertain_do_not_wait(category: AMDCategory) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        speech_started(detector)
        speech_ended(detector, 0)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, category)
        assert await hooks.should_reply(llm.ChatContext())
        assert detector._turns[1].prediction.category == category
        assert detector._turns[1].prediction.delay < 0.01


@pytest.mark.asyncio
async def test_elapsed_silence_and_slow_inference_do_not_add_another_wait() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=3) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 0.5)
        hooks = await commit(detector, session, classifier)
        await asyncio.sleep(1.2)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert await hooks.should_reply(llm.ChatContext())
        assert detector._turns[1].prediction.delay == pytest.approx(1.2, abs=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["prediction", "inference_timeout", "inference_error", "reused"])
async def test_machine_fallbacks_and_uncertain_keep_the_silence_wait(reason: str) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert await first.should_reply(llm.ChatContext())
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        if reason == "reused":
            hooks = commit_turn(detector, end_of_turn(""))
        else:
            hooks = await commit(detector, session, classifier)
            if reason == "prediction":
                classifier.prediction(2, AMDCategory.UNCERTAIN)
            elif reason == "inference_error":
                classifier.respond(2, "invalid JSON")
        ctx = llm.ChatContext()
        reply = asyncio.create_task(hooks.should_reply(ctx))
        await asyncio.sleep(0.9)
        prediction = detector._turns[2].prediction
        assert prediction is not None
        assert len(events) == 1
        assert prediction.reason == reason
        # uncertain keeps the screening stage, so its reply waits for silence too;
        # a turn without a prediction is uncertain, and an empty turn repeats the last
        assert prediction.category == (
            AMDCategory.MACHINE_SCREENING if reason == "reused" else AMDCategory.UNCERTAIN
        )
        assert prediction.stage == AMDCategory.MACHINE_SCREENING
        assert await reply
        assert ctx.items[-1].extra["amd_stage"] == "machine-screening"
        assert not playback_allowed(session)
        await asyncio.sleep(0.11)
        assert playback_allowed(session)
        assert classifier.requests.empty()


@pytest.mark.asyncio
async def test_resumed_speech_invalidates_release_and_preserves_history() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.sleep(0.5)
        speech_started(detector)
        await asyncio.sleep(1)
        speech_ended(detector, 0.5)
        await asyncio.sleep(0.5)
        assert [e.category for e in events] == [AMDCategory.MACHINE_VM]
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        assert detector._turns[1].prediction is not None
        commit_turn(detector, end_of_turn("Hello, can you hear me?"))
        request = await classifier.request()
        assert request.earlier_turns[0].text_content == "hello"
        assert request.stage == AMDCategory.MACHINE_VM
        assert AMDCategory.MACHINE_SCREENING not in request.recommended_next_categories
        assert not await hooks.should_reply(llm.ChatContext())
        classifier.prediction(2, AMDCategory.HUMAN)
        assert (await detector.execute()).category == AMDCategory.HUMAN
        assert [e.category for e in events] == [AMDCategory.MACHINE_VM, AMDCategory.HUMAN]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["vad", "stt"])
async def test_untranscribed_speech_rearms_the_committed_prediction(source: str) -> None:
    classifier = ClassifierLLM()
    model = RecordingLLM()
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        tts=FakeTTS(fake_audio_duration=0.05),
        vad=FakeVAD() if source == "vad" else None,
        turn_handling={"turn_detection": "vad" if source == "vad" else "stt"},
        aec_warmup_duration=None,
    )
    session.output.audio = FakeAudioOutput()
    await session.start(CustomerAgent())
    try:
        async with AMD(
            session,
            llm=classifier,
            stt=None,
            machine_silence_threshold=1.5,
            inference_timeout=3,
        ) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            recognition = session._activity._audio_recognition
            if source == "vad":
                await asyncio.wait_for(recognition._vad_atask, 2)
            events = []
            speeches = []
            detector.on("amd_prediction", events.append)
            session.on("speech_created", lambda event: speeches.append(event.speech_handle))
            await commit(detector, session, classifier, reply=True)
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
            await asyncio.wait_for(model.calls.get(), 2)
            await asyncio.sleep(0.5)
            assert detector._turns[1].prediction is not None
            assert detector._reply_held_at(asyncio.get_running_loop().time())
            assert not playback_allowed(session)
            assert next_deadline(detector) is not None
            assert next_deadline(detector) - asyncio.get_running_loop().time() == pytest.approx(
                1.0, abs=0.01
            )

            if source == "vad":
                await recognition._on_vad_event(
                    vad.VADEvent(
                        type=vad.VADEventType.START_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=0,
                        silence_duration=0,
                    )
                )
            else:
                recognition._process_stt_event(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )
            await asyncio.sleep(1.1)
            assert len(events) == 1
            assert detector._reply_held_at(asyncio.get_running_loop().time())
            assert not playback_allowed(session)
            assert session.agent_state != "speaking"

            if source == "vad":
                await recognition._on_vad_event(
                    vad.VADEvent(
                        type=vad.VADEventType.END_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=0.6,
                        silence_duration=0.5,
                    )
                )
            else:
                recognition._process_stt_event(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        speech_end_time=time.time() - 0.5,
                    )
                )
            await asyncio.sleep(0.9)
            assert len(events) == 1
            assert detector._reply_held_at(asyncio.get_running_loop().time())
            assert not playback_allowed(session)
            assert session.agent_state != "speaking"
            await asyncio.sleep(0.11)
            assert playback_allowed(session)
            assert [(event.turn_id, event.category, event.transcript) for event in events] == [
                (1, AMDCategory.MACHINE_SCREENING, "hello")
            ]
            assert detector._turn_id == 1
            assert classifier.requests.empty()
            assert len(speeches) == 1
            await asyncio.wait_for(speeches[0], 2)
            assert not speeches[0].interrupted
            assert model.calls.empty()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_late_human_cannot_replace_a_held_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    original_classify = _inference.classify
    response = asyncio.get_running_loop().create_future()
    cancelled = asyncio.Event()

    async def classify(model, request, **kwargs):
        if request.chat_ctx.messages()[-1].extra["turn_id"] != 2:
            return await original_classify(model, request, **kwargs)
        while True:
            try:
                return await asyncio.shield(response)
            except asyncio.CancelledError:
                cancelled.set()

    monkeypatch.setattr(_inference, "classify", classify)
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await first.should_reply(llm.ChatContext())
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0)
        commit_turn(detector, end_of_turn())
        task = detector._classifier_atask
        try:
            await asyncio.wait_for(cancelled.wait(), 0.3)
            assert not response.done()
            assert detector._reply_held_at(asyncio.get_running_loop().time())
            assert [event.reason for event in events] == ["inference_timeout"]
            response.set_result(_inference.AMDResponse(category=AMDCategory.HUMAN))
            await task
            await asyncio.sleep(1.31)
            assert detector.lifecycle is AMDLifecycle.ACTIVE
            assert detector._state == AMDCategory.MACHINE_VM
            assert [event.category for event in events] == [AMDCategory.UNCERTAIN]
            assert events[0].reason == "inference_timeout"
        finally:
            if not response.done():
                response.set_result(_inference.AMDResponse(category=AMDCategory.HUMAN))


@pytest.mark.asyncio
async def test_late_machine_prediction_cannot_replace_an_empty_turn_hold() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2) as (
        detector,
        session,
        classifier,
        model,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        assert await first.should_reply(llm.ChatContext())
        handle = SpeechHandle.create()
        first.on_agent_turn_committed(handle)
        output = session.output.audio
        await output.capture_frame(rtc.AudioFrame.create(24000, 1, 24))
        output.flush()
        await output.wait_for_playout()
        handle._mark_done()
        await asyncio.wait_for(handle, 2)

        speech_started(detector)
        speech_ended(detector, 1.5)
        second = await commit(detector, session, classifier)
        assert not await second.should_reply(llm.ChatContext())
        fallback = detector._turns[2].prediction
        assert fallback is not None
        assert fallback.reason == "inference_timeout"
        fallback_snapshot = fallback.model_dump()

        speech_started(detector)
        speech_ended(detector, 0)
        activity = session._activity
        assert activity is not None
        assert activity.on_end_of_turn(end_of_turn(""))
        reply_task = activity._user_turn_completed_atask
        assert reply_task is not None
        release_at = next_deadline(detector)
        assert release_at is not None
        assert detector._turn_id == 3
        assert detector._turns[3].prediction.reason == "reused"
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        await asyncio.sleep(0.4)
        classifier.prediction(2, AMDCategory.MACHINE_UNAVAILABLE)
        await asyncio.sleep(0.1)
        assert next_deadline(detector) == release_at
        assert detector._turns[2].prediction is fallback
        assert fallback.model_dump() == fallback_snapshot

        await asyncio.sleep(release_at - asyncio.get_running_loop().time() - 0.1)
        assert model.calls.empty()
        assert [(event.turn_id, event.reason) for event in events] == [
            (1, "prediction"),
            (2, "inference_timeout"),
            (3, "reused"),
        ]
        await asyncio.sleep(0.11)
        assert detector._state == AMDCategory.MACHINE_VM
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._turns[3].prediction is not None
        assert detector._turns[2].prediction is fallback
        assert fallback.model_dump() == fallback_snapshot
        assert [(event.turn_id, event.reason) for event in events] == [
            (1, "prediction"),
            (2, "inference_timeout"),
            (3, "reused"),
        ]
        await asyncio.wait_for(reply_task, 2)
        assert model.calls.empty()
        assert classifier.requests.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 3, 4])
async def test_inference_timeout_limit_waits_for_silence_before_completion(limit: int) -> None:
    async with running(
        machine_silence_threshold=1.5, inference_timeout=0.2, max_inference_timeouts=limit
    ) as (
        detector,
        session,
        classifier,
        _,
    ):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await first.should_reply(llm.ChatContext())
        for turn_id in range(2, limit + 2):
            speech_started(detector)
            speech_ended(detector, 0.5)
            await commit(detector, session, classifier)
            await asyncio.sleep(0.3)
            assert detector.lifecycle is AMDLifecycle.ACTIVE, (
                session._recorded_events,
                detector._completion(),
            )
            assert detector._turns[turn_id].prediction.reason == "inference_timeout"
            assert detector._reply_held_at(asyncio.get_running_loop().time())
            await asyncio.sleep(0.71)
            assert detector._turns[turn_id].prediction is not None
        assert (await detector.execute()).reason == "inference_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("inference_ready", [True, False])
async def test_empty_eot_after_resumed_speech_rearms_a_useful_prediction(
    inference_ready: bool,
) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=3) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier)
        if inference_ready:
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.2)
        speech_started(detector)
        speech_ended(detector, 0.5)
        commit_turn(detector, end_of_turn(""))
        if not inference_ready:
            classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        # an empty turn reuses the prediction and emits its own reused event,
        # whether it was committed after the prediction or while it was pending
        expected = [(1, "prediction"), (2, "reused")]
        await asyncio.sleep(0.9)
        assert [(e.turn_id, e.reason) for e in events] == expected
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        await asyncio.sleep(0.11)
        assert [(e.turn_id, e.reason) for e in events] == expected
        assert detector._turns[2].prediction is not None
        assert classifier.requests.empty()


@pytest.mark.asyncio
async def test_empty_turn_does_not_delay_the_next_prediction() -> None:
    stt = DrainingSTT()
    async with running(stt=stt, machine_silence_threshold=1.5) as (detector, _, classifier, _):
        events = []
        detector.on("amd_prediction", events.append)
        speech_started(detector)
        stream = push_audio(detector, stt)
        speech_ended(detector, 0)
        commit_turn(detector, end_of_turn(""))
        await asyncio.sleep(_LATE_AMD_FINAL_WAIT + 0.01)  # no AMD final arrives for the empty turn
        assert [(e.turn_id, e.reason) for e in events] == [(1, "reused")]
        assert detector._turns[1].prediction.reason == "reused"

        await asyncio.sleep(0.1)
        speech_started(detector)
        transcript = "Please state your name and why you are calling."
        stream.send_fake_transcript(transcript)
        await eventually(lambda: detector._resources.stt._current.get("amd") == transcript)
        speech_ended(detector, 0)
        info = end_of_turn(transcript)
        hooks = commit_turn(detector, info)
        assert (await classifier.request()).current_turn.extra["turn_id"] == 2
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)

        await asyncio.sleep(0.5)
        assert len(events) == 2
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        await asyncio.sleep(1.01)
        assert [(event.turn_id, event.category) for event in events] == [
            (1, AMDCategory.UNCERTAIN),
            (2, AMDCategory.MACHINE_SCREENING),
        ]
        assert not stream.flushed.is_set()
        assert await hooks.should_reply(llm.ChatContext())


@pytest.mark.asyncio
async def test_new_inference_discards_stale_turn_waits() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        speech_started(detector)
        speech_ended(detector, 1.5)
        first = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await first.should_reply(llm.ChatContext())
        speech_started(detector)
        speech_ended(detector, 0)
        await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.1)
        commit_turn(detector, end_of_turn(""))
        speech_started(detector)
        speech_ended(detector, 0)
        fourth = await commit(detector, session, classifier)
        await detector._should_reply(2, llm.ChatContext())
        await detector._should_reply(3, llm.ChatContext())
        assert detector._turns[2].prediction.reason == "prediction"
        assert detector._turns[3].prediction.reason == "reused"
        classifier.prediction(4, AMDCategory.MACHINE_SCREENING)
        await fourth.should_reply(llm.ChatContext())
        await playback_released(session)
        assert next_deadline(detector) is not None
        assert next_deadline(detector) - asyncio.get_running_loop().time() == pytest.approx(
            10, abs=0.01
        )


@pytest.mark.asyncio
async def test_prediction_listener_failure_cleans_up_a_deferred_release() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):

        def on_prediction(_event: object) -> None:
            raise TypeError("listener failed")

        detector.on("amd_prediction", on_prediction)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        assert (await detector.execute()).reason == "internal_error"
        assert detector._turns[1].prediction is not None
        assert session.amd is None


@pytest.mark.asyncio
async def test_late_machine_prediction_does_not_postpone_idle_timeout() -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.01, idle_timeout=0.1) as (
        detector,
        session,
        classifier,
        model,
    ):
        events = []
        replies = []
        detector.on("amd_prediction", events.append)
        session.on("speech_created", lambda event: replies.append(event.speech_handle))
        await commit(detector, session, classifier, reply=True)
        await asyncio.wait_for(model.calls.get(), 2)
        assert events[-1].reason == "inference_timeout"
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.wait_for(replies[0], 2)
        await asyncio.sleep(0)
        release_at = next_deadline(detector)
        assert release_at is not None
        await asyncio.sleep(0.2)
        assert detector.lifecycle is AMDLifecycle.FINISHED
        assert [(event.category, event.reason) for event in events] == [
            (AMDCategory.UNCERTAIN, "inference_timeout"),
        ]
        assert (await asyncio.wait_for(detector.execute(), 0.2)).reason == "idle_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [None, 0.7])
async def test_eot_without_speech_edges_uses_available_timing(elapsed: float | None) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        info = end_of_turn()
        info.metrics.end_of_turn_delay = elapsed
        commit_turn(detector, info)
        await classifier.request()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        started = asyncio.get_running_loop().time()
        await detector._should_reply(1, llm.ChatContext())
        assert asyncio.get_running_loop().time() == started
        await playback_released(session)
        waited = asyncio.get_running_loop().time() - started
        assert waited == pytest.approx(1.5 - (elapsed or 0), abs=0.01)
        assert detector._turns[1].prediction.delay < 0.01


@pytest.mark.asyncio
async def test_eot_before_speech_end_waits_for_the_speech_end_anchor() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        speech_started(detector)
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.6)
        assert detector._turns[1].prediction is not None
        assert detector._reply_held_at(asyncio.get_running_loop().time())
        speech_ended(detector, 0.5)
        started = asyncio.get_running_loop().time()
        await detector._should_reply(1, llm.ChatContext())
        await playback_released(session)
        assert asyncio.get_running_loop().time() - started == pytest.approx(1.0, abs=0.01)
        assert detector._turns[1].prediction.delay < 0.01


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("speech_delay", "remaining_silence"), [(0.7, 0.8), (None, 1.5), (-0.5, 1.5)]
)
async def test_speech_timing_is_independent_of_event_creation(
    speech_delay: float | None, remaining_silence: float
) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        now = time.time()
        detector._on_user_state_changed(
            UserStateChangedEvent(
                old_state="listening",
                new_state="speaking",
                speech_timestamp=now - speech_delay if speech_delay is not None else None,
                created_at=now - 0.2,
            )
        )
        await asyncio.sleep(1)
        now = time.time()
        detector._on_user_state_changed(
            UserStateChangedEvent(
                old_state="speaking",
                new_state="listening",
                speech_timestamp=now - speech_delay if speech_delay is not None else None,
                created_at=now - 0.2,
            )
        )
        await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        started = asyncio.get_running_loop().time()
        await detector._should_reply(1, llm.ChatContext())
        await playback_released(session)
        assert detector._turns[1].prediction.speech_duration == pytest.approx(1.0, abs=0.01)
        assert asyncio.get_running_loop().time() - started == pytest.approx(
            remaining_silence, abs=0.01
        )
        assert detector._turns[1].prediction.delay < 0.01


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["vad", "stt", "stt_with_vad"])
async def test_audio_recognition_supplies_speech_edges_and_elapsed_silence(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, _, _):
        recognition = session._activity._audio_recognition
        monkeypatch.setattr(recognition, "_run_eou_detection", Mock())
        state_events = []
        session.on("user_state_changed", state_events.append)
        if source == "vad":
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.1,
                    silence_duration=0,
                )
            )
        else:
            monkeypatch.setattr(
                recognition, "_vad", FakeVAD() if source == "stt_with_vad" else None
            )
            recognition._turn_detection_mode = "stt"
            recognition._process_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH, speech_start_time=time.time() - 0.1
                )
            )
        assert detector._speech_started_at is not None
        assert detector._speech_since_commit
        assert state_events[-1].speech_timestamp == pytest.approx(time.time() - 0.1, abs=0.01)
        await asyncio.sleep(1)
        if source == "vad":
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.END_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.3,
                    silence_duration=0.5,
                    inference_duration=0.2,
                )
            )
        else:
            recognition._process_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH, speech_end_time=time.time() - 0.7
                )
            )
        assert detector._speech_started_at is None
        assert detector._speech_ended_at == pytest.approx(
            asyncio.get_running_loop().time() - 0.7, abs=0.01
        )
        assert state_events[-1].speech_timestamp == pytest.approx(time.time() - 0.7, abs=0.01)
        assert state_events[-1].created_at == pytest.approx(time.time(), abs=0.01)
        if source == "stt_with_vad":
            speech_ended_at = detector._speech_ended_at
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.END_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.3,
                    silence_duration=0.1,
                )
            )
            assert len(state_events) == 2
            assert detector._speech_ended_at == speech_ended_at
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0.1,
                    silence_duration=0,
                )
            )
            assert detector._speech_started_at is not None
            assert detector._speech_since_commit


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["vad", "stt"])
async def test_untranscribed_speech_restarts_idle_timer(source: str) -> None:
    session = AgentSession(
        llm=FakeLLM(),
        stt=FakeSTT(),
        vad=FakeVAD() if source == "vad" else None,
        turn_handling={"turn_detection": "vad" if source == "vad" else "stt"},
    )
    await session.start(Agent(instructions="Answer the call."))
    try:
        async with AMD(session, llm=None, stt=None, idle_timeout=0.2, timeout=2) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            recognition = session._activity._audio_recognition
            if source == "vad":
                # Let the empty fake stream finish before injecting VAD events.
                await recognition._vad_atask
                await recognition._on_vad_event(
                    vad.VADEvent(
                        type=vad.VADEventType.START_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=0,
                        silence_duration=0,
                    )
                )
            else:
                recognition._process_stt_event(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )
            await asyncio.sleep(0.3)
            assert detector.lifecycle is AMDLifecycle.ACTIVE

            if source == "vad":
                await recognition._on_vad_event(
                    vad.VADEvent(
                        type=vad.VADEventType.END_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=0.3,
                        silence_duration=0,
                    )
                )
            else:
                recognition._process_stt_event(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                )
            await asyncio.sleep(0.1)
            assert detector.lifecycle is AMDLifecycle.ACTIVE
            result = await asyncio.wait_for(detector.execute(), 0.2)
            assert result.reason == "idle_timeout"
            assert result.last_prediction is None
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_speech_end_does_not_start_idle_while_customer_hook_runs() -> None:
    agent = CustomerAgent(stop=True)
    agent.hook_release.clear()
    async with running(agent=agent, idle_timeout=0.1) as (detector, session, classifier, _):
        await commit(detector, session, classifier, reply=True)
        await agent.hook_started.wait()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await detector._should_reply(1, llm.ChatContext())
        speech_started(detector)
        speech_ended(detector, 0)
        await asyncio.sleep(0.2)
        assert detector.lifecycle is AMDLifecycle.ACTIVE

        agent.hook_release.set()
        result = await asyncio.wait_for(detector.execute(), 0.2)
        assert result.reason == "idle_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["timeout", "cancelled", "participant_disconnected"])
async def test_completion_cancels_silence_wait_and_releases_the_held_reply(reason: str) -> None:
    async with running(machine_silence_threshold=1.5, inference_timeout=0.2, timeout=0.5) as (
        detector,
        session,
        classifier,
        _,
    ):
        events = []
        detector.on("amd_prediction", events.append)
        await commit(detector, session, classifier)
        committed_at = asyncio.get_running_loop().time()
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await eventually(
            lambda: (
                next_deadline(detector) is not None and next_deadline(detector) > committed_at + 0.4
            )
        )
        assert (await detector._should_reply(1, llm.ChatContext())).allow
        assert not playback_allowed(session)
        if reason == "cancelled":
            await detector.aclose()
        elif reason == "participant_disconnected":
            detector._participant_identity = "callee"
            detector._on_disconnected(SimpleNamespace(identity="callee"))
        assert (await detector.execute()).reason == reason
        assert playback_allowed(session)
        assert next_deadline(detector) is None
        await asyncio.sleep(1.6)
        assert detector._turns[1].prediction.category == AMDCategory.MACHINE_VM
        assert [e.category for e in events] == [AMDCategory.MACHINE_VM]
        assert not detector._reply_held_at(asyncio.get_running_loop().time())
        assert session.amd is None
        assert not detector._tasks


@pytest.mark.asyncio
async def test_silence_deadline_crossed_while_rearming_still_wakes_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice.amd import detector as detector_module

    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, _):
        hooks = await commit(detector, session, classifier)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: detector._turns[1].prediction is not None)
        assert await hooks.should_reply(llm.ChatContext())
        assert not playback_allowed(session)
        release_at = detector._speech_ended_at + 1.5
        readings = iter((release_at - 0.01, release_at - 0.01))

        def clock() -> float:
            return next(readings, release_at + 0.01)

        with monkeypatch.context() as crossed_deadline:
            crossed_deadline.setattr(detector_module.time, "monotonic", clock)
            detector._update_idle()
            assert detector._timer.when() == pytest.approx(
                asyncio.get_running_loop().time(), abs=1e-6
            )
            await asyncio.wait_for(playback_released(session), 0.1)
        assert detector.lifecycle is AMDLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_machine_reply_generates_during_the_hold_and_plays_after_it() -> None:
    tool_ran_at: list[float] = []

    class ToolAgent(CustomerAgent):
        @llm.function_tool
        async def note_screening(self) -> None:
            """Record that the screener asked a question."""
            tool_ran_at.append(asyncio.get_running_loop().time())

    async with running(agent=ToolAgent(), machine_silence_threshold=1.5) as (
        detector,
        session,
        classifier,
        model,
    ):
        loop = asyncio.get_running_loop()
        model.fake_response_map[_DEFAULT_SCREENING_INSTRUCTIONS].tool_calls.append(
            llm.FunctionToolCall(name="note_screening", arguments="{}", call_id="note")
        )
        speaking_at: list[float] = []
        session.on(
            "agent_state_changed",
            lambda event: event.new_state == "speaking" and speaking_at.append(loop.time()),
        )
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier, reply=True)
        release_at = detector._speech_ended_at + 1.5
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert loop.time() < release_at - 0.9
        assert call["chat_ctx"].items[-1].text_content == _DEFAULT_SCREENING_INSTRUCTIONS

        await asyncio.sleep(release_at - loop.time() - 0.01)
        assert not playback_allowed(session)
        assert not speaking_at
        assert not tool_ran_at
        await eventually(lambda: speaking_at and tool_ran_at)
        assert speaking_at[0] >= release_at
        assert tool_ran_at[0] >= release_at


@pytest.mark.asyncio
async def test_resumed_speech_extends_the_hold_for_a_generated_reply() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, model):
        loop = asyncio.get_running_loop()
        speeches: list[SpeechHandle] = []
        session.on("speech_created", lambda event: speeches.append(event.speech_handle))
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.wait_for(model.calls.get(), 2)
        await asyncio.sleep(0.5)
        speech_started(detector)
        await asyncio.sleep(2)
        assert not playback_allowed(session)
        assert session.agent_state != "speaking"

        speech_ended(detector, 0)
        release_at = loop.time() + 1.5
        await asyncio.sleep(1.49)
        assert not playback_allowed(session)
        assert session.agent_state != "speaking"
        await playback_released(session)
        assert loop.time() == pytest.approx(release_at, abs=0.01)
        assert len(speeches) == 1
        await asyncio.wait_for(speeches[0], 2)
        assert not speeches[0].interrupted
        assert session.output.audio.captured_playout_segments == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category", [AMDCategory.MACHINE_SCREENING, AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE]
)
async def test_newer_turn_cancels_a_held_reply(category: AMDCategory) -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, model):
        speeches: list[SpeechHandle] = []
        session.on("speech_created", lambda event: speeches.append(event.speech_handle))
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.wait_for(model.calls.get(), 2)
        assert len(speeches) == 1
        held = speeches[0]

        if category is AMDCategory.HUMAN:
            # a first human after screening is likely human, counted as uncertain: its reply carries no
            # AMD instructions, so the hook's edit is the model's last input
            model.fake_response_map["Customer hook edit"] = FakeLLMResponse(
                input="Customer hook edit", content="Please call me back.", ttft=0, duration=0
            )
        speech_started(detector)
        speech_ended(detector, 1.5)
        await commit(detector, session, classifier, reply=True)
        await asyncio.wait_for(held, 2)
        assert held.interrupted
        classifier.prediction(2, category)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)

        if category is AMDCategory.MACHINE_UNAVAILABLE:
            assert (await detector.execute()).category == category
            assert len(speeches) == 1
        else:
            assert len(speeches) == 2
            await asyncio.wait_for(speeches[1], 2)
            assert not speeches[1].interrupted
        assert session.output.audio.captured_playout_segments == (
            0 if category is AMDCategory.MACHINE_UNAVAILABLE else 1
        )


@pytest.mark.asyncio
async def test_newer_turn_without_a_reply_cancels_a_held_reply() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, model):
        activity = session._activity
        assert activity is not None
        speeches: list[SpeechHandle] = []
        session.on("speech_created", lambda event: speeches.append(event.speech_handle))
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.wait_for(model.calls.get(), 2)
        assert len(speeches) == 1

        assert activity.on_end_of_turn(end_of_turn(skip_reply=True))
        await asyncio.wait_for(speeches[0], 2)
        assert speeches[0].interrupted
        assert session.output.audio.captured_playout_segments == 0


@pytest.mark.asyncio
async def test_superseded_held_voicemail_does_not_consume_the_stage_message() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, model):
        speeches: list[SpeechHandle] = []
        session.on("speech_created", lambda event: speeches.append(event.speech_handle))
        speech_started(detector)
        speech_ended(detector, 0.5)
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(model.calls.get(), 2)
        assert detector._voicemail_handle is speeches[0]

        speech_started(detector)
        speech_ended(detector, 1.5)
        await commit(detector, session, classifier, reply=True)
        await asyncio.wait_for(speeches[0], 2)
        assert speeches[0].interrupted
        assert detector._voicemail_handle is None
        assert not detector._voicemail_message_played

        classifier.prediction(2, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        assert detector._voicemail_handle is speeches[1]
        await asyncio.wait_for(speeches[1], 2)
        assert detector._voicemail_message_played
        assert session.output.audio.captured_playout_segments == 1


# endregion


# region: Realtime models


@asynccontextmanager
async def running_realtime(
    *,
    machine_silence_threshold: float = 0,
    agent: Agent | None = None,
    session_stt: bool = True,
    amd_stt: stt.STT | None = None,
    user_transcription: bool = True,
    turn_detection: Literal["manual", "vad"] = "manual",
) -> AsyncIterator[tuple[AMD, AgentSession, ClassifierLLM, FakeRealtimeSession]]:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            can_disable_turn_detection=True,
            auto_tool_reply_generation=False,
            user_transcription=user_transcription,
        )
    )
    async with AgentSession(
        llm=model,
        stt=FakeSTT() if session_stt else None,
        vad=FakeVAD() if turn_detection == "vad" else None,
        turn_handling={"turn_detection": turn_detection, "interruption": {"mode": "vad"}},
        aec_warmup_duration=None,
    ) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(agent or Agent(instructions="Call about an appointment."))
        classifier = ClassifierLLM()
        async with AMD(
            session,
            llm=classifier,
            stt=amd_stt,
            machine_silence_threshold=machine_silence_threshold,
        ) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            yield detector, session, classifier, model.active_session


async def reply(
    session: AgentSession,
    classifier: ClassifierLLM,
    rt: FakeRealtimeSession,
    category: AMDCategory,
) -> SpeechHandle:
    handle = asyncio.get_running_loop().create_future()
    session.once("speech_created", lambda ev: handle.set_result(ev.speech_handle))
    count = rt.generate_reply_calls
    assert session._activity.on_end_of_turn(end_of_turn())
    request = await classifier.request()
    classifier.prediction(request.current_turn.extra["turn_id"], category)
    await eventually(lambda: rt.generate_reply_calls == count + 1)
    return await handle


def respond(
    rt: FakeRealtimeSession, *, tool: llm.FunctionCall | None = None, duration: float = 0.01
) -> None:
    messages = utils.aio.Chan[llm.MessageGeneration]()
    functions = utils.aio.Chan[llm.FunctionCall]()
    if tool is not None:
        functions.send_nowait(tool)
    else:
        text = utils.aio.Chan[str]()
        audio = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])
        text.send_nowait("Please call me back.")
        text.close()
        audio.send_nowait(_audio_frame(duration))
        audio.close()
        messages.send_nowait(
            llm.MessageGeneration(
                message_id=utils.shortuuid("message_"),
                text_stream=text,
                audio_stream=audio,
                modalities=modalities,
            )
        )
    messages.close()
    functions.close()
    rt._reply_futs[-1].set_result(
        llm.GenerationCreatedEvent(
            message_stream=messages, function_stream=functions, user_initiated=True
        )
    )


@pytest.mark.parametrize(
    ("configuration", "error"),
    [
        ("server_turn_detection", "client-side turn detection"),
        ("auto_tool_reply", "client-controlled tool replies"),
        ("session_tools_only", "per-response tools"),
        ("missing_stt", "session STT"),
        ("missing_classifier", "LLM or a decision model for classification"),
    ],
)
async def test_unsupported_configuration_does_not_install_amd(
    configuration: str, error: str
) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            can_disable_turn_detection=configuration != "server_turn_detection",
            auto_tool_reply_generation=configuration == "auto_tool_reply",
            per_response_tool_choice=configuration != "session_tools_only",
        )
    )
    async with AgentSession(
        llm=model,
        stt=None if configuration == "missing_stt" else FakeSTT(),
        vad=None,
        turn_handling={"turn_detection": "manual"},
    ) as session:
        await session.start(Agent(instructions="Call about an appointment."))
        detector = AMD(
            session,
            llm=None if configuration == "missing_classifier" else ClassifierLLM(),
            stt=None,
        )
        with pytest.raises(ValueError, match=error):
            await detector.__aenter__()
        assert session.amd is None
        assert session._turn_hooks is None
        assert session._activity._authorization_allowed.is_set()
        assert detector.lifecycle is AMDLifecycle.INITIALIZED


async def test_reply_waits_for_classification_and_silence() -> None:
    async with running_realtime(machine_silence_threshold=1.5) as (
        detector,
        session,
        classifier,
        rt,
    ):
        handles: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        speech_started(detector)
        speech_ended(detector, 0.5)
        assert session._activity.on_end_of_turn(end_of_turn())
        await classifier.request()
        await asyncio.sleep(0.1)
        assert rt.generate_reply_calls == 0
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.5)
        assert rt.generate_reply_calls == 0
        await eventually(lambda: rt.generate_reply_calls == 1)
        assert rt.committed
        assert rt.reply_instructions == [_DEFAULT_SCREENING_INSTRUCTIONS]
        respond(rt)
        await asyncio.wait_for(handles[0], 2)


async def test_untranscribed_speech_does_not_revoke_reply_authorization() -> None:
    async with running_realtime(machine_silence_threshold=0.5, turn_detection="vad") as (
        detector,
        session,
        classifier,
        rt,
    ):
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_SCREENING)
        activity = session._activity
        recognition = activity._audio_recognition
        output = session.output.audio
        await asyncio.wait_for(recognition._vad_atask, 2)
        await recognition._on_vad_event(
            vad.VADEvent(
                type=vad.VADEventType.START_OF_SPEECH,
                samples_index=0,
                timestamp=0,
                speech_duration=0,
                silence_duration=0,
            )
        )
        respond(rt)
        await asyncio.sleep(0.1)
        assert not handle.done()
        assert output.captured_playout_segments == 0

        await recognition._on_vad_event(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=0,
                timestamp=0,
                speech_duration=0.1,
                silence_duration=0,
            )
        )
        await asyncio.wait_for(handle, 0.2)
        assert not handle.interrupted
        assert output.captured_playout_segments == 1
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._turn_id == 1
        assert classifier.requests.empty()

        assert activity.on_end_of_turn(end_of_turn("Please hold."))
        await classifier.request()
        assert not activity._authorization_allowed.is_set()
        assert rt.generate_reply_calls == 1
        classifier.prediction(2, AMDCategory.WAIT)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        assert rt.generate_reply_calls == 1
        assert output.captured_playout_segments == 1


async def test_amd_interrupts_a_pending_reply_despite_disabled_interruptions() -> None:
    agent = Agent(
        instructions="Call about an appointment.",
        turn_handling={"interruption": {"enabled": False}},
    )
    async with running_realtime(agent=agent) as (_, session, classifier, rt):
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_SCREENING)
        activity = session._activity
        assert activity.on_end_of_turn(end_of_turn("Please hold."))
        await classifier.request()
        classifier.prediction(2, AMDCategory.WAIT)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        await asyncio.wait_for(handle, 0.1)
        assert handle.interrupted
        assert rt.generate_reply_calls == 1
        assert session.output.audio.captured_playout_segments == 0


async def test_amd_preserves_false_interruption_resume() -> None:
    agent = Agent(
        instructions="Call about an appointment.",
        turn_handling={"interruption": {"enabled": False}},
    )
    async with running_realtime(agent=agent, turn_detection="vad") as (
        detector,
        session,
        classifier,
        rt,
    ):
        output = FakeAudioOutput(can_pause=True)
        session.output.audio = output
        session.options.interruption["min_words"] = 0
        session.options.interruption["false_interruption_timeout"] = 0.1
        resumed = []
        session.on("agent_false_interruption", lambda event: resumed.append(event.resumed))
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_SCREENING)
        activity = session._activity
        await activity._audio_recognition._vad_atask
        respond(rt, duration=1)
        await eventually(lambda: session.agent_state == "speaking")
        activity.on_start_of_speech(None, time.time())
        activity._interrupt_by_audio_activity()
        assert output._paused_at is not None
        assert handle.allow_interruptions
        activity.on_end_of_speech(None, speech_end_time=time.time())
        await asyncio.wait_for(handle, 2)
        assert resumed == [True]
        assert not handle.interrupted
        assert output.captured_playout_segments == 1
        assert detector._turn_id == 1


async def test_stage_instructions_expire_after_first_human_reply() -> None:
    async with running_realtime() as (_, session, classifier, rt):
        for category, instructions in (
            (AMDCategory.MACHINE_SCREENING, _DEFAULT_SCREENING_INSTRUCTIONS),
            # a first human after screening is likely human, counted as uncertain, with no stage instructions
            (AMDCategory.HUMAN, NOT_GIVEN),
            (AMDCategory.HUMAN, _DEFAULT_HUMAN_INSTRUCTIONS),
        ):
            handle = await reply(session, classifier, rt, category)
            assert rt.reply_instructions[-1] == instructions
            respond(rt)
            await asyncio.wait_for(handle, 2)

        assert session.amd is None
        assert session._turn_hooks is None
        handle = session.generate_reply()
        await eventually(lambda: rt.generate_reply_calls == 4)
        assert rt.reply_instructions[-1] is NOT_GIVEN
        respond(rt)
        await asyncio.wait_for(handle, 2)
        assert rt.updated_instructions == "Call about an appointment."
        for context in (session.history, session.current_agent.chat_ctx, rt.chat_ctx):
            assert not any(
                "Call state:" in (message.text_content or "") for message in context.messages()
            )


@pytest.mark.parametrize("category", [AMDCategory.WAIT, AMDCategory.MACHINE_UNAVAILABLE])
@pytest.mark.parametrize("transcription_before_verdict", [False, True])
async def test_suppressed_reply_keeps_only_the_realtime_transcript(
    category: AMDCategory, transcription_before_verdict: bool
) -> None:
    async with running_realtime() as (_, session, classifier, rt):
        assert session._activity.on_end_of_turn(end_of_turn())
        await classifier.request()
        transcript = llm.InputTranscriptionCompleted(
            item_id="user-audio", transcript="hello", is_final=True
        )
        if transcription_before_verdict:
            rt.emit("input_audio_transcription_completed", transcript)
        classifier.prediction(1, category)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        assert rt.committed
        assert rt.generate_reply_calls == 0
        if not transcription_before_verdict:
            rt.emit("input_audio_transcription_completed", transcript)
        for context in (session.current_agent.chat_ctx, session.history):
            messages = [m for m in context.messages() if m.role == "user"]
            assert [(m.id, m.text_content) for m in messages] == [("user-audio", "hello")]


@pytest.mark.parametrize("source", ["session", "amd", "both"])
@pytest.mark.parametrize(
    "category", [AMDCategory.WAIT, AMDCategory.MACHINE_UNAVAILABLE, AMDCategory.HUMAN]
)
async def test_external_stt_preserves_history_without_native_transcription(
    source: str, category: AMDCategory
) -> None:
    amd_stt = DrainingSTT() if source != "session" else None
    async with running_realtime(
        session_stt=source != "amd", amd_stt=amd_stt, user_transcription=False
    ) as (
        detector,
        session,
        classifier,
        rt,
    ):
        handles: list[SpeechHandle] = []
        added: list[llm.ChatMessage] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        session.on("conversation_item_added", lambda ev: added.append(ev.item))
        activity = session._activity
        activity.push_audio(_audio_frame(0.1))
        if source != "amd":
            await activity._audio_recognition._on_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(text="Session transcript.", language=LanguageCode("en"))
                    ],
                )
            )
        if amd_stt is not None and source == "amd":
            amd_stt.streams[0].send_fake_transcript("AMD transcript.")
            await eventually(
                lambda: detector._resources.stt._current.get("amd") == "AMD transcript."
            )

        await session.commit_user_turn()
        request = await classifier.request()
        assert (
            request.current_turn.text_content
            == {
                "amd": "AMD transcript.",
                "session": "Session transcript.",
                "both": '{"transcript_1": "Session transcript.", "transcript_2": ""}',
            }[source]
        )
        if source == "both":
            # no race to lose: AMD's stream keeps listening alongside the session's
            assert detector._resources.stt.amd_stt_active
        assert rt.generate_reply_calls == 0
        classifier.prediction(1, category)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        if category is AMDCategory.HUMAN:
            await eventually(lambda: rt.generate_reply_calls == 1)
            respond(rt)
            await asyncio.wait_for(handles[0], 2)
        else:
            assert rt.generate_reply_calls == 0

        expected = "AMD transcript." if source == "amd" else "Session transcript."
        for context in (session.current_agent.chat_ctx, session.history):
            assert [m.text_content for m in context.messages() if m.role == "user"] == [expected]
        assert [m.text_content for m in added if m.role == "user"] == [expected]
        assert not [m for m in rt.chat_ctx.messages() if m.role == "user"]
        assert rt.committed


@pytest.mark.parametrize("category", [AMDCategory.WAIT, AMDCategory.HUMAN])
async def test_session_stt_classifies_before_realtime_transcription(category: AMDCategory) -> None:
    async with running_realtime() as (_, session, classifier, rt):
        handles: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        recognition = session._activity._audio_recognition
        await recognition._on_stt_event(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text="Please hold.", language=LanguageCode("en"))],
            )
        )
        assert await session.commit_user_turn() == "Please hold."
        request = await classifier.request()
        assert request.current_turn.text_content == "Please hold."
        assert not session.current_agent.chat_ctx.messages()
        assert not handles
        classifier.prediction(1, category)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        if category is AMDCategory.HUMAN:
            await eventually(lambda: rt.generate_reply_calls == 1)
            assert rt.reply_instructions[-1] is NOT_GIVEN
            respond(rt)
            await asyncio.wait_for(handles[0], 2)
        else:
            assert rt.generate_reply_calls == 0


@pytest.mark.parametrize("session_stt", [False, True])
@pytest.mark.parametrize("skip_reply", [False, True])
async def test_manual_realtime_commit_without_amd(session_stt: bool, skip_reply: bool) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True))
    async with AgentSession(
        llm=model,
        stt=FakeSTT() if session_stt else None,
        turn_handling={"turn_detection": "manual"},
    ) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="Call about an appointment."))
        rt = model.active_session
        rt.commit_audio = Mock(wraps=rt.commit_audio)
        handles: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        session._activity.push_audio(_audio_frame(0.1))
        if session_stt:
            await session._activity._audio_recognition._on_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="Hello.", language=LanguageCode("en"))],
                )
            )

        committed = session.commit_user_turn(skip_reply=skip_reply)
        rt.commit_audio.assert_called_once_with()
        assert await committed == ("Hello." if session_stt else "")
        recognition = session._activity._audio_recognition
        if recognition._end_of_turn_task is not None:
            await asyncio.wait_for(recognition._end_of_turn_task, 2)
        if session._activity._user_turn_completed_atask is not None:
            await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        if not skip_reply:
            await eventually(lambda: rt.generate_reply_calls == 1)
            respond(rt)
            await asyncio.wait_for(handles[0], 2)
        assert rt.generate_reply_calls == (0 if skip_reply else 1)
        rt.commit_audio.assert_called_once_with()
    if not session_stt:
        assert not [m for m in session.history.messages() if m.role == "user"]


@pytest.mark.parametrize("session_stt", [False, True])
@pytest.mark.parametrize("turn_detection", ["manual", "vad"])
async def test_amd_stt_supplies_transcripts_at_eot(
    session_stt: bool, turn_detection: Literal["manual", "vad"]
) -> None:
    amd_stt = DrainingSTT()
    async with running_realtime(
        session_stt=session_stt, amd_stt=amd_stt, turn_detection=turn_detection
    ) as (
        detector,
        session,
        classifier,
        rt,
    ):
        handles: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        activity = session._activity
        recognition = activity._audio_recognition
        activity.push_audio(_audio_frame(0.1))
        stream = amd_stt.streams[0]
        if not session_stt:
            rt.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id="previous-audio", transcript="Late realtime transcript.", is_final=True
                ),
            )
        assert detector._resources.stt.amd_stt_active

        if turn_detection == "vad":
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0,
                    silence_duration=0,
                )
            )
        if session_stt:
            await recognition._on_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(text="Session transcript.", language=LanguageCode("en"))
                    ],
                )
            )
        stream.send_fake_transcript("Please state your name.")
        await eventually(
            lambda: detector._resources.stt._current.get("amd") == "Please state your name."
        )
        assert rt.generate_reply_calls == 0
        assert classifier.requests.empty()

        if turn_detection == "manual":
            transcript = await session.commit_user_turn()
            assert transcript == ("Session transcript." if session_stt else "")
        else:
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.END_OF_SPEECH,
                    samples_index=0,
                    timestamp=0.5,
                    speech_duration=0.5,
                    silence_duration=0.5,
                )
            )
        request = await classifier.request()
        assert request.current_turn.text_content == (
            '{"transcript_1": "Session transcript.", "transcript_2": "Please state your name."}'
            if session_stt
            else "Please state your name."
        )
        assert rt.generate_reply_calls == 0
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: rt.generate_reply_calls == 1)
        assert rt.reply_instructions[-1] == _DEFAULT_SCREENING_INSTRUCTIONS
        assert rt.pushed_audio
        assert rt.committed
        respond(rt)
        await asyncio.wait_for(handles[0], 2)


async def test_realtime_tool_handoff_is_cancelled_while_amd_is_running() -> None:
    replacement = Agent(instructions="Continue the call.")

    class HandoffAgent(Agent):
        @llm.function_tool
        async def transfer(self) -> Agent:
            """Transfer the call to the next agent."""
            return replacement

    original = HandoffAgent(instructions="Call about an appointment.")
    async with running_realtime(agent=original) as (detector, session, classifier, rt):
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_SCREENING)
        respond(rt, tool=llm.FunctionCall(call_id="move", name="transfer", arguments="{}"))
        await asyncio.wait_for(handle, 2)

        assert session.current_agent is original
        assert session.amd is detector
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert replacement._activity is None
        outputs = [item for item in rt.chat_ctx.items if item.type == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0].call_id == "move"
        assert outputs[0] in session.history.items


@pytest.mark.parametrize("publish_fails", [False, True])
async def test_ivr_dtmf_only_generates_a_reply_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    publish_fails: bool,
) -> None:
    publish_dtmf = AsyncMock(side_effect=RuntimeError("publish failed") if publish_fails else None)
    monkeypatch.setattr(
        "livekit.agents.beta.tools.send_dtmf.get_job_context",
        lambda: SimpleNamespace(
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publish_dtmf))
        ),
    )
    async with running_realtime() as (_, session, classifier, rt):
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_IVR)
        assert [tool.info.name for tool in rt.reply_tools[0]] == ["send_dtmf_events"]
        assert session.current_agent.tools == []
        respond(
            rt,
            tool=llm.FunctionCall(
                call_id="dtmf-1", name="send_dtmf_events", arguments='{"events": ["1"]}'
            ),
        )
        if publish_fails:
            await eventually(lambda: rt.generate_reply_calls == 2)
            assert rt.reply_instructions == [_DEFAULT_IVR_INSTRUCTIONS, _DEFAULT_IVR_INSTRUCTIONS]
            assert rt.reply_tools[1] == rt.reply_tools[0]
            respond(rt)
        await asyncio.wait_for(handle, 2)
        publish_dtmf.assert_awaited_once_with(code=1, digit="1")
        assert rt.generate_reply_calls == (2 if publish_fails else 1)
        outputs = [i for i in rt.chat_ctx.items if i.type == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0].is_error == publish_fails
        assert outputs[0].reply_required == publish_fails
        assert session.agent_state == "listening"

        held = await reply(session, classifier, rt, AMDCategory.HUMAN)
        # a first human after a menu is likely human, counted as uncertain and keeps the keypad
        assert [tool.info.name for tool in rt.reply_tools[-1]] == ["send_dtmf_events"]
        respond(rt)
        await asyncio.wait_for(held, 2)
        human = await reply(session, classifier, rt, AMDCategory.HUMAN)
        assert rt.reply_tools[-1] is NOT_GIVEN
        respond(rt)
        await asyncio.wait_for(human, 2)


@pytest.mark.parametrize("interrupted", [False, True])
async def test_voicemail_is_retried_only_after_interrupted_playout(interrupted: bool) -> None:
    async with running_realtime() as (detector, session, classifier, rt):
        handles: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_VM)
        assert rt.reply_instructions[-1] == _DEFAULT_VOICEMAIL_INSTRUCTIONS
        respond(rt, duration=10 if interrupted else 0.01)
        if interrupted:
            await eventually(lambda: session.output.audio._pushed_duration > 0)
            session.interrupt()
        await asyncio.wait_for(handle, 2)
        assert detector._voicemail_message_played is not interrupted

        assert session._activity.on_end_of_turn(end_of_turn())
        await classifier.request()
        classifier.prediction(2, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        if interrupted:
            await eventually(lambda: rt.generate_reply_calls == 2)
            respond(rt)
            await asyncio.wait_for(handles[-1], 2)
        else:
            assert rt.generate_reply_calls == 1


async def test_customer_stop_response_prevents_realtime_generation() -> None:
    async with running_realtime(agent=CustomerAgent(stop=True)) as (_, session, classifier, rt):
        assert session._activity.on_end_of_turn(end_of_turn())
        await classifier.request()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        assert rt.generate_reply_calls == 0


# endregion


# region: examples/telephony/amd.py


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        "answered",
        "voicemail",
        "voicemail_menu",
        "voicemail_failed",
        "human_after_voicemail",
        "disconnected",
        "missing",
        "timeout",
        "cancelled",
        "console",
    ],
)
async def test_outbound_call_setup(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    monkeypatch.setenv("SIP_PHONE_NUMBER", "+15555550100")
    monkeypatch.setenv("SIP_PARTICIPANT_IDENTITY", "callee")
    monkeypatch.setenv("SIP_OUTBOUND_TRUNK_ID", "test-trunk")
    monkeypatch.setattr("dotenv.load_dotenv", Mock())
    for name in ("STT", "LLM", "TTS"):
        monkeypatch.setattr(agents.inference, name, Mock())
    session = Mock(start=AsyncMock())
    if outcome == "console":
        type(session).room_io = PropertyMock(
            side_effect=RuntimeError(
                "Cannot access room_io: the AgentSession was not started with a room."
            )
        )
    monkeypatch.setattr(agents, "AgentSession", Mock(return_value=session))
    category = {
        "voicemail": AMDCategory.MACHINE_VM,
        "voicemail_menu": AMDCategory.MACHINE_IVR,
        "voicemail_failed": AMDCategory.MACHINE_VM,
    }.get(outcome, AMDCategory.HUMAN)
    result = AMDCompletedEvent(
        category=category,
        reason=AMDReason.FINISHED if category == AMDCategory.HUMAN else AMDReason.IDLE_TIMEOUT,
        voicemail_message_played=outcome
        in {"voicemail", "voicemail_menu", "human_after_voicemail"},
    )
    detector = MagicMock(execute=AsyncMock(return_value=result))
    monkeypatch.setattr(agents, "AMD", Mock(return_value=detector))
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "telephony" / "amd.py")
    )

    room = SimpleNamespace(
        name="test-room",
        remote_participants={},
        isconnected=Mock(return_value=outcome != "console"),
    )

    async def create_participant(*args: object, **kwargs: object) -> None:
        detector.__aenter__.assert_awaited_once()
        if outcome == "timeout":
            raise asyncio.TimeoutError
        if outcome == "cancelled":
            raise asyncio.CancelledError
        if outcome != "missing":
            room.remote_participants["callee"] = SimpleNamespace(identity="callee")
        if outcome == "disconnected":
            room.remote_participants.pop("callee")

    create = AsyncMock(side_effect=create_participant)
    ctx = SimpleNamespace(
        room=room,
        api=SimpleNamespace(sip=SimpleNamespace(create_sip_participant=create)),
        shutdown=Mock(),
        add_shutdown_callback=Mock(),
    )
    if outcome == "console":
        with pytest.raises(RuntimeError, match="not started with a room"):
            await example["entrypoint"](ctx)
        detector.__aenter__.assert_not_awaited()
        create.assert_not_awaited()
        return
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await example["entrypoint"](ctx)
    else:
        await example["entrypoint"](ctx)

    create.assert_awaited_once()
    assert create.call_args.args[0].wait_until_answered
    detector.__aexit__.assert_awaited_once()
    if outcome in {
        "answered",
        "voicemail",
        "voicemail_menu",
        "voicemail_failed",
        "human_after_voicemail",
    }:
        detector.execute.assert_awaited_once()
        if outcome in {"voicemail", "voicemail_menu"}:
            ctx.shutdown.assert_called_once_with("voicemail completed")
        else:
            ctx.shutdown.assert_not_called()
    else:
        detector.execute.assert_not_awaited()
        if outcome == "cancelled":
            ctx.shutdown.assert_not_called()
        else:
            ctx.shutdown.assert_called_once_with(
                "call not answered" if outcome == "timeout" else "participant missing"
            )


@pytest.fixture
def example_agent(monkeypatch: pytest.MonkeyPatch) -> agents.Agent:
    monkeypatch.setattr("dotenv.load_dotenv", Mock())
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "telephony" / "amd.py")
    )
    return example["MyAgent"]()


@pytest.mark.asyncio
@pytest.mark.virtual_time
@pytest.mark.parametrize("menu_arrives", [False, True])
async def test_voicemail_keeps_call_open_until_amd_finishes(
    monkeypatch: pytest.MonkeyPatch, example_agent: agents.Agent, menu_arrives: bool
) -> None:
    job = Mock()
    monkeypatch.setattr("livekit.agents.beta.tools.end_call.get_job_context", lambda: job)
    publish = AsyncMock()
    monkeypatch.setattr(
        "livekit.agents.beta.tools.send_dtmf.get_job_context",
        lambda: SimpleNamespace(
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publish))
        ),
    )
    async with running(agent=example_agent, voicemail_idle_timeout=10.0) as (
        detector,
        session,
        classifier,
        model,
    ):
        activity = session._activity
        model.fake_response_map[_DEFAULT_VOICEMAIL_INSTRUCTIONS].tool_calls = [
            llm.FunctionToolCall(name="end_call", arguments="{}", call_id="too-early")
        ]
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_VM)
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert "end_call" not in {tool.id for tool in call["tools"]}
        await eventually(lambda: detector._voicemail_message_played and activity._no_pending_speech)
        assert next_deadline(detector) - asyncio.get_running_loop().time() == pytest.approx(
            10, abs=0.1
        )
        await asyncio.sleep(5)
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        job.shutdown.assert_not_called()

        if not menu_arrives:
            result = await asyncio.wait_for(detector.execute(), 6)
            assert result.reason == AMDReason.IDLE_TIMEOUT
            assert result.category == AMDCategory.MACHINE_VM
            assert result.voicemail_message_played
            job.shutdown.assert_not_called()
            return

        # The rejected tool call can produce an error follow-up before the next menu.
        while not model.calls.empty():
            call = model.calls.get_nowait()
            assert "end_call" not in {tool.id for tool in call["tools"]}

        instructions = detector._get_instructions(AMDCategory.MACHINE_IVR)
        model.fake_response_map[instructions] = FakeLLMResponse(
            input=instructions,
            content="",
            ttft=0,
            duration=0,
            tool_calls=[
                llm.FunctionToolCall(
                    name="send_dtmf_events", arguments='{"events":["1"]}', call_id="save-message"
                )
            ],
        )
        speech_started(detector)
        speech_ended(detector, 0)
        activity.on_end_of_turn(end_of_turn("To save your message, press 1."))
        await classifier.request()
        classifier.prediction(2, AMDCategory.MACHINE_IVR)
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert {tool.id for tool in call["tools"]} == {"send_dtmf_events"}
        result = await asyncio.wait_for(detector.execute(), 12)
        publish.assert_awaited_once_with(code=1, digit="1")
        assert result.reason == AMDReason.IDLE_TIMEOUT
        assert result.category == AMDCategory.MACHINE_IVR
        assert result.voicemail_message_played
        job.shutdown.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.virtual_time
async def test_human_can_end_call_after_amd(
    monkeypatch: pytest.MonkeyPatch, example_agent: agents.Agent
) -> None:
    job = Mock()
    monkeypatch.setattr("livekit.agents.beta.tools.end_call.get_job_context", lambda: job)
    async with running(agent=example_agent) as (detector, session, classifier, model):
        model.fake_response_map["hello"].tool_calls = [
            llm.FunctionToolCall(name="end_call", arguments="{}", call_id="conversation-complete")
        ]
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.HUMAN)
        call = await asyncio.wait_for(model.calls.get(), 2)
        assert "end_call" in {tool.id for tool in call["tools"]}
        await eventually(lambda: job.shutdown.called)
        job.shutdown.assert_called_once_with(reason="user_initiated")


# endregion
