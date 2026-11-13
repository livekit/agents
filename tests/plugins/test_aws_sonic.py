"""Nova Sonic realtime sessions over a fake Bedrock duplex stream (real SDK client and session)."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from aws_sdk_bedrock_runtime.client import AsyncBedrockRuntimeClient
from aws_sdk_bedrock_runtime.models import (
    BidirectionalOutputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOutputChunk,
    ModelStreamErrorException,
    ValidationException,
)
from botocore.credentials import RefreshableCredentials
from smithy_aws_event_stream.exceptions import InvalidEventBytes

from livekit import rtc
from livekit.agents import llm
from livekit.plugins.aws.experimental.realtime import realtime_model as sonic
from tests.fakes import until

# one Nova Sonic protocol event by kind; Any: the SDK carries its JSON bodies as untyped bytes
Event = dict[str, dict[str, Any]]

SPECULATIVE = '{"generationStage":"SPECULATIVE"}'


class Duplex:
    """One InvokeModelWithBidirectionalStream call: what the client sent, what Bedrock answers."""

    def __init__(self, ready: asyncio.Event) -> None:
        self.sent: list[Event] = []
        self.inbox: asyncio.Queue[Event | BaseException | None] = asyncio.Queue()
        self.changed = asyncio.Condition()
        self.input_stream = self
        self.output_stream = None
        self.stream_ready = ready
        self.text_gate: asyncio.Event | None = None

    async def send(self, chunk: InvokeModelWithBidirectionalStreamInputChunk) -> None:
        event = json.loads(chunk.value.bytes_)["event"]
        if "textInput" in event and self.text_gate is not None:
            await self.text_gate.wait()
        async with self.changed:
            self.sent.append(event)
            self.changed.notify_all()
        if "sessionEnd" in event:
            self.inbox.put_nowait(None)

    async def close(self) -> None:
        self.inbox.put_nowait(None)

    async def await_output(self) -> tuple[None, Duplex]:
        await self.stream_ready.wait()
        return None, self

    async def receive(self) -> InvokeModelWithBidirectionalStreamOutputChunk | None:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        if item is None:
            return None
        payload = json.dumps({"event": item}).encode()
        return InvokeModelWithBidirectionalStreamOutputChunk(
            value=BidirectionalOutputPayloadPart(bytes_=payload)
        )

    def kinds(self) -> list[str]:
        return [next(iter(event)) for event in self.sent]

    async def wait_until(self, predicate: Callable[[list[Event]], bool]) -> None:
        async with asyncio.timeout(3), self.changed:
            await self.changed.wait_for(lambda: predicate(self.sent))

    async def wait_for(self, kind: str) -> None:
        await self.wait_until(lambda sent: any(kind in event for event in sent))


class Bedrock:
    """Stands in for the Bedrock service behind the real AsyncBedrockRuntimeClient."""

    def __init__(self) -> None:
        self.duplexes: list[Duplex] = []
        self.opened = asyncio.Condition()
        self.ready = asyncio.Event()
        self.ready.set()

    async def invoke(self, _input: object) -> Duplex:
        duplex = Duplex(self.ready)
        async with self.opened:
            self.duplexes.append(duplex)
            self.opened.notify_all()
        return duplex

    async def stream(self, index: int) -> Duplex:
        async with asyncio.timeout(3), self.opened:
            await self.opened.wait_for(lambda: len(self.duplexes) > index)
        return self.duplexes[index]

    async def answer(self, **event: dict[str, object]) -> None:
        duplex = self.duplexes[-1]
        duplex.inbox.put_nowait(event)
        while not duplex.inbox.empty():
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)


@pytest.fixture
async def bedrock(monkeypatch: pytest.MonkeyPatch) -> Bedrock:
    service = Bedrock()
    for name, value in {
        "AWS_ACCESS_KEY_ID": "key",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_EC2_METADATA_DISABLED": "true",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        AsyncBedrockRuntimeClient, "invoke_model_with_bidirectional_stream", service.invoke
    )
    return service


@pytest.fixture
async def session(bedrock: Bedrock) -> AsyncIterator[sonic.RealtimeSession]:
    model = sonic.RealtimeModel(region="us-east-1")
    realtime = model.session()
    try:
        yield realtime
    finally:
        bedrock.ready.set()
        for duplex in bedrock.duplexes:
            if duplex.text_gate is not None:
                duplex.text_gate.set()
        await realtime.aclose()
        await model.aclose()


async def start(
    session: sonic.RealtimeSession,
    bedrock: Bedrock,
    chat_ctx: llm.ChatContext | None = None,
    tools: list[llm.Tool] | None = None,
) -> Duplex:
    await session.update_instructions("Be brief.")
    await session.update_chat_ctx(chat_ctx or llm.ChatContext())
    if tools:
        await session.update_tools(tools)
    duplex = await bedrock.stream(0)
    if bedrock.ready.is_set():
        await duplex.wait_until(
            lambda sent: any(e.get("contentStart", {}).get("type") == "AUDIO" for e in sent)
        )
    else:
        await duplex.wait_for("promptStart")
    await asyncio.sleep(0.1)  # the session goes live once audio input has started
    return duplex


def frame(value: int = 1) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=value.to_bytes(2, "little", signed=True) * 512,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=512,
    )


async def handled(session: sonic.RealtimeSession, duplex: Duplex) -> None:
    """Everything queued before it was handled: the input loop sends tool results and audio in
    order, so two later frames sent mean an earlier tool result was sent or dropped."""
    for value in (5, 6):
        session.push_audio(frame(value))
        sent = base64.b64encode(frame(value).data.tobytes()).decode()
        await duplex.wait_until(
            lambda events, sent=sent: any(
                e.get("audioInput", {}).get("content") == sent for e in events
            )
        )


def tool_use(call_id: str, arguments: str) -> list[Event]:
    return [
        {"contentStart": {"contentId": f"tool-{call_id}", "type": "TOOL", "role": "TOOL"}},
        {"toolUse": {"toolUseId": call_id, "toolName": "lookup", "content": arguments}},
    ]


def tool_output(call_id: str, output: str) -> llm.ChatContext:
    ctx = llm.ChatContext()
    ctx.items.append(
        llm.FunctionCallOutput(call_id=call_id, name="lookup", output=output, is_error=False)
    )
    return ctx


async def test_session_starts_over_the_installed_bedrock_sdk(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """The installed bedrock-runtime SDK builds a signed duplex session that opens with sessionStart; https://github.com/livekit/agents/issues/7042 https://github.com/livekit/agents/issues/3590 https://github.com/livekit/agents/issues/3629 https://github.com/livekit/agents/pull/4134"""
    duplex = await start(session, bedrock)
    await duplex.wait_for("contentStart")
    assert duplex.kinds()[:2] == ["sessionStart", "promptStart"]
    assert duplex.sent[2]["contentStart"]["role"] == "SYSTEM"


@llm.function_tool
async def lookup(city: str) -> str:
    """Look up a city."""
    return city


async def test_tool_schema_with_nested_parameters_reaches_prompt_start(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """Tool parameters, nested models included, reach the promptStart tool spec; https://github.com/livekit/agents/pull/4177 https://github.com/livekit/agents/pull/2910"""
    from pydantic import BaseModel

    class Address(BaseModel):
        city: str

    async def find(address: Address, unit: str) -> str:
        """Find an address."""
        return address.city

    find.__annotations__.update(address=Address, unit=str)
    duplex = await start(session, bedrock, tools=[llm.function_tool(find)])
    spec = duplex.sent[1]["promptStart"]["toolConfiguration"]["tools"][0]["toolSpec"]
    schema = json.loads(spec["inputSchema"]["json"])
    assert set(schema["properties"]) == {"address", "unit"}
    assert schema["required"] == ["address", "unit"]
    nested = schema["properties"]["address"]["$ref"].rsplit("/", 1)[-1]
    assert set(schema["$defs"][nested]["properties"]) == {"city"}


@pytest.mark.parametrize(
    "output, expected",
    [
        pytest.param(
            "Email recorded. Ask the user to confirm.",
            {"tool_result": "Email recorded. Ask the user to confirm."},
            id="plain-string",
        ),
        pytest.param('{"temperature": 21}', {"temperature": 21}, id="json-object"),
        pytest.param(
            "42",
            42,
            id="json-number",
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/issues/4348 reproduces: a tool "
                "result that parses as a non-object JSON value is never sent",
            ),
        ),
        pytest.param(
            "[1, 2]",
            [1, 2],
            id="json-array",
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/issues/4348 reproduces: a tool "
                "result that parses as a non-object JSON value is never sent",
            ),
        ),
    ],
)
async def test_every_string_tool_result_reaches_sonic_as_json(
    session: sonic.RealtimeSession, bedrock: Bedrock, output: str, expected: object
) -> None:
    """Any string a tool returns is sent back as a JSON tool result for its call; https://github.com/livekit/agents/issues/4348 https://github.com/livekit/agents/issues/2874 https://github.com/livekit/agents/pull/2859"""
    duplex = await start(session, bedrock, tools=[lookup])
    for event in tool_use("call-1", '{"city": "Paris"}'):
        await bedrock.answer(**event)
    await session.update_chat_ctx(tool_output("call-1", output))
    await handled(session, duplex)
    start_event = next(
        e["contentStart"] for e in duplex.sent if e.get("contentStart", {}).get("type") == "TOOL"
    )
    assert start_event["toolResultInputConfiguration"]["toolUseId"] == "call-1"
    result = next(e["toolResult"] for e in duplex.sent if "toolResult" in e)
    assert json.loads(result["content"]) == expected


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ('{"order_id":"1234"}', '{"order_id":"1234"}'),
        (json.dumps('{"order_id":"1234"}'), '{"order_id":"1234"}'),
        ('"hello"', '"hello"'),
        ('"[1,2]"', '"[1,2]"'),
    ],
)
async def test_double_encoded_tool_arguments_are_unwrapped_only_for_objects(
    session: sonic.RealtimeSession, bedrock: Bedrock, arguments: str, expected: str
) -> None:
    """A doubly encoded JSON object reaches the tool as an object, other values stay as sent; https://github.com/livekit/agents/pull/5411 https://github.com/livekit/agents/pull/5409"""
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)
    await start(session, bedrock)
    for event in tool_use("call-1", arguments):
        await bedrock.answer(**event)
    calls = [call async for call in generations[0].function_stream]
    assert [(call.call_id, call.arguments) for call in calls] == [("call-1", expected)]


async def test_history_is_context_and_new_user_text_is_sent_once(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """Handed-off history is sent as non-interactive context; only new user text is sent interactively, once; https://github.com/livekit/agents/issues/4899 https://github.com/livekit/agents/issues/4578"""
    history = llm.ChatContext()
    history.add_message(role="user", content="old question")
    history.add_message(role="assistant", content="old answer")
    duplex = await start(session, bedrock, chat_ctx=history)
    await duplex.wait_until(lambda sent: sum("textInput" in e for e in sent) >= 3)
    texts = [e for e in duplex.sent if "contentStart" in e and e["contentStart"]["type"] == "TEXT"]
    assert [e["contentStart"]["role"] for e in texts] == ["SYSTEM", "USER", "ASSISTANT"]
    assert not any(e["contentStart"].get("interactive") for e in texts)

    current = history.copy()
    current.add_message(role="user", content="new question")
    await session.update_chat_ctx(current)
    await session.update_chat_ctx(current)
    await duplex.wait_until(lambda sent: sum("textInput" in e for e in sent) >= 4)
    await asyncio.sleep(0.05)
    contents = [e["textInput"]["content"] for e in duplex.sent if "textInput" in e]
    assert contents[1:] == ["old question", "old answer", "new question"]

    await bedrock.answer(contentStart={"contentId": "u1", "type": "TEXT", "role": "USER"})
    await bedrock.answer(textOutput={"contentId": "u1", "role": "USER", "content": "spoken"})
    assert session.chat_ctx.messages()[-1].raw_text_content == "spoken"


async def test_one_generation_per_assistant_turn_with_complete_text_and_audio(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """From an empty context, a multi-sentence reply is one generation carrying all its text and audio; https://github.com/livekit/agents/issues/4855 https://github.com/livekit/agents/pull/3704 https://github.com/livekit/agents/issues/2800"""
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)
    await start(session, bedrock)
    await bedrock.answer(contentStart={"contentId": "u1", "type": "TEXT", "role": "USER"})
    await bedrock.answer(textOutput={"contentId": "u1", "role": "USER", "content": "hello"})
    pcm = b"\x01\x00" * 160
    for index, sentence in enumerate(["Hello. ", "How can I help?"]):
        content_id = f"a{index}"
        await bedrock.answer(
            contentStart={
                "contentId": content_id,
                "type": "TEXT",
                "role": "ASSISTANT",
                "additionalModelFields": SPECULATIVE,
            }
        )
        await bedrock.answer(
            textOutput={"contentId": content_id, "role": "ASSISTANT", "content": sentence}
        )
        await bedrock.answer(contentStart={"contentId": f"audio{index}", "type": "AUDIO"})
        await bedrock.answer(
            audioOutput={"contentId": f"audio{index}", "content": base64.b64encode(pcm).decode()}
        )
    await bedrock.answer(contentEnd={"type": "AUDIO", "stopReason": "END_TURN"})

    assert len({generation.response_id for generation in generations}) == 1
    messages = [message async for message in generations[0].message_stream]
    assert len(messages) == 1
    assert [text async for text in messages[0].text_stream] == ["Hello. ", "How can I help?"]
    audio = b"".join([frame.data.tobytes() async for frame in messages[0].audio_stream])
    assert audio == pcm * 2


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/8260 reproduces: completionStart "
    "announces a generation before the user speaks",
)
async def test_completion_start_does_not_announce_a_generation_before_the_user_turn(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """The first assistant turn is announced once, after the user starts speaking; https://github.com/livekit/agents/issues/8260"""
    order: list[str] = []
    session.on("input_speech_started", lambda _: order.append("user"))
    session.on("generation_created", lambda event: order.append(f"gen:{event.response_id}"))
    await start(session, bedrock)
    await bedrock.answer(completionStart={"completionId": "c1"})
    await bedrock.answer(contentStart={"contentId": "u1", "type": "TEXT", "role": "USER"})
    await bedrock.answer(textOutput={"contentId": "u1", "role": "USER", "content": "hi"})
    await bedrock.answer(
        contentStart={
            "contentId": "a1",
            "type": "TEXT",
            "role": "ASSISTANT",
            "additionalModelFields": SPECULATIVE,
        }
    )
    assert order[0] == "user"
    assert len(order) == 2


async def test_audio_waits_for_the_stream_and_never_splits_a_text_block(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """Audio starts only once the stream is up and never lands inside a TEXT block; https://github.com/livekit/agents/pull/5626 https://github.com/livekit/agents/pull/6720"""
    bedrock.ready.clear()
    duplex = await start(session, bedrock)
    session.push_audio(frame())
    await asyncio.sleep(0.1)
    assert "audioInput" not in duplex.kinds()
    assert not any(e.get("contentStart", {}).get("type") == "AUDIO" for e in duplex.sent)

    bedrock.ready.set()
    await duplex.wait_for("audioInput")
    duplex.text_gate = asyncio.Event()
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="hello")
    await session.update_chat_ctx(ctx)
    await duplex.wait_until(
        lambda sent: sent[-1].get("contentStart", {}).get("interactive") is True
    )
    session.push_audio(frame(2))
    await asyncio.sleep(0.05)
    duplex.text_gate.set()
    await duplex.wait_until(lambda sent: sent[-1].get("audioInput") is not None)
    kinds = duplex.kinds()
    text_start = max(i for i, e in enumerate(duplex.sent) if "contentStart" in e)
    assert kinds[text_start : text_start + 4] == [
        "contentStart",
        "textInput",
        "contentEnd",
        "audioInput",
    ]


async def test_audio_reaches_the_new_stream_after_a_session_recycle(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """After a graceful recycle user audio flows to the replacement stream; https://github.com/livekit/agents/issues/5603"""

    @llm.function_tool
    async def first() -> str:
        """First tool."""
        return ""

    @llm.function_tool
    async def second() -> str:
        """Second tool."""
        return ""

    await start(session, bedrock, tools=[first])
    old = bedrock.duplexes[0]
    await old.wait_until(
        lambda sent: any(e.get("contentStart", {}).get("type") == "AUDIO" for e in sent)
    )
    await session.update_tools([second])
    new = await bedrock.stream(1)
    await new.wait_until(
        lambda sent: any(e.get("contentStart", {}).get("type") == "AUDIO" for e in sent)
    )
    session.push_audio(frame(3))
    await new.wait_for("audioInput")
    audio = [e["audioInput"]["content"] for e in new.sent if "audioInput" in e]
    assert base64.b64decode(audio[0]) == frame(3).data.tobytes()


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(lambda: ModelStreamErrorException("unexpected error"), id="model-stream"),
        pytest.param(InvalidEventBytes, id="invalid-event-bytes"),
    ],
)
async def test_transient_stream_faults_restart_the_session(
    session: sonic.RealtimeSession, bedrock: Bedrock, fault: Callable[[], Exception]
) -> None:
    """A transient Bedrock stream fault reopens the stream, which stays live.

    https://github.com/livekit/agents/issues/3646
    https://github.com/livekit/agents/issues/4530 https://github.com/livekit/agents/issues/6793
    """
    errors: list[llm.RealtimeModelError] = []
    transcripts: list[llm.InputTranscriptionCompleted] = []
    session.on("error", errors.append)
    session.on("input_audio_transcription_completed", transcripts.append)
    await start(session, bedrock)
    bedrock.duplexes[0].inbox.put_nowait(fault())
    restarted = await bedrock.stream(1)
    await restarted.wait_for("promptStart")
    await bedrock.answer(contentStart={"contentId": "u1", "type": "TEXT", "role": "USER"})
    await bedrock.answer(textOutput={"contentId": "u1", "role": "USER", "content": "still here"})
    await bedrock.answer(
        contentStart={
            "contentId": "a1",
            "type": "TEXT",
            "role": "ASSISTANT",
            "additionalModelFields": SPECULATIVE,
        }
    )
    assert [t.transcript for t in transcripts if t.is_final] == ["still here"]
    assert not errors


async def test_a_system_instability_reopens_the_stream_with_one_receive_loop(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """Bedrock's retryable "System instability" validation error reopens the stream, which one
    receive loop then reads, as after any other transient fault."""
    duplex = await start(session, bedrock)
    duplex.inbox.put_nowait(
        ValidationException("System instability detected. Please retry your request.")
    )
    restarted = await bedrock.stream(1)
    await restarted.wait_for("promptStart")
    await bedrock.answer(contentStart={"contentId": "u1", "type": "TEXT", "role": "USER"})

    receive_loops = [
        t
        for t in asyncio.all_tasks()
        if t.get_name() == "RealtimeSession._process_responses" and not t.done()
    ]
    for task in receive_loops[1:]:  # stop the extra loop so teardown reports this assertion
        task.cancel()
    assert len(receive_loops) == 1


async def test_tool_result_is_delivered_while_post_tool_speech_streams(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """A tool result is sent while Sonic is still speaking after the call; https://github.com/livekit/agents/issues/4887"""
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)
    duplex = await start(session, bedrock)
    for event in tool_use("call-1", "{}"):
        await bedrock.answer(**event)
    assert [call.call_id async for call in generations[0].function_stream] == ["call-1"]
    await bedrock.answer(
        contentStart={
            "contentId": "a1",
            "type": "TEXT",
            "role": "ASSISTANT",
            "additionalModelFields": SPECULATIVE,
        }
    )
    await bedrock.answer(
        textOutput={"contentId": "a1", "role": "ASSISTANT", "content": "Checking now."}
    )
    await session.update_chat_ctx(tool_output("call-1", '{"found": true}'))
    await duplex.wait_for("toolResult")
    result = next(e["toolResult"] for e in duplex.sent if "toolResult" in e)
    assert json.loads(result["content"]) == {"found": True}


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/7052 reproduces: aclose leaves a "
    "pending user-text send running",
)
async def test_close_cancels_a_pending_user_text_send(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """Closing the session cancels user text still waiting for the stream; https://github.com/livekit/agents/pull/7052"""
    bedrock.ready.clear()
    await start(session, bedrock)
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="must not be sent after close")
    await session.update_chat_ctx(ctx)
    await session.aclose()
    pending = [t.get_name() for t in asyncio.all_tasks() if not t.done()]
    assert "RealtimeSession._send_user_text" not in pending


async def test_user_transcript_is_interim_then_final_once_the_assistant_answers(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """The user's text is an interim transcript, made final when the assistant starts."""
    transcripts: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", transcripts.append)
    await start(session, bedrock)

    await bedrock.answer(textOutput={"role": "USER", "content": "hello", "contentId": "u1"})
    await bedrock.answer(
        contentStart={
            "contentId": "a1",
            "type": "TEXT",
            "role": "ASSISTANT",
            "additionalModelFields": SPECULATIVE,
        }
    )

    assert [(t.transcript, t.is_final) for t in transcripts] == [("hello", False), ("hello", True)]


@pytest.mark.parametrize(
    ("model", "nested"),
    [
        ("amazon.nova-sonic-v1:0", False),
        ("us.amazon.nova-sonic-v1:0", False),
        ("amazon.nova-2-sonic-v1:0", True),
        ("us.amazon.nova-2-sonic-v1:0", True),
        ("amazon.nova-2-5-sonic", True),
        ("us.amazon.nova-2-5-sonic", True),
    ],
)
async def test_turn_detection_is_nested_for_nova_2_and_later(
    bedrock: Bedrock, model: str, nested: bool
) -> None:
    """Nova Sonic 2+ gets endpointing under ``turnDetectionConfiguration``, Sonic 1 flat.

    https://github.com/livekit/agents/issues/6200
    """
    realtime_model = sonic.RealtimeModel(model=model, region="us-east-1", turn_detection="HIGH")
    session = realtime_model.session()
    try:
        duplex = await start(session, bedrock)
    finally:
        await session.aclose()
        await realtime_model.aclose()

    sent = duplex.sent[0]["sessionStart"]
    if nested:
        assert sent["turnDetectionConfiguration"] == {"endpointingSensitivity": "HIGH"}
        assert "endpointingSensitivity" not in sent
    else:
        assert sent["endpointingSensitivity"] == "HIGH"
        assert "turnDetectionConfiguration" not in sent


async def test_credentials_refresh_before_they_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nova Sonic signs with fresh credentials once the cached ones near expiry.

    https://github.com/livekit/agents/issues/5484
    """
    now = [datetime(2030, 1, 1, tzinfo=UTC)]
    credentials = RefreshableCredentials(
        "old-key",
        "old-secret",
        "old-token",
        datetime(2030, 1, 1, 1, tzinfo=UTC),
        refresh_using=lambda: {
            "access_key": "new-key",
            "secret_key": "new-secret",
            "token": "new-token",
            "expiry_time": "2030-01-01T03:00:00+00:00",
        },
        method="assume-role",
        time_fetcher=lambda: now[0],
    )
    monkeypatch.setattr(
        boto3, "Session", lambda: SimpleNamespace(get_credentials=lambda: credentials)
    )
    monkeypatch.setattr(sonic, "time", SimpleNamespace(time=lambda: now[0].timestamp()))
    resolver = sonic.Boto3CredentialsResolver()

    first = await resolver.get_identity()
    now[0] = datetime(2030, 1, 1, 0, 30, tzinfo=UTC)
    cached = await resolver.get_identity()
    now[0] = datetime(2030, 1, 1, 0, 58, tzinfo=UTC)
    fresh = await resolver.get_identity()

    assert (first.access_key_id, first.session_token) == ("old-key", "old-token")
    assert cached.access_key_id == "old-key"
    assert (fresh.access_key_id, fresh.secret_access_key, fresh.session_token) == (
        "new-key",
        "new-secret",
        "new-token",
    )


@pytest.mark.xfail(
    strict=True,
    reason="after a fatal stream error aclose() returns at once, leaving the audio input and "
    "session recycle tasks running",
)
async def test_close_after_a_fatal_error_stops_every_session_task(
    session: sonic.RealtimeSession, bedrock: Bedrock
) -> None:
    """Closing a session that a fatal Bedrock error ended still stops all of its tasks."""
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    before = asyncio.all_tasks()
    duplex = await start(session, bedrock)
    duplex.inbox.put_nowait(RuntimeError("fatal"))
    await until(lambda: errors, "the fatal error is reported")

    await session.aclose()

    running = [t for t in asyncio.all_tasks() - before if not t.done()]
    for task in running:  # stop them so the leak check reports this assertion, not teardown
        task.cancel()
    await asyncio.gather(*running, return_exceptions=True)
    assert not errors[0].recoverable
    assert [t.get_name() for t in running] == []
